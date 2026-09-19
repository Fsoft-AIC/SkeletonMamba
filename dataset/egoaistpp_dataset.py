"""Synchronized prepared EgoAIST++ clips for training and evaluation.

Standalone inference loads conditions separately to avoid target-motion access.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.manifest import load_manifest, resolve_path, sha256_file
from dataset.cache_integrity import validate_content_checksum
from dataset.motion_representation import MotionNormalizer, REPRESENTATION, read_metadata


def feature_identity(metadata, kind):
    """Only model-wide extractor semantics; never sequence-specific source hashes."""
    extractor = metadata.get('extractor')
    feature_type = metadata.get('feature_type')
    if kind not in ('audio', 'video') or metadata.get('kind') != kind:
        raise ValueError(f'Invalid {kind} feature metadata')
    if not isinstance(extractor, (str, dict)) or not extractor:
        raise ValueError(f'{kind} features require a nonempty extractor identity')
    if feature_type is not None and (not isinstance(feature_type, str) or not feature_type):
        raise ValueError(f'{kind} feature_type must be a nonempty string when supplied')
    # Older pooled-video and explicitly synthetic caches have no feature_type.
    # Absence is itself part of the contract, never guessed from feature width.
    return {'extractor': copy.deepcopy(extractor), 'feature_type': feature_type}


def validate_feature_contract(metadata, kind, expected_feature_contract):
    if not isinstance(expected_feature_contract, dict) or kind not in expected_feature_contract:
        raise ValueError(f'Missing checkpoint/training {kind} feature extractor contract')
    if feature_identity(metadata, kind) != expected_feature_contract[kind]:
        raise ValueError(f'{kind} feature extractor/type differs from the training feature contract')


def infer_feature_contract(record, manifest_path):
    """Read small metadata entries from the first clip without materializing features."""
    result = {}
    for kind in ('audio', 'video'):
        prepared = record.get('prepared', {}).get(kind)
        if not prepared:
            raise FileNotFoundError(f'{record["sequence_id"]}: no prepared {kind}; run preprocessing')
        path = resolve_path(prepared, manifest_path)
        if not path.is_file():
            raise FileNotFoundError(f'Missing prepared {kind}: {path}')
        validate_prepared_file_checksum(path, record, kind)
        with np.load(path, allow_pickle=False) as archive:
            metadata = read_metadata(archive)
        if metadata.get('sequence_id') != record['sequence_id']:
            raise ValueError(f'{kind}: wrong sequence metadata')
        result[kind] = feature_identity(metadata, kind)
    return result


def load_prepared(path, kind, sequence_id):
    with np.load(path, allow_pickle=False) as data:
        metadata = read_metadata(data)
        key = 'motion' if kind == 'motion' else ('labels' if kind == 'contacts' else 'features')
        arrays = {name: np.array(data[name]) for name in data.files if name != 'metadata'}
    validate_content_checksum(arrays, metadata, path)
    values = arrays[key].astype(np.float32, copy=False)
    timestamps = arrays['timestamps'].astype(np.float64, copy=False)
    if metadata.get('sequence_id') != sequence_id:
        raise ValueError(f'{path}: wrong sequence metadata')
    if kind == 'motion':
        if metadata.get('representation') != REPRESENTATION or values.ndim != 3 or values.shape[1:] != (24,9):
            raise ValueError(f'{path}: incompatible motion representation')
    elif kind == 'contacts':
        if (metadata.get('kind') != 'contacts' or values.ndim != 2 or values.shape[1] != 4
                or not np.isin(values, [0, 1]).all() or not isinstance(metadata.get('policy'), dict)
                or not metadata['policy'] or metadata.get('foot_indices') != [7, 8, 10, 11]
                or metadata.get('units') != 'm' or metadata.get('up_axis') not in ('x', 'y', 'z')
                or not metadata.get('coordinate_frame') or not metadata.get('coordinate_evidence')):
            raise ValueError(f'{path}: incompatible contact labels/policy/coordinate metadata')
    elif metadata.get('kind') != kind or not metadata.get('extractor') or values.ndim != 2 or values.shape[1] < 1:
        raise ValueError(f'{path}: incompatible {kind} feature metadata/shape')
    if len(values) == 0 or timestamps.shape != (len(values),):
        raise ValueError(f'{path}: inconsistent feature/timestamp length')
    if not np.isfinite(values).all() or not np.isfinite(timestamps).all():
        raise ValueError(f'{path}: nonfinite prepared values')
    if abs(timestamps[0]) > 1e-4 or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f'{path}: expected increasing per-clip timestamps starting at zero')
    return torch.from_numpy(values.astype(bool) if kind == 'contacts' else values), timestamps, metadata


def validate_prepared_file_checksum(path, record, kind):
    """Bind each provided manifest digest to the exact artifact read at runtime."""
    hashes = record.get('prepared_sha256', {})
    if kind in hashes:
        expected = hashes[kind]
        if not isinstance(expected, str) or expected != sha256_file(path):
            raise ValueError(f'{path}: prepared {kind} file checksum differs from the manifest')


def validate_audio_alignment(record, metadata):
    """Verify source segment alignment in addition to local timestamps."""
    expected=record.get('source_start_time_seconds')
    actual=metadata.get('source_start_time_seconds')
    if record.get('source_offset_status') not in ('verified','clip_aligned_verified'):
        raise ValueError('Source music offset is unverified; supply alignment evidence before training')
    if metadata.get('alignment_verified') is not True:
        raise ValueError('Audio cache lacks verified alignment provenance')
    evidence=record.get('source_offset_evidence')
    if not isinstance(evidence,(str,dict)) or not evidence or metadata.get('source_offset_evidence')!=evidence:
        raise ValueError('Audio source-offset evidence is missing or differs from the manifest')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<0
           for v in (expected,actual)) or abs(expected-actual)>1e-6:
        raise ValueError('Audio source offset does not match the verified manifest offset')


class EgoAISTppDataset(Dataset):
    def __init__(self, manifest_path, statistics_path, window=150, training=False,
                 expected_audio_dim=None, expected_video_dim=None, expected_feature_contract=None):
        self.manifest_path = manifest_path
        self.records = load_manifest(manifest_path)
        if not self.records:
            raise ValueError('Manifest contains no usable sequences')
        if window < 1:
            raise ValueError('Window must be positive')
        if training and any(r['split'] != 'train' for r in self.records):
            raise ValueError('Training dataset may only consume the training split')
        self.normalizer = (statistics_path if isinstance(statistics_path, MotionNormalizer)
                           else MotionNormalizer.from_npz(statistics_path))
        self.training_motion_hashes = {}
        for entry in self.normalizer.metadata.get('prepared_motion_hashes', []):
            sequence, digest = entry.get('sequence_id'), entry.get('sha256')
            if not sequence or not isinstance(digest, str) or sequence in self.training_motion_hashes:
                raise ValueError('Invalid or duplicate normalization motion provenance')
            self.training_motion_hashes[sequence] = digest
        first_contract = infer_feature_contract(self.records[0], manifest_path)
        if expected_feature_contract is not None and first_contract != expected_feature_contract:
            raise ValueError('First sequence feature extractor/type differs from the training feature contract')
        self.expected_feature_contract = copy.deepcopy(
            first_contract if expected_feature_contract is None else expected_feature_contract)
        contact_presence = [bool(r.get('prepared', {}).get('contacts')) for r in self.records]
        if any(contact_presence) and not all(contact_presence):
            raise ValueError('Contact caches must be prepared for every sequence in a selected manifest')
        self.has_contacts = all(contact_presence)
        self.window, self.training = window, training
        self.expected_dims = {'audio':expected_audio_dim, 'video':expected_video_dim}
        # Evaluation enumerates deterministic, non-overlapping windows, including tails.
        self.index = [(i,start) for i,r in enumerate(self.records)
                      for start in ([None] if training else range(0,r['num_frames'],window))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self,index):
        record_idx,start = self.index[index]
        record = self.records[record_idx]
        values, times, metadata = {}, {}, {}
        kinds = ('motion', 'audio', 'video', 'contacts') if self.has_contacts else ('motion', 'audio', 'video')
        for kind in kinds:
            path = record.get('prepared',{}).get(kind)
            if not path:
                raise FileNotFoundError(f'{record["sequence_id"]}: no prepared {kind}; run preprocessing')
            path = resolve_path(path,self.manifest_path)
            if not path.is_file():
                raise FileNotFoundError(f'Missing prepared {kind}: {path}')
            validate_prepared_file_checksum(path, record, kind)
            values[kind],times[kind],metadata[kind] = load_prepared(path,kind,record['sequence_id'])
            if kind == 'motion' and record['split'] == 'train':
                expected_hash = self.training_motion_hashes.get(record['sequence_id'])
                if expected_hash is None:
                    if metadata[kind].get('synthetic') is not True:
                        raise ValueError('Training motion is absent from normalization provenance')
                elif expected_hash != sha256_file(path):
                    raise ValueError('Training motion/statistics checksum mismatch; regenerate normalization')
            if kind in ('audio', 'video'):
                validate_feature_contract(metadata[kind], kind, self.expected_feature_contract)
            if kind in self.expected_dims and self.expected_dims[kind] is not None and values[kind].shape[-1] != self.expected_dims[kind]:
                raise ValueError(f'{kind} feature width does not match model configuration')
        n = len(values['motion'])
        if n != record['num_frames']:
            raise ValueError('Prepared motion length differs from manifest')
        for kind in kinds[1:]:
            if times[kind].shape != times['motion'].shape or not np.allclose(times[kind],times['motion'],atol=1e-4,rtol=0):
                raise ValueError(f'{record["sequence_id"]}: {kind} timestamps are not synchronized')
        validate_audio_alignment(record,metadata['audio'])
        if record.get('fps') is not None and not np.allclose(times['motion'],np.arange(n)/record['fps'],atol=1e-4,rtol=0):
            raise ValueError('Prepared timestamps disagree with the declared motion rate')
        for key in ('geometry_hash','canonicalization'):
            expected = self.normalizer.metadata.get(key)
            if expected is None or metadata['motion'].get(key) != expected:
                raise ValueError(f'Motion/statistics {key} metadata mismatch')
        if self.has_contacts:
            contact_meta = metadata['contacts']
            for key in ('geometry_hash', 'canonicalization'):
                if contact_meta.get(key) != metadata['motion'].get(key):
                    raise ValueError(f'Contact/motion {key} metadata mismatch')
            geometry = metadata['motion'].get('geometry', {})
            if geometry.get('units') != 'm' or geometry.get('up_axis') != contact_meta['up_axis']:
                raise ValueError('Contact labels require matching verified motion units/up axis')
            motion_path = resolve_path(record['prepared']['motion'], self.manifest_path)
            if contact_meta.get('source_motion_sha256') != sha256_file(motion_path):
                raise ValueError('Contact labels are stale: prepared motion changed')
            contact_path = resolve_path(record['prepared']['contacts'], self.manifest_path)
            if record.get('prepared_sha256', {}).get('contacts') != sha256_file(contact_path):
                raise ValueError('Contact cache checksum is missing or differs from the manifest')
        if start is None:
            start = int(torch.randint(max(n-self.window+1,1),(1,)).item())
        length = min(self.window,n-start)
        result = {}
        for kind,value in values.items():
            value = value[start:start+length]
            if kind == 'motion':
                value = self.normalizer.normalize(value)
            output = torch.zeros((self.window,)+value.shape[1:],dtype=value.dtype)
            output[:length] = value
            result[kind] = output
        result.update(mask=torch.arange(self.window)<length, length=length,
                      sequence_id=record['sequence_id'],start_frame=start)
        return result
