"""Standard-library data provenance helpers, usable before Torch installation."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import os
import re
import tempfile
import zipfile

SCHEMA_VERSION = 1


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def parse_sequence_id(name):
    parts = name.split('_')
    if len(parts) < 8 or not re.fullmatch(r'\d+', parts[-1]):
        raise ValueError(f'Invalid sequence name: {name}')
    prefixes = ('g', 's', 'c', 'd', 'm', 'ch')
    if not all(value.startswith(prefix) and len(value) > len(prefix)
               for value, prefix in zip(parts[:6], prefixes)):
        raise ValueError(f'Invalid AIST motion identifier: {name}')
    return dict(source_motion_id='_'.join(parts[:6]), music_id=parts[4],
                scene_id='_'.join(parts[6:-1]), segment_id=int(parts[-1]),
                candidate_choreography_id='_'.join(parts[i] for i in (0, 1, 4, 5)))


def resolve_path(path, manifest_path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path(manifest_path).resolve().parent / path


def load_manifest(path):
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    seen = set()
    for r in records:
        if r.get('schema_version') != SCHEMA_VERSION:
            raise ValueError(f'Unsupported manifest schema: {r.get("schema_version")}')
        if r['sequence_id'] in seen:
            raise ValueError(f'Duplicate sequence ID in manifest: {r["sequence_id"]}')
        seen.add(r['sequence_id'])
    return records


def atomic_json(path, value):
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True)+'\n')


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_manifest(path, records):
    _atomic_text(path, ''.join(json.dumps(r, sort_keys=True)+'\n' for r in records))


def npz_headers(path):
    """Read shape/dtype metadata without importing NumPy or unpickling objects."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        for entry in archive.namelist():
            if not entry.endswith('.npy'):
                continue
            with archive.open(entry) as f:
                magic = f.read(8)
                if len(magic) != 8 or magic[:6] != b'\x93NUMPY' or magic[6] not in (1, 2, 3):
                    raise ValueError(f'Invalid NPY header: {entry}')
                n = int.from_bytes(f.read(2 if magic[6] == 1 else 4), 'little')
                if n > 1024*1024:
                    raise ValueError('Excessive NPY header size')
                header = ast.literal_eval(f.read(n).decode('utf-8' if magic[6] == 3 else 'latin1'))
                if 'O' in str(header['descr']):
                    raise ValueError('Object arrays are not accepted')
                result[entry[:-4]] = header
    return result


def audit_records(root, fps, hash_files=False):
    root = Path(root).resolve()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('FPS must be positive and explicitly declared')
    split_ids = {split: set((root/'split'/f'{split}.txt').read_text().split())
                 for split in ('train', 'test')}
    if split_ids['train'] & split_ids['test']:
        raise ValueError('Official train/test motion IDs overlap')
    records, issues = [], []
    for tree in ('train', 'test'):
        for folder in sorted((root/tree).iterdir()):
            if not folder.is_dir():
                continue
            try:
                parsed = parse_sequence_id(folder.name)
                motion, video = folder/'motion.npz', folder/'egocentric.mp4'
                header = npz_headers(motion)
                body_keys = [k for k in ('pose_body', 'body_pose') if k in header]
                if len(body_keys) != 1:
                    raise ValueError('Expected exactly one body-pose alias in raw archive')
                key = body_keys[0]
                n = header[key]['shape'][0]
                if n <= 0 or header[key]['shape'] not in ((n,69), (n,23,3)):
                    raise ValueError('Invalid body-pose dimensions')
                if any(header[k]['shape'] != (n,3) for k in ('trans','root_orient')):
                    raise ValueError('Root and body dimensions disagree')
                flags = []
                if not video.is_file():
                    flags.append('missing_video')
                membership = tree if parsed['source_motion_id'] in split_ids[tree] else 'auxiliary'
                if membership == 'auxiliary':
                    flags.append('outside_official_directory_split')
                if n < 150:
                    flags.append('shorter_than_reference_window')
                records.append(dict(schema_version=SCHEMA_VERSION, sequence_id=folder.name,
                    **parsed, split=membership, raw_tree=tree,
                    raw={'motion':str(motion),'video':str(video)}, num_frames=n,
                    fps=float(fps), timing_status='declared_fps_not_yet_video_verified',
                    source_start_time_seconds=None, source_offset_status='unverified',
                    body_pose_key=key, quality_flags=flags,
                    raw_motion_sha256=sha256_file(motion) if hash_files else None,
                    prepared={}))
            except (ValueError, KeyError, OSError, zipfile.BadZipFile, SyntaxError) as e:
                issues.append({'raw_tree':tree,'sequence_id':folder.name,'reason':str(e)})
    return records, issues
