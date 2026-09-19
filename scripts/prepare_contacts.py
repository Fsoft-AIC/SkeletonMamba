"""Prepare contact labels from denormalized ground-truth motion.

Supply the floor height and thresholds in metres and metres per second in the
motion's coordinate frame. The paper does not specify these thresholds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dataset.egoaistpp_dataset import load_prepared
from dataset.manifest import sha256_file
from dataset.motion_representation import load_geometry
from scripts.feature_utils import (arrays_hash, preparation_records, save_feature_cache,
                                   safe_name, mark_prepared, finish_manifest)

FOOT_INDICES = (7, 8, 10, 11)


def contact_labels(positions, fps, *, floor_height, height_threshold, speed_threshold, up_axis=2):
    """Per-frame labels: near floor AND both adjacent 3D speeds below threshold.

    Boundary frames use their sole adjacent transition; single-frame clips have
    no observable speed and receive no positive contact labels.
    """
    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (24, 3) or not len(values) or not np.isfinite(values).all():
        raise ValueError('positions must be finite [T,24,3] in metres')
    if (not all(np.isfinite(v) for v in (fps, floor_height, height_threshold, speed_threshold))
            or fps <= 0 or height_threshold < 0 or speed_threshold < 0 or up_axis not in (0, 1, 2)):
        raise ValueError('declare valid FPS, floor, height/speed thresholds and up axis')
    feet = values[:, FOOT_INDICES, :]
    if len(feet) < 2:
        return np.zeros((len(feet), 4), dtype=bool)
    speed = np.linalg.norm(np.diff(feet, axis=0), axis=-1) * fps
    incoming = np.concatenate((speed[:1], speed), axis=0)
    outgoing = np.concatenate((speed, speed[-1:]), axis=0)
    planted = np.maximum(incoming, outgoing) <= speed_threshold
    near_floor = np.abs(feet[..., up_axis] - floor_height) <= height_threshold
    return planted & near_floor


def prepare_contacts(manifest_path, geometry_path, output_dir, output_manifest, *,
                     floor_height, height_threshold, speed_threshold, coordinate_frame,
                     coordinate_evidence, overwrite=False):
    if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
        raise ValueError('declare the coordinate frame of the floor and motion')
    if not isinstance(coordinate_evidence, str) or not coordinate_evidence.strip():
        raise ValueError('provide evidence that motion coordinates/floor have been verified')
    _, _, geometry = load_geometry(geometry_path)
    geometry_hash = sha256_file(geometry_path)
    records = preparation_records(manifest_path, output_manifest)
    policy = dict(name='ground_truth_height_and_max_adjacent_3d_speed_v1',
                  floor_height_m=float(floor_height), height_threshold_m=float(height_threshold),
                  speed_threshold_m_s=float(speed_threshold), boundary='sole_adjacent_transition',
                  singleton='no_positive_contact_without_velocity')
    for record in records:
        source = record.get('prepared', {}).get('motion')
        if not source:
            raise ValueError('prepare denormalized motion before contacts')
        motion, timestamps, motion_meta = load_prepared(source, 'motion', record['sequence_id'])
        fps = float(record['fps'])
        if len(motion) != record['num_frames'] or not np.allclose(timestamps, np.arange(len(motion)) / fps, atol=1e-5, rtol=0):
            raise ValueError('contact preparation requires verified uniform motion timestamps/FPS')
        if motion_meta.get('geometry_hash') != geometry_hash:
            raise ValueError('supplied geometry hash does not match prepared motion')
        if motion_meta.get('geometry') != geometry:
            raise ValueError('prepared motion geometry metadata differs from verified asset')
        canonicalization = motion_meta.get('canonicalization')
        if canonicalization not in ('none', 'head_translation'):
            raise ValueError('contact preparation requires a verified supported coordinate transform')
        labels = contact_labels(motion[..., :3].numpy(), fps, floor_height=floor_height,
                                height_threshold=height_threshold, speed_threshold=speed_threshold,
                                up_axis=('x', 'y', 'z').index(geometry['up_axis']))
        fingerprint = dict(schema_version=1, kind='contacts', sequence_id=record['sequence_id'],
                           source_motion_sha256=sha256_file(source), geometry_hash=geometry_hash,
                           canonicalization=canonicalization, units='m', up_axis=geometry['up_axis'],
                           coordinate_frame=coordinate_frame, coordinate_evidence=coordinate_evidence,
                           policy=policy, foot_indices=list(FOOT_INDICES), fps=fps,
                           timestamps_sha256=arrays_hash({'timestamps': timestamps}))
        output = Path(output_dir) / f'{safe_name(record["sequence_id"])}.npz'
        save_feature_cache(output, {'labels': labels, 'timestamps': timestamps}, fingerprint,
                           fingerprint, overwrite=overwrite)
        mark_prepared(record, 'contacts', output)
    return finish_manifest(output_manifest, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'geometry', 'output-dir', 'output-manifest', 'coordinate-frame', 'coordinate-evidence'):
        parser.add_argument(f'--{name}', required=True)
    for name in ('floor-height', 'height-threshold', 'speed-threshold'):
        parser.add_argument(f'--{name}', type=float, required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    count = prepare_contacts(args.manifest, args.geometry, args.output_dir, args.output_manifest,
                             floor_height=args.floor_height, height_threshold=args.height_threshold,
                             speed_threshold=args.speed_threshold, coordinate_frame=args.coordinate_frame,
                             coordinate_evidence=args.coordinate_evidence, overwrite=args.overwrite)
    print(f'Prepared/verified {count} contact caches')


if __name__ == '__main__':
    main()
