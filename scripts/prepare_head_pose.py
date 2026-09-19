"""Validate and import calibrated head estimates.

Each SEQUENCE_ID.npz must contain positions[T,3], rotations[T,3,3] or [T,6],
confidence[T], timestamps[T], and metadata with schema_version=1,
sequence_id, source_kind='estimated', coordinate_frame, calibration_id and
extractor, units='m', and rotation6d_convention='first_two_matrix_rows' for 6D
rotations. Inputs must already match the clip timestamps and coordinate frame;
ground-truth trajectories are rejected.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dataset.manifest import sha256_file
from dataset.motion_representation import read_metadata
from scripts.feature_utils import (arrays_hash, finish_manifest, mark_prepared,
    preparation_records, safe_name, save_feature_cache, target_timestamps, validate_timestamps)
from utils.kinematics import rotation_6d_to_matrix


def load_estimated_head(path, sequence_id, coordinate_frame):
    with np.load(path, allow_pickle=False) as archive:
        metadata = read_metadata(archive)
        positions = np.array(archive["positions"], dtype=np.float32)
        rotations = np.array(archive["rotations"], dtype=np.float32)
        confidence = np.array(archive["confidence"], dtype=np.float32)
        times = np.array(archive["timestamps"], dtype=np.float64)
    if metadata.get("source_kind") != "estimated":
        raise ValueError("head inputs must have source_kind='estimated'; ground-truth/oracle input is forbidden")
    if metadata.get("sequence_id") != sequence_id:
        raise ValueError("head estimate sequence_id mismatch")
    if metadata.get("coordinate_frame") != coordinate_frame or not coordinate_frame:
        raise ValueError("head estimate coordinate_frame differs from explicitly selected model frame")
    if not metadata.get("calibration_id") or not metadata.get("extractor"):
        raise ValueError("head estimate needs calibration_id and extractor identity")
    if metadata.get("units") != "m":
        raise ValueError("head estimates must explicitly declare units='m'")
    length = len(positions)
    if positions.shape != (length, 3) or confidence.shape != (length,):
        raise ValueError("head positions must be [T,3] and confidence [T]")
    if not all(np.isfinite(v).all() for v in (positions, rotations, confidence)):
        raise ValueError("head arrays contain nonfinite values")
    if np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("head confidence must lie in [0,1]")
    if rotations.shape == (length, 6):
        if metadata.get("rotation6d_convention") != "first_two_matrix_rows":
            raise ValueError("6D head rotations require the explicit first_two_matrix_rows convention")
        if (np.linalg.norm(rotations[:, :3], axis=-1) < 1e-6).any() or (
            np.linalg.norm(np.cross(rotations[:, :3], rotations[:, 3:]), axis=-1) < 1e-6
        ).any():
            raise ValueError("degenerate head rotation6d input")
        rotations = rotation_6d_to_matrix(torch.from_numpy(rotations)).numpy()
    if rotations.shape != (length, 3, 3):
        raise ValueError("head rotations must be [T,3,3] or [T,6]")
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=2e-4, rtol=0) or not np.allclose(
        np.linalg.det(rotations), 1, atol=2e-4, rtol=0
    ):
        raise ValueError("head rotations must be proper orthonormal rotation matrices")
    return dict(positions=positions, rotations=rotations, confidence=confidence,
                timestamps=validate_timestamps(times, length, clip=True)), metadata


def prepare_head_pose(manifest_path, source_dir, output_dir, output_manifest, *,
                      coordinate_frame, overwrite=False):
    records = preparation_records(manifest_path, output_manifest)
    for record in records:
        sequence = record["sequence_id"]
        source = Path(source_dir) / f"{safe_name(sequence)}.npz"
        arrays, metadata = load_estimated_head(source, sequence, coordinate_frame)
        target_times = target_timestamps(record)
        if arrays["timestamps"].shape != target_times.shape or not np.allclose(
            arrays["timestamps"], target_times, atol=1e-4, rtol=0
        ):
            raise ValueError("head estimates must already align with clip timestamps; resample in the external estimator")
        arrays["timestamps"] = target_times
        fingerprint = dict(schema_version=1, kind="head_pose", sequence_id=sequence,
                           source_sha256=sha256_file(source), extractor=metadata["extractor"],
                           source_kind="estimated", units="m",
                           calibration_id=metadata["calibration_id"], coordinate_frame=coordinate_frame,
                           target_timestamps_sha256=arrays_hash({"timestamps": target_times}))
        output = Path(output_dir) / f"{sequence}.npz"
        output_metadata = dict(kind="head_pose", sequence_id=sequence, source_kind="estimated",
                               extractor=metadata["extractor"], calibration_id=metadata["calibration_id"],
                               coordinate_frame=coordinate_frame, timestamp_origin="clip_start",
                               units="m",
                               rotation6d_convention="first_two_matrix_rows", source_path=str(source.resolve()))
        save_feature_cache(output, arrays, output_metadata, fingerprint, overwrite=overwrite)
        mark_prepared(record, "head_pose", output)
    return finish_manifest(output_manifest, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "source-dir", "output-dir", "output-manifest", "coordinate-frame"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    count = prepare_head_pose(args.manifest, args.source_dir, args.output_dir, args.output_manifest,
                              coordinate_frame=args.coordinate_frame, overwrite=args.overwrite)
    print(f"Prepared/verified {count} estimated head trajectories")


if __name__ == "__main__":
    main()
