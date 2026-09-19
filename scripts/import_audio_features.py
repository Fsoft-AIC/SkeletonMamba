"""Import EDGE-style slice features after numerical motion matching.

Requires a unique manifest motion_match and NPYs named after the matched source
motion. The caller supplies the extractor identity, feature rate, delay, and
evidence of shared timing. Alignment is relative to the matched slice unless
full-song timing has been independently verified.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dataset.manifest import resolve_path, sha256_file
from scripts.feature_utils import (arrays_hash, finish_manifest, interpolate_features,
    mark_prepared, preparation_records, safe_name, save_feature_cache, target_timestamps,
    validate_timestamps)


def import_audio_features(manifest_path, source_dir, output_dir, output_manifest, *,
                          feature_fps, feature_delay_seconds, extractor_id,
                          timing_evidence, feature_type="jukebox", overwrite=False):
    if feature_type not in ("jukebox", "baseline35"):
        raise ValueError("feature_type must be jukebox or baseline35")
    if (not np.isfinite([feature_fps, feature_delay_seconds]).all()
            or feature_fps <= 0 or feature_delay_seconds < 0):
        raise ValueError("feature fps must be positive and delay finite/nonnegative")
    if not isinstance(extractor_id, str) or not extractor_id.strip():
        raise ValueError("explicit extractor identity is required for metadata-free source NPYs")
    if not isinstance(timing_evidence, str) or not timing_evidence.strip():
        raise ValueError("explicit evidence of paired motion/feature timing is required")
    width = 4800 if feature_type == "jukebox" else 35
    records = preparation_records(manifest_path, output_manifest)
    extractor = dict(name=extractor_id, identity_status="caller_declared_archive_cache",
        feature_type=feature_type, feature_fps=float(feature_fps),
        feature_delay_seconds=float(feature_delay_seconds), paired_timing_evidence=timing_evidence)
    for record in records:
        sequence = record["sequence_id"]
        match = record.get("motion_match", {})
        if match.get("unique_match") is not True:
            raise ValueError(f"{sequence}: a unique numerical motion match is required before cache import")
        motion_source = resolve_path(match["source_path"], manifest_path)
        if sha256_file(motion_source) != match.get("source_sha256"):
            raise ValueError(f"{sequence}: matched source motion checksum differs from verified evidence")
        stem = safe_name(match["source_stem"])
        if motion_source.stem != stem:
            raise ValueError("matched motion path and source_stem disagree")
        clip_times = target_timestamps(record)
        requested = validate_timestamps(match["source_frame_times_seconds"], len(clip_times))
        indices = np.asarray(match["source_frame_indices"])
        fps = float(match["source_fps"])
        if (not np.isfinite(fps) or fps <= 0 or indices.shape != requested.shape
                or not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0)
                or not np.allclose(requested, indices / fps, atol=1e-6, rtol=0)):
            raise ValueError("matched source indices, fps and source timestamps disagree")
        source = Path(source_dir) / f"{stem}.npy"
        features = np.asarray(np.load(source, allow_pickle=False), dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != width or not np.isfinite(features).all():
            raise ValueError(f"{source}: expected finite features[T,{width}]")
        source_times = float(feature_delay_seconds) + np.arange(len(features), dtype=np.float64) / feature_fps
        aligned = interpolate_features(features, source_times, requested)
        source_hash = sha256_file(source)
        source_evidence = dict(motion_match=match, paired_timing_evidence=timing_evidence,
                               feature_source_sha256=source_hash)
        if "source_audio_timestamps_seconds" in record:
            absolute_times = validate_timestamps(record["source_audio_timestamps_seconds"], len(clip_times))
            offset = record.get("source_start_time_seconds")
            if (record.get("source_offset_status") != "verified" or not record.get("source_offset_evidence")
                    or not isinstance(offset, (int, float)) or isinstance(offset, bool)
                    or not np.isfinite(offset) or abs(absolute_times[0] - offset) > 1e-6
                    or not np.allclose(absolute_times - requested, absolute_times[0] - requested[0],
                                       atol=1e-6, rtol=0)):
                raise ValueError("absolute source-music timestamps require consistent verified origin evidence")
            record["audio_reference_kind"] = "full_source_music"
            record["audio_reference_id"] = record["music_id"]
        else:
            record["source_start_time_seconds"] = float(requested[0])
            record["source_offset_status"] = "clip_aligned_verified"
            record["source_offset_evidence"] = source_evidence
            record["audio_reference_kind"] = "matched_source_slice"
            record["audio_reference_id"] = stem
        fingerprint = dict(schema_version=1, kind="audio", sequence_id=sequence,
            source_sha256=source_hash, feature_type=feature_type, feature_dim=width,
            extractor=extractor, source_start_time_seconds=float(record["source_start_time_seconds"]),
            offset_evidence=record["source_offset_evidence"],
            audio_reference_kind=record["audio_reference_kind"], audio_reference_id=record["audio_reference_id"],
            matched_feature_evidence=source_evidence,
            source_timestamps_sha256=arrays_hash({"timestamps": requested}),
            target_timestamps_sha256=arrays_hash({"timestamps": clip_times}), interpolation="linear_no_extrapolation")
        metadata = dict(schema_version=1, kind="audio", sequence_id=sequence, music_id=record["music_id"],
            extractor=extractor, feature_type=feature_type, feature_dim=width,
            source_path=str(source.resolve()), alignment_verified=True,
            source_start_time_seconds=float(record["source_start_time_seconds"]),
            source_offset_evidence=record["source_offset_evidence"],
            audio_reference_kind=record["audio_reference_kind"], audio_reference_id=record["audio_reference_id"],
            timestamp_origin="clip_start")
        output = Path(output_dir) / f"{safe_name(sequence)}.npz"
        save_feature_cache(output, {"features": aligned, "timestamps": clip_times},
                           metadata, fingerprint, overwrite=overwrite)
        mark_prepared(record, "audio", output)
    return finish_manifest(output_manifest, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "source-dir", "output-dir", "output-manifest", "extractor-id", "timing-evidence"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--feature-fps", type=float, required=True)
    parser.add_argument("--feature-delay-seconds", type=float, required=True)
    parser.add_argument("--feature-type", choices=("jukebox", "baseline35"), default="jukebox")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    count = import_audio_features(args.manifest, args.source_dir, args.output_dir, args.output_manifest,
        feature_fps=args.feature_fps, feature_delay_seconds=args.feature_delay_seconds,
        extractor_id=args.extractor_id, timing_evidence=args.timing_evidence,
        feature_type=args.feature_type, overwrite=args.overwrite)
    print(f"Imported/verified {count} matched slice audio caches")


if __name__ == "__main__":
    main()
