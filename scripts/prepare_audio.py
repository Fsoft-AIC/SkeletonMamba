"""Import timestamped source-music features using verified clip offsets.

Source files are MUSIC_ID.npz with features, timestamps, and scalar JSON
metadata: schema_version=1, kind='audio', music_id, extractor, feature_type
('jukebox' or 'baseline35').
Optional offsets JSON maps sequence IDs to {source_start_time_seconds,evidence}.
Evidence must establish alignment to the source music.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np

from dataset.manifest import sha256_file
from scripts.feature_utils import (arrays_hash, finish_manifest, interpolate_features,
    load_feature_source, mark_prepared, preparation_records, safe_name,
    save_feature_cache, target_timestamps, validate_timestamps)


def prepare_audio(manifest_path, source_dir, output_dir, output_manifest, *,
                  offsets_path=None, feature_type="jukebox", overwrite=False):
    if feature_type not in ("jukebox", "baseline35"):
        raise ValueError("feature_type must be jukebox or baseline35")
    width = 4800 if feature_type == "jukebox" else 35
    records = preparation_records(manifest_path, output_manifest)
    offsets = json.loads(Path(offsets_path).read_text()) if offsets_path else {}
    offset_hash = sha256_file(offsets_path) if offsets_path else None
    loaded = OrderedDict()  # Bound memory when many full-length songs are imported.
    for record in records:
        sequence = record["sequence_id"]
        if sequence in offsets:
            entry = offsets[sequence]
            if not isinstance(entry, dict) or not entry.get("evidence"):
                raise ValueError("offset entries require explicit alignment evidence")
            record["source_start_time_seconds"] = entry["source_start_time_seconds"]
            record["source_offset_status"] = "verified"
            record["source_offset_evidence"] = {"description": entry["evidence"], "offsets_sha256": offset_hash}
        offset = record.get("source_start_time_seconds")
        if (record.get("source_offset_status") not in ("verified", "clip_aligned_verified")
                or not isinstance(offset, (int, float)) or isinstance(offset, bool)
                or not np.isfinite(offset) or offset < 0):
            raise ValueError(f"{sequence}: a verified source_start_time_seconds is required; no suffix inference")
        if not record.get("source_offset_evidence"):
            raise ValueError(f"{sequence}: source_offset_evidence is required for verified alignment")
        music_id = safe_name(record["music_id"])
        source = Path(source_dir) / f"{music_id}.npz"
        if music_id not in loaded:
            values, times, metadata = load_feature_source(source, kind="audio", width=width, music_id=music_id)
            if metadata.get("feature_type") != feature_type:
                raise ValueError(f"{source}: feature_type differs from requested {feature_type}")
            loaded[music_id] = (values, times, metadata, sha256_file(source))
            if len(loaded) > 2:
                loaded.popitem(last=False)
        loaded.move_to_end(music_id)
        values, source_times, metadata, source_hash = loaded[music_id]
        clip_times = target_timestamps(record)
        requested_times = validate_timestamps(record.get("source_audio_timestamps_seconds",
                                                       clip_times + float(offset)), len(clip_times))
        if abs(requested_times[0] - offset) > 1e-6:
            raise ValueError("source audio timestamp origin differs from verified source offset")
        if record.get("audio_reference_kind") == "matched_source_slice":
            raise ValueError("full-song imports require full-song offsets, not matched source-slice offsets")
        aligned = interpolate_features(values, source_times, requested_times)
        fingerprint = dict(schema_version=1, kind="audio", sequence_id=sequence,
                           source_sha256=source_hash, feature_type=feature_type,
                           feature_dim=width,
                           extractor=metadata["extractor"], source_start_time_seconds=float(offset),
                           target_timestamps_sha256=arrays_hash({"timestamps": clip_times}),
                           requested_source_timestamps_sha256=arrays_hash({"timestamps": requested_times}),
                           offset_evidence=record.get("source_offset_evidence"),
                           interpolation="linear_no_extrapolation")
        output = Path(output_dir) / f"{safe_name(sequence)}.npz"
        output_metadata = dict(kind="audio", sequence_id=sequence, music_id=music_id,
                               extractor=metadata["extractor"], feature_type=feature_type,
                               feature_dim=width, source_path=str(source.resolve()),
                               alignment_verified=True, source_start_time_seconds=float(offset),
                               source_offset_evidence=record["source_offset_evidence"],
                               timestamp_origin="clip_start")
        save_feature_cache(output, {"features": aligned, "timestamps": clip_times},
                           output_metadata, fingerprint, overwrite=overwrite)
        mark_prepared(record, "audio", output)
    return finish_manifest(output_manifest, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "source-dir", "output-dir", "output-manifest"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--offsets", help="JSON with verified offsets and evidence")
    parser.add_argument("--feature-type", choices=("jukebox", "baseline35"), default="jukebox")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    count = prepare_audio(args.manifest, args.source_dir, args.output_dir, args.output_manifest,
                          offsets_path=args.offsets, feature_type=args.feature_type, overwrite=args.overwrite)
    print(f"Prepared/verified {count} audio feature caches")


if __name__ == "__main__":
    main()
