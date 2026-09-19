"""Feature-cache validation and manifest provenance helpers."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from dataset.manifest import load_manifest, resolve_path, sha256_file, write_manifest
from dataset.cache_integrity import arrays_hash
from dataset.motion_representation import atomic_save_npz, read_metadata


def safe_name(name):
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("cache identifiers must be nonempty path-free names")
    return name


def preparation_records(manifest_path, output_manifest):
    if Path(manifest_path).resolve() == Path(output_manifest).resolve():
        raise ValueError("write a new output manifest; preserve the input manifest")
    records = copy.deepcopy(load_manifest(manifest_path))
    for record in records:
        safe_name(record["sequence_id"])
        for field in ("raw", "prepared"):
            record[field] = {key: str(resolve_path(value, manifest_path).resolve())
                             for key, value in record.get(field, {}).items() if value is not None}
    return records


def validate_timestamps(timestamps, length, *, clip=False):
    times = np.asarray(timestamps, dtype=np.float64)
    if (times.shape != (length,) or length < 1 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("timestamps must be finite, strictly increasing and match array length")
    if times[0] < -1e-6 or (clip and abs(times[0]) > 1e-6):
        raise ValueError("timestamps must be nonnegative and clip timestamps must start at zero")
    return times


def target_timestamps(record):
    length, fps = int(record["num_frames"]), float(record["fps"])
    if length < 1 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("manifest needs positive num_frames and fps")
    path = record.get("prepared", {}).get("motion")
    if path and Path(path).is_file():
        with np.load(path, allow_pickle=False) as archive:
            metadata = read_metadata(archive)
            if metadata.get("sequence_id") != record["sequence_id"]:
                raise ValueError("prepared motion sequence metadata does not match")
            return validate_timestamps(archive["timestamps"], length, clip=True)
    return np.arange(length, dtype=np.float64) / fps


def load_feature_source(path, *, kind, width, sequence_id=None, music_id=None):
    if width < 1:
        raise ValueError("feature width must be positive")
    with np.load(path, allow_pickle=False) as archive:
        features = np.array(archive["features"], dtype=np.float32)
        times = np.array(archive["timestamps"], dtype=np.float64)
        metadata = read_metadata(archive)
    if features.ndim != 2 or features.shape[-1] != width or not np.isfinite(features).all():
        raise ValueError(f"{path}: expected finite features[T,{width}]")
    times = validate_timestamps(times, len(features), clip=kind == "video")
    if metadata.get("kind") != kind or not metadata.get("extractor"):
        raise ValueError(f"{path}: explicit kind and extractor identity are required")
    if sequence_id is not None and metadata.get("sequence_id") != sequence_id:
        raise ValueError(f"{path}: sequence_id mismatch")
    if music_id is not None and metadata.get("music_id") != music_id:
        raise ValueError(f"{path}: music_id mismatch")
    return features, times, metadata


def interpolate_features(features, source_times, target_times):
    """Linear timestamp interpolation; never extrapolate missing coverage."""
    source_times = validate_timestamps(source_times, len(features))
    target_times = np.asarray(target_times, dtype=np.float64)
    if (target_times.ndim != 1 or not len(target_times) or not np.isfinite(target_times).all()
            or np.any(np.diff(target_times) <= 0)):
        raise ValueError("target timestamps must be finite and increasing")
    if target_times[0] < source_times[0] - 1e-6 or target_times[-1] > source_times[-1] + 1e-6:
        raise ValueError("source features do not cover the requested clip timestamps; extrapolation is forbidden")
    if len(source_times) == 1:
        if not np.allclose(target_times, source_times[0], atol=1e-6, rtol=0):
            raise ValueError("single-frame source cannot cover this clip")
        return np.repeat(features, len(target_times), axis=0)
    target_times = np.clip(target_times, source_times[0], source_times[-1])
    upper = np.searchsorted(source_times, target_times, side="right").clip(1, len(source_times) - 1)
    lower = upper - 1
    fraction = ((target_times - source_times[lower]) / (source_times[upper] - source_times[lower]))[:, None]
    return (features[lower] * (1 - fraction) + features[upper] * fraction).astype(np.float32)


def verified_cache(path, fingerprint):
    """Verify a cache's provenance and content checksum before reuse."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = read_metadata(archive)
        arrays = {key: np.array(archive[key]) for key in archive.files if key != "metadata"}
    if metadata.get("fingerprint") != fingerprint:
        raise ValueError(f"stale cache at {path}; inspect provenance or use --overwrite deliberately")
    for key in ("schema_version", "sequence_id", "kind", "extractor", "feature_dim",
                "source_kind", "coordinate_frame", "calibration_id", "units",
                "source_start_time_seconds", "source_offset_evidence", "audio_reference_kind",
                "audio_reference_id"):
        if key in fingerprint and metadata.get(key) != fingerprint[key]:
            raise ValueError(f"cache metadata {key} mismatch: {path}")
    if fingerprint.get("kind") == "audio" and (
        metadata.get("alignment_verified") is not True
        or metadata.get("source_start_time_seconds") != fingerprint["source_start_time_seconds"]
        or metadata.get("source_offset_evidence") != fingerprint["offset_evidence"]
    ):
        raise ValueError(f"cache audio alignment metadata mismatch: {path}")
    if metadata.get("content_sha256") != arrays_hash(arrays):
        raise ValueError(f"cache content checksum mismatch: {path}")
    for array in arrays.values():
        if not np.isfinite(array).all():
            raise ValueError(f"nonfinite cached values: {path}")
    return metadata


def save_feature_cache(path, arrays, metadata, fingerprint, *, overwrite=False):
    path = Path(path)
    if path.is_file() and not overwrite:
        verified_cache(path, fingerprint)
        return
    metadata = dict(metadata, schema_version=1, fingerprint=fingerprint,
                    content_sha256=arrays_hash(arrays))
    atomic_save_npz(path, **arrays, metadata=np.array(json.dumps(metadata, sort_keys=True)))


def mark_prepared(record, kind, output):
    record.setdefault("prepared", {})[kind] = str(Path(output).resolve())
    record.setdefault("prepared_sha256", {})[kind] = sha256_file(output)


def finish_manifest(output_manifest, records):
    write_manifest(output_manifest, records)
    return len(records)
