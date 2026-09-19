"""Import clip features or extract frozen ResNet50 pooled features.

Extraction requires ffprobe, OpenCV, torchvision and either local ResNet50
weights or --download-weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from urllib.parse import urlparse

import numpy as np

from dataset.manifest import sha256_file
from scripts.feature_utils import (arrays_hash, finish_manifest, interpolate_features,
    load_feature_source, mark_prepared, preparation_records, safe_name,
    save_feature_cache, target_timestamps, validate_timestamps, verified_cache)


def probe_frame_times(video_path):
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
               "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(video_path)]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("video extraction requires ffprobe on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"ffprobe failed for {video_path}: {exc.stderr.strip()}") from exc
    frames = json.loads(result.stdout).get("frames", [])
    if not frames or any("best_effort_timestamp_time" not in frame for frame in frames):
        raise ValueError("ffprobe did not provide a timestamp for every decoded video frame")
    times = np.array([float(frame["best_effort_timestamp_time"]) for frame in frames])
    start = float(times[0])
    return validate_timestamps(times - start, len(times), clip=True), start


def load_resnet50(*, weights_path=None, download_weights=False, device="cpu"):
    if bool(weights_path) == bool(download_weights):
        raise ValueError("choose exactly one of local weights or explicitly authorized weight download")
    try:
        import torch
        from torchvision.models import ResNet50_Weights, resnet50
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("video extraction needs a Torch-compatible torchvision installation") from exc
    weights = ResNet50_Weights.IMAGENET1K_V2
    if weights_path:
        model = resnet50(weights=None)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        identity = {"architecture": "resnet50", "weights_sha256": sha256_file(weights_path),
                    "preprocessing": "IMAGENET1K_V2", "output": "avgpool_2048"}
    else:
        model = resnet50(weights=weights)
        identity = {"architecture": "resnet50", "weights": "IMAGENET1K_V2",
                    "weights_url": weights.url, "preprocessing": "IMAGENET1K_V2", "output": "avgpool_2048"}
        downloaded = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
        identity["weights_sha256"] = sha256_file(downloaded)
    model.fc = torch.nn.Identity()
    model.eval().requires_grad_(False).to(device)
    return model, weights.transforms(), identity


def extract_video_features(video_path, model, transform, *, device="cpu", batch_size=32):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    try:
        import cv2
        import torch
    except ImportError as exc:
        raise RuntimeError("video extraction requires OpenCV and Torch") from exc
    times, pts_origin = probe_frame_times(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot decode video: {video_path}")
    batch, chunks, count = [], [], 0

    def flush():
        nonlocal batch
        if batch:
            with torch.inference_mode():
                result = model(torch.stack(batch).to(device))
            chunks.append(result.float().cpu().numpy())
            batch = []

    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
            batch.append(transform(tensor))
            count += 1
            if len(batch) == batch_size:
                flush()
        flush()
    finally:
        capture.release()
    if count != len(times) or not chunks:
        raise ValueError(f"decoded frame count {count} differs from ffprobe count {len(times)}")
    features = np.concatenate(chunks)
    if features.shape != (count, 2048) or not np.isfinite(features).all():
        raise ValueError("ResNet50 must produce finite pooled features[T,2048]")
    return features, times, pts_origin


def prepare_video(manifest_path, output_dir, output_manifest, *, source_dir=None,
                  extract=False, weights_path=None, download_weights=False,
                  device="cpu", batch_size=32, feature_dim=2048, overwrite=False):
    if bool(source_dir) == bool(extract):
        raise ValueError("choose precomputed --source-dir or explicit --extract")
    if not extract and (weights_path or download_weights):
        raise ValueError("weight options only apply to --extract")
    if extract and feature_dim != 2048:
        raise ValueError("ResNet50 pooled output has exactly 2048 features")
    records = preparation_records(manifest_path, output_manifest)
    model_info = None
    for record in records:
        sequence = record["sequence_id"]
        target_times = target_timestamps(record)
        output = Path(output_dir) / f"{safe_name(sequence)}.npz"
        if source_dir:
            source = Path(source_dir) / f"{sequence}.npz"
            features, times, metadata = load_feature_source(
                source, kind="video", width=feature_dim, sequence_id=sequence)
            extractor, source_hash = metadata["extractor"], sha256_file(source)
            feature_type = metadata.get("feature_type")
            if feature_type is not None and (not isinstance(feature_type, str) or not feature_type):
                raise ValueError("video feature_type must be a nonempty string when supplied")
            fingerprint = dict(schema_version=1, kind="video", sequence_id=sequence,
                               extractor=extractor, feature_type=feature_type,
                               source_sha256=source_hash, feature_dim=feature_dim,
                               target_timestamps_sha256=arrays_hash({"timestamps": target_times}),
                               interpolation="linear_no_extrapolation")
            pts_origin = metadata.get("video_pts_origin_seconds", 0.0)
        else:
            source = Path(record["raw"]["video"])
            if model_info is None:
                model_info = load_resnet50(weights_path=weights_path,
                                           download_weights=download_weights, device=device)
            model, transform, extractor = model_info
            feature_type = "resnet50_avgpool"
            source_hash = sha256_file(source)
            fingerprint = dict(schema_version=1, kind="video", sequence_id=sequence,
                               extractor=extractor, feature_type=feature_type,
                               source_sha256=source_hash, feature_dim=2048,
                               target_timestamps_sha256=arrays_hash({"timestamps": target_times}),
                               interpolation="linear_no_extrapolation")
            if output.is_file() and not overwrite:
                verified_cache(output, fingerprint)
                mark_prepared(record, "video", output)
                continue
            features, times, pts_origin = extract_video_features(
                source, model, transform, device=device, batch_size=batch_size)
        aligned = interpolate_features(features, times, target_times)
        metadata = dict(kind="video", sequence_id=sequence, extractor=extractor,
                        feature_type=feature_type, feature_dim=feature_dim, timestamp_origin="clip_start",
                        source_path=str(source.resolve()), video_pts_origin_seconds=pts_origin,
                        timestamps_verified_by="ffprobe_and_decode" if extract else "source_cache")
        save_feature_cache(output, {"features": aligned, "timestamps": target_times},
                           metadata, fingerprint, overwrite=overwrite)
        mark_prepared(record, "video", output)
    return finish_manifest(output_manifest, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output-dir", "output-manifest"):
        parser.add_argument(f"--{name}", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source-dir")
    mode.add_argument("--extract", action="store_true")
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--weights")
    weights.add_argument("--download-weights", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    count = prepare_video(args.manifest, args.output_dir, args.output_manifest,
        source_dir=args.source_dir, extract=args.extract, weights_path=args.weights,
        download_weights=args.download_weights, device=args.device, batch_size=args.batch_size,
        feature_dim=args.feature_dim, overwrite=args.overwrite)
    print(f"Prepared/verified {count} video feature caches")


if __name__ == "__main__":
    main()
