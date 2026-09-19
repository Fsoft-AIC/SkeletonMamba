"""Extract MMV beats with librosa and pretrained RAFT (or baseline Farneback).

Music offsets require verified timing; slice WAVs also require unique motion
matching. Empty beat sets are retained for the evaluator's empty-input policy.
Exact RAFT and librosa settings are not specified in the paper.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import importlib.metadata
from pathlib import Path

import numpy as np

from dataset.manifest import resolve_path, sha256_file
from evaluation.mmv import optical_flow_beats
from scripts.feature_utils import (arrays_hash, finish_manifest, mark_prepared,
    preparation_records, safe_name, save_feature_cache, target_timestamps, validate_timestamps,
    verified_cache)
from scripts.prepare_video import probe_frame_times
from scripts.optical_flow import RaftFlow


def crop_music_beats(source_beats, *, source_start_seconds, clip_duration_seconds):
    """Select the source half-open interval and convert to clip-relative seconds."""
    beats = np.asarray(source_beats, dtype=np.float64)
    if (beats.ndim != 1 or not np.isfinite(beats).all() or np.any(beats < 0)
            or np.any(np.diff(beats) <= 0)):
        raise ValueError("source beats must be finite, nonnegative, strictly increasing seconds")
    if (not np.isfinite([source_start_seconds, clip_duration_seconds]).all()
            or source_start_seconds < 0 or clip_duration_seconds <= 0):
        raise ValueError("finite nonnegative offset and positive clip duration are required")
    selected = beats[(beats >= source_start_seconds)
                     & (beats < source_start_seconds + clip_duration_seconds)]
    return selected - source_start_seconds


def extract_music_beats(source_path, *, sample_rate=22050, hop_length=512):
    """Run librosa's dynamic-programming beat tracker on the full source song."""
    if sample_rate < 1 or hop_length < 1:
        raise ValueError("sample_rate and hop_length must be positive")
    try:
        import librosa
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("music beat extraction requires librosa in this Python environment") from exc
    audio, rate = librosa.load(str(source_path), sr=sample_rate, mono=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("source music must contain finite audio samples")
    onset = librosa.onset.onset_strength(y=audio, sr=rate, hop_length=hop_length, center=True)
    tempo, frames = librosa.beat.beat_track(onset_envelope=onset, sr=rate,
        hop_length=hop_length, start_bpm=120., tightness=100., trim=False, units="frames")
    beats = np.asarray(frames, dtype=np.float64) * hop_length / rate
    duration = len(audio) / rate
    beats = beats[(beats >= 0) & (beats < duration)]
    return beats, duration


def extract_video_beats(video_path, *, smooth_window=1, max_width=640,
                        video_backend="raft", flow_estimator=None):
    """Mean-flow minima at actual adjacent-frame midpoint PTS."""
    if video_backend not in ("raft", "farneback"):
        raise ValueError("video_backend must be raft or farneback")
    if flow_estimator is not None and video_backend != "raft":
        raise ValueError("flow_estimator applies only to RAFT")
    if max_width < 1 or smooth_window < 1 or smooth_window % 2 != 1:
        raise ValueError("max_width must be positive and smooth_window a positive odd integer")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("video beat extraction requires OpenCV") from exc
    if video_backend == "raft":
        flow_estimator = flow_estimator or RaftFlow(max_width=max_width)
    times, pts_origin = probe_frame_times(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot decode input video: {video_path}")
    previous, magnitudes, count = None, [], 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frame = bgr
            if video_backend == "farneback":
                frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                if frame.shape[1] > max_width:
                    height = max(1, round(frame.shape[0] * max_width / frame.shape[1]))
                    frame = cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)
            if previous is not None:
                if previous.shape != frame.shape:
                    raise ValueError("video resolution changed during decoding")
                if video_backend == "raft":
                    magnitude = flow_estimator.magnitude(previous, frame)
                else:
                    flow = cv2.calcOpticalFlowFarneback(previous, frame, None,
                        pyr_scale=.5, levels=3, winsize=15, iterations=3, poly_n=5,
                        poly_sigma=1.2, flags=0)
                    magnitude = np.sqrt(np.square(flow).sum(axis=-1)).mean()
                if not np.isfinite(magnitude):
                    raise ValueError("nonfinite optical flow")
                magnitudes.append(float(magnitude))
            previous = frame
            count += 1
    finally:
        capture.release()
    if count != len(times):
        raise ValueError(f"decoded frame count {count} differs from ffprobe count {len(times)}")
    if count < 2:
        raise ValueError("at least two decoded video frames are required for optical flow")
    midpoint_times = (times[:-1] + times[1:]) / 2
    magnitudes = np.asarray(magnitudes, dtype=np.float64)
    beats = optical_flow_beats(magnitudes, timestamps=midpoint_times, smooth_window=smooth_window)
    return beats, times, pts_origin


def _extractor_identity(*, sample_rate, hop_length, smooth_window, max_width,
                        video_backend="raft", flow_estimator=None):
    try:
        import cv2
        librosa_version = importlib.metadata.version("librosa")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError("beat preparation requires librosa and OpenCV; no fallback inputs are generated") from exc
    video = (flow_estimator.identity() if video_backend == "raft" else
        {"name": "opencv.Farneback_mean_flow_strict_minima", "backend": "farneback", "version": cv2.__version__,
         "max_width": max_width, "pyr_scale": .5, "levels": 3, "winsize": 15,
         "iterations": 3, "poly_n": 5, "poly_sigma": 1.2, "flags": 0})
    video.update(smooth_window=smooth_window, timestamps="ffprobe_adjacent_frame_midpoints")
    return dict(music={"name": "librosa.beat.beat_track", "version": librosa_version,
        "sample_rate": sample_rate, "hop_length": hop_length, "onset_center": True,
        "start_bpm": 120., "tightness": 100., "trim": False,
        "timestamp_rule": "beat_frame * hop_length / sample_rate"},
        video=video,
        implementation_sha256=sha256_file(__file__),
        minima_implementation_sha256=sha256_file(Path(__file__).resolve().parents[1] / "evaluation/mmv.py"))


def _music_reference(record, target_times, manifest_path, *, music_dir,
                     paired_slices_dir, paired_timing_evidence):
    """Resolve the declared audio reference without interpreting slice numbers."""
    sequence = record["sequence_id"]
    if paired_slices_dir is None:
        offset = record.get("source_start_time_seconds")
        if record.get("audio_reference_kind") == "matched_source_slice":
            raise ValueError(f"{sequence}: full-song beats need full-song alignment evidence; slice-relative offsets are insufficient")
        if (record.get("source_offset_status") not in ("verified", "clip_aligned_verified")
                or not isinstance(offset, (int, float)) or isinstance(offset, bool)
                or not np.isfinite(offset) or offset < 0 or not record.get("source_offset_evidence")):
            raise ValueError(f"{sequence}: verified music offset and source_offset_evidence are required")
        source_grid = np.asarray(record.get("source_audio_timestamps_seconds", target_times + offset), dtype=np.float64)
        source = Path(music_dir) / f"{safe_name(record['music_id'])}.wav"
        evidence = record["source_offset_evidence"]
        reference_kind, reference_id = "full_source_music", record["music_id"]
        basis = "manifest_verified_full_song_offset"
    else:
        match = record.get("motion_match", {})
        if match.get("unique_match") is not True:
            raise ValueError(f"{sequence}: paired slices require a unique verified numerical motion match")
        source_path = match.get("source_path", match.get("source_motion_path"))
        if not source_path:
            raise ValueError(f"{sequence}: matched source motion path is missing")
        motion_path = resolve_path(source_path, manifest_path)
        stem = safe_name(match.get("source_stem", motion_path.stem))
        if motion_path.stem != stem or sha256_file(motion_path) != match.get("source_sha256"):
            raise ValueError(f"{sequence}: source motion identity differs from verified matching evidence")
        if (record.get("audio_reference_kind") == "matched_source_slice"
                and record.get("audio_reference_id") != stem):
            raise ValueError(f"{sequence}: audio reference and matched source motion disagree")
        source_grid = validate_timestamps(match.get("source_frame_times_seconds"), len(target_times))
        indices = np.asarray(match.get("source_frame_indices"))
        source_fps = match.get("source_fps")
        if (not isinstance(source_fps, (int, float)) or isinstance(source_fps, bool)
                or not np.isfinite(source_fps) or source_fps <= 0
                or indices.shape != source_grid.shape or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0)
                or not np.allclose(source_grid, indices / source_fps, atol=1e-6, rtol=0)):
            raise ValueError(f"{sequence}: matched source frame indices, FPS and timestamps disagree")
        offset = float(source_grid[0])
        source = Path(paired_slices_dir) / f"{stem}.wav"
        evidence = dict(motion_match=match, paired_waveform_timing_evidence=paired_timing_evidence,
                        source_offset_scope="matched_source_slice_only")
        reference_kind, reference_id = "matched_source_slice", stem
        basis = "numerical_motion_match_plus_caller_declared_waveform_pairing"
    if (source_grid.shape != target_times.shape or not np.isfinite(source_grid).all()
            or not np.allclose(source_grid, target_times + offset, atol=1e-6, rtol=0)
            or not np.allclose(target_times, np.arange(len(target_times)) / record["fps"], atol=1e-6, rtol=0)):
        raise ValueError(f"{sequence}: nonuniform source sampling requires an explicit beat-time warp; scalar offset cropping is insufficient")
    return source, float(offset), dict(source_offset_evidence=evidence,
        audio_reference_kind=reference_kind, audio_reference_id=reference_id,
        alignment_basis=basis, source_timestamps_sha256=arrays_hash({"timestamps": source_grid}))


def prepare_beats(manifest_path, music_dir, output_dir, output_manifest, *,
                  paired_slices_dir=None, paired_timing_evidence=None,
                  sample_rate=22050, hop_length=512, smooth_window=1, max_width=640,
                  video_backend="raft", raft_weights=None, download_weights=False,
                  device="cpu", raft_updates=12,
                  overwrite=False):
    if bool(music_dir) == bool(paired_slices_dir):
        raise ValueError("choose exactly one of full-source music_dir or paired_slices_dir")
    if paired_slices_dir is not None and (not isinstance(paired_timing_evidence, str)
                                         or not paired_timing_evidence.strip()):
        raise ValueError("paired slices require explicit waveform/motion timing evidence or declaration")
    if music_dir is not None and paired_timing_evidence is not None:
        raise ValueError("paired timing evidence only applies to paired_slices_dir")
    if min(sample_rate, hop_length, max_width, smooth_window) < 1 or smooth_window % 2 != 1:
        raise ValueError("rates/width must be positive and smooth_window a positive odd integer")
    if video_backend not in ("raft", "farneback"):
        raise ValueError("video_backend must be raft or farneback")
    if video_backend == "farneback" and (raft_weights or download_weights):
        raise ValueError("RAFT weights/download options require video_backend=raft")
    flow_estimator = (RaftFlow(weights_path=raft_weights, download_weights=download_weights,
        device=device, max_width=max_width, num_flow_updates=raft_updates) if video_backend == "raft" else None)
    records = preparation_records(manifest_path, output_manifest)
    source_beats = OrderedDict()
    extractor = None
    for record in records:
        sequence = record["sequence_id"]
        target_times = target_timestamps(record)
        source, offset, alignment = _music_reference(record, target_times, manifest_path,
            music_dir=music_dir, paired_slices_dir=paired_slices_dir,
            paired_timing_evidence=paired_timing_evidence)
        video = Path(record["raw"]["video"])
        if not source.is_file() or not video.is_file():
            raise FileNotFoundError(f"{sequence}: both source WAV and raw egocentric video must exist")
        if extractor is None:
            extractor = _extractor_identity(sample_rate=sample_rate, hop_length=hop_length,
                smooth_window=smooth_window, max_width=max_width,
                video_backend=video_backend, flow_estimator=flow_estimator)
        duration = float(target_times[-1] + 1 / record["fps"])
        source_hash, video_hash = sha256_file(source), sha256_file(video)
        fingerprint = dict(schema_version=1, kind="beats", sequence_id=sequence,
            source_kind="inputs", extractor=extractor, source_music_sha256=source_hash,
            source_video_sha256=video_hash, source_start_time_seconds=float(offset),
            **alignment, asset_provenance=record.get("asset_provenance"), clip_duration_seconds=duration,
            target_timestamps_sha256=arrays_hash({"timestamps": target_times}))
        output = Path(output_dir) / f"{safe_name(sequence)}.npz"
        if output.is_file() and not overwrite:
            verified_cache(output, fingerprint)
            mark_prepared(record, "beats", output)
            continue
        key = (str(source.resolve()), source_hash)
        if key not in source_beats:
            source_beats[key] = extract_music_beats(source, sample_rate=sample_rate, hop_length=hop_length)
            if len(source_beats) > 2:
                source_beats.popitem(last=False)
        source_beats.move_to_end(key)
        beats, source_duration = source_beats[key]
        if offset + duration > source_duration + 1e-6:
            raise ValueError(f"{sequence}: source music does not cover the clip duration")
        music_beats = crop_music_beats(beats, source_start_seconds=offset, clip_duration_seconds=duration)
        video_beats, video_times, pts_origin = extract_video_beats(video,
            smooth_window=smooth_window, max_width=max_width,
            video_backend=video_backend, flow_estimator=flow_estimator)
        if target_times[-1] > video_times[-1] + 1e-6:
            raise ValueError(f"{sequence}: decoded video does not cover requested clip timestamps")
        video_beats = crop_music_beats(video_beats, source_start_seconds=0, clip_duration_seconds=duration)
        metadata = dict(schema_version=1, kind="beats", sequence_id=sequence, source_kind="inputs",
            music_id=record["music_id"], extractor=extractor, alignment_verified=True,
            source_start_time_seconds=float(offset), **alignment,
            asset_provenance=record.get("asset_provenance"),
            source_music_path=str(source.resolve()), source_video_path=str(video.resolve()),
            video_pts_origin_seconds=float(pts_origin), timestamp_origin="clip_start",
            clip_duration_seconds=duration,
            reconstruction_choice=f"librosa tracker and {video_backend} mean-flow minima; exact published settings unavailable",
            empty_beats_policy="preserve_empty_arrays_no_substitution")
        save_feature_cache(output, {"music_beats_seconds": music_beats, "video_beats_seconds": video_beats},
                           metadata, fingerprint, overwrite=overwrite)
        mark_prepared(record, "beats", output)
    return finish_manifest(output_manifest, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output-dir", "output-manifest"):
        parser.add_argument(f"--{name}", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--music-dir", help="full music WAVs keyed by music_id; requires full-song offsets")
    source.add_argument("--paired-slices-dir", help="WAVs keyed by uniquely matched source-motion stem")
    parser.add_argument("--paired-timing-evidence", help="required declaration connecting the paired WAV and source-motion time zero/rate")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--max-width", type=int, default=640)
    parser.add_argument("--video-backend", choices=("raft", "farneback"), default="raft")
    parser.add_argument("--raft-weights", help="Local pinned RAFT-Large C_T_SKHT_V2 checkpoint")
    parser.add_argument("--download-weights", action="store_true", help="Allow downloading pinned pretrained RAFT weights")
    parser.add_argument("--device", default="auto", help="RAFT inference device (auto/cpu/cuda)")
    parser.add_argument("--raft-updates", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    count = prepare_beats(args.manifest, args.music_dir, args.output_dir, args.output_manifest,
        paired_slices_dir=args.paired_slices_dir, paired_timing_evidence=args.paired_timing_evidence,
        sample_rate=args.sample_rate, hop_length=args.hop_length, smooth_window=args.smooth_window,
        max_width=args.max_width, video_backend=args.video_backend, raft_weights=args.raft_weights,
        download_weights=args.download_weights, device=args.device, raft_updates=args.raft_updates,
        overwrite=args.overwrite)
    print(f"Prepared/verified {count} input-only beat caches")


if __name__ == "__main__":
    main()
