"""Evaluate a checkpoint under an explicit, unaligned motion protocol."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

from dataset.egoaistpp_dataset import EgoAISTppDataset
from dataset.cache_integrity import validate_content_checksum
from dataset.manifest import atomic_json, sha256_file
from dataset.motion_representation import MotionNormalizer, REPRESENTATION, atomic_save_npz, read_metadata
from evaluation.motion_metrics import motion_metrics
from evaluation.mmv import compute_mmv, kinematic_beats
from infer import load_head_guidance
from model.factory import load_system_checkpoint
from model.head_guidance import GUIDANCE_PROTOCOLS
from trainer import choose_device
from utils.config import model_protocols


def metric_denominator(name, metrics):
    if name.startswith("acceleration_"):
        return int(metrics["acceleration_frames"])
    if name == "foot_skating_mm_s":
        return int(metrics["foot_contact_transitions"])
    if name in ("mm", "mv", "mmv"):
        return 1
    return int(metrics["valid_frames"])


def evaluate(checkpoint_path, manifest_path, output_dir, *, device="cpu", seed=0,
             floor_height=None, head_estimates_dir=None, guidance_strength=0.01,
             beat_features_dir=None, beat_sigma_seconds=0.1, use_ema=True,
             up_axis=2, contact_height_threshold=0.05, guidance_protocol="posterior_log_v1",
             require_raft_beats=False):
    if guidance_protocol not in GUIDANCE_PROTOCOLS:
        raise ValueError("Unknown guidance protocol")
    if require_raft_beats and beat_features_dir is None:
        raise ValueError("require_raft_beats requires beat_features_dir")
    device = choose_device(device)
    system, checkpoint = load_system_checkpoint(checkpoint_path, device, use_ema=use_ema)
    config = checkpoint["config"]
    if not system.feature_contract:
        raise ValueError("Checkpoint lacks the feature extractor contract")
    torch.set_num_threads(config["training"].get("cpu_threads", 4))
    normalization = MotionNormalizer(system.normalizer.mean.detach().cpu(), system.normalizer.std.detach().cpu(),
                                     system.normalizer.metadata)
    dataset = EgoAISTppDataset(manifest_path, normalization, window=config["data"]["window"], training=False,
                              expected_audio_dim=config["conditioning"]["audio_dim"],
                              expected_video_dim=config["conditioning"]["video_dim"],
                              expected_feature_contract=system.feature_contract)
    if any(r["split"] not in ("test", "val") for r in dataset.records):
        raise ValueError("Evaluation requires a declared test or validation manifest")
    if len({r["split"] for r in dataset.records}) != 1:
        raise ValueError("Do not mix validation and test in one report")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fps = config["diffusion"].get("fps", 30)
    if any(record.get("fps") is None or not math.isclose(record["fps"], fps, abs_tol=1e-8) for record in dataset.records):
        raise ValueError("Evaluation manifest FPS differs from the checkpoint motion rate")
    frame = config.get("assumptions", {}).get("coordinates", "unspecified")
    sums, counts, per_window = {}, {}, []
    for index in range(len(dataset)):
        sample = dataset[index]
        sequence_id, start, length = sample["sequence_id"], sample["start_frame"], sample["length"]
        audio, video, mask = (sample[k][None].to(device) for k in ("audio", "video", "mask"))
        times = (start + np.arange(length, dtype=np.float64)) / fps
        guidance = None
        input_hashes = {}
        if head_estimates_dir:
            head_path = Path(head_estimates_dir) / f"{sequence_id}.npz"
            guidance = load_head_guidance(head_path, times, device,
                                          sequence_id=sequence_id, coordinate_frame=frame, strength=guidance_strength,
                                          padded_length=mask.shape[1], protocol=guidance_protocol)
            input_hashes["head_estimates_sha256"] = sha256_file(head_path)
        generator = torch.Generator(device=device).manual_seed(seed + index)
        # No target or target-derived transform is supplied to sampling.
        prediction = system.sample(audio, video, mask, head_guidance=guidance, generator=generator)
        target = system.normalizer.denormalize(sample["motion"][None].to(device))
        metrics = motion_metrics(prediction, target, mask, fps=fps, floor_height=floor_height,
                                 up_axis=up_axis, contact_height_threshold=contact_height_threshold)
        values = {name: float(value) for name, value in metrics.items()}
        values.update(mm=float("nan"), mv=float("nan"), mmv=float("nan"))
        if beat_features_dir:
            beat_path = Path(beat_features_dir) / f"{sequence_id}.npz"
            with np.load(beat_path, allow_pickle=False) as data:
                metadata = read_metadata(data)
                if "content_sha256" in metadata:
                    validate_content_checksum({key: np.array(data[key]) for key in data.files if key != "metadata"}, metadata, beat_path)
                if metadata.get("source_kind") != "inputs" or metadata.get("sequence_id") != sequence_id:
                    raise ValueError("Beat references must be extracted from input music/video for this sequence")
                video_extractor = metadata.get("extractor", {}).get("video", {})
                if require_raft_beats and (video_extractor.get("backend") != "raft"
                        or not re.fullmatch(r"[0-9a-f]{64}", str(video_extractor.get("weights_sha256", "")))):
                    raise ValueError("RAFT beat evaluation requires recorded RAFT backend and pretrained weight hash")
                music = np.array(data["music_beats_seconds"])
                vision = np.array(data["video_beats_seconds"])
            input_hashes["beat_features_sha256"] = sha256_file(beat_path)
            input_hashes["beat_extractor"] = metadata.get("extractor", {})
            if any(array.ndim != 1 or not np.isfinite(array).all() or (array < 0).any() for array in (music, vision)):
                raise ValueError("Beat references must be finite nonnegative timestamps")
            begin, end = start / fps, (start + length) / fps
            music = music[(music >= begin) & (music < end)] - begin
            vision = vision[(vision >= begin) & (vision < end)] - begin
            positions = prediction[0, :length, :, :3].cpu().numpy()
            body_beats = kinematic_beats(positions, fps=fps)
            head_beats = kinematic_beats(positions[:, 15:16], fps=fps)
            values.update(compute_mmv(body_beats, music, head_beats, vision, sigma_seconds=beat_sigma_seconds))
        for name, value in values.items():
            if name in ("valid_frames", "acceleration_frames", "foot_contact_transitions"):
                continue
            denominator = metric_denominator(name, metrics)
            if math.isfinite(value) and denominator:
                sums[name] = sums.get(name, 0.0) + value * denominator
                counts[name] = counts.get(name, 0) + denominator
            else:
                counts.setdefault(name, 0)
        metadata = {"schema_version": 1, "sequence_id": sequence_id, "start_frame": start,
                    "representation": REPRESENTATION, "coordinate_frame": frame,
                    "checkpoint_sha256": sha256_file(checkpoint_path), "seed": seed + index,
                    "guidance": "estimated_head" if guidance else "unguided",
                    "guidance_strength": guidance_strength if guidance else 0.0,
                    "guidance_protocol": guidance.protocol if guidance else None,
                    "model_protocols": model_protocols(config),
                    "weights": "ema" if use_ema else "model", **input_hashes}
        filename = output / "predictions" / f"{sequence_id}__{start:06d}.npz"
        atomic_save_npz(filename, motion=prediction[0, :length].cpu().numpy(), timestamps=times,
                        metadata=np.array(json.dumps(metadata)))
        per_window.append({"sequence_id": sequence_id, "start_frame": start, "length": length,
                           "input_hashes": input_hashes,
                           "metrics": {k: v if math.isfinite(v) else None for k, v in values.items()}})
    report = {"schema_version": 1, "protocol": "reconstruction_global_unaligned_single_sample_v1",
              "checkpoint_sha256": sha256_file(checkpoint_path), "manifest_sha256": sha256_file(manifest_path),
              "weights": "ema" if use_ema else "model", "seed": seed, "fps": fps,
              "guidance": "estimated_head" if head_estimates_dir else "unguided",
              "coordinate_frame": frame, "floor_height_m": floor_height,
              "up_axis": up_axis, "contact_height_threshold_m": contact_height_threshold,
              "contact_policy": "predicted_feet_near_declared_floor" if floor_height is not None else "unavailable",
              "guidance_strength": guidance_strength if head_estimates_dir else 0.0,
              "guidance_protocol": guidance_protocol if head_estimates_dir else None,
              "model_protocols": model_protocols(config), "require_raft_beats": require_raft_beats,
              "beat_sigma_seconds": beat_sigma_seconds, "requested_sequences": len(dataset.records),
              "evaluated_windows": len(per_window), "evaluated_frames": sum(r["length"] for r in per_window),
              "aggregation": "frame/contact weighted motion metrics; window mean beat agreement; no cross-window derivatives",
              "metrics": {name: sums[name] / count if count else None for name, count in counts.items()},
              "denominators": counts, "windows": per_window}
    atomic_json(output / "metrics.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--floor-height", type=float, help="Declared floor height in model coordinates, metres")
    parser.add_argument("--up-axis", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--contact-height-threshold", type=float, default=0.05)
    parser.add_argument("--head-estimates-dir")
    parser.add_argument("--guidance-strength", type=float, default=0.01)
    parser.add_argument("--guidance-protocol", choices=GUIDANCE_PROTOCOLS, default="posterior_log_v1")
    parser.add_argument("--require-raft-beats", action="store_true", help="Reject baseline or unversioned video beat caches")
    parser.add_argument("--beat-features-dir")
    parser.add_argument("--beat-sigma-seconds", type=float, default=0.1)
    parser.add_argument("--raw-weights", action="store_true")
    args = parser.parse_args()
    report = evaluate(args.checkpoint, args.manifest, args.output_dir, device=args.device, seed=args.seed,
                      floor_height=args.floor_height, head_estimates_dir=args.head_estimates_dir,
                      guidance_strength=args.guidance_strength, guidance_protocol=args.guidance_protocol,
                      require_raft_beats=args.require_raft_beats, beat_features_dir=args.beat_features_dir,
                      beat_sigma_seconds=args.beat_sigma_seconds, use_ema=not args.raw_weights,
                      up_axis=args.up_axis, contact_height_threshold=args.contact_height_threshold)
    print(json.dumps({k: v for k, v in report.items() if k != "windows"}, indent=2))


if __name__ == "__main__":
    main()
