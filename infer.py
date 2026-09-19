"""Generate motion from prepared video/music features without target motion."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dataset.egoaistpp_dataset import load_prepared, validate_feature_contract
from dataset.cache_integrity import validate_content_checksum
from dataset.manifest import sha256_file
from dataset.motion_representation import REPRESENTATION, atomic_save_npz, read_metadata
from model.factory import load_system_checkpoint
from model.head_guidance import EstimatedHeadGuidance, GUIDANCE_PROTOCOLS
from trainer import choose_device
from utils.kinematics import rotation_6d_to_matrix
from utils.config import model_protocols


def load_conditions(audio_path, video_path, config, device, *, feature_contract=None):
    with np.load(audio_path, allow_pickle=False) as data:
        audio_metadata = read_metadata(data)
    sequence_id = audio_metadata.get("sequence_id")
    if not sequence_id:
        raise ValueError("Input features require a sequence_id")
    audio, timestamps, audio_metadata = load_prepared(audio_path, "audio", sequence_id)
    video, video_timestamps, video_metadata = load_prepared(video_path, "video", sequence_id)
    if feature_contract is not None:
        validate_feature_contract(audio_metadata, "audio", feature_contract)
        validate_feature_contract(video_metadata, "video", feature_contract)
    if timestamps.shape != video_timestamps.shape or not np.allclose(timestamps, video_timestamps, atol=1e-4, rtol=0):
        raise ValueError("Audio and video must have matching physical timestamps")
    if audio_metadata.get("alignment_verified") is not True:
        raise ValueError("Audio features require verified source alignment")
    offset = audio_metadata.get("source_start_time_seconds")
    if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not np.isfinite(offset) or offset < 0 or not audio_metadata.get("source_offset_evidence"):
        raise ValueError("Audio features require a finite verified source offset and its evidence")
    if audio.shape[-1] != config["conditioning"]["audio_dim"] or video.shape[-1] != config["conditioning"]["video_dim"]:
        raise ValueError("Feature widths disagree with the checkpoint")
    if len(audio) > config["data"]["window"]:
        raise ValueError("This fixed-window inference command requires a clip no longer than the trained window")
    fps = config["diffusion"].get("fps", 30)
    if not np.allclose(timestamps, np.arange(len(audio)) / fps, atol=1e-4, rtol=0):
        raise ValueError("Input timestamps disagree with the checkpoint motion rate")
    mask = torch.ones((1, len(audio)), dtype=torch.bool, device=device)
    return audio[None].to(device), video[None].to(device), mask, timestamps, sequence_id


def load_head_guidance(path, timestamps, device, *, sequence_id, coordinate_frame, strength, padded_length=None,
                       protocol="posterior_log_v1"):
    """Read input-estimated trajectories, never target-derived dataset fields."""
    with np.load(path, allow_pickle=False) as data:
        metadata = read_metadata(data)
        if "content_sha256" in metadata:
            validate_content_checksum({key: np.array(data[key]) for key in data.files if key != "metadata"}, metadata, path)
        if metadata.get("source_kind") != "estimated":
            raise ValueError("Default inference accepts estimated head trajectories only")
        if metadata.get("coordinate_frame") != coordinate_frame:
            raise ValueError("Estimated head and model coordinate frames disagree")
        if metadata.get("sequence_id") != sequence_id:
            raise ValueError("Estimated head trajectory belongs to a different sequence")
        if metadata.get("units") != "m" or not metadata.get("calibration_id"):
            raise ValueError("Estimated heads require metre units and an explicit calibration identity")
        if not metadata.get("extractor"):
            raise ValueError("Estimated heads require an estimator identity")
        source_times = np.array(data["timestamps"], dtype=np.float64)
        positions = np.array(data["positions"], dtype=np.float32)
        rotations = np.array(data["rotations"], dtype=np.float32) if "rotations" in data else None
        confidence = np.array(data["confidence"], dtype=np.float32)
    if source_times.ndim != 1 or len(source_times) == 0 or not np.isfinite(source_times).all() or np.any(np.diff(source_times) <= 0):
        raise ValueError("Invalid head-estimate timestamps")
    indices = np.searchsorted(source_times, timestamps)
    indices = np.clip(indices, 0, len(source_times) - 1)
    previous = np.maximum(indices - 1, 0)
    indices = np.where(np.abs(source_times[previous] - timestamps) < np.abs(source_times[indices] - timestamps), previous, indices)
    if not np.allclose(source_times[indices], timestamps, atol=1e-4, rtol=0):
        raise ValueError("Estimated heads are not synchronized to the input clip")
    if positions.shape != (len(source_times), 3) or confidence.shape != (len(source_times),):
        raise ValueError("Invalid estimated-head position/confidence shape")
    n = len(timestamps)
    padded_length = n if padded_length is None else padded_length
    if padded_length < n:
        raise ValueError("Head padding length is shorter than valid input")
    p = torch.zeros(1, padded_length, 3, device=device)
    c = torch.zeros(1, padded_length, device=device)
    p[0, :n] = torch.from_numpy(positions[indices]).to(device)
    c[0, :n] = torch.from_numpy(confidence[indices]).to(device)
    r = None
    if rotations is not None:
        rotations = torch.from_numpy(rotations).to(device)
        if rotations.shape == (len(source_times), 6):
            if metadata.get("rotation6d_convention") != "first_two_matrix_rows":
                raise ValueError("Estimated rotation6d convention is missing or incompatible")
            if (torch.linalg.vector_norm(rotations[:, :3], dim=-1) < 1e-6).any() or (torch.linalg.vector_norm(torch.linalg.cross(rotations[:, :3], rotations[:, 3:]), dim=-1) < 1e-6).any():
                raise ValueError("Degenerate estimated head rotations")
            rotations = rotation_6d_to_matrix(rotations)
        if rotations.shape != (len(source_times), 3, 3):
            raise ValueError("Estimated rotations must be matrices or rotation6d")
        identity = torch.eye(3, device=device).expand_as(rotations)
        if not torch.isfinite(rotations).all() or not torch.allclose(rotations @ rotations.transpose(-1, -2), identity, atol=2e-4, rtol=0) or not torch.allclose(torch.linalg.det(rotations), torch.ones(len(source_times), device=device), atol=2e-4, rtol=0):
            raise ValueError("Estimated rotations must be proper orthonormal matrices")
        r = torch.eye(3, device=device).expand(1, padded_length, 3, 3).clone()
        r[0, :n] = rotations[torch.from_numpy(indices).to(device)]
    return EstimatedHeadGuidance(p, r, c, strength=strength, coordinate_frame=coordinate_frame, protocol=protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", required=True, help="Prepared aligned audio NPZ")
    parser.add_argument("--video", required=True, help="Prepared video-feature NPZ")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--head-estimates")
    parser.add_argument("--guidance-strength", type=float, default=0.01)
    parser.add_argument("--guidance-protocol", choices=GUIDANCE_PROTOCOLS, default="posterior_log_v1")
    parser.add_argument("--raw-weights", action="store_true", help="Use model weights instead of EMA")
    args = parser.parse_args()
    device = choose_device(args.device)
    system, checkpoint = load_system_checkpoint(args.checkpoint, device, use_ema=not args.raw_weights)
    config = checkpoint["config"]
    torch.set_num_threads(config["training"].get("cpu_threads", 4))
    if not system.feature_contract:
        raise ValueError("Checkpoint lacks the feature extractor contract")
    audio, video, mask, timestamps, sequence_id = load_conditions(
        args.audio, args.video, config, device, feature_contract=system.feature_contract)
    frame = config.get("assumptions", {}).get("coordinates", "unspecified")
    guidance = None if not args.head_estimates else load_head_guidance(
        args.head_estimates, timestamps, device, sequence_id=sequence_id,
        coordinate_frame=frame, strength=args.guidance_strength, protocol=args.guidance_protocol)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = system.sample(audio, video, mask, head_guidance=guidance, generator=generator)
    metadata = {"schema_version": 1, "sequence_id": sequence_id, "representation": REPRESENTATION,
                "coordinate_frame": frame, "checkpoint_sha256": sha256_file(args.checkpoint),
                "audio_sha256": sha256_file(args.audio), "video_sha256": sha256_file(args.video),
                "weights": "model" if args.raw_weights else "ema", "seed": args.seed,
                "guidance": "estimated_head" if guidance else "unguided",
                "guidance_strength": args.guidance_strength if guidance else 0.0,
                "guidance_protocol": guidance.protocol if guidance else None,
                "model_protocols": model_protocols(config),
                "head_estimates_sha256": sha256_file(args.head_estimates) if guidance else None,
                "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                "normalization": checkpoint.get("normalization_metadata", {})}
    atomic_save_npz(args.output, motion=result[0].cpu().numpy(), timestamps=timestamps,
                    metadata=np.array(json.dumps(metadata)))
    print(f"Saved {len(timestamps)} motion frames to {args.output}")


if __name__ == "__main__":
    main()
