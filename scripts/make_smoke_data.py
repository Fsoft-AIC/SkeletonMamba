"""Create a small synthetic dataset for end-to-end smoke tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dataset.manifest import write_manifest
from dataset.motion_representation import REPRESENTATION, atomic_save_npz
from scripts.compute_statistics import compute_statistics


def create_smoke_data(output, *, seed=123):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    random = np.random.default_rng(seed)
    paths = {}
    for split, count in (("train", 4), ("val", 2), ("test", 2)):
        records = []
        for index in range(count):
            sequence_id = f"synthetic_{split}_{index:03d}"
            folder = output / split / sequence_id
            frames = 5 + index
            times = np.arange(frames, dtype=np.float64) / 30
            phase = times.astype(np.float32) * 5 + index
            motion = np.zeros((frames, 24, 9), np.float32)
            motion[:, :, 0] = np.sin(phase)[:, None] * 0.1 + np.arange(24)[None] * 0.01
            motion[:, :, 1] = np.cos(phase)[:, None] * 0.1
            motion[:, :, 2] = np.arange(24)[None] * 0.03
            motion[:, :, 3] = 1
            motion[:, :, 7] = 1
            features = {"motion": str(folder / "motion.npz"), "audio": str(folder / "audio.npz"),
                        "video": str(folder / "video.npz")}
            metadata = {"schema_version": 1, "sequence_id": sequence_id, "representation": REPRESENTATION,
                        "geometry_hash": "synthetic_engineering_fixture", "canonicalization": "none",
                        "synthetic": True, "coordinates": "synthetic_meters_z_up"}
            atomic_save_npz(features["motion"], motion=motion, timestamps=times,
                            metadata=np.array(json.dumps(metadata)))
            for kind, width in (("audio", 4), ("video", 6)):
                values = random.normal(scale=0.05, size=(frames, width)).astype(np.float32)
                values[:, 0] = np.sin(phase)
                values[:, 1] = np.cos(phase)
                metadata = {"schema_version": 1, "sequence_id": sequence_id, "kind": kind,
                            "extractor": "synthetic_engineering_fixture", "synthetic": True,
                            "alignment_verified": True, "source_start_time_seconds": 0.0,
                            "source_offset_evidence": "synthetic fixture generated on a common clock"}
                atomic_save_npz(features[kind], features=values, timestamps=times,
                                metadata=np.array(json.dumps(metadata)))
            records.append({"schema_version": 1, "sequence_id": sequence_id, "split": split,
                            "source_motion_id": sequence_id, "music_id": sequence_id,
                            "scene_id": "synthetic", "num_frames": frames, "fps": 30.0,
                            "source_start_time_seconds": 0.0, "source_offset_status": "clip_aligned_verified",
                            "source_offset_evidence": "synthetic fixture generated on a common clock",
                            "prepared": features, "synthetic": True})
        manifest = output / f"{split}.jsonl"
        write_manifest(manifest, records)
        paths[f"{split}_manifest"] = str(manifest)
    statistics = output / "statistics.npz"
    compute_statistics(paths["train_manifest"], statistics)
    paths["statistics"] = str(statistics)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/smoke")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    print(json.dumps(create_smoke_data(args.output, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
