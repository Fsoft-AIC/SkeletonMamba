"""Run EDGE/Jukemirlib extraction using a supplied interpreter and checkpoints.

Upstream implementations and licenses:
https://github.com/Stanford-TML/EDGE/blob/main/data/audio_extraction/jukebox_features.py
https://github.com/rodrigo-castellon/jukemirlib

Layer 66 produces 4800-dimensional features at 30 Hz. Context resets every five
seconds, so arbitrary clip boundaries may yield different features. The caller
must supply the feature-to-waveform delay; upstream does not calibrate it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from dataset.manifest import sha256_file
from dataset.motion_representation import atomic_save_npz
from scripts.feature_utils import arrays_hash, load_feature_source, safe_name


CHECKPOINTS = ("vqvae.pth.tar", "prior_level_2.pth.tar")

# The supplied interpreter isolates upstream dependencies from this project's Torch.
_WORKER = r'''
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

job = json.loads(Path(sys.argv[1]).read_text())
sys.path.insert(0, job["edge_root"])
module_path = Path(job["edge_root"]) / "data/audio_extraction/jukebox_features.py"
spec = importlib.util.spec_from_file_location("official_edge_jukebox_features", str(module_path))
edge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(edge)
if edge.FPS != 30 or edge.LAYER != 66:
    raise ValueError("This adapter requires the inspected EDGE layer=66, FPS=30 API")
juke = importlib.import_module("jukemirlib")
lib = importlib.import_module("jukemirlib.lib")
setup = importlib.import_module("jukemirlib.setup_models")
constants = importlib.import_module("jukemirlib.constants")

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for data in iter(lambda: handle.read(1048576), b""):
            h.update(data)
    return h.hexdigest()

identity = {"layer": edge.LAYER, "feature_fps": edge.FPS,
            "python_version": sys.version.split()[0],
            "modules": {}}
for name in ("jukemirlib", "jukemirlib.lib", "jukemirlib.setup_models",
             "jukemirlib.constants", "jukebox.prior.autoregressive", "torch", "librosa", "soundfile"):
    module = importlib.import_module(name)
    identity["modules"][name] = {"version": getattr(module, "__version__", None),
                                "source_sha256": digest(module.__file__)}
Path(job["identity_path"]).write_text(json.dumps(identity, sort_keys=True))
if job["mode"] == "probe":
    sys.exit(0)
if identity != job["expected_identity"]:
    raise ValueError("Extractor environment changed after provenance inspection")

# Resolve checkpoints locally instead of using the upstream download resolver.
cache = Path(job["checkpoint_dir"]).resolve()
allowed = {"vqvae.pth.tar", "prior_level_2.pth.tar"}
def local_checkpoint(local_path, remote_prefix):
    path = Path(local_path).resolve()
    if path.name not in allowed or path.parent != cache:
        raise ValueError("Unexpected upstream checkpoint request: " + str(local_path))
    if not path.is_file():
        raise FileNotFoundError("Local checkpoint is required: " + str(path))
    return str(path)
setup.get_checkpoint = local_checkpoint
constants.CACHE_DIR = str(cache)
constants.DEVICE = job["device"]
juke.DEVICE = job["device"]
lib.VQVAE, lib.TOP_PRIOR = setup.setup_models(cache_dir=str(cache), device=job["device"])

import soundfile as sf
for item in job["items"]:
    features, timestamps, windows = [], [], []
    with sf.SoundFile(item["source"]) as audio:
        sample_rate, total_samples = audio.samplerate, len(audio)
        window_samples = 5 * sample_rate
        if total_samples < sample_rate / edge.FPS:
            raise ValueError("Audio is shorter than one feature frame")
        with tempfile.TemporaryDirectory(prefix="edge-windows-") as scratch:
            scratch = Path(scratch)
            for start in range(0, total_samples, window_samples):
                count = min(window_samples, total_samples - start)
                if count < sample_rate / edge.FPS:
                    # Record the sub-frame tail as uncovered in worker metadata.
                    break
                audio.seek(start)
                samples = audio.read(count, dtype="float32", always_2d=True)
                if len(samples) != count or not np.isfinite(samples).all():
                    raise ValueError("Audio decode returned missing/nonfinite samples")
                window_path = scratch / "window.wav"
                sf.write(str(window_path), samples, sample_rate, subtype="FLOAT")
                rep, unused = edge.extract(str(window_path), skip_completed=False,
                                           dest_dir=str(scratch / "unused"))
                rep = np.asarray(rep, dtype=np.float32)
                if rep.ndim != 2 or rep.shape[1] != 4800 or not len(rep) or not np.isfinite(rep).all():
                    raise ValueError("Official extractor must return finite nonempty [T,4800]")
                duration = count / sample_rate
                if abs(len(rep) - duration * edge.FPS) > 1.01:
                    raise ValueError("Unexpected feature coverage; refusing inferred/truncated timing")
                local_times = np.arange(len(rep), dtype=np.float64) / edge.FPS
                keep = local_times < duration - 1e-9
                rep, local_times = rep[keep], local_times[keep]
                features.append(rep)
                timestamps.append(start / sample_rate + job["feature_delay_seconds"] + local_times)
                windows.append({"start_sample": start, "num_samples": count,
                                "num_features": len(rep)})
    staged_result = item["result"] + ".partial.npz"
    np.savez_compressed(staged_result, features=np.concatenate(features),
        timestamps=np.concatenate(timestamps),
        worker_metadata=np.array(json.dumps({"sample_rate": sample_rate,
            "total_samples": total_samples, "windows": windows,
            "uncovered_tail_samples": total_samples - (windows[-1]["start_sample"] + windows[-1]["num_samples"])})))
    Path(staged_result).replace(item["result"])
'''


def feature_timestamps(length, *, fps, source_start_seconds, delay_seconds):
    """Build timestamps from the supplied frame rate, origin, and delay."""
    values = (fps, source_start_seconds, delay_seconds)
    if (not isinstance(length, int) or length < 1 or not np.isfinite(values).all()
            or fps <= 0 or source_start_seconds < 0 or delay_seconds < 0):
        raise ValueError("positive length/rate and finite nonnegative timing values are required")
    return source_start_seconds + delay_seconds + np.arange(length, dtype=np.float64) / fps


def _git_revision(root):
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _run_worker(python, worker, job_path, job):
    job_path.write_text(json.dumps(job, sort_keys=True))
    subprocess.run([str(python), str(worker), str(job_path)], check=True,
                   cwd=job["edge_root"])
    return json.loads(Path(job["identity_path"]).read_text())


def _verify_source_cache(path, fingerprint, music_id):
    features, times, metadata = load_feature_source(path, kind="audio", width=4800, music_id=music_id)
    if metadata.get("fingerprint") != fingerprint or metadata.get("feature_type") != "jukebox":
        raise ValueError(f"stale source cache: {path}; inspect provenance or use --overwrite")
    if metadata.get("extractor") != fingerprint["extractor"]:
        raise ValueError(f"source cache extractor metadata mismatch: {path}")
    if metadata.get("content_sha256") != arrays_hash({"features": features, "timestamps": times}):
        raise ValueError(f"source cache checksum mismatch: {path}")


def extract_audio(source_dir, output_dir, *, edge_root, python, checkpoint_dir,
                  feature_fps, feature_delay_seconds, timing_evidence, window_policy,
                  device="cuda", music_ids=None, overwrite=False):
    if feature_fps != 30 or window_policy != "fixed_5s":
        raise ValueError("inspected EDGE adapter requires 30 Hz and explicit fixed_5s window policy")
    feature_timestamps(1, fps=feature_fps, source_start_seconds=0,
                       delay_seconds=feature_delay_seconds)
    if not isinstance(timing_evidence, str) or not timing_evidence.strip():
        raise ValueError("nonempty timing evidence is required; feature delay is not inferred")
    python_path = shutil.which(str(python))
    if python_path is None:
        raise ValueError(f"external Python executable does not exist: {python}")
    edge_root, checkpoint_dir = Path(edge_root).resolve(), Path(checkpoint_dir).resolve()
    module_path = edge_root / "data/audio_extraction/jukebox_features.py"
    if not module_path.is_file():
        raise ValueError("edge_root must contain the official data/audio_extraction/jukebox_features.py")
    checkpoint_hashes = {}
    for name in CHECKPOINTS:
        path = checkpoint_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"provide local checkpoint {path}; this command never downloads weights")
        checkpoint_hashes[name] = sha256_file(path)
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    sources = ([source_dir / f"{safe_name(music_id)}.wav" for music_id in music_ids]
               if music_ids else sorted(source_dir.glob("*.wav")))
    if not sources or len({source.stem for source in sources}) != len(sources):
        raise ValueError("provide at least one uniquely named source WAV")
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    extractor = dict(name="official_EDGE_jukemirlib", edge_git_sha=_git_revision(edge_root),
                     edge_module_sha256=sha256_file(module_path), checkpoint_sha256=checkpoint_hashes,
                     layer=66, feature_fps=30, feature_dim=4800)
    with tempfile.TemporaryDirectory(prefix="skeletonmamba-jukebox-") as temporary:
        scratch = Path(temporary)
        worker, job_path = scratch / "worker.py", scratch / "job.json"
        worker.write_text(_WORKER)
        job = dict(mode="probe", edge_root=str(edge_root), checkpoint_dir=str(checkpoint_dir),
                   identity_path=str(scratch / "identity.json"), device=device,
                   feature_delay_seconds=float(feature_delay_seconds))
        extractor["environment"] = _run_worker(python_path, worker, job_path, job)
        extractor["adapter_sha256"] = sha256_file(__file__)
        pending = []
        for index, source in enumerate(sources):
            music_id = safe_name(source.stem)
            output = output_dir / f"{music_id}.npz"
            fingerprint = dict(schema_version=1, music_id=music_id, source_sha256=sha256_file(source),
                extractor=extractor, window_policy=window_policy, window_seconds=5,
                feature_delay_seconds=float(feature_delay_seconds), timing_evidence=timing_evidence,
                timestamp_rule="window_start + explicit_delay + frame_index / 30")
            if output.is_file() and not overwrite:
                _verify_source_cache(output, fingerprint, music_id)
                continue
            pending.append((source, output, fingerprint, scratch / f"result-{index}.npz"))
        worker_error = None
        if pending:
            job.update(mode="extract", expected_identity=extractor["environment"], items=[
                {"source": str(source), "result": str(result)} for source, _, _, result in pending])
            try:
                _run_worker(python_path, worker, job_path, job)
            except subprocess.CalledProcessError as exc:
                # Cache completed songs so retries can resume after worker failure.
                worker_error = exc
        for source, output, fingerprint, result in pending:
            if not result.is_file() and worker_error is not None:
                continue
            with np.load(result, allow_pickle=False) as archive:
                arrays = {key: np.array(archive[key]) for key in ("features", "timestamps")}
                worker_metadata = json.loads(str(archive["worker_metadata"].item()))
            # Validate external results before publishing an atomic reusable cache.
            from scripts.feature_utils import validate_timestamps
            validate_timestamps(arrays["timestamps"], len(arrays["features"]))
            if arrays["features"].shape != (len(arrays["timestamps"]), 4800) or not np.isfinite(arrays["features"]).all():
                raise ValueError("external worker returned invalid Jukebox features")
            metadata = dict(schema_version=1, kind="audio", music_id=source.stem,
                feature_type="jukebox", feature_dim=4800, extractor=extractor,
                source_path=str(source), fingerprint=fingerprint, source_kind="inputs",
                feature_delay_seconds=float(feature_delay_seconds), timing_evidence=timing_evidence,
                timestamp_origin="source_music_start", window_policy=window_policy,
                reconstruction_choice="independent fixed 5 s Jukebox contexts, final partial window",
                coverage=worker_metadata, content_sha256=arrays_hash(arrays))
            atomic_save_npz(output, **arrays, metadata=np.array(json.dumps(metadata, sort_keys=True)))
        if worker_error is not None:
            raise worker_error
    return len(sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "output-dir", "edge-root", "python", "checkpoint-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--feature-fps", type=float, required=True)
    parser.add_argument("--feature-delay-seconds", type=float, required=True)
    parser.add_argument("--timing-evidence", required=True)
    parser.add_argument("--window-policy", choices=("fixed_5s",), required=True)
    parser.add_argument("--music-id", action="append", help="repeat to restrict extraction")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    count = extract_audio(args.source_dir, args.output_dir, edge_root=args.edge_root,
        python=args.python, checkpoint_dir=args.checkpoint_dir, feature_fps=args.feature_fps,
        feature_delay_seconds=args.feature_delay_seconds, timing_evidence=args.timing_evidence,
        window_policy=args.window_policy, music_ids=args.music_id, device=args.device, overwrite=args.overwrite)
    print(f"Extracted/verified {count} timestamped source Jukebox caches")


if __name__ == "__main__":
    main()
