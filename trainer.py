"""Resumable trainer with CUDA mixed precision and CPU support."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from dataset.egoaistpp_dataset import EgoAISTppDataset
from model.factory import build_system
from utils.config import save_config, validate_config
from utils.precision import amp_dtype, autocast_context, check_cuda_precision


def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def choose_device(value="auto"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return device


def move_batch(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def optimizer_groups(model, weight_decay):
    decay, no_decay, seen = [], [], set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise RuntimeError(f"Duplicate parameter in optimizer: {name}")
        seen.add(id(parameter))
        excluded = getattr(parameter, "_no_weight_decay", False) or parameter.ndim < 2 or name.endswith("bias")
        (no_decay if excluded else decay).append(parameter)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


class ResumableBatchSampler(Sampler):
    """Cursor advances only when a batch is requested (requires num_workers=0)."""
    def __init__(self, size, batch_size, seed):
        if size < 1:
            raise ValueError("Training manifest is empty")
        self.size, self.batch_size = size, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator).tolist()
        self.cursor, self.epoch = 0, 0

    def __iter__(self):
        if self.cursor >= self.size:
            self.order = torch.randperm(self.size, generator=self.generator).tolist()
            self.cursor = 0
            self.epoch += 1
        while self.cursor < self.size:
            start = self.cursor
            self.cursor = min(self.cursor + self.batch_size, self.size)
            yield self.order[start:self.cursor]

    def __len__(self):
        return math.ceil(self.size / self.batch_size)

    def state_dict(self):
        return {"size": self.size, "batch_size": self.batch_size, "order": self.order,
                "cursor": self.cursor, "epoch": self.epoch, "generator": self.generator.get_state()}

    def load_state_dict(self, state):
        if state["size"] != self.size or state["batch_size"] != self.batch_size:
            raise ValueError("Dataset size or batch size changed across resume")
        if sorted(state["order"]) != list(range(self.size)) or not 0 <= state["cursor"] <= self.size:
            raise ValueError("Invalid saved data order")
        self.order, self.cursor, self.epoch = state["order"], state["cursor"], state["epoch"]
        self.generator.set_state(state["generator"].cpu())


def fingerprints(config):
    result = {}
    for name in ("train_manifest", "val_manifest", "statistics"):
        path = Path(config["data"][name])
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[name] = digest.hexdigest()
    return result


def runtime_metadata(device):
    result = {"python": platform.python_version(), "torch": torch.__version__,
            "torch_path": torch.__file__, "cuda_build": torch.version.cuda, "device": str(device),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "triton_f32_default_override": os.environ.get("TRITON_F32_DEFAULT"),
            "cpu_threads": torch.get_num_threads(), "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    if torch.device(device).type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(gpu_name=properties.name, gpu_memory_bytes=properties.total_memory,
                      compute_capability=[properties.major, properties.minor])
    return result


class Trainer:
    def __init__(self, config):
        self.config = validate_config(config)
        self.options = self.config["training"]
        self.device = choose_device(self.options.get("device", "auto"))
        if self.config["model"].get("backend", "reference") == "mamba2" and self.device.type != "cuda":
            raise RuntimeError("The fused Mamba2 backend requires CUDA; select the CUDA environment and device")
        if self.config["model"].get("backend", "reference") == "mamba2" and self.options.get("deterministic", True):
            raise ValueError("Official CUDA Mamba2 backward requires training.deterministic: false")
        self.amp = self.options.get("amp", False)
        self.amp_dtype = amp_dtype(self.options)
        if self.amp:
            check_cuda_precision(self.device, self.amp_dtype)
        torch.set_num_threads(self.options.get("cpu_threads", 4))
        seed_everything(self.options.get("seed", 42), self.options.get("deterministic", True))
        self.output = Path(self.options["output_dir"])
        self.output.mkdir(parents=True, exist_ok=True)
        data = self.config["data"]
        kwargs = {"statistics_path": data["statistics"], "window": data.get("window", 150),
                  "expected_audio_dim": self.config["conditioning"]["audio_dim"],
                  "expected_video_dim": self.config["conditioning"]["video_dim"]}
        self.dataset = EgoAISTppDataset(data["train_manifest"], training=True, **kwargs)
        self.feature_contract = copy.deepcopy(self.dataset.expected_feature_contract)
        self.validation_dataset = EgoAISTppDataset(data["val_manifest"], training=False,
                                                  expected_feature_contract=self.feature_contract, **kwargs)
        train_records = getattr(self.dataset, "records", [])
        validation_records = getattr(self.validation_dataset, "records", [])
        if any(record.get("split") != "val" for record in validation_records):
            raise ValueError("Checkpoint selection requires a validation-only manifest, never training or test")
        for key in ("sequence_id", "source_motion_id"):
            train_ids = {record[key] for record in train_records if record.get(key)}
            validation_ids = {record[key] for record in validation_records if record.get(key)}
            if train_ids & validation_ids:
                raise ValueError(f"Training/validation {key} overlap would leak validation targets")
        # Enforce provisional candidate grouping only when selected in manifests.
        records = train_records + validation_records
        if records and all(record.get("validation_grouping") == "candidate_choreography" for record in records):
            key = "candidate_choreography_id"
            train_ids = {record[key] for record in train_records if record.get(key)}
            validation_ids = {record[key] for record in validation_records if record.get(key)}
            if train_ids & validation_ids:
                raise ValueError("Training/validation overlap violates the selected candidate choreography grouping")
        fps = self.config["diffusion"].get("fps", 30)
        for dataset in (self.dataset, self.validation_dataset):
            for record in getattr(dataset, "records", []):
                actual_fps = record.get("fps")
                if (isinstance(actual_fps, bool) or not isinstance(actual_fps, (int, float))
                        or not math.isclose(actual_fps, fps, abs_tol=1e-8)):
                    raise ValueError("Training/validation manifest FPS differs from the diffusion motion rate")
        if self.config["diffusion"].get("loss_weights", {}).get("contact", 0) and not (
            self.dataset.has_contacts and self.validation_dataset.has_contacts
        ):
            raise ValueError("Contact loss requires prepared contact labels in both training and validation manifests")
        if len(self.validation_dataset) == 0:
            raise ValueError("A nonempty held-out validation manifest is required")
        self.data_fingerprints = fingerprints(self.config)
        self.sampler = ResumableBatchSampler(len(self.dataset), self.options["batch_size"], self.options["seed"])
        self.loader_generator = torch.Generator().manual_seed(self.options["seed"] + 1)
        self.loader = DataLoader(self.dataset, batch_sampler=self.sampler, num_workers=0,
                                 generator=self.loader_generator)
        self.iterator = None
        self.model = build_system(self.config).to(self.device)
        self.model.feature_contract = copy.deepcopy(self.feature_contract)
        self.ema = copy.deepcopy(self.model).requires_grad_(False).eval()
        self.optimizer = torch.optim.AdamW(
            optimizer_groups(self.model, self.options.get("weight_decay", 0.01)),
            lr=self.options["learning_rate"], betas=tuple(self.options.get("betas", [0.9, 0.999])))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.options["max_steps"])
        # BF16 has FP32's exponent range and does not need FP16 loss scaling.
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp and self.amp_dtype == torch.float16)
        self.step, self.attempts = 0, 0
        self.best_validation = float("inf")

    def next_batch(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        return move_batch(batch, self.device)

    @torch.no_grad()
    def update_ema(self):
        decay = self.options.get("ema_decay", 0.995) if self.step > self.options.get("ema_warmup", 100) else 0.0
        source = self.model.state_dict()
        for name, tensor in self.ema.state_dict().items():
            if tensor.is_floating_point():
                tensor.lerp_(source[name], 1.0 - decay)
            else:
                tensor.copy_(source[name])

    def train_step(self):
        """Return metrics and whether an optimizer update actually occurred."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        batches = [self.next_batch() for _ in range(self.options.get("accumulation_steps", 1))]
        examples = sum(batch["motion"].shape[0] for batch in batches)
        totals = {}
        self.attempts += 1
        for batch in batches:
            weight = batch["motion"].shape[0] / examples
            with autocast_context(self.device, enabled=self.amp, dtype=self.amp_dtype):
                losses = self.model(batch)
                loss = losses["loss"] * weight
            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                return {"skipped_nonfinite_loss": 1.0}, False
            self.scaler.scale(loss).backward()
            for name, value in losses.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    totals[name] = totals.get(name, 0.0) + float(value.detach()) * weight
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.options.get("gradient_clip", 1.0))
        finite = bool(torch.isfinite(grad_norm))
        old_scale = self.scaler.get_scale()
        if self.scaler.is_enabled() and finite:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            updated = self.scaler.get_scale() >= old_scale
        elif self.scaler.is_enabled():
            # A norm can overflow even when each gradient was finite. Do not let
            # AdamW update weights/decay if GradScaler did not notice that case.
            self.scaler.update(new_scale=old_scale * self.scaler.get_backoff_factor())
            updated = False
        elif finite:
            self.optimizer.step()
            updated = True
        else:
            updated = False
        self.optimizer.zero_grad(set_to_none=True)
        if updated:
            self.step += 1
            self.scheduler.step()
            self.update_ema()
        totals.update(gradient_norm=float(grad_norm), updated=updated, learning_rate=self.scheduler.get_last_lr()[0])
        return totals, updated

    @torch.no_grad()
    def validate(self):
        saved_rng = rng_state()
        try:
            seed_everything(self.options["seed"] + 100, self.options.get("deterministic", True))
            loader = DataLoader(self.validation_dataset, batch_size=self.options["batch_size"], shuffle=False,
                                num_workers=0, generator=torch.Generator().manual_seed(0))
            totals, count = {}, 0
            self.ema.eval()
            for index, batch in enumerate(loader):
                if index >= self.options.get("validation_batches", 8):
                    break
                batch = move_batch(batch, self.device)
                with autocast_context(self.device, enabled=self.amp, dtype=self.amp_dtype):
                    losses = self.ema(batch)
                n = batch["motion"].shape[0]
                count += n
                for name, value in losses.items():
                    if isinstance(value, torch.Tensor) and value.numel() == 1:
                        totals[name] = totals.get(name, 0.0) + float(value) * n
            result = {k: v / count for k, v in totals.items()}
            if not all(math.isfinite(v) for v in result.values()):
                raise RuntimeError("Validation produced nonfinite losses")
            return result
        finally:
            restore_rng(saved_rng)

    def save(self, name="latest.pt"):
        checkpoint = {"schema_version": 1, "config": self.config, "step": self.step,
                      "attempts": self.attempts, "best_validation": self.best_validation,
                      "model": self.model.state_dict(), "ema": self.ema.state_dict(),
                      "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                      "scaler": self.scaler.state_dict(), "rng": rng_state(),
                      "sampler": self.sampler.state_dict(), "loader_generator": self.loader_generator.get_state(),
                      "data_fingerprints": self.data_fingerprints, "runtime": runtime_metadata(self.device)}
        checkpoint["normalization_metadata"] = getattr(getattr(self.model, "normalizer", None), "metadata", {})
        checkpoint["feature_contract"] = copy.deepcopy(self.feature_contract)
        checkpoint["reconstruction_choices"] = getattr(getattr(self.model, "denoiser", None), "reconstruction_choices", {})
        with tempfile.NamedTemporaryFile(dir=self.output, prefix="checkpoint-", suffix=".tmp", delete=False) as stream:
            temp = Path(stream.name)
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, self.output / name)
        return self.output / name

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("schema_version") != 1:
            raise ValueError("Unsupported checkpoint schema")
        for section in ("model", "conditioning", "diffusion"):
            if checkpoint["config"][section] != self.config[section]:
                raise ValueError(f"Resume {section} configuration differs")
        if checkpoint["config"]["data"].get("window", 150) != self.config["data"].get("window", 150):
            raise ValueError("Resume data.window differs")
        if checkpoint["config"].get("assumptions") != self.config.get("assumptions"):
            raise ValueError("Resume representation/coordinate/reconstruction assumptions differ")
        for key in ("batch_size", "accumulation_steps", "max_steps", "seed", "amp", "learning_rate", "weight_decay", "betas", "ema_decay", "ema_warmup", "gradient_clip", "deterministic", "cpu_threads"):
            if checkpoint["config"]["training"].get(key) != self.options.get(key):
                raise ValueError(f"Resume training.{key} differs; use a new run for changed training settings")
        if checkpoint["config"]["training"].get("amp_dtype", "float16") != self.options.get("amp_dtype", "float16"):
            raise ValueError("Resume training.amp_dtype differs; use a new run for changed precision")
        if checkpoint["data_fingerprints"] != self.data_fingerprints:
            raise ValueError("Training/validation manifests or normalization statistics changed")
        if checkpoint.get("feature_contract") != self.feature_contract:
            raise ValueError("Feature extractors changed across resume")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.ema.load_state_dict(checkpoint["ema"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.sampler.load_state_dict(checkpoint["sampler"])
        self.loader_generator.set_state(checkpoint["loader_generator"].cpu())
        self.step, self.attempts = checkpoint["step"], checkpoint["attempts"]
        self.best_validation = checkpoint["best_validation"]
        self.iterator = None
        restore_rng(checkpoint["rng"])

    def log(self, event, metrics):
        record = {"event": event, "step": self.step, "attempts": self.attempts, **metrics}
        print(json.dumps(record), flush=True)
        with (self.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

    def train(self, *, stop_after=None):
        save_config(self.config, self.output / "config.yaml")
        (self.output / "runtime.json").write_text(json.dumps(runtime_metadata(self.device), indent=2))
        target = min(self.options["max_steps"], stop_after) if stop_after is not None else self.options["max_steps"]
        consecutive_skips = 0
        while self.step < target:
            metrics, updated = self.train_step()
            if not updated:
                consecutive_skips += 1
                self.log("skipped_update", metrics)
                if consecutive_skips >= 20:
                    self.save()
                    raise RuntimeError("20 consecutive nonfinite updates; checkpoint saved for diagnosis")
                continue
            consecutive_skips = 0
            if self.step % self.options.get("log_every", 10) == 0:
                self.log("train", metrics)
            if self.step % self.options.get("validate_every", 1000) == 0:
                validation = self.validate()
                self.log("validation", validation)
                if validation["loss"] < self.best_validation:
                    self.best_validation = validation["loss"]
                    self.save("best.pt")
            if self.step % self.options.get("save_every", 1000) == 0:
                self.save()
        self.save()
        self.save("final.pt" if self.step == self.options["max_steps"] else "paused.pt")
