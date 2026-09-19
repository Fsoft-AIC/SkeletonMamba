"""Extract video flow with pretrained torchvision RAFT-Large C_T_SKHT_V2.

The paper does not specify its RAFT checkpoint or preprocessing settings.
"""
from __future__ import annotations

import importlib.metadata
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dataset.manifest import sha256_file


RAFT_WEIGHTS = "C_T_SKHT_V2"
RAFT_FILENAME = "raft_large_C_T_SKHT_V2-ff5fadd5.pth"
RAFT_URL = "https://download.pytorch.org/models/" + RAFT_FILENAME
RAFT_CACHE = Path("data/cache/torch/hub/checkpoints") / RAFT_FILENAME


class RaftFlow:
    """Reuse one RAFT-Large model across all adjacent frame pairs and videos."""

    def __init__(self, *, weights_path=None, download_weights=False, device="cpu",
                 max_width=640, num_flow_updates=12):
        if (isinstance(max_width, bool) or not isinstance(max_width, int) or max_width < 1
                or isinstance(num_flow_updates, bool) or not isinstance(num_flow_updates, int)
                or num_flow_updates < 1):
            raise ValueError("RAFT width and update count must be positive integers")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.weights_path = Path(weights_path) if weights_path else RAFT_CACHE
        self.download_weights = download_weights
        self.max_width = max_width
        self.num_flow_updates = num_flow_updates
        self._model = None
        self._weights_sha256 = None

    def _resolve_weights(self):
        if self._weights_sha256 is not None:
            return
        if not self.weights_path.is_file():
            if not self.download_weights:
                raise FileNotFoundError(
                    f"RAFT pretrained weights missing: {self.weights_path}. Supply --raft-weights "
                    "or explicitly pass --download-weights; no random or Farneback fallback is used.")
            self.weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(RAFT_URL, str(self.weights_path), hash_prefix="ff5fadd5")
        digest = sha256_file(self.weights_path)
        if not digest.startswith("ff5fadd5"):
            raise ValueError("RAFT weights do not match pinned torchvision RAFT-Large C_T_SKHT_V2")
        self._weights_sha256 = digest

    def identity(self):
        self._resolve_weights()
        return dict(name="torchvision.RAFT_Large_mean_flow_strict_minima", backend="raft",
                    version=importlib.metadata.version("torchvision"), weights=RAFT_WEIGHTS,
                    weights_url=RAFT_URL, weights_sha256=self._weights_sha256,
                    num_flow_updates=self.num_flow_updates, max_width=self.max_width,
                    input="RGB float32 [-1,1]", resize="aspect ratio, width cap, bilinear antialias=False",
                    padding="replicate bottom/right to multiple of 8 and minimum 128; exclude from mean",
                    flow_units="pixels on resized image", device=str(self.device),
                    implementation_sha256=sha256_file(__file__))

    def _load_model(self):
        if self._model is None:
            self._resolve_weights()
            from torchvision.models.optical_flow import raft_large
            model = raft_large(weights=None, progress=False)
            model.load_state_dict(torch.load(self.weights_path, map_location="cpu", weights_only=True), strict=True)
            self._model = model.to(self.device).eval()
        return self._model

    def _image(self, bgr):
        if bgr.ndim != 3 or bgr.shape[-1] != 3 or bgr.dtype != np.uint8 or min(bgr.shape[:2]) < 1:
            raise ValueError("RAFT expects decoded uint8 BGR images [H,W,3]")
        height, width = bgr.shape[:2]
        scale = min(1., self.max_width / width)
        height, width = max(1, round(height * scale)), max(1, round(width * scale))
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        value = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(self.device, dtype=torch.float32)
        if value.shape[-2:] != (height, width):
            value = F.interpolate(value, size=(height, width), mode="bilinear", align_corners=False, antialias=False)
        value = value / 127.5 - 1.
        padded_h, padded_w = max(128, math.ceil(height / 8) * 8), max(128, math.ceil(width / 8) * 8)
        return F.pad(value, (0, padded_w - width, 0, padded_h - height), mode="replicate"), height, width

    @torch.inference_mode()
    def magnitude(self, previous_bgr, current_bgr):
        if previous_bgr.shape != current_bgr.shape:
            raise ValueError("video resolution changed during decoding")
        first, height, width = self._image(previous_bgr)
        second, _, _ = self._image(current_bgr)
        predictions = self._load_model()(first, second, num_flow_updates=self.num_flow_updates)
        if not predictions:
            raise ValueError("RAFT returned no flow estimates")
        flow = predictions[-1]
        if flow.shape != (1, 2, *first.shape[-2:]) or not torch.isfinite(flow).all():
            raise ValueError("invalid or nonfinite RAFT optical flow")
        return float(torch.linalg.vector_norm(flow[0, :, :height, :width], dim=0).mean())
