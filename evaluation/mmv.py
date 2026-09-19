"""MMV Eq.30 with explicit timestamps in seconds.

The score follows the supplement. Extraction helpers use strict local minima
of mean joint speed or supplied optical-flow magnitude, with optional box
smoothing. Exact paper extraction settings are unpublished.
"""
import numpy as np


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def beat_agreement(generated_beats, reference_beats, *, sigma_seconds, empty_policy="nan"):
    """Average over REFERENCE beats, nearest generated beat (direction matters)."""
    if not np.isfinite(sigma_seconds) or sigma_seconds <= 0:
        raise ValueError("sigma_seconds must be finite and positive")
    if empty_policy not in ("nan", "zero", "error"):
        raise ValueError("empty_policy must be nan, zero or error")
    generated, reference = _array(generated_beats), _array(reference_beats)
    if generated.ndim != 1 or reference.ndim != 1:
        raise ValueError("beat timestamps must be one-dimensional seconds")
    if not np.isfinite(generated).all() or not np.isfinite(reference).all():
        raise ValueError("beat timestamps must be finite")
    if not len(generated) or not len(reference):
        if empty_policy == "error":
            raise ValueError("MMV component undefined for empty beat sets")
        return 0.0 if empty_policy == "zero" else float("nan")
    # Search sorted neighbours instead of materializing an O(N*M) distance array.
    generated = np.sort(generated)
    indices = np.searchsorted(generated, reference)
    left = generated[np.maximum(indices - 1, 0)]
    right = generated[np.minimum(indices, len(generated) - 1)]
    distances = np.minimum(np.abs(reference - left), np.abs(reference - right))
    return float(np.exp(-distances**2 / (2 * sigma_seconds**2)).mean())


def compute_mmv(motion_beats, music_beats, head_beats, video_beats, *,
                sigma_seconds, empty_policy="nan"):
    mm = beat_agreement(motion_beats, music_beats, sigma_seconds=sigma_seconds, empty_policy=empty_policy)
    mv = beat_agreement(head_beats, video_beats, sigma_seconds=sigma_seconds, empty_policy=empty_policy)
    return {"mm": mm, "mv": mv, "mmv": 0.5 * (mm + mv)}


def _minima(values, timestamps, valid, smooth_window):
    if smooth_window < 1 or smooth_window % 2 != 1:
        raise ValueError("smooth_window must be a positive odd integer")
    if len(values) < max(3, smooth_window):
        return np.empty(0, dtype=np.float64)
    if not np.isfinite(values[valid]).all():
        raise ValueError("valid magnitudes must be finite")
    clean = np.where(valid, values, 0)
    if smooth_window > 1:
        kernel = np.ones(smooth_window)
        clean = np.convolve(clean, kernel / smooth_window, mode="same")
        valid = np.convolve(valid.astype(float), kernel, mode="same") == smooth_window
    minima = (clean[1:-1] < clean[:-2]) & (clean[1:-1] < clean[2:])
    minima &= valid[:-2] & valid[1:-1] & valid[2:]
    return timestamps[1:-1][minima]


def optical_flow_beats(magnitudes, fps=None, mask=None, *, timestamps=None, smooth_window=1):
    """Extract strict minima; caller defines flow aggregation and time alignment."""
    values = _array(magnitudes)
    if values.ndim != 1:
        raise ValueError("optical-flow magnitude must be [T]")
    if timestamps is None:
        if fps is None or not np.isfinite(fps) or fps <= 0:
            raise ValueError("provide positive fps or explicit timestamps")
        timestamps = np.arange(len(values)) / fps
    else:
        timestamps = _array(timestamps)
        if timestamps.shape != values.shape or not np.isfinite(timestamps).all() or (np.diff(timestamps) <= 0).any():
            raise ValueError("timestamps must be finite and strictly increasing")
    valid = np.ones(len(values), dtype=bool) if mask is None else _array(mask).astype(bool)
    if valid.shape != values.shape:
        raise ValueError("mask must have shape [T]")
    return _minima(values, timestamps, valid, smooth_window)


def kinematic_beats(positions, fps, mask=None, *, smooth_window=1):
    """Mean-joint speed minima at transition midpoint times, in seconds."""
    positions = _array(positions)
    if positions.ndim == 2 and positions.shape[-1] == 3:
        positions = positions[:, None, :]
    if positions.ndim != 3 or positions.shape[-1] != 3 or fps <= 0 or not np.isfinite(fps):
        raise ValueError("positions must be [T,J,3] and fps positive")
    valid = np.ones(len(positions), dtype=bool) if mask is None else _array(mask).astype(bool)
    if valid.shape != (len(positions),):
        raise ValueError("mask must be [T]")
    if not np.isfinite(positions[valid]).all():
        raise ValueError("valid joint positions must be finite")
    positions = np.where(valid[:, None, None], positions, 0)
    speed = np.linalg.norm(np.diff(positions, axis=0), axis=-1).mean(axis=-1) * fps
    times = (np.arange(len(speed)) + 0.5) / fps
    return _minima(speed, times, valid[:-1] & valid[1:], smooth_window)
