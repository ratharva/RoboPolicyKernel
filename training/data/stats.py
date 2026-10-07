"""Approximate normalization statistics (mean/std per feature), sampled from a
subset of the built Ray Dataset -- not a full pass, which isn't needed for
stable mean/std estimates.

Image stats are computed over raw pixel values (RGB: [0, 255]; depth
cameras: whatever the raw sensor units are, e.g. millimeters), NOT rescaled
to [0, 1] -- this must match training/vendor/util.py's NumpyToTorchCollate, which
widens uint8/uint16 images to float32 with no rescale. Only consumed by
cameras using image_normalization.py's "mean_std" mode -- others (unit01,
depth, log) use a fixed formula instead, see that module.

Call this on the dataset BEFORE any HWC->CHW transpose stage (see
training/data/ray_dataset.py's build_lerobot_v3_dataset vs.
transpose_for_training) -- the transpose below assumes HWC input.
"""
from __future__ import annotations

import numpy as np

from training.common.config import DataConfig


def compute_dataset_stats(ds, cfg: DataConfig, n_samples: int = 500) -> dict:
    print(f"Sampling {n_samples} rows to compute normalization stats ...")
    rows = ds.take(n_samples)
    if not rows:
        raise ValueError("dataset produced zero rows -- can't compute normalization stats")

    states = np.stack([r["observation.state"] for r in rows])  # (N, state_dim)
    actions = np.stack([r["action"] for r in rows])  # (N, chunk_size, action_dim)
    is_pad = np.stack([r["action_is_pad"] for r in rows])  # (N, chunk_size)
    valid_actions = actions[~is_pad]  # (M, action_dim) -- excludes padded chunk tail

    stats: dict = {
        "observation.state": {
            "mean": states.mean(axis=0).tolist(),
            "std": states.std(axis=0).tolist(),
        },
        "action": {
            "mean": valid_actions.mean(axis=0).tolist(),
            "std": valid_actions.std(axis=0).tolist(),
        },
    }

    image_cols = [f"observation.images.{k}" for k in cfg.robot.camera_keys]
    image_cols += [f"observation.images.depth_{k}" for k in cfg.robot.depth_camera_keys]
    for col in image_cols:
        imgs = np.stack([r[col] for r in rows]).astype(np.float32)  # (N,H,W,C), range [0,255] for RGB
        imgs = imgs.transpose(0, 3, 1, 2)  # (N,C,H,W)
        c = imgs.shape[1]  # 3 for RGB; depth cameras are single-channel, not hardcoded here
        stats[col] = {
            "mean": imgs.mean(axis=(0, 2, 3)).reshape(c, 1, 1).tolist(),
            "std": imgs.std(axis=(0, 2, 3)).reshape(c, 1, 1).tolist(),
        }

    print(f"  computed stats for: {list(stats.keys())}")
    return stats


def compute_quantile_stats(
    ds, cfg: DataConfig, n_samples: int = 500, quantiles: tuple[float, float] = (0.01, 0.99)
) -> dict:
    """Same 500-row sampling shape as compute_dataset_stats -- only the
    reduction differs (np.quantile instead of mean/std). Key names match
    lerobot's NormalizerProcessorStep._apply_transform: QUANTILES wants
    "q01"/"q99", QUANTILE10 wants "q10"/"q90" -- derived generically from
    whatever quantiles tuple is passed. Only STATE/ACTION -- images always
    stay VISUAL="IDENTITY" (see training/model/image_normalization.py)."""
    lo, hi = quantiles
    lo_key, hi_key = f"q{round(lo * 100):02d}", f"q{round(hi * 100):02d}"

    print(f"Sampling {n_samples} rows to compute {lo_key}/{hi_key} quantile stats ...")
    rows = ds.take(n_samples)
    if not rows:
        raise ValueError("dataset produced zero rows -- can't compute normalization stats")

    states = np.stack([r["observation.state"] for r in rows])  # (N, state_dim)
    actions = np.stack([r["action"] for r in rows])  # (N, chunk_size, action_dim)
    is_pad = np.stack([r["action_is_pad"] for r in rows])  # (N, chunk_size)
    valid_actions = actions[~is_pad]  # (M, action_dim) -- excludes padded chunk tail

    stats: dict = {
        "observation.state": {
            lo_key: np.quantile(states, lo, axis=0).tolist(),
            hi_key: np.quantile(states, hi, axis=0).tolist(),
        },
        "action": {
            lo_key: np.quantile(valid_actions, lo, axis=0).tolist(),
            hi_key: np.quantile(valid_actions, hi, axis=0).tolist(),
        },
    }
    print(f"  computed {lo_key}/{hi_key} quantile stats for: {list(stats.keys())}")
    return stats
