"""Shared save_checkpoint/load_checkpoint for any accelerate-FSDP2-wrapped
policy (MolmoAct2, PI05) -- fully generic over `accelerator`/paths, no
policy-specific logic, so kept in one place rather than duplicated per
policy file.
"""
from __future__ import annotations

import json
import os


def save_checkpoint(
    accelerator, out_dir: str, epoch: int, step: int, epoch_complete: bool,
    extra_meta: dict | None = None,
) -> None:
    """Every rank must call this -- FSDP2-sharded save needs all ranks'
    shards, unlike ACT/DDP's rank-0-only pickle save.

    epoch_complete distinguishes a step-windowed (mid-epoch) checkpoint from
    an end-of-epoch one; train_loop.py's resume logic needs this to compute
    the correct start_epoch.

    extra_meta: merged into meta.json as-is -- e.g. train_loop.py passes
    {"normalization": normalization_snapshot}. None -> no change."""
    accelerator.save_state(out_dir)
    if accelerator.is_main_process:
        meta = {"epoch": epoch, "step": step, "epoch_complete": epoch_complete}
        if extra_meta:
            meta.update(extra_meta)
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f)


def load_checkpoint(accelerator, in_dir: str) -> dict:
    accelerator.load_state(in_dir)
    with open(os.path.join(in_dir, "meta.json")) as f:
        return json.load(f)
