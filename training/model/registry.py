"""Dispatch table between policy families (ACT, MolmoAct2, pi0.5) for
train_loop.py, so the shared per-worker training loop doesn't hardcode
which policy it's running.

Factories, not eager module-level instances -- POLICY_ADAPTER_FACTORIES'
values are called lazily inside get_adapter(), so importing this module
never imports a policy's module unless it's actually selected at runtime.
This lets a policy with extra/conflicting dependencies fail fast with a
clear ModuleNotFoundError rather than breaking import for scripts that
never asked for it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass
class PolicyAdapter:
    # (data_cfg, overrides, train_cfg, dataset_stats, device) -> (policy, preprocessor)
    build: Callable[..., tuple[Any, Any]]
    # (policy, inputs) -> (loss, metrics)
    forward_loss: Callable[[Any, dict], tuple[torch.Tensor, dict[str, float]]]
    needs_task: bool
    # Called once right after build(), before any distributed wrapping.
    post_build_hook: Callable[[Any, Any], None] | None = None
    # None -> train_loop.py falls back to ray.train.torch.prepare_model(policy)
    # + the pickle-state_dict save/load path. Set (MolmoAct2 fsdp2 only) to
    # swap in accelerate-based FSDP2 wrapping and sharded checkpointing.
    wrap_for_training: Callable[..., Any] | None = None
    save_checkpoint: Callable[..., None] | None = None
    load_checkpoint: Callable[..., dict] | None = None
    # Extra kwargs for ray.train.torch.prepare_model's parallel_strategy_kwargs
    # -- only meaningful when wrap_for_training is None (the DDP path).
    prepare_model_kwargs: dict | None = None
    # True when a Ray Data preprocessing stage already ran the policy's
    # preprocessor upstream (training/data/ray_dataset.py) -- train_loop.py
    # then skips calling preprocessor(batch) itself for this policy.
    preprocessing_offloaded: bool = False
    # (policy, inputs) -> {camera_or_output_name: (T,H,W,C) uint8 frames} or
    # None. Pure groundwork for a future policy that actually predicts
    # visual frames (e.g. a world model) -- ACT/MolmoAct2/PI05 all only
    # ever predict actions (confirmed: every one of their output_features
    # dicts declares "action" and nothing visual), so this is None for all
    # three today and train_loop.py's val pass silently skips the
    # predicted-frames GIF entirely when it's unset. A future adapter that
    # DOES predict frames sets this and gets "watch the generated frames
    # improve across training" for free -- see train_loop.py's val-pass
    # block, logged under val_gif/<key> at the same step as the val pass,
    # so W&B's own per-step media history is what shows the improvement
    # over time, no extra "compare across steps" mechanism needed.
    predict_frames: Callable[[Any, dict], dict[str, Any] | None] | None = None
    # (policy, inputs) -> {metric_name: value}. None (every policy today)
    # -- train_loop.py skips this entirely. A custom/future adapter that
    # computes something beyond forward_loss's own metrics dict sets this;
    # its keys get merged into the SAME step_metrics/eval_step_metrics
    # dict forward_loss's own metrics already flow through, so they
    # automatically show up under train/*, window/*, epoch/*, val/*, AND
    # test/* (both TensorBoard and W&B, subject to the same
    # --wandb-metrics/--wandb-metric-groups filtering) -- no new plumbing needed.
    extra_metrics: Callable[[Any, dict], dict[str, float]] | None = None
    # (overrides) -> PretrainedNormalization. None for ACT only.
    # Implemented by pi05 and MolmoAct2 -- see training/model/
    # normalization.py's resolve_normalization, which calls this.
    get_pretrained_normalization: Callable[[Any], Any] | None = None
    # Overrides DataConfig.default_image_normalization's dataclass default
    # ("mean_std") unless --default-image-normalization was passed
    # explicitly. None for ACT/MolmoAct2. pi05 sets "unit01" -- lerobot's
    # pi05 model unconditionally expects [0,1] image input.
    preferred_visual_normalization: str | None = None


def _act_adapter() -> PolicyAdapter:
    from training.model import act
    return PolicyAdapter(build=act.build_policy_and_processor, forward_loss=act.forward_loss, needs_task=False)


def _molmoact2_adapter(distributed_strategy: str, offload_tokenization: bool) -> PolicyAdapter:
    from training.model import molmoact2
    is_fsdp2 = distributed_strategy == "fsdp2"
    return PolicyAdapter(
        build=molmoact2.build_policy_and_processor,
        forward_loss=molmoact2.forward_loss,
        needs_task=True,
        post_build_hook=molmoact2.enable_training_optimizations,
        wrap_for_training=molmoact2.wrap_for_training if is_fsdp2 else None,
        save_checkpoint=molmoact2.save_checkpoint if is_fsdp2 else None,
        load_checkpoint=molmoact2.load_checkpoint if is_fsdp2 else None,
        # LoRA + gradient checkpointing under plain DDP can fail with "did
        # not receive grad for all parameters" when not every
        # requires_grad=True parameter participates in every forward pass.
        prepare_model_kwargs=None if is_fsdp2 else {"find_unused_parameters": True},
        preprocessing_offloaded=offload_tokenization,
        get_pretrained_normalization=molmoact2.get_pretrained_normalization,
    )


def _pi05_adapter(distributed_strategy: str) -> PolicyAdapter:
    from training.model import pi05
    is_fsdp2 = distributed_strategy == "fsdp2"
    return PolicyAdapter(
        build=pi05.build_policy_and_processor,
        forward_loss=pi05.forward_loss,
        needs_task=True,
        # No post_build_hook -- gradient checkpointing auto-wires from
        # PI05Config's own flag at construction time.
        wrap_for_training=pi05.wrap_for_training if is_fsdp2 else None,
        save_checkpoint=pi05.save_checkpoint if is_fsdp2 else None,
        load_checkpoint=pi05.load_checkpoint if is_fsdp2 else None,
        prepare_model_kwargs=None if is_fsdp2 else {"find_unused_parameters": True},
        # Both orthogonal to distributed_strategy -- wired unconditionally.
        get_pretrained_normalization=pi05.get_pretrained_normalization,
        preferred_visual_normalization="unit01",
    )


def get_adapter(policy_type: str, distributed_strategy: str = "ddp", offload_tokenization: bool = False) -> PolicyAdapter:
    if policy_type == "act":
        return _act_adapter()
    if policy_type == "molmoact2":
        return _molmoact2_adapter(distributed_strategy, offload_tokenization)
    if policy_type == "pi05":
        return _pi05_adapter(distributed_strategy)
    raise ValueError(f"unknown policy_type {policy_type!r}, expected one of ('act', 'molmoact2', 'pi05')")
