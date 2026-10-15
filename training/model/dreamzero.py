"""DreamZero world action model (NVIDIA GEAR, Apache-2.0, github.com/dreamzero0/dreamzero).

The model, its Hydra configs, DreamTransform/collate (view tiling, prompt template,
umt5 tokenization, padding) and its checkpoint layout are DreamZero's own; this module
only wires them into our Ray Data + Ray Train loop. Install DreamZero as an editable
checkout so its YAML configs ship with it (a plain pip install drops them):

    git clone https://github.com/dreamzero0/dreamzero && git -C dreamzero checkout <DREAMZERO_COMMIT>
    pip install --no-deps --ignore-requires-python -e dreamzero
"""
from __future__ import annotations

import gc
import os

import numpy as np
import torch

from training.config import DreamZeroConfigOverrides

DREAMZERO_COMMIT = "ab790c198fbce33503358efbbd4187ce9a89adf3"

# Only verified pairings: droid's 8 dims are joint_0..6 + gripper in both state and action,
# DreamZero-DROID's own layout (joints relative to block-start state, gripper absolute).
DATASET_PRESETS = {
    "droid": {"embodiment_tag": "oxe_droid", "relative_action_dims": list(range(7))},
}

# Batch keys DreamZero's action head reads in training forward.
_FORWARD_KEYS = (
    "images", "text", "text_attention_mask", "state", "state_mask", "action", "action_mask",
    "embodiment_id", "has_real_action",
)


def resolve_embodiment(overrides: DreamZeroConfigOverrides, dataset_source: str) -> tuple[str, list[int]]:
    preset = DATASET_PRESETS.get(dataset_source, {})
    tag = overrides.embodiment_tag or preset.get("embodiment_tag")
    rel = overrides.relative_action_dims if overrides.relative_action_dims is not None else preset.get(
        "relative_action_dims"
    )
    if not tag or rel is None:
        raise ValueError(
            f"no DreamZero preset for dataset_source={dataset_source!r} -- pass --dreamzero-embodiment-tag "
            f"and --dreamzero-relative-action-dims explicitly (verified presets: {sorted(DATASET_PRESETS)})"
        )
    return tag, list(rel)


def video_steps(o: DreamZeroConfigOverrides) -> list[int]:
    return list(range(0, o.chunk_size + 1, o.frame_stride))


def state_steps(o: DreamZeroConfigOverrides) -> list[int]:
    return [b * o.action_per_block for b in range(o.num_blocks)]


def config_dir() -> str:
    import groot

    path = os.path.join(os.path.dirname(groot.__file__), "vla", "configs")
    if not os.path.isfile(os.path.join(path, "conf.yaml")):
        raise FileNotFoundError(
            f"DreamZero's Hydra configs not found at {path} -- install DreamZero as an editable checkout "
            "(pip install -e), a plain pip install drops its YAML files."
        )
    return path


def compose_dreamzero_cfg(o: DreamZeroConfigOverrides, num_views: int):
    from hydra import compose, initialize_config_dir

    tiled = num_views > 1
    frame_seqlen = (o.image_height * (2 if tiled else 1) // 16) * (o.image_width * (2 if tiled else 1) // 16)
    wan = o.wan_ckpt_dir
    overrides = [
        "model=dreamzero/vla",
        "data=dreamzero/droid_relative",
        "model/dreamzero/action_head=wan_flow_matching_action_tf",
        "model/dreamzero/transform=dreamzero_cotrain",
        f"train_architecture={'lora' if o.train_mode == 'lora' else 'full'}",
        f"num_frames={len(video_steps(o))}",
        f"action_horizon={o.action_per_block}",
        f"num_views={num_views}",
        "num_frame_per_block=2",
        f"num_action_per_block={o.action_per_block}",
        "num_state_per_block=1",
        f"max_chunk_size={o.num_blocks}",
        f"frame_seqlen={frame_seqlen}",
        f"max_state_dim={o.max_state_dim}",
        f"max_action_dim={o.max_action_dim}",
        f"image_resolution_width={o.image_width}",
        f"image_resolution_height={o.image_height}",
        f"tokenizer_path={o.tokenizer_path}",
        f"action_head_cfg.config.lora_rank={o.lora_rank}",
        f"action_head_cfg.config.lora_alpha={o.lora_alpha}",
        "droid_data_root=unused", "output_dir=unused", "wandb_project=unused",
    ]
    if wan:
        overrides += [
            f"dit_version={wan}",
            f"text_encoder_pretrained_path={wan}/models_t5_umt5-xxl-enc-bf16.pth",
            f"image_encoder_pretrained_path={wan}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            f"vae_pretrained_path={wan}/Wan2.1_VAE.pth",
        ]
    if o.pretrained_path:
        # Same pair their own new-embodiment scripts pass when starting from a full checkpoint.
        overrides += [
            "++action_head_cfg.config.skip_component_loading=true",
            "++action_head_cfg.config.defer_lora_injection=true",
        ]
    overrides += list(o.hydra_overrides)
    with initialize_config_dir(config_dir=config_dir(), version_base=None):
        return compose(config_name="conf", overrides=overrides)


def _load_full_checkpoint(model, ckpt_dir: str) -> None:
    """Mirrors DreamZero's BaseExperiment.create_model: shards with strict=False, then deferred LoRA."""
    import json

    from safetensors.torch import load_file

    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    single_path = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(index_path):
        with open(index_path) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        for shard in shards:
            model.load_state_dict(load_file(os.path.join(ckpt_dir, shard)), strict=False)
    elif os.path.exists(single_path):
        model.load_state_dict(load_file(single_path), strict=False)
    else:
        raise FileNotFoundError(f"no model.safetensors(.index.json) under {ckpt_dir}")
    head = model.action_head
    if getattr(head.config, "defer_lora_injection", False):
        head.inject_lora_after_loading()


def _unpack_block_outputs_without_grad(model) -> None:
    """Upstream bug: CausalWanModel._forward_train does `x = block(x, **kwargs)` when grad is off,
    but blocks return (x, kv_cache) -- only its grad-checkpoint branch unpacks. Their trainer never
    runs that forward without grad; our val pass (inference_mode) does. Training-forward calls pass
    no kv_cache, so only those get unpacked; their kv-cached inference paths are untouched."""
    for block in model.modules():
        if type(block).__name__ != "CausalWanAttentionBlock":
            continue
        orig = block.forward

        def forward(*args, _orig=orig, **kwargs):
            out = _orig(*args, **kwargs)
            if not torch.is_grad_enabled() and kwargs.get("kv_cache") is None and isinstance(out, tuple):
                return out[0]
            return out

        block.forward = forward


def _return_plain_dict(model) -> None:
    """VLA.forward returns a BatchFeature (a UserDict); DDP's _find_tensors only walks real dicts,
    so it sees no outputs, marks every param unused, then backward marks them ready twice."""
    orig = model.forward
    model.forward = lambda *a, **k: dict(orig(*a, **k))


def build_policy_and_processor(data_cfg, overrides: DreamZeroConfigOverrides, train_cfg, dataset_stats, device):
    import torch.distributed as dist

    # One local rank at a time: each CPU build peaks at a full fp32 copy (~80GB) before moving to GPU.
    if not dist.is_initialized():
        return _build(data_cfg, overrides, device)
    import ray.train

    ctx = ray.train.get_context()
    model = None
    for turn in range(ctx.get_local_world_size()):
        if turn == ctx.get_local_rank():
            model = _build(data_cfg, overrides, device)[0]
            gc.collect()
        dev = torch.device(device)
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        dist.barrier(device_ids=[idx] if dev.type == "cuda" else None)
    return model, None


def _build(data_cfg, overrides: DreamZeroConfigOverrides, device):
    from hydra.utils import instantiate

    cfg = compose_dreamzero_cfg(overrides, num_views=len(data_cfg.robot.camera_keys))
    model = instantiate(cfg.model)
    _unpack_block_outputs_without_grad(model)
    _return_plain_dict(model)
    if overrides.pretrained_path:
        _load_full_checkpoint(model, overrides.pretrained_path)
    # DreamZero's head takes its working dtype from its first parameter and expects bf16
    # (their recipe runs DeepSpeed bf16); trainable params stay fp32 like DeepSpeed's master copy.
    model = model.to(device=device, dtype=torch.bfloat16)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.train()
    return model, None


def forward_loss(policy, inputs: dict):
    batch = {k: inputs[k] for k in _FORWARD_KEYS if k in inputs}
    # Our collate turns non-camera uint8 into long/float; DreamZero only rescales uint8 video itself.
    if batch["images"].dtype != torch.uint8:
        batch["images"] = batch["images"].round().clamp(0, 255).to(torch.uint8)
    out = policy(batch)
    return out["loss"], {
        "dynamics_loss": float(out["dynamics_loss"].detach()),
        "action_loss": float(out["action_loss"].detach()),
    }


def valid_episodes(v3_root: str, require_success: bool) -> list[int]:
    """Episodes with a non-empty instruction (and is_episode_successful, when that column exists) --
    DreamZero's own DROID data drops unannotated and unsuccessful episodes."""
    import fsspec
    import pyarrow.parquet as pq
    import ray

    with fsspec.open(f"{v3_root}/meta/tasks.parquet", "rb") as f:
        tasks = pq.read_table(f).to_pandas()  # index = task text, column = task_index
    empty = {int(i) for name, i in tasks["task_index"].items() if not str(name).strip()}
    data = ray.data.read_parquet(f"{v3_root}/data")
    cols = ["episode_index", "task_index"]
    if require_success and "is_episode_successful" in data.schema().names:
        cols.append("is_episode_successful")
    df = data.select_columns(cols).to_pandas()
    ok = ~df.task_index.isin(empty)
    if "is_episode_successful" in df.columns:
        ok &= df.is_episode_successful.astype(bool)
    good = df[ok].groupby("episode_index").size()
    total = df.groupby("episode_index").size()
    return sorted(int(e) for e in good.index if good[e] == total[e])


def to_relative(state: np.ndarray, action: np.ndarray, rel_dims: list[int], per_block: int) -> np.ndarray:
    """DreamZero's relative actions: each per_block chunk minus the state at that block's start.
    state [..., num_blocks, S], action [..., num_blocks * per_block, A]; dims are positional."""
    out = action.astype(np.float32, copy=True)
    for b in range(state.shape[-2]):
        sl = slice(b * per_block, (b + 1) * per_block)
        out[..., sl, rel_dims] -= state[..., b : b + 1, rel_dims]
    return out


def q99_normalize(x: np.ndarray, stats: dict) -> np.ndarray:
    """DreamZero's q99 mode: 2*(x-q01)/(q99-q01)-1 clamped to [-1,1]; dims with q01==q99 pass through."""
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    span = q99 - q01
    ok = span != 0
    out = x.astype(np.float32, copy=True)
    out[..., ok] = 2 * (x[..., ok] - q01[ok]) / span[ok] - 1
    return np.clip(out, -1, 1)


class DreamZeroPreprocessStage:
    """Ray Data map_batches actor: relative actions + q99 normalization (ours), then DreamZero's
    own DreamTransform.apply (view tiling, padding, masks, prompt template, umt5 tokenization)."""

    def __init__(self, camera_columns: list[str], overrides: DreamZeroConfigOverrides, embodiment_tag: str,
                 relative_action_dims: list[int], dataset_stats: dict):
        from hydra.utils import instantiate

        from groot.vla.data.schema.embodiment_tags import EmbodimentTag

        cfg = compose_dreamzero_cfg(overrides, num_views=len(camera_columns))
        self.transform = instantiate(cfg.model_specific_transform)
        self.transform.embodiment_tag = EmbodimentTag(embodiment_tag)
        self.transform.num_views = len(camera_columns)
        self.camera_columns = camera_columns
        self.o = overrides
        self.rel_dims = relative_action_dims
        self.state_stats = dataset_stats["observation.state"]
        self.action_stats = dataset_stats["action"]

    def _resize(self, frames: np.ndarray) -> np.ndarray:
        # [B, T, H, W, C] uint8 -> [B, T, h, w, C] uint8
        b, t, h, w, c = frames.shape
        if (h, w) == (self.o.image_height, self.o.image_width):
            return frames
        frames = np.ascontiguousarray(frames)
        if not frames.flags.writeable:  # Arrow buffers are read-only
            frames = frames.copy()
        x = torch.from_numpy(frames).reshape(b * t, h, w, c).permute(0, 3, 1, 2).float()
        x = torch.nn.functional.interpolate(
            x, size=(self.o.image_height, self.o.image_width), mode="bilinear", antialias=True, align_corners=False,
        )
        x = x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        return x.reshape(b, t, self.o.image_height, self.o.image_width, c).numpy()

    def __call__(self, batch: dict) -> dict:
        video = np.stack([self._resize(np.asarray(batch[c])) for c in self.camera_columns], axis=2)  # [B,T,V,H,W,C]
        state = np.asarray(batch["observation.state"], dtype=np.float32)
        action = to_relative(state, np.asarray(batch["action"], dtype=np.float32), self.rel_dims, self.o.action_per_block)
        tasks = np.asarray(batch["task"], dtype=object)
        out = self.transform.apply({
            "video": video,
            "state": q99_normalize(state, self.state_stats),
            "action": q99_normalize(action, self.action_stats),
            "annotation.task": tasks,
        })
        result = {k: (v.numpy() if torch.is_tensor(v) else np.asarray(v)) for k, v in out.items() if k in _FORWARD_KEYS}
        result["task"] = tasks
        return result


def compute_dreamzero_stats(v3_root: str, episodes: list[int], overrides: DreamZeroConfigOverrides,
                            relative_action_dims: list[int]) -> dict:
    """q01/q99 over block-start states and relative actions of every full window in `episodes`,
    from parquet only (no video decode)."""
    import ray

    cols = ["episode_index", "frame_index", "observation.state", "action"]
    df = ray.data.read_parquet(f"{v3_root}/data", columns=cols).to_pandas()
    df = df[df.episode_index.isin(set(episodes))].sort_values(["episode_index", "frame_index"])
    s_steps = np.asarray(state_steps(overrides))
    a_steps = np.arange(overrides.chunk_size)
    states, actions = [], []
    for _, ep in df.groupby("episode_index"):
        st = np.stack(ep["observation.state"].to_numpy()).astype(np.float32)
        ac = np.stack(ep["action"].to_numpy()).astype(np.float32)
        anchors = np.arange(0, len(ep) - overrides.chunk_size + 1, overrides.anchor_stride)
        if len(anchors) == 0:
            continue
        w_state = st[anchors[:, None] + s_steps]
        w_action = to_relative(w_state, ac[anchors[:, None] + a_steps], relative_action_dims, overrides.action_per_block)
        states.append(w_state.reshape(-1, st.shape[-1]))
        actions.append(w_action.reshape(-1, ac.shape[-1]))
    if not states:
        raise ValueError("no episode is long enough for one full DreamZero window -- can't compute stats")
    s, a = np.concatenate(states), np.concatenate(actions)
    return {
        "observation.state": {"q01": np.quantile(s, 0.01, axis=0).tolist(), "q99": np.quantile(s, 0.99, axis=0).tolist()},
        "action": {"q01": np.quantile(a, 0.01, axis=0).tolist(), "q99": np.quantile(a, 0.99, axis=0).tolist()},
    }


def wrap_for_training(policy, optimizer, overrides: DreamZeroConfigOverrides, device):
    """FSDP2 via accelerate (same path as molmoact2.wrap_for_training): shards each DiT block.
    Returns (policy, optimizer, accelerator)."""
    from accelerate import Accelerator
    from accelerate.utils import FullyShardedDataParallelPlugin

    wrap = "CausalWanAttentionBlock"
    if wrap not in {type(m).__name__ for m in policy.modules()}:
        raise RuntimeError(f"{wrap} not found on the built DreamZero model -- its DiT block class name changed")
    fsdp_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        reshard_after_forward=True,
        cpu_offload=overrides.fsdp_cpu_offload,
        state_dict_type=overrides.fsdp_state_dict_type,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=[wrap],
    )
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin, mixed_precision="bf16")
    policy, optimizer = accelerator.prepare(policy, optimizer)
    return policy, optimizer, accelerator


from training.model.fsdp2_checkpoint import load_checkpoint, save_checkpoint  # noqa: E402,F401
