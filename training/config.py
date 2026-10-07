"""Training-time configuration: hyperparameters and per-policy overrides.
DataConfig (the dataset/robot description shared with data prep) lives in
training/common/config.py -- see that module and
training/data_prep/schema_loader.py for how a --dataset-source populates it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import draccus

from training.common.config import DataConfig


@dataclass
class PolicyOverrides(draccus.ChoiceRegistry):
    """Shared base for the three *ConfigOverrides dataclasses below, so
    draccus's choice-registry mechanism can select one from a --config-file
    YAML's `model: {type: act|molmoact2|pi05, ...}` section (see
    training/train.py's --config-file). Purely a draccus-integration
    detail -- runtime code (train_loop.py, model/*.py) keeps accessing
    RunConfig.model by plain attribute access, unaffected by this base."""

    # Populated by train.py's resolve_normalization(), driver-side, before
    # TorchTrainer construction -- not a CLI/YAML input itself. None for
    # ACT/MolmoAct2 and for pi05 when normalization resolution is unused.
    # Shape matches lerobot's normalization_mapping (FeatureType name ->
    # NormalizationMode name).
    resolved_normalization_mode: dict[str, str] | None = None


@PolicyOverrides.register_subclass("act")
@dataclass
class ACTConfigOverrides(PolicyOverrides):
    """Values passed to lerobot's ACTConfig -- field names must match the
    installed lerobot version; see training/model/act.py."""
    chunk_size: int = 100
    n_action_steps: int = 100
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4
    kl_weight: float = 10.0
    dropout: float = 0.1


@PolicyOverrides.register_subclass("molmoact2")
@dataclass
class MolmoAct2ConfigOverrides(PolicyOverrides):
    """Values passed to lerobot's MolmoAct2Config; see training/model/molmoact2.py."""
    checkpoint_path: str = "allenai/MolmoAct2"
    # Pins weight loading to a specific Hub revision/commit/tag -- wires into
    # MolmoAct2Config's own real (previously unwired) checkpoint_revision field.
    revision: str | None = None
    # Selects a real, checkpoint-published normalization/config tag (e.g.
    # "franka_droid") from the checkpoint's own norm_stats.json -- wires into
    # MolmoAct2Config's own real (previously unwired) norm_tag field. Explicit
    # only, no auto-derivation from --dataset-source (the mapping isn't 1:1 --
    # e.g. "franka_droid" matches --dataset-source droid, not droid_100). When
    # set, MolmoAct2Policy's own _apply_norm_tag_metadata silently overwrites
    # chunk_size/n_action_steps to the tag's values at construction time --
    # see training/model/normalization.py's validation guard for this.
    norm_tag: str | None = None
    chunk_size: int = 30          # MolmoAct2Config's own default -- NOT ACT's 100
    n_action_steps: int = 30
    action_mode: str = "continuous"  # real default is "both"; narrowed to skip the
                                      # discrete FAST-tokenizer dependency entirely
    num_flow_timesteps: int = 8
    # "lora" | "fft" | "freeze" -- our own CLI-level convenience concept, not a
    # real MolmoAct2Config field. Translated to the real enable_lora_vlm/
    # enable_lora_action_expert/train_action_expert_only booleans in
    # model/molmoact2.py's build_molmoact2_config.
    train_mode: str = "lora"
    lora_rank: int = 64    # MolmoAct2Config's own default
    lora_alpha: int = 16   # MolmoAct2Config's own default (alpha < rank here)
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    gradient_checkpointing: bool = True   # our own default; real MolmoAct2Config default is False
    # "bfloat16" (real MolmoAct2Config default already) or "float32"/"float16".
    # Unlike PI05, MolmoAct2Config.model_dtype already defaults to bfloat16
    # upstream and drives a real torch.autocast(dtype=model_dtype) context
    # (confirmed in modeling_molmoact2.py) -- this pipeline was already
    # getting that benefit by not touching the field; this just makes it a
    # real, user-selectable override (e.g. float32 to debug a numerics issue).
    dtype: str = "bfloat16"
    image_keys: list[str] | None = None   # None -> derived from data_cfg.robot.camera_keys at build time
    setup_type: str = ""     # REQUIRED free-text embodiment prompt, validated non-empty at build time
    control_mode: str = ""   # REQUIRED free-text control-mode prompt, validated non-empty at build time

    # Per-group learning rates -- MolmoAct2's get_optim_params() returns 4
    # groups (vlm/vit/connector/action_expert), unlike ACT's 2. None falls
    # back to TrainConfig.lr at build time.
    optimizer_vit_lr: float | None = None
    optimizer_connector_lr: float | None = None
    optimizer_action_expert_lr: float | None = None

    # --- Full-finetune / FSDP2 -- only sane with train_mode="fft" ---
    distributed_strategy: str = "ddp"   # "ddp" | "fsdp2"
    fsdp_cpu_offload: bool = False      # trades speed for fitting on fewer/smaller GPUs
    fsdp_state_dict_type: str = "sharded_state_dict"  # accelerate's own default is "full_state_dict",
                                                       # which gathers the whole model onto one rank

    # --- Ray Data preprocessing offload (training/data/ray_dataset.py) ---
    offload_tokenization: bool = False


@PolicyOverrides.register_subclass("pi05")
@dataclass
class Pi05ConfigOverrides(PolicyOverrides):
    """Values passed to lerobot's PI05Config; see training/model/pi05.py.

    Deliberately no single "train_mode" convenience string like
    MolmoAct2ConfigOverrides -- PI05Config exposes two independent booleans
    directly (freeze_vision_encoder, train_expert_only) and has no LoRA/peft
    support.
    """
    # REQUIRED, no safe default -- find a real pretrained checkpoint on the
    # HF Hub and pass it via --pi05-pretrained-path.
    pretrained_path: str = ""
    # Pins both weight loading and normalization-config loading to a
    # specific Hub revision/commit/tag. None -> Hub's default (latest).
    revision: str | None = None
    chunk_size: int = 50          # PI05Config's own default
    n_action_steps: int = 50      # PI05Config's own default
    freeze_vision_encoder: bool = False   # freezes the vision tower only
    train_expert_only: bool = False       # only the action expert trains
    gradient_checkpointing: bool = True   # our own default; real PI05Config default is False
    # Precision options: "bfloat16" or "float32". Never wired through before
    # this field existed -- PI05Config itself defaults to "float32" (its
    # OWN dataclass default), so every run silently trained in full
    # fp32 on H100/A100 hardware that gets real speedup from bf16 tensor
    # cores. bfloat16 casts most of the model (confirmed via
    # to_bfloat16_for_selected_params's real source) but deliberately keeps
    # vision_tower/multi_modal_projector/layernorms/model.norm in float32
    # for numerical stability -- this is the library's own designed mixed-
    # precision split, not a blunt whole-model cast.
    dtype: str = "bfloat16"
    # Pads input_features with dummy observation.images.empty_camera_{i}
    # VISUAL features up to a target camera count, for when a pretrained
    # checkpoint expects more camera slots than this dataset has.
    empty_cameras: int = 0

    # --- Full-finetune / FSDP2 -- unlike MolmoAct2, NOT restricted to any
    # particular freeze_vision_encoder/train_expert_only combination: PI05
    # has no LoRA to already solve memory (which is why MolmoAct2 restricts
    # fsdp2 to train_mode=="fft"), and DDP always fully replicates the
    # whole model regardless of what's frozen -- so FSDP2's memory-sharding
    # benefit applies to every PI05 training mode, not just a "full
    # finetune" one. ---
    distributed_strategy: str = "ddp"   # "ddp" | "fsdp2"
    fsdp_cpu_offload: bool = False      # trades speed for fitting on fewer/smaller GPUs
    fsdp_state_dict_type: str = "sharded_state_dict"  # accelerate's own default is "full_state_dict",
                                                       # which gathers the whole model onto one rank

    # --- Speed flags (see training/README.md's "π0.5 training speed") ---

    # Recommended: closes most of the native-vs-Ray per-step compute gap
    # (~1.12s -> ~0.46s). Wires PI05Config's own compile_model/compile_mode
    # (lerobot already does `torch.compile(self.forward, mode=...)` when
    # set). Use compile_mode "default" or "reduce-overhead", not
    # "max-autotune" -- its warmup cost is severe.
    compile_model: bool = False
    compile_mode: str = "max-autotune"   # PI05Config's own default -- see caveat above

    # Recommended: ~1.12s -> ~0.60s alone, ~0.35s combined with
    # compile_model. Monkey-patches vision_tower/multi_modal_projector to
    # bf16 (lerobot keeps them float32 by default) -- see
    # training/model/pi05.py's _apply_vision_bf16_override. Not yet
    # confirmed stable over a long training run.
    vision_bf16: bool = False

    # No measured speed effect -- kept as a documented dead end. Monkey-
    # patches lerobot's _apply_checkpoint to skip checkpointing the tiny
    # action_out_proj layer. See _install_narrow_checkpoint_patch.
    narrow_checkpoint: bool = False

    # Diagnostic, zero risk -- prints the real attn_implementation
    # (confirmed "sdpa", not "eager", for training). See
    # training/model/pi05.py's _print_attn_implementation.
    print_attn_impl: bool = False


@dataclass
class TrainConfig:
    num_epochs: int = 1
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 1e-5
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    # AdamW betas/eps -- PyTorch's own defaults, used by every policy today.
    # A real optimizer recipe (e.g. AllenAI's own MolmoAct2 finetuning code)
    # can need different values (betas=(0.9, 0.95), eps=1e-6) -- generic
    # fields here rather than a MolmoAct2-only concept, since any policy's
    # optimizer construction (train_loop.py) already goes through one shared
    # torch.optim.AdamW(...) call.
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    max_train_steps: int | None = None
    # Step-windowed reporting of TRAINING loss: finer-grained than
    # per-epoch, so early stopping gets more than one data point even with
    # num_epochs=1. Unrelated to the val/test split below -- this is
    # purely a training-loss cadence.
    window_every_steps: int = 200
    # Stop once the windowed TRAINING loss hasn't improved for this many
    # windows in a row. None disables early stopping. Deliberately stays
    # training-loss-based even when a val split is active.
    early_stop_patience: int | None = 5
    # False (default): a checkpoint is written on every report, scored by
    # loss. Ray Train's checkpoint manager keeps the best N by score PLUS
    # the single most recently written one, which is what a resume picks up
    # from. True: only checkpoint when the loss improves -- less write I/O,
    # but a resume can lose progress back to the last improvement.
    # NOTE: whenever a val split is active, checkpoint attachment moves to
    # the val pass (see train_loop.py) -- this flag then gates val's OWN
    # improvement (val loss, not training loss); the windowed block stops
    # attaching checkpoints at all in that case (val owns retention).
    save_only_on_improvement: bool = False
    checkpoint_max_to_keep: int = 3
    # Opt-in, plain-DDP path only (no effect under FSDP2, which already
    # uses accelerate's own save_state/load_state) -- copies model/optimizer
    # state to CPU synchronously (fast), then writes it via pickle.dump in
    # a background thread while training continues, instead of blocking the
    # training loop for the full write duration. Carries real,
    # not-fully-verified risk: a save is only reported to Ray (eligible for
    # checkpoint_score_attribute scoring or resume) one checkpoint-cycle
    # late, once its write is CONFIRMED finished -- see train_loop.py's
    # _AsyncCheckpointer. Test with a real
    # crash+resume before trusting this for a long run.
    async_checkpoint: bool = False

    # Opt-in perf instrumentation (training/perf_logging.py) -- off by
    # default so a normal run pays zero cost: accurate step-timing needs
    # torch.cuda.synchronize() calls, which serialize async CUDA work and
    # cost real throughput whenever they're on. See train_loop.py.
    log_perf_metrics: bool = False
    # (start_step, end_step) inclusive range to capture a real
    # torch.profiler trace for -- only meaningful when log_perf_metrics is
    # also True. None disables profiling entirely.
    profile_steps: tuple[int, int] | None = None

    # TensorBoard is unconditional otherwise -- True (default) preserves
    # that exactly. False (--no-tensorboard) skips creating tb_writer
    # entirely, e.g. when only W&B is wanted.
    tensorboard: bool = True

    # --- W&B (training/wandb_logging.py), via ray.air.integrations.wandb's
    # setup_wandb() -- off by default, additive: everything TensorBoard
    # already logs also goes to W&B at the same cadences when enabled.
    wandb: bool = False
    # None (default): don't pass a `mode` to wandb.init() at all -- the
    # WANDB_MODE env var (or wandb's own "online" default) decides, same
    # as before this flag existed. A real value ("online" | "offline" |
    # "disabled") passed explicitly OVERRIDES the env var (wandb's own
    # kwarg-beats-env-var behavior) -- confirmed directly that defaulting
    # this to "online" instead of None broke WANDB_MODE=offline entirely,
    # since an explicit kwarg always wins.
    wandb_mode: str | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None
    # Allowlist (fnmatch globs OK, e.g. "perf/*") -- None (default) logs
    # every metric already being computed, so turning on --wandb doesn't
    # silently hide anything unless explicitly filtered. Named groups
    # below expand into and merge with this at setup time -- see
    # training/wandb_logging.py's METRIC_GROUPS/expand_metric_groups.
    wandb_metrics: list[str] | None = None
    # Denylist (fnmatch globs OK), applied after the allowlist.
    wandb_exclude_metrics: list[str] = field(default_factory=list)
    # Friendly names for common wandb_metrics glob groups (e.g. "core",
    # "perf", "media") -- see training/wandb_logging.py's METRIC_GROUPS.
    # Use --list-wandb-metrics to see every real group/metric name without
    # reading source or running a training job.
    wandb_metric_groups: list[str] = field(default_factory=list)
    # Episode-preview GIFs -- which cameras to sample (empty = off, opt-in
    # per camera), how often (None -> reuse the effective val cadence,
    # val_every_steps or window_every_steps), how many frames per GIF.
    # Sampling pool is VAL's episodes (never test's) whenever a val split
    # is active; anywhere in the dataset otherwise.
    wandb_gif_cameras: list[str] = field(default_factory=list)
    wandb_gif_every_steps: int | None = None
    wandb_gif_frames: int = 30
    # Predicted-frames GIFs (PolicyAdapter.predict_frames -- see
    # training/model/registry.py; inert for every policy today, groundwork
    # for a future world-model-style policy) -- how often to log them,
    # relative to the VAL pass they're generated inside (see
    # train_loop.py): None (default) logs one every val pass; a real value
    # only logs one every Nth val pass at that step multiple. Can only
    # ever be a multiple of the EFFECTIVE val cadence (val_every_steps or,
    # when unset, window_every_steps) -- generating these needs a real val
    # batch, which only exists when the val pass itself runs, unlike
    # wandb_gif_every_steps above (real recorded episodes, no val batch
    # needed, so that one can use any cadence).
    wandb_predict_frames_every_steps: int | None = None

    # Val: on by default (0.1 = 10% of episodes held out of TRAINING
    # entirely, used for a real periodic eval pass that drives checkpoint
    # retention -- see train_loop.py). 0/None disables it: falls back to
    # byte-for-byte today's behavior (no held-out split, windowed/epoch-end
    # blocks keep attaching checkpoints from training loss, GIFs sample
    # from anywhere). Ignored (must be left at 0/None) when val_v3_root is
    # set instead -- see train.py's CLI validation.
    val_split_fraction: float | None = 0.1
    # Val's own report/checkpoint cadence; None (default) reuses
    # window_every_steps so the common case needs no extra flag.
    val_every_steps: int | None = None
    # Caps the periodic val pass to this many batches per rank per window
    # (tune down if val is taking too long relative to window_every_steps;
    # tune up -- or leave -- for a more thorough val loss estimate). The
    # one-time end-of-training test pass (below) is deliberately NOT
    # capped by this -- it evaluates its whole split exactly once.
    val_max_batches: int = 50
    # None (default): reuse batch_size. Eval has no optimizer-state/
    # gradient memory overhead, so a larger batch is often safe here and
    # reduces per-batch Python/dispatch overhead -- used for both the
    # periodic val pass and the one-time test pass.
    val_batch_size: int | None = None

    # Test: fully opt-in (None/0 = disabled, same shape as val_split_fraction).
    # Evaluated exactly once, after training completes, against episodes
    # NEVER touched during training or val -- never influences checkpoint
    # retention (see train_loop.py's one-time test block). Ignored (must be
    # left at 0/None) when test_v3_root is set instead.
    test_split_fraction: float | None = None


@dataclass
class RunConfig:
    tasks: list[str] = field(default_factory=list)
    max_episodes_per_task: int = 300
    run_name: str | None = None
    # Kept inside training/ deliberately -- train.py's runtime_env excludes
    # it from the working_dir upload by its resolved path (see
    # build_runtime_env in train.py); a path outside the repo root can miss
    # that exclude and sweep checkpoints into the upload past Ray's 512MiB
    # package cap. On a shared cluster, override to shared storage (e.g. a
    # mounted network filesystem) so a worker restart on a different node
    # can still see checkpoints.
    storage_root: str = "training/runs"
    # "act" (default), "molmoact2", or "pi05" -- selects which policy
    # model/registry.py builds; train.py's main() overwrites `model` with
    # the right override type once --policy-type is parsed.
    policy_type: str = "act"
    data: DataConfig = field(default_factory=DataConfig)
    model: PolicyOverrides = field(default_factory=ACTConfigOverrides)
    train: TrainConfig = field(default_factory=TrainConfig)
