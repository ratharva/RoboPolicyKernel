"""DataConfig -- shared between data prep and training. Every tunable here
describes the dataset being used, independent of how it was prepared or how
a policy trains on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from training.common.robots import RobotSchema


@dataclass
class NormalizationConfig:
    """How this run resolves STATE/ACTION normalization mode + stats --
    shared across policies, kept separate from image normalization above.
    See training/model/normalization.py. Defaults are a no-op (each
    policy's own hardcoded mapping, dataset-computed mean/std stats)."""
    # "explicit": use explicit_mode below (None -> the policy's own
    # hardcoded default). "checkpoint": read the mode from the pretrained
    # checkpoint's own saved config -- requires the adapter to implement
    # get_pretrained_normalization (pi05 does; ACT/MolmoAct2 don't).
    mode_source: str = "explicit"   # "explicit" | "checkpoint"
    # Only used when mode_source == "explicit". One of lerobot's
    # NormalizationMode values (MEAN_STD | MIN_MAX | QUANTILES |
    # QUANTILE10 | IDENTITY) -- a generic override, not pi05-only.
    explicit_mode: str | None = None

    # "dataset": compute from this run's own training data. "checkpoint":
    # use stats the checkpoint publishes (errors if it declares a mode but
    # ships none). "explicit_file": load a stats JSON from explicit_stats_file.
    stats_source: str = "dataset"   # "dataset" | "checkpoint" | "explicit_file"
    explicit_stats_file: str | None = None   # required iff stats_source == "explicit_file"


@dataclass
class DataConfig:
    # "hf://datasets/<repo_id>" (gated repos need HF_TOKEN), "s3://<bucket>/<prefix>",
    # or "gs://<bucket>/<prefix>". See training/data_prep/source.py for backend
    # details -- only the hf:// path has been run against real data.
    source_uri: str = ""
    cache_dir: str | None = None  # None -> huggingface_hub's default cache

    # Which dataset source_uri's data follows -- camera keys, state/action
    # dims, and tick rate live on this object (see training/common/robots.py).
    # Populated by training/data_prep/schema_loader.py from a --dataset-source
    # selection, not hardcoded to any one dataset -- see
    # training/data_prep/strategies/registry.py.
    robot: RobotSchema | None = None

    # Training-time preprocessing choice, not a property of the dataset itself.
    image_size: tuple[int, int] = (224, 224)  # (H, W), resized on decode

    # Per-camera image-normalization mode (camera key -> one of
    # training/model/image_normalization.py's MODES) -- a camera missing
    # from this dict uses default_image_normalization. "mean_std" (the
    # default) replicates pre-existing behavior exactly: per-feature
    # mean/std from training/data/stats.py's compute_dataset_stats.
    image_normalization: dict[str, str] = field(default_factory=dict)
    default_image_normalization: str = "mean_std"
    # Required per-camera clip/scale ceiling for the "depth"/"log" modes
    # (real sensor units, e.g. max depth in millimeters) -- no guessed
    # default, see image_normalization.py's _require_max.
    image_normalization_max: dict[str, float] = field(default_factory=dict)

    # Which named subset of robot.action_components to train on (e.g.
    # "joint" vs "end_effector" for agibot_alpha, which records both) --
    # None (default) trains on every action_component, current behavior
    # exactly. Must be a key of robot.action_space_components; see
    # RobotSchema.select_action_space. Applied once, in training/train.py,
    # to the Ray Dataset before training/data/stats.py's
    # compute_dataset_stats -- NOT a train_loop.py per-batch concern.
    action_space: str | None = None

    # "absolute" (default, current behavior) or "delta": whether the
    # `action` column holds absolute targets or offsets from the current
    # observation.state (action -= state, per masked dim) -- see
    # training/data/ray_dataset.py's to_relative_action_space, applied to
    # the SAME Ray Dataset stage as action_space above, before stats.
    action_representation: str = "absolute"
    # Component names (from robot.action_components) to keep absolute even
    # when action_representation="delta" and a same-named state component
    # exists (e.g. a gripper open/close value) -- empty by default.
    action_delta_exclude: list[str] = field(default_factory=list)

    # STATE/ACTION normalization mode+stats resolution -- see NormalizationConfig.
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
