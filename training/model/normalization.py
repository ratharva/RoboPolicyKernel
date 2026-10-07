"""Resolves STATE/ACTION normalization mode + stats generically across
policies, driven by DataConfig.normalization (training/common/config.py) and
PolicyAdapter.get_pretrained_normalization (training/model/registry.py).

Default DataConfig.normalization (NormalizationConfig(), see common/config.py)
makes every function here a no-op -- existing ACT/MolmoAct2/pi05 behavior is
untouched unless a run opts in via --normalization-mode-source/--normalization-
stats-source.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from math import isfinite
from numbers import Real

from training.common.config import DataConfig, NormalizationConfig
from training.common.robots import RobotSchema

# FeatureType name -> the dataset-side column name it corresponds to. Only
# STATE/ACTION are ever resolved here -- VISUAL always stays IDENTITY (see
# training/model/image_normalization.py, which owns 100% of image scaling).
_FEATURE_COLUMNS = {"STATE": "observation.state", "ACTION": "action"}
_STAT_KEYS = {
    "MEAN_STD": ("mean", "std"),
    "MIN_MAX": ("min", "max"),
    "QUANTILES": ("q01", "q99"),
    "QUANTILE10": ("q10", "q90"),
    "IDENTITY": (),
}


@dataclass
class PretrainedNormalization:
    """What a pretrained checkpoint's own saved config declares, returned by
    PolicyAdapter.get_pretrained_normalization."""

    mode: dict[str, str]  # e.g. {"VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"}
    # Per-feature numeric stats the checkpoint publishes -- {} (not None)
    # means the checkpoint declares a mode but ships no stat values, which
    # callers must treat as an actionable error, not a silent fallback.
    stats: dict[str, dict]
    action_relative: bool
    action_relative_exclude: list[str]
    # Padded capacity (e.g. PI05Config's max_state_dim/max_action_dim), not
    # a per-dataset dim -- a "must fit within" ceiling, not equality.
    max_state_dim: int | None
    max_action_dim: int | None
    revision: str | None


@dataclass
class ResolvedNormalization:
    mode: dict[str, str] | None  # None -> defer to policy's own hardcoded default
    stats: dict[str, dict] | None  # None -> defer to compute_dataset_stats's existing output
    pretrained: PretrainedNormalization | None  # for validate_normalization's cross-checks


def inspect_pretrained_normalization(policy_type: str, model_cfg) -> PretrainedNormalization | None:
    """Read-only, no training-data access. Used by --inspect-normalization
    and internally by resolve_normalization -- same function both call, so
    the two stay in sync. None when the adapter has no
    get_pretrained_normalization hook (currently ACT)."""
    from training.model.registry import get_adapter

    adapter = get_adapter(policy_type)
    if adapter.get_pretrained_normalization is None:
        return None
    return adapter.get_pretrained_normalization(model_cfg)


def fetch_processor_stats(pretrained_path: str, revision: str | None) -> dict:
    """Per-feature numeric stats a checkpoint's own saved
    policy_preprocessor.json publishes, if any -- lives in the
    normalizer_processor step's config["features"]. Not pi05-specific --
    this is lerobot's general DataProcessorPipeline save format. Returns {}
    (never raises) when the file is missing/unparseable or publishes no
    stats -- both mean the same thing to a caller."""
    filename = "policy_preprocessor.json"
    if os.path.isdir(pretrained_path):
        path = os.path.join(pretrained_path, filename)
        if not os.path.exists(path):
            return {}
    else:
        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(repo_id=pretrained_path, filename=filename, revision=revision)
        except Exception:
            return {}

    try:
        with open(path) as f:
            doc = json.load(f)
        for step in doc.get("steps", []):
            if step.get("registry_name") == "normalizer_processor":
                return step.get("config", {}).get("features", {}) or {}
    except (json.JSONDecodeError, OSError):
        return {}
    return {}


def _resolve_dataset_stats_for_mode(raw_ds, data_cfg: DataConfig, mode: dict[str, str]) -> dict:
    """Computes whatever training/data/stats.py functions the resolved mode
    needs. Features resolved to MEAN_STD/IDENTITY are left out -- they
    defer to train.py's already-computed compute_dataset_stats output."""
    from training.data.stats import compute_quantile_stats

    quantile_needs: dict[tuple[float, float], list[str]] = {}
    for feature_type, column in _FEATURE_COLUMNS.items():
        feature_mode = mode.get(feature_type)
        if feature_mode == "QUANTILES":
            quantile_needs.setdefault((0.01, 0.99), []).append(column)
        elif feature_mode == "QUANTILE10":
            quantile_needs.setdefault((0.10, 0.90), []).append(column)
        elif feature_mode not in (None, "MEAN_STD", "IDENTITY"):
            raise ValueError(
                f"resolved {feature_type} normalization mode {feature_mode!r} has no dataset-stats "
                "computation implemented -- training/data/stats.py only supports MEAN_STD/QUANTILES/"
                "QUANTILE10. Use --normalization-stats-source checkpoint or explicit_file instead."
            )

    stats: dict = {}
    for quantiles, columns in quantile_needs.items():
        computed = compute_quantile_stats(raw_ds, data_cfg, quantiles=quantiles)
        for column in columns:
            stats[column] = computed[column]
    return stats


def _load_explicit_stats_file(path: str | None) -> dict:
    if not path:
        raise ValueError(
            "--normalization-stats-source explicit_file requires --normalization-explicit-stats-file "
            "to also be set."
        )
    with open(path) as f:
        return json.load(f)


def resolve_normalization(
    data_cfg: DataConfig, policy_type: str, model_cfg, raw_ds,
) -> ResolvedNormalization:
    """Called once from train.py right after compute_dataset_stats. No-op
    when data_cfg.normalization is left at its defaults."""
    norm_cfg = data_cfg.normalization

    if norm_cfg.mode_source == "explicit" and norm_cfg.explicit_mode is None and norm_cfg.stats_source == "dataset":
        return ResolvedNormalization(mode=None, stats=None, pretrained=None)

    pretrained = None
    if norm_cfg.mode_source == "checkpoint" or norm_cfg.stats_source == "checkpoint":
        pretrained = inspect_pretrained_normalization(policy_type, model_cfg)

    if norm_cfg.mode_source == "checkpoint":
        # Some adapters (pi05) read mode straight off the real checkpoint's
        # own saved config, which may legitimately declare a non-IDENTITY
        # VISUAL scheme (that checkpoint's own pretraining normalized
        # images itself). This codebase's own image_normalization.py
        # already owns 100% of per-camera image scaling -- wiring a
        # non-IDENTITY VISUAL through here into normalization_mapping would
        # silently double-normalize every frame. Force it back to IDENTITY
        # here, the one shared, policy-agnostic place every mode_source
        # path funnels through, rather than trusting each adapter to
        # re-derive this invariant on its own (MolmoAct2's hook happens to
        # hardcode it safely; pi05's doesn't, which is what let this
        # through in the first place).
        mode = {**pretrained.mode, "VISUAL": "IDENTITY"} if pretrained is not None else None
    elif norm_cfg.explicit_mode is not None:
        mode = {"VISUAL": "IDENTITY", "STATE": norm_cfg.explicit_mode, "ACTION": norm_cfg.explicit_mode}
    else:
        mode = None

    if norm_cfg.stats_source == "checkpoint":
        stats = pretrained.stats if pretrained is not None else {}
    elif norm_cfg.stats_source == "explicit_file":
        stats = _load_explicit_stats_file(norm_cfg.explicit_stats_file)
    else:  # "dataset"
        stats = _resolve_dataset_stats_for_mode(raw_ds, data_cfg, mode) if mode is not None else None
        if stats == {}:
            stats = None  # every resolved feature was MEAN_STD/IDENTITY -- true no-op

    return ResolvedNormalization(mode=mode, stats=stats, pretrained=pretrained)


def validate_normalization(
    resolved: ResolvedNormalization, data_cfg: DataConfig, robot: RobotSchema, policy_type: str, model_cfg,
) -> None:
    """Hard-fails with ValueError, called driver-side before TorchTrainer
    construction so a bad config fails in seconds."""
    norm_cfg = data_cfg.normalization

    if norm_cfg.mode_source == "checkpoint" and resolved.pretrained is None:
        raise ValueError(
            f"--normalization-mode-source checkpoint requires policy_type={policy_type!r} to declare "
            "a pretrained checkpoint's normalization scheme, but its adapter has no "
            "get_pretrained_normalization hook (pi05 and molmoact2 do today) -- use "
            "--normalization-mode-source explicit instead."
        )

    if norm_cfg.stats_source == "checkpoint":
        if resolved.pretrained is None:
            raise ValueError(
                f"--normalization-stats-source checkpoint requires policy_type={policy_type!r} to "
                "declare a get_pretrained_normalization hook, but none exists (pi05 and molmoact2 do "
                "today) -- use --normalization-stats-source dataset or explicit_file instead."
            )
        if not resolved.pretrained.stats:
            raise ValueError(
                f"--normalization-stats-source checkpoint was requested for policy_type={policy_type!r} "
                f"(pretrained_path={getattr(model_cfg, 'pretrained_path', None)!r}, "
                f"revision={getattr(model_cfg, 'revision', None)!r}), but that checkpoint's own saved "
                f"config declares mode={resolved.pretrained.mode!r} without publishing any numeric "
                "statistics -- this checkpoint only lets you recover the SCHEME, not the stat values. "
                "Pass --normalization-stats-source dataset (compute from this run's own training data) "
                "or --normalization-stats-source explicit_file <path> instead."
            )

    external_stats = norm_cfg.stats_source in ("checkpoint", "explicit_file")
    if external_stats and resolved.mode is None:
        raise ValueError(
            "Checkpoint or explicit-file statistics require a resolved normalization mode. "
            "Set --normalization-mode-source checkpoint or --normalization-explicit-mode."
        )
    if external_stats and not isinstance(resolved.stats, dict):
        raise ValueError("Normalization statistics must be a mapping of feature columns to statistics.")

    for feature_type, column in _FEATURE_COLUMNS.items():
        mode = (resolved.mode or {}).get(feature_type)
        if external_stats and mode not in _STAT_KEYS:
            raise ValueError(f"A supported normalization mode is required for {feature_type}, got {mode!r}.")
        if norm_cfg.stats_source == "checkpoint" and resolved.pretrained is not None:
            checkpoint_mode = resolved.pretrained.mode.get(feature_type)
            if mode != checkpoint_mode:
                raise ValueError(
                    f"Resolved {feature_type} mode {mode!r} differs from checkpoint mode "
                    f"{checkpoint_mode!r}; checkpoint statistics must use their declared mode."
                )
        if mode == "IDENTITY":
            continue
        # Dataset resolution intentionally supplies only the quantile features;
        # MEAN_STD features use train.py's previously computed dataset statistics.
        if not external_stats and (not resolved.stats or column not in resolved.stats):
            continue
        feature_stats = (resolved.stats or {}).get(column)
        if not isinstance(feature_stats, dict):
            raise ValueError(f"Missing normalization statistics for {column!r} ({mode}).")
        expected_dim = robot.state_dim if column == "observation.state" else robot.action_dim
        for key in _STAT_KEYS.get(mode, ()):
            values = feature_stats.get(key)
            if values is None:
                raise ValueError(f"Normalization statistics for {column!r} ({mode}) require {key!r}.")
            if not isinstance(values, (list, tuple)) or len(values) != expected_dim or any(
                not isinstance(value, Real) or isinstance(value, bool) or not isfinite(value)
                for value in values
            ):
                raise ValueError(
                    f"Normalization statistic {column!r}.{key} must be a finite numeric vector "
                    f"with dim {expected_dim}. Check --action-space/dataset schema against the stats source."
                )

    # MolmoAct2-specific, independent of mode_source/stats_source: setting
    # --molmoact2-norm-tag is wired unconditionally into MolmoAct2Config, so
    # MolmoAct2Policy's own _apply_norm_tag_metadata silently overwrites
    # chunk_size/n_action_steps to the tag's own values at construction
    # time regardless of the normalization resolution path taken -- fail
    # loudly here instead of letting the run silently diverge from what
    # was configured.
    if policy_type == "molmoact2" and getattr(model_cfg, "norm_tag", None):
        from training.model.molmoact2 import get_norm_tag_metadata

        tag_meta = get_norm_tag_metadata(model_cfg.checkpoint_path, model_cfg.revision, model_cfg.norm_tag)
        for field, attr in (("action_horizon", "chunk_size"), ("n_action_steps", "n_action_steps")):
            tag_value = tag_meta.get(field)
            configured_value = getattr(model_cfg, attr)
            if tag_value is not None and tag_value != configured_value:
                raise ValueError(
                    f"--molmoact2-norm-tag {model_cfg.norm_tag!r} declares {field}={tag_value!r}, but "
                    f"this run configures {attr}={configured_value!r} -- MolmoAct2Policy's own "
                    f"_apply_norm_tag_metadata will silently overwrite {attr} to {tag_value!r} at "
                    f"construction time. Set --molmoact2-{attr.replace('_', '-')} {tag_value} to match, "
                    "or the run's real behavior will diverge from what you configured."
                )

    if policy_type == "pi05" and resolved.pretrained is not None and (
        norm_cfg.mode_source == "checkpoint" or norm_cfg.stats_source == "checkpoint"
    ):
        p = resolved.pretrained
        if p.max_state_dim is not None and robot.state_dim > p.max_state_dim:
            raise ValueError(
                f"this run's RobotSchema state_dim ({robot.state_dim}) exceeds the checkpoint's "
                f"max_state_dim capacity ({p.max_state_dim}) -- the pretrained model can't accept a "
                "state vector this large."
            )
        if p.max_action_dim is not None and robot.action_dim > p.max_action_dim:
            raise ValueError(
                f"this run's RobotSchema action_dim ({robot.action_dim}) exceeds the checkpoint's "
                f"max_action_dim capacity ({p.max_action_dim}) -- the pretrained model can't accept an "
                "action vector this large."
            )

        want_delta = data_cfg.action_representation == "delta"
        if p.action_relative != want_delta:
            raise ValueError(
                f"checkpoint was trained with use_relative_actions={p.action_relative!r} "
                f"(relative_exclude_joints={p.action_relative_exclude!r}), but this run requests "
                f"--action-representation {data_cfg.action_representation!r} -- these must match or "
                "the checkpoint's action semantics are corrupted."
            )
        if p.action_relative and set(p.action_relative_exclude) != set(data_cfg.action_delta_exclude):
            raise ValueError(
                f"checkpoint's relative-action exclude list {p.action_relative_exclude!r} doesn't match "
                f"this run's --action-delta-exclude {data_cfg.action_delta_exclude!r} -- these must "
                "match or some action dims will be delta-transformed inconsistently with pretraining."
            )

        # Not automatable -- no per-component name/unit metadata exists in
        # the checkpoint's saved config. Print a side-by-side for a human
        # to eyeball instead of a pass/fail check.
        print(
            "normalization: this run's RobotSchema state/action components "
            f"(name, dim), in order -- state: {robot.state_components}, action: {robot.action_components}. "
            f"Checkpoint declares max_state_dim={p.max_state_dim}, max_action_dim={p.max_action_dim} "
            "(padded capacity, not per-component names/units) -- verify these are compatible by eye; "
            "this can't be checked automatically with lerobot's real checkpoint format."
        )


def build_normalization_snapshot(mode: dict[str, str] | None, stats: dict) -> dict:
    """{"mode": ..., "stats": {"observation.state": ..., "action": ...}} --
    persisted into every checkpoint (train_loop.py) so inference/resume use
    the same normalization that was actually used at train time."""
    return {
        "mode": mode,
        "stats": {
            "observation.state": stats.get("observation.state"),
            "action": stats.get("action"),
        },
    }
