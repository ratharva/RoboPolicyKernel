"""Ray Train per-worker loop -- policy-agnostic (ACT, MolmoAct2, or pi05; see
training/model/registry.py). Policy-specific concerns (model construction,
forward/loss, optional distributed-wrapping strategy) are dispatched through
a PolicyAdapter rather than hardcoded here. Also implements Lightning-style
best-N-checkpoints + step-windowed early stopping.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import random
import tempfile
import time

import torch
import torch.distributed
import ray.train
import ray.train.torch
from torch.utils.tensorboard import SummaryWriter

from training import perf_logging
from training.vendor.util import NumpyToTorchCollate
from training.config import RunConfig, TrainConfig
from training.model.image_normalization import apply_image_normalization
from training.model.normalization import build_normalization_snapshot
from training.model.registry import PolicyAdapter, get_adapter
from training.wandb_logging import filter_metrics, sample_episode_frames

log = logging.getLogger("act_train")


def _gather_eval_stats(loss_sum: float, n: int, metrics_sum: dict[str, float]) -> tuple[float, int, dict[str, float]]:
    """Sums each rank's local val/test loss/n/metrics so val/*//test/* cover
    the whole held-out split, not just rank 0's shard slice. Collective --
    every rank must call this."""
    if not torch.distributed.is_initialized():
        return loss_sum, n, metrics_sum
    world_size = torch.distributed.get_world_size()
    gathered = [None] * world_size
    torch.distributed.all_gather_object(gathered, (loss_sum, n, metrics_sum))
    total_loss = sum(g[0] for g in gathered)
    total_n = sum(g[1] for g in gathered)
    total_metrics: dict[str, float] = {}
    for _, _, m in gathered:
        for k, v in m.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
    return total_loss, total_n, total_metrics


def _sync_should_stop(should_stop: bool, device: torch.device) -> bool:
    """Broadcasts rank 0's early-stop decision to every rank -- each rank
    trains on a different data shard and could otherwise disagree, which
    would hang DDP's collective ops. Every rank must call this."""
    if not torch.distributed.is_initialized():
        return should_stop
    flag = torch.tensor([1.0 if should_stop else 0.0], device=device)
    torch.distributed.broadcast(flag, src=0)
    return bool(flag.item())


def _log_all(
    tb_writer, wandb_run, prefix: str, values: dict, step: int, train_cfg: TrainConfig,
    wandb_step: int | None = None,
) -> None:
    """Writes `values` (bare keys, e.g. {"loss": ...}) to both loggers
    under `f"{prefix}/{key}"` -- TensorBoard unfiltered (as today), W&B
    subject to train_cfg.wandb_metrics/wandb_exclude_metrics (see
    training/wandb_logging.py's filter_metrics). Either logger being None
    (--no-tensorboard / --wandb not passed) is a no-op for that logger.

    `wandb_step` (defaults to `step`) exists because W&B's `step` is ONE
    shared counter across every key in a run, unlike TensorBoard's
    per-tag x-axis -- confirmed by reproducing it directly: logging
    epoch/* at `step=epoch` (0, 1, 2, ...) after window/*/train/* already
    used much larger training-step values silently DROPPED the epoch/*
    point entirely (wandb only accepts non-decreasing steps; it does not
    error, it just discards). The epoch-report call site passes the real
    training step here instead, keeping `epoch` as a value inside the
    dict rather than as the step."""
    prefixed = {f"{prefix}/{k}": v for k, v in values.items()}
    if tb_writer is not None:
        for k, v in prefixed.items():
            tb_writer.add_scalar(k, v, step)
    if wandb_run is not None:
        filtered = filter_metrics(prefixed, train_cfg.wandb_metrics, train_cfg.wandb_exclude_metrics)
        if filtered:
            wandb_run.log(filtered, step=wandb_step if wandb_step is not None else step)


def _trainable_state_dict(module: torch.nn.Module) -> dict:
    """Full state_dict() includes every parameter and buffer regardless of
    requires_grad. Filters out only frozen (requires_grad=False) PARAMETERS
    -- buffers are always kept (typically tiny; safer than auditing every
    policy's buffers for train-time mutation). Frozen parameters don't need
    saving: adapter.build() deterministically reconstructs them from
    model_cfg.checkpoint_path/revision on every construction, resume or
    fresh -- no policy's build_policy_and_processor branches on resume
    state. Real, confirmed benefit: a --molmoact2-train-mode freeze run
    (only ~620M of ~5.49B params trainable) was producing a 13GB checkpoint
    -- the full frozen VLM included -- before this fix."""
    frozen = {name for name, param in module.named_parameters() if not param.requires_grad}
    return {k: v for k, v in module.state_dict().items() if k not in frozen}


def _clone_state_to_cpu(obj):
    """Recursively clones every torch.Tensor found in a nested dict/list/
    tuple structure to a detached CPU copy -- a stable snapshot safe to
    read from a background thread while training continues mutating the
    LIVE GPU tensors (the next optimizer.step() only touches those, never
    this copy). Non-tensor leaves (ints, strings, small dicts inside
    optimizer state, etc.) pass through unchanged."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _clone_state_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clone_state_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_state_to_cpu(v) for v in obj)
    return obj


def _write_pickle_checkpoint(cpu_state: dict, ckpt_dir: str) -> None:
    """Runs in the background thread -- the slow part (was previously done
    synchronously in _report_with_checkpoint's default DDP path). Writes
    the exact same state.pkl format the synchronous path already uses, so
    the resume/load logic needs no format-detection branch at all -- a
    checkpoint written with --async-checkpoint on or off looks identical
    on disk."""
    with open(os.path.join(ckpt_dir, "state.pkl"), "wb") as f:
        pickle.dump(cpu_state, f)


class _AsyncCheckpointer:
    """--async-checkpoint (opt-in, plain-DDP path only -- see training/README.md).
    Backgrounds the slow pickle.dump()+write behind a plain single-worker
    thread pool, instead of torch.distributed.checkpoint.async_save.

    DCP's own async_save was tried first and rejected after a real run hit
    `AssertionError: A CPU backend must be enabled for async save` --
    DCP's async machinery needs a CPU-capable process-group backend
    (cpu:gloo,cuda:nccl) for its own internal coordination, which Ray
    Train's NCCL-only process group doesn't provide. That coordination
    exists to support genuinely SHARDED/distributed checkpoints (FSDP2,
    tensor-parallel) -- overkill here: a plain-DDP checkpoint is an
    unsharded full replica and only rank 0 ever saves, so there's no
    cross-rank coordination need for the save itself. A plain background
    thread sidesteps the question entirely -- no torch.distributed
    involvement at all.

    Tensors are copied to CPU SYNCHRONOUSLY first (fast, purely local --
    this is what makes the subsequent background write safe, since the
    next optimizer.step() only mutates the LIVE GPU tensors, never this
    snapshot), then the slow pickle.dump()+write happens in the background
    thread. Only one save is ever in flight (the thread pool has exactly
    one worker) -- a save kicked off at step N is only reported to Ray
    (eligible for checkpoint_score_attribute scoring or resume) one
    checkpoint-cycle late, once its write is CONFIRMED finished via the
    Future, not assumed. In practice a save takes far less than a full
    window_every_steps/val_every_steps cycle, so `_future.result()` below
    should return immediately, not actually block."""

    def __init__(self):
        import concurrent.futures

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._future = None
        self._pending: tuple[dict, str, int, int, bool] | None = None

    def _flush_pending(self) -> tuple[dict, str, int, int, bool] | None:
        if self._future is None:
            return None
        self._future.result()  # blocks -- see class docstring on why this should be a no-op in practice
        self._future = None
        pending, self._pending = self._pending, None
        return pending

    def save(
        self, state: dict, ckpt_dir: str, metrics: dict, epoch: int, step: int, epoch_complete: bool,
    ) -> tuple[dict, str, int, int, bool] | None:
        """Waits for any previous save to finish (returning ITS
        (metrics, ckpt_dir, epoch, step, epoch_complete) for the caller to
        report to Ray), then kicks off a NEW background save for `state`
        and remembers it as pending. Caller must gate this to rank 0 only."""
        finished = self._flush_pending()
        os.makedirs(ckpt_dir, exist_ok=True)
        cpu_state = _clone_state_to_cpu(state)
        self._future = self._executor.submit(_write_pickle_checkpoint, cpu_state, ckpt_dir)
        self._pending = (metrics, ckpt_dir, epoch, step, epoch_complete)
        return finished

    def drain(self) -> tuple[dict, str, int, int, bool] | None:
        """Call once at the very end of training so the LAST in-flight save
        still gets reported to Ray instead of silently vanishing."""
        return self._flush_pending()


def _attach_finished_async_checkpoint(finished: tuple[dict, str, int, int, bool] | None) -> None:
    """Reports a finished async-checkpoint's own metrics + directory to
    Ray, then removes the directory -- mirrors the old code's
    tempfile.TemporaryDirectory() auto-cleanup, just done manually since
    this directory has to outlive the call that created it. No-op if
    nothing was pending. Rank 0 only, caller must gate."""
    if finished is None:
        return
    import shutil

    metrics, ckpt_dir, epoch, step, epoch_complete = finished
    ray.train.report(metrics, checkpoint=ray.train.Checkpoint.from_directory(ckpt_dir))
    shutil.rmtree(ckpt_dir, ignore_errors=True)


def _report_with_checkpoint(
    metrics: dict, adapter: PolicyAdapter, unwrapped_policy, optimizer, dist_ctx,
    run_cfg: RunConfig, epoch: int, step: int, epoch_complete: bool, rank: int, save_checkpoint: bool,
    async_checkpointer: "_AsyncCheckpointer | None" = None, normalization_snapshot: dict | None = None,
) -> None:
    """dist_ctx is the accelerator from adapter.wrap_for_training (FSDP2 path)
    or None (DDP path). epoch_complete distinguishes a step-windowed
    (mid-epoch) checkpoint from an end-of-epoch one -- required for the
    resume logic in train_loop_per_worker to pick the correct epoch.
    async_checkpointer is only ever non-None for the plain-DDP path (see
    _AsyncCheckpointer) -- the FSDP2 branch below always uses accelerate's
    own save_state/load_state, a separate mechanism this doesn't touch.
    normalization_snapshot (training/model/normalization.py's
    build_normalization_snapshot) is persisted into every checkpoint format
    below so resume/inference read back the same normalization used at
    train time."""
    if not save_checkpoint:
        ray.train.report(metrics)
        return

    if adapter.save_checkpoint:
        # Sharded FSDP2 save needs every rank to participate and all ranks to
        # write to the same directory, so this uses one fixed, deterministic
        # path derived from run_cfg rather than a per-rank temp dir.
        ckpt_dir = os.path.join(run_cfg.storage_root, run_cfg.run_name, "_fsdp2_checkpoint_tmp")
        os.makedirs(ckpt_dir, exist_ok=True)
        adapter.save_checkpoint(
            dist_ctx, ckpt_dir, epoch, step, epoch_complete,
            extra_meta={"normalization": normalization_snapshot},
        )
        if rank == 0:
            ray.train.report(metrics, checkpoint=ray.train.Checkpoint.from_directory(ckpt_dir))
        else:
            ray.train.report(metrics)
        # Rank 0's report() must finish reading ckpt_dir before any rank
        # starts overwriting it for the next checkpoint.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return

    if async_checkpointer is not None:
        # Deliberately exactly ONE ray.train.report() call here, matching
        # every other call site in this file -- calling it twice (once for
        # a just-finished previous checkpoint, once more for THIS cycle's
        # live metrics) would make rank 0's report() call COUNT differ from
        # every other rank's for this same event, an unverified risk on top
        # of an already-experimental feature. The real, user-visible cost:
        # whenever a previous save just finished, Ray's OWN tracked metrics
        # (result.metrics, history.jsonl) report that PREVIOUS cycle's
        # values, not this one -- TensorBoard/W&B already have the current
        # cycle's real numbers from _log_all above, so nothing is actually
        # lost, but Ray's own history looks one checkpoint-cycle stale.
        if rank == 0:
            ckpt_dir = os.path.join(
                run_cfg.storage_root, run_cfg.run_name, f"_async_checkpoint_tmp_{step}",
            )
            state = {
                "model": _trainable_state_dict(unwrapped_policy),
                "optim": optimizer.state_dict(),
                "epoch": epoch, "step": step, "epoch_complete": epoch_complete,
                "normalization": normalization_snapshot,
            }
            finished = async_checkpointer.save(state, ckpt_dir, metrics, epoch, step, epoch_complete)
            if finished is not None:
                _attach_finished_async_checkpoint(finished)
            else:
                ray.train.report(metrics)
        else:
            ray.train.report(metrics)
        return

    # Default (synchronous DDP) path: pickle the unwrapped model/optimizer
    # state_dict, rank 0 only. This is what --async-checkpoint replaces.
    if rank == 0:
        state = {
            "model": _trainable_state_dict(unwrapped_policy),
            "optim": optimizer.state_dict(),
            "epoch": epoch, "step": step, "epoch_complete": epoch_complete,
            "normalization": normalization_snapshot,
        }
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "state.pkl"), "wb") as f:
                pickle.dump(state, f)
            ray.train.report(metrics, checkpoint=ray.train.Checkpoint.from_directory(d))
    else:
        ray.train.report(metrics)


def _run_eval_pass(
    shard, adapter: PolicyAdapter, policy, preprocessor, data_cfg, dataset_stats: dict,
    dist_ctx, collate, batch_size: int, max_batches: int | None, device: torch.device,
) -> tuple[float, int, dict[str, float], dict | None]:
    """Read-only forward+loss pass over `shard`, capped at `max_batches` per
    rank (None = unbounded -- the one-time TEST pass). Shared by the
    periodic VAL pass and the one-time TEST pass; caller wraps this in
    policy.eval()/policy.train() and reduces the result across ranks via
    _gather_eval_stats (this function only returns THIS rank's local
    contribution). torch.inference_mode() is strictly stronger than
    torch.no_grad() for a read-only pass like this (skips autograd
    version-counter bookkeeping no_grad still does).

    Each rank's own shard (ray.train.get_dataset_shard) isn't guaranteed to
    have the same row count as every other rank's -- so without the
    cross-rank sync below, a rank that runs out of local batches first
    would exit into the collective _gather_eval_stats/all_gather_object
    while another rank is still inside adapter.forward_loss, itself a
    collective for DDP (buffer broadcast, on by default) or FSDP2 (param
    all-gather) -- two different ranks stuck in two different collective
    ops is a real deadlock, not a theoretical one. Every rank all_reduces
    (MIN) a "do I still have a batch" flag before each forward call, so
    all ranks agree to stop together and call forward() the exact same
    number of times. Trade-off: the pass stops as soon as the SMALLEST
    rank's shard (or max_batches) is exhausted -- it may not process every
    row of an unevenly-split shard, which is preferable to hanging."""
    loss_sum, n = 0.0, 0
    metrics_sum: dict[str, float] = {}
    last_inputs = None
    distributed = torch.distributed.is_initialized()
    it = iter(shard.iter_torch_batches(batch_size=batch_size, collate_fn=collate))
    i = 0
    with torch.inference_mode():
        while max_batches is None or i < max_batches:
            batch = next(it, None)
            has_batch = batch is not None
            if distributed:
                flag = torch.tensor([1.0 if has_batch else 0.0], device=device)
                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
                has_batch = bool(flag.item())
            if not has_batch:
                break
            if not adapter.needs_task:
                batch.pop("task", None)
            if not adapter.preprocessing_offloaded:
                batch = apply_image_normalization(
                    batch, data_cfg.image_normalization, data_cfg.default_image_normalization,
                    dataset_stats, data_cfg.image_normalization_max,
                )
            inputs = batch if adapter.preprocessing_offloaded else preprocessor(batch)
            last_inputs = inputs
            if dist_ctx is not None and hasattr(dist_ctx, "autocast"):
                with dist_ctx.autocast():
                    loss, step_metrics = adapter.forward_loss(policy, inputs)
            else:
                loss, step_metrics = adapter.forward_loss(policy, inputs)
            if adapter.extra_metrics is not None:
                step_metrics = {**step_metrics, **adapter.extra_metrics(policy, inputs)}
            loss_sum += loss.item()
            n += 1
            for k, v in step_metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
            i += 1
    return loss_sum, n, metrics_sum, last_inputs


def _finish_val_pass(
    loss_sum: float, n: int, metrics_sum: dict[str, float], adapter: PolicyAdapter,
    unwrapped_policy, optimizer, dist_ctx, run_cfg: RunConfig, train_cfg: TrainConfig,
    epoch: int, step: int, epoch_complete: bool, rank: int, tb_writer, wandb_run,
    best_val_loss_seen: float, async_checkpointer: "_AsyncCheckpointer | None" = None,
    normalization_snapshot: dict | None = None,
) -> float:
    """Shared tail of a val pass -- gather this rank's _run_eval_pass output
    across ranks, log under val/*, and attach a checkpoint scored by THIS
    pass's loss (never training loss) when it improves best_val_loss_seen
    (or unconditionally, per save_only_on_improvement). Returns the updated
    best_val_loss_seen. Used by both the periodic VAL block and the
    epoch-end safety-net val pass in train_loop_per_worker -- a short run
    (fewer steps than the val cadence) would otherwise end with NO
    checkpoint at all, now that neither the windowed nor epoch-end
    training-loss reports attach one while val is active."""
    loss_sum, n, metrics_sum = _gather_eval_stats(loss_sum, n, metrics_sum)
    val_report_metrics = {
        "epoch": epoch, "steps": step, "report_kind": "val",
        "loss": loss_sum / max(n, 1),
        **{k: v / max(n, 1) for k, v in metrics_sum.items()},
    }
    if rank == 0 and n > 0:
        _log_all(
            tb_writer, wandb_run, "val",
            {k: v for k, v in val_report_metrics.items() if k not in ("epoch", "steps", "report_kind")},
            step, train_cfg,
        )
        if tb_writer is not None:
            tb_writer.flush()

    val_improved = val_report_metrics["loss"] < best_val_loss_seen
    if val_improved:
        best_val_loss_seen = val_report_metrics["loss"]

    save_checkpoint = val_improved if train_cfg.save_only_on_improvement else True
    with perf_logging.Timer() as t_ckpt:
        _report_with_checkpoint(
            val_report_metrics, adapter, unwrapped_policy, optimizer, dist_ctx,
            run_cfg, epoch, step, epoch_complete=epoch_complete, rank=rank, save_checkpoint=save_checkpoint,
            async_checkpointer=async_checkpointer, normalization_snapshot=normalization_snapshot,
        )
    if train_cfg.log_perf_metrics and rank == 0 and save_checkpoint:
        _log_all(tb_writer, wandb_run, "perf", {"checkpoint_save_s": t_ckpt.elapsed}, step, train_cfg)
    return best_val_loss_seen


def train_loop_per_worker(config: dict) -> None:
    # Real wall-clock marker -- paired with train.py's own _mark_phase prints,
    # this brackets Ray Train's own worker-launch overhead (time between the
    # driver calling trainer.fit() and a worker actually starting to run).
    print(f"  (worker started at {time.time():.1f})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_cfg: RunConfig = config["run_cfg"]
    data_cfg = run_cfg.data
    model_cfg = run_cfg.model
    train_cfg = run_cfg.train

    adapter = get_adapter(
        run_cfg.policy_type,
        distributed_strategy=getattr(model_cfg, "distributed_strategy", "ddp"),
        offload_tokenization=getattr(model_cfg, "offload_tokenization", False),
    )

    dataset_stats = config["dataset_stats"]

    # Resume normalization cross-check -- must run BEFORE adapter.build()
    # since dataset_stats/model_cfg.resolved_normalization_mode feed into
    # it. Peeks at a resumed checkpoint's saved normalization snapshot and
    # prefers its exact stats/mode on ordinary numeric drift (dataset-mode
    # sampling isn't seeded); hard-fails only on a structural mismatch
    # (different dims/mode keys). The checkpoint directory gets staged
    # twice as a result (again below for the real model/optimizer restore,
    # which needs dist_ctx, not yet constructed here) -- a one-time resume
    # cost, not a hot-path concern.
    resume_checkpoint = ray.train.get_checkpoint()
    if resume_checkpoint is not None:
        with resume_checkpoint.as_directory() as d:
            if adapter.save_checkpoint:  # FSDP2 -- meta.json sidecar
                meta_path = os.path.join(d, "meta.json")
                saved_normalization = None
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        saved_normalization = json.load(f).get("normalization")
            else:  # plain DDP (sync or async -- identical on-disk format) -- state.pkl
                state_path = os.path.join(d, "state.pkl")
                saved_normalization = None
                if os.path.exists(state_path):
                    with open(state_path, "rb") as f:
                        saved_normalization = pickle.load(f).get("normalization")
        if saved_normalization:
            for column, expected_dim in (
                ("observation.state", data_cfg.robot.state_dim), ("action", data_cfg.robot.action_dim),
            ):
                saved_stat = (saved_normalization.get("stats") or {}).get(column)
                fresh_stat = dataset_stats.get(column)
                if not saved_stat or not fresh_stat:
                    continue
                saved_len = len(next(iter(saved_stat.values())))
                if set(saved_stat) != set(fresh_stat) or saved_len != expected_dim:
                    raise RuntimeError(
                        f"resuming {run_cfg.run_name!r}: checkpoint's saved normalization for {column!r} "
                        f"(keys={sorted(saved_stat)}, dim={saved_len}) is structurally incompatible with "
                        f"this run's freshly-resolved normalization (keys={sorted(fresh_stat)}, expected "
                        f"dim={expected_dim}) -- looks like a genuinely different dataset/config is being "
                        "resumed against."
                    )
                dataset_stats[column] = saved_stat
            saved_mode = saved_normalization.get("mode")
            if saved_mode is not None:
                model_cfg.resolved_normalization_mode = saved_mode
            log.info(
                "resuming %s with checkpoint's own saved normalization (mode=%s)",
                run_cfg.run_name, saved_mode,
            )

    normalization_snapshot = build_normalization_snapshot(
        getattr(model_cfg, "resolved_normalization_mode", None), dataset_stats,
    )

    policy, preprocessor = adapter.build(data_cfg, model_cfg, train_cfg, dataset_stats, device=str(device))
    policy = policy.to(device)
    if adapter.post_build_hook:
        adapter.post_build_hook(policy, model_cfg)

    # Built from the unwrapped policy, before any distributed wrapping: DDP
    # doesn't change parameter tensor identity, and accelerate's FSDP2 path
    # (adapter.wrap_for_training) wraps model+optimizer together via
    # accelerator.prepare(), which rebinds the optimizer's param references
    # to the sharded parameters -- so the optimizer must exist first.
    optim_params = policy.get_optim_params() if hasattr(policy, "get_optim_params") else policy.parameters()
    optimizer = torch.optim.AdamW(
        optim_params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay,
        betas=train_cfg.adam_betas, eps=train_cfg.adam_eps,
    )

    dist_ctx = None
    if adapter.wrap_for_training:
        policy, optimizer, dist_ctx = adapter.wrap_for_training(policy, optimizer, model_cfg, device)
        unwrapped_policy = dist_ctx.unwrap_model(policy)  # accelerate's own unwrap, not the DDP .module pattern
    else:
        policy = ray.train.torch.prepare_model(
            policy, parallel_strategy_kwargs=adapter.prepare_model_kwargs or {},
        )
        # prepare_model only wraps in DistributedDataParallel when world_size
        # > 1, so `policy` may have no `.module` -- unwrap defensively.
        unwrapped_policy = policy.module if hasattr(policy, "module") else policy

    # --async-checkpoint: only meaningful for the plain-DDP path -- FSDP2
    # (adapter.save_checkpoint set) always uses accelerate's own
    # save_state/load_state instead, a separate mechanism this doesn't
    # touch, so _report_with_checkpoint's FSDP2 branch ignores this anyway.
    async_checkpointer = (
        _AsyncCheckpointer() if train_cfg.async_checkpoint and not adapter.save_checkpoint else None
    )

    policy.train()  # load-bearing for MolmoAct2's train_mode="freeze" (see model/molmoact2.py)

    start_epoch, step = 0, 0
    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        if adapter.load_checkpoint:
            with checkpoint.as_directory() as d:
                state = adapter.load_checkpoint(dist_ctx, d)
        else:
            # Same state.pkl format regardless of --async-checkpoint --
            # _AsyncCheckpointer writes via a background thread instead of
            # inline, but the on-disk format (and this load path) is
            # identical either way, so resuming a run that switches
            # --async-checkpoint on/off between checkpoints just works.
            with checkpoint.as_directory() as d:
                with open(os.path.join(d, "state.pkl"), "rb") as f:
                    state = pickle.load(f)
                # strict=False: state["model"] may only contain trainable
                # params (see _trainable_state_dict) -- frozen params are
                # already correctly set by adapter.build() above. Old-format
                # checkpoints (full state dict, predating this) still load
                # fine here too (zero missing keys). The real failure this
                # guards against is resuming into a run with a DIFFERENT
                # trainable/frozen split than the one that saved this
                # checkpoint (e.g. train_mode changed between save and
                # resume) -- that would silently load a wrong/incomplete
                # model without this check.
                load_result = unwrapped_policy.load_state_dict(state["model"], strict=False)
                frozen_now = {name for name, param in unwrapped_policy.named_parameters() if not param.requires_grad}
                bad_missing = set(load_result.missing_keys) - frozen_now
                if bad_missing or load_result.unexpected_keys:
                    raise RuntimeError(
                        f"resume checkpoint's saved model state doesn't match this run's trainable/"
                        f"frozen split -- missing keys that should be trainable now: {sorted(bad_missing)}; "
                        f"unexpected keys: {sorted(load_result.unexpected_keys)}. This usually means "
                        f"resuming into a run with a different train_mode/frozen-parameter configuration "
                        f"than the one that saved this checkpoint."
                    )
                optimizer.load_state_dict(state["optim"])
        # Only advance to the next epoch when the checkpoint actually
        # finished one; otherwise resume the same epoch (its data iterator
        # restarts from row 0). Checkpoints predating epoch_complete default
        # to True, preserving epoch-boundary-only resume behavior.
        epoch_complete = state.get("epoch_complete", True)
        start_epoch = state["epoch"] + 1 if epoch_complete else state["epoch"]
        step = state["step"]

    if start_epoch >= train_cfg.num_epochs:
        log.warning(
            "RESUMED A COMPLETED RUN: checkpoint is at epoch %d and num_epochs=%d, "
            "so no training steps will run. Raise num_epochs or use a new run_name "
            "to retrain.", start_epoch - 1, train_cfg.num_epochs,
        )
        ray.train.report({"epoch": start_epoch - 1, "steps": step, "skipped_resumed_complete": True})
        return

    rank = ray.train.get_context().get_world_rank()
    shard = ray.train.get_dataset_shard("train")
    image_keys = {f"observation.images.{k}" for k in data_cfg.robot.camera_keys}
    image_keys |= {f"observation.images.depth_{k}" for k in data_cfg.robot.depth_camera_keys}
    collate = NumpyToTorchCollate(device, image_keys=image_keys)

    # Held-out val split (on by default -- --val-split-fraction/--val-v3-root)
    # and test split (opt-in -- --test-split-fraction/--test-v3-root):
    # second/third named datasets registered by train.py alongside "train"
    # when active, absent otherwise. "val_active"/"test_active" (not part
    # of RunConfig's own schema, threaded through train_loop_config the
    # same way v3_root already is) cover BOTH ways a split can be active --
    # a hash-based fraction slice of train's own root, or a wholly separate
    # --val-v3-root/--test-v3-root -- train_loop.py doesn't need to know
    # which. Val drives checkpoint retention (see the VAL block below);
    # test is evaluated exactly once, after training completes, and never
    # attaches a checkpoint.
    val_shard = ray.train.get_dataset_shard("val") if config.get("val_active") else None
    test_shard = ray.train.get_dataset_shard("test") if config.get("test_active") else None
    effective_val_every_steps = train_cfg.val_every_steps or train_cfg.window_every_steps

    # Only rank 0 writes TensorBoard -- every worker's shard is a different
    # data slice, so only rank 0's loss curve is coherent, and multiple
    # workers writing to the same run would corrupt one events file.
    # train_cfg.tensorboard=False (--no-tensorboard) skips this entirely,
    # e.g. when only W&B is wanted.
    # Always computed -- --profile-steps needs this dir even with --no-tensorboard.
    tb_dir = os.path.join(run_cfg.storage_root, run_cfg.run_name, "tensorboard")
    tb_writer = None
    if rank == 0 and train_cfg.tensorboard:
        os.makedirs(tb_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_dir, flush_secs=10)
        print(f"TensorBoard logs: {tb_dir}  (run: tensorboard --logdir {tb_dir})")

    # W&B (training/wandb_logging.py) -- setup_wandb() does its own
    # rank-zero check internally (via Ray's own session), so it's called on
    # EVERY rank, not gated by `if rank == 0:` the way tb_writer is above;
    # non-zero ranks get back a disabled no-op run object. Imported lazily
    # (like every other optional dependency in this codebase -- see
    # perf_logging.py's nvidia-ml-py handling) so wandb never needs to be
    # installed unless --wandb is actually passed.
    wandb_run = None
    if train_cfg.wandb:
        try:
            from ray.air.integrations.wandb import setup_wandb
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "--wandb needs the `wandb` package -- it's commented out in training/requirements.txt "
                "(optional) since most runs don't use it. Run `pip install wandb`, then `wandb login` "
                "(or pass --wandb-mode offline to skip login for a smoke test)."
            ) from e
        # --wandb-mode sets the WANDB_MODE env var itself, rather than
        # passing mode= as a kwarg to setup_wandb() -- confirmed by reading
        # ray.air.integrations.wandb._set_api_key's real source that Ray's
        # OWN pre-check for "should this run require a real API key" only
        # ever looks at the WANDB_MODE env var, never at a `mode` kwarg
        # forwarded to wandb.init() -- passing mode="offline" as a kwarg
        # alone still hit "No WandB API key found" in a real test, because
        # that pre-check runs BEFORE wandb.init() ever sees the kwarg. Only
        # set when explicit, so an unset --wandb-mode leaves any
        # externally-set WANDB_MODE (e.g. from the shell) untouched.
        if train_cfg.wandb_mode is not None:
            os.environ["WANDB_MODE"] = train_cfg.wandb_mode
        try:
            wandb_run = setup_wandb(
                config={"run_name": run_cfg.run_name, "policy_type": run_cfg.policy_type},
                project=train_cfg.wandb_project, entity=train_cfg.wandb_entity, name=run_cfg.run_name,
            )
        except Exception as e:
            # wandb's own exceptions (e.g. wandb.errors.UsageError for "not
            # logged in") already have a clear message -- just make sure it
            # isn't buried in Ray's own WorkerGroupError traceback noise,
            # and point at the no-login escape hatch.
            raise RuntimeError(
                f"--wandb failed to start a W&B run: {e} -- pass --wandb-mode offline for a smoke test "
                f"that needs no login, or run `wandb login` first."
            ) from e

        # Named metric groups (--wandb-metric-groups) are a convenience
        # layer over the same allowlist filter_metrics already applies --
        # expand and merge into wandb_metrics ONCE here, in place. train_cfg
        # is already a per-worker-local object (Ray re-serializes
        # train_loop_config per worker), so mutating it is safe and needs
        # no changes to _log_all/filter_metrics below, which keep reading
        # train_cfg.wandb_metrics exactly as they already do.
        if train_cfg.wandb_metric_groups:
            from training.wandb_logging import expand_metric_groups

            group_patterns = expand_metric_groups(train_cfg.wandb_metric_groups)
            train_cfg.wandb_metrics = list(dict.fromkeys(group_patterns + (train_cfg.wandb_metrics or [])))

    # Episode-preview GIFs (training/wandb_logging.py's sample_episode_frames)
    # -- opt-in per camera. Reads directly from the converted v3 root's own
    # video files, not from Ray Data, so it needs v3_root (plumbed in by
    # train.py, not part of RunConfig's own schema -- see that file) and,
    # when a real val split exists, the held-out episode-index set so GIFs
    # come specifically from data the model never trained on (val's pool,
    # never test's).
    v3_root = config.get("v3_root")
    val_episode_indices = config.get("val_episode_indices")
    gif_rng = random.Random(0)
    # Computed once here, not per-window: the held-out VAL set when active
    # (so GIFs come specifically from data never trained on), else every
    # real episode_index in the dataset (one lightweight meta/episodes
    # read, not a Ray Data query). None/empty when GIFs aren't configured
    # at all, so no extra I/O happens in the common case.
    gif_episode_pool: list[int] | None = None
    if rank == 0 and train_cfg.wandb_gif_cameras and v3_root:
        if val_episode_indices:
            gif_episode_pool = list(val_episode_indices)
        else:
            from training.vendor.lerobot_datasource import LeRobotDatasourceMetadata

            gif_episode_pool = list(range(LeRobotDatasourceMetadata(v3_root).total_episodes))

    # Shared by early-stop patience and checkpoint-improvement-gating -- one
    # running "best loss seen this run", updated at both window and epoch
    # reports. Not restored from a resumed checkpoint.
    best_loss_seen = float("inf")
    best_val_loss_seen = float("inf")
    windows_without_improvement = 0
    should_stop = False
    # Tracks the last step the periodic VAL block actually ran at, so the
    # epoch-end safety-net val pass (below) can skip itself when it would
    # be a redundant, back-to-back re-run of the SAME step's val set (e.g.
    # --max-train-steps landing exactly on a --val-every-steps boundary) --
    # a real cross-rank ray.train.report() stall was observed from two val
    # passes firing back-to-back at the same step, not just wasted compute.
    last_val_step: int | None = None

    # --- perf instrumentation (training/perf_logging.py), opt-in via
    # --log-perf-metrics -- see that module's docstring for why it's not
    # always on. `sync_device` is passed to every perf_logging.Timer below;
    # None (the off case) skips torch.cuda.synchronize() entirely, so the
    # only always-on cost is a couple of cheap perf_counter() calls per
    # step, not the real synchronization overhead this flag exists to make
    # opt-in.
    sync_device = device if train_cfg.log_perf_metrics else None
    step_ref = [0]  # mutated every step below; read by the GPU poller thread
    gpu_poll_stop = None
    world_size = ray.train.get_context().get_world_size()
    if train_cfg.log_perf_metrics and rank == 0:
        eff_bs = perf_logging.effective_batch_size(train_cfg, world_size)
        _log_all(tb_writer, wandb_run, "perf", {"effective_batch_size": eff_bs}, 0, train_cfg)
        print(
            f"Effective batch size: {eff_bs} (batch_size={train_cfg.batch_size} x "
            f"grad_accum={train_cfg.grad_accum} x world_size={world_size})"
        )
        # CUDA-only -- gpu_snapshot() raises unconditionally on a CPU-only device.
        if device.type == "cuda":
            gpu_poll_stop = perf_logging.start_gpu_poller(
                lambda snap, step: _log_all(tb_writer, wandb_run, "perf", snap, step, train_cfg),
                device, step_ref,
            )

    prof = None
    prof_started = False
    if train_cfg.log_perf_metrics and train_cfg.profile_steps and rank == 0:
        # torch.profiler is already reachable off the module-level `import
        # torch` above -- an `import torch.profiler` statement here would
        # make `torch` itself a local name for this WHOLE function (Python
        # decides local-vs-global at compile time from any local
        # import/assignment anywhere in the function body), breaking the
        # very first line's `torch.device(...)` with an UnboundLocalError.
        # Confirmed by hitting exactly that in a real run.
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_dir),
        )
    # Single continuous timeline across the whole run (NOT reset per epoch,
    # unlike the window_* sums below) -- inter-epoch time (epoch-end
    # reporting/logging) is real idle time and legitimately belongs in the
    # next epoch's first data_wait_s measurement, not discarded.
    t_iter_end = time.perf_counter()

    for epoch in range(start_epoch, train_cfg.num_epochs):
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        loss_sum = 0.0
        extra_sums: dict[str, float] = {}
        n = 0
        window_loss_sum = 0.0
        window_extra_sums: dict[str, float] = {}
        window_n = 0
        window_data_wait_sum = window_preprocess_sum = window_compute_sum = 0.0
        window_optimizer_step_sum, window_optimizer_step_n = 0.0, 0
        window_wall_start = time.perf_counter()

        for batch in shard.iter_torch_batches(batch_size=train_cfg.batch_size, collate_fn=collate):
            data_wait_s = time.perf_counter() - t_iter_end

            if not adapter.needs_task:
                batch.pop("task", None)  # language conditioning this policy doesn't use

            if prof is not None and not prof_started and step + 1 == train_cfg.profile_steps[0]:
                # Start before this iteration's compute -- `step` is still the previous count.
                prof.start()
                prof_started = True

            # If a Ray Data stage already ran the policy's full preprocessor
            # upstream (see offload_molmoact2_preprocessing), the batch is
            # already image-normalized/normalized/tokenized -- skip doing
            # any of that again here.
            with perf_logging.Timer(sync_device) as t_prep:
                if not adapter.preprocessing_offloaded:
                    batch = apply_image_normalization(
                        batch, data_cfg.image_normalization, data_cfg.default_image_normalization,
                        dataset_stats, data_cfg.image_normalization_max,
                    )
                inputs = batch if adapter.preprocessing_offloaded else preprocessor(batch)

            with perf_logging.Timer(sync_device) as t_compute:
                if dist_ctx is not None and hasattr(dist_ctx, "autocast"):
                    # FSDP2/accelerate path only. Accelerator(mixed_precision="bf16")
                    # casts model parameters to bf16 but not input batches, so the
                    # forward needs autocast() to cast activations on the fly.
                    with dist_ctx.autocast():
                        loss, step_metrics = adapter.forward_loss(policy, inputs)
                else:
                    loss, step_metrics = adapter.forward_loss(policy, inputs)

                (loss / train_cfg.grad_accum).backward()

            # Brand-new metrics a custom/future adapter computes beyond its
            # own forward_loss (e.g. something a world model wants to
            # track) -- None for every policy today. Merged into
            # step_metrics so it automatically flows into train/*/window/*/
            # epoch/* below, no separate logging path needed. Kept outside
            # the t_compute timer above -- perf/compute_s should reflect
            # real forward+backward cost, not an arbitrary extra metric.
            if adapter.extra_metrics is not None:
                step_metrics = {**step_metrics, **adapter.extra_metrics(policy, inputs)}

            step += 1
            step_ref[0] = step
            accum += 1
            loss_sum += loss.item()
            n += 1
            window_loss_sum += loss.item()
            window_n += 1
            window_data_wait_sum += data_wait_s
            window_preprocess_sum += t_prep.elapsed
            window_compute_sum += t_compute.elapsed
            for k, v in step_metrics.items():
                extra_sums[k] = extra_sums.get(k, 0.0) + v
                window_extra_sums[k] = window_extra_sums.get(k, 0.0) + v

            ran_optimizer_step = False
            if accum % train_cfg.grad_accum == 0:
                with perf_logging.Timer(sync_device) as t_opt:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                ran_optimizer_step = True
                window_optimizer_step_sum += t_opt.elapsed
                window_optimizer_step_n += 1
                accum = 0

            if prof is not None and prof_started:
                prof.step()
                if step == train_cfg.profile_steps[1]:
                    prof.stop()
                    prof_started = False

            if step % 10 == 0 and rank == 0:
                # Explicit timestamp -- Ray's own log capture for a worker's
                # stdout only prefixes "(RayTrainWorker pid=...)", it doesn't
                # add one the way the driver's own INFO lines do.
                log.info(
                    "%s epoch=%d step=%d loss=%.4f %s",
                    time.strftime("%Y-%m-%d %H:%M:%S"), epoch, step, loss.item(),
                    " ".join(f"{k}={v:.4f}" for k, v in step_metrics.items()),
                )
                train_values = {"loss": loss.item(), **step_metrics}
                for i, g in enumerate(optimizer.param_groups):
                    train_values[f"lr_group{i}"] = g["lr"]
                _log_all(tb_writer, wandb_run, "train", train_values, step, train_cfg)
                if train_cfg.log_perf_metrics:
                    perf_values = {
                        "data_wait_s": data_wait_s, "preprocess_s": t_prep.elapsed, "compute_s": t_compute.elapsed,
                    }
                    if ran_optimizer_step:
                        perf_values["optimizer_step_s"] = t_opt.elapsed
                    _log_all(tb_writer, wandb_run, "perf", perf_values, step, train_cfg)
                if tb_writer is not None:
                    tb_writer.flush()

            # Step-windowed report: finer-grained than epoch, so early
            # stopping has more than one data point even within a single
            # epoch or a max_train_steps-capped run. Training-loss-based
            # only -- best-checkpoint scoring lives in the VAL block below
            # whenever a val split is active.
            if step % train_cfg.window_every_steps == 0:
                window_metrics = {
                    "epoch": epoch, "steps": step, "report_kind": "window",
                    "loss": window_loss_sum / max(window_n, 1),
                    **{k: v / max(window_n, 1) for k, v in window_extra_sums.items()},
                }
                window_loss_sum = 0.0
                window_extra_sums = {}
                window_n_for_perf = window_n  # window_n gets zeroed right below
                window_n = 0

                improved = False
                if rank == 0:
                    _log_all(
                        tb_writer, wandb_run, "window",
                        {k: v for k, v in window_metrics.items() if k not in ("epoch", "steps", "report_kind")},
                        step, train_cfg,
                    )
                    if train_cfg.log_perf_metrics:
                        window_wall_s = time.perf_counter() - window_wall_start
                        io_bound_denom = window_data_wait_sum + window_preprocess_sum + window_compute_sum
                        perf_window_values = {
                            "io_bound_fraction": window_data_wait_sum / io_bound_denom if io_bound_denom > 0 else 0.0,
                            "samples_per_sec": (
                                window_n_for_perf * train_cfg.batch_size * world_size / window_wall_s
                                if window_wall_s > 0 else 0.0
                            ),
                        }
                        if window_optimizer_step_n > 0:
                            perf_window_values["optimizer_step_s_avg"] = (
                                window_optimizer_step_sum / window_optimizer_step_n
                            )
                        _log_all(tb_writer, wandb_run, "perf", perf_window_values, step, train_cfg)
                    if tb_writer is not None:
                        tb_writer.flush()

                    improved = window_metrics["loss"] < best_loss_seen
                    if improved:
                        best_loss_seen = window_metrics["loss"]

                    if train_cfg.early_stop_patience is not None:
                        windows_without_improvement = 0 if improved else windows_without_improvement + 1
                        should_stop = windows_without_improvement >= train_cfg.early_stop_patience
                        if should_stop:
                            log.info(
                                "EARLY STOP at step=%d: loss hasn't improved for %d windows (best=%.4f)",
                                step, windows_without_improvement, best_loss_seen,
                            )

                window_data_wait_sum = window_preprocess_sum = window_compute_sum = 0.0
                window_optimizer_step_sum, window_optimizer_step_n = 0.0, 0
                window_wall_start = time.perf_counter()

                should_stop = _sync_should_stop(should_stop, device)  # every rank must call this
                # Whenever a val split is active, val owns checkpoint
                # retention entirely (see the VAL block below) -- the
                # windowed block becomes pure training-loss logging/early-
                # stop, no checkpoint attached here. Disabled (val_shard is
                # None): byte-for-byte today's behavior, unchanged.
                save_checkpoint = (
                    False if val_shard is not None
                    else (improved if train_cfg.save_only_on_improvement else True)
                )
                with perf_logging.Timer() as t_ckpt:
                    _report_with_checkpoint(
                        window_metrics, adapter, unwrapped_policy, optimizer, dist_ctx,
                        run_cfg, epoch, step, epoch_complete=False, rank=rank, save_checkpoint=save_checkpoint,
                        async_checkpointer=async_checkpointer, normalization_snapshot=normalization_snapshot,
                    )
                if train_cfg.log_perf_metrics and rank == 0 and save_checkpoint:
                    _log_all(tb_writer, wandb_run, "perf", {"checkpoint_save_s": t_ckpt.elapsed}, step, train_cfg)

            # Periodic VAL pass -- own top-level cadence check, not nested
            # in the window block above (val_every_steps can differ from
            # window_every_steps). Drives checkpoint retention: unlike the
            # windowed/epoch-end blocks (training loss), this reports the
            # real held-out val loss under checkpoint_score_attribute="loss",
            # so a good-looking training loss that masks real overfitting no
            # longer wins retention. Early stopping (above) stays
            # training-loss-based, unchanged.
            if val_shard is not None and step % effective_val_every_steps == 0:
                last_val_step = step
                policy.eval()
                val_loss_sum, val_n, val_metrics_sum, last_val_inputs = _run_eval_pass(
                    val_shard, adapter, policy, preprocessor, data_cfg, dataset_stats, dist_ctx, collate,
                    train_cfg.val_batch_size or train_cfg.batch_size, train_cfg.val_max_batches, device,
                )

                # Predicted-frames GIFs -- pure groundwork for a future
                # policy that actually predicts visual frames (e.g. a world
                # model); see PolicyAdapter.predict_frames's docstring.
                # Inert no-op for ACT/MolmoAct2/PI05 today (adapter.
                # predict_frames is None for all three). Logged at this val
                # pass's own step, same key across the run, so W&B's own
                # per-step media history is what shows the generated frames
                # improving over training -- no extra "compare across
                # steps" logic needed. --wandb-predict-frames-every-steps
                # controls how often, relative to val passes (can only be a
                # multiple of the EFFECTIVE val cadence -- generating one
                # needs a real val batch, which only exists here).
                predict_frames_every = train_cfg.wandb_predict_frames_every_steps or effective_val_every_steps
                if (
                    adapter.predict_frames is not None and rank == 0
                    and wandb_run is not None and last_val_inputs is not None
                    and step % predict_frames_every == 0
                ):
                    # last_val_inputs' tensors were created under
                    # _run_eval_pass's own torch.inference_mode() -- an
                    # inference tensor used in a graph built OUTSIDE
                    # inference_mode can raise at runtime (autograd refuses
                    # to save it for backward), so keep this call inside
                    # the same mode too, not just policy.eval().
                    with torch.inference_mode():
                        predicted = adapter.predict_frames(policy, last_val_inputs)
                    if predicted:
                        import wandb

                        media = {
                            f"val_gif/{name}": wandb.Video(
                                frames.transpose(0, 3, 1, 2), format="gif", fps=data_cfg.robot.tick_fps,
                            )
                            for name, frames in predicted.items()
                        }
                        # Apply the same allow/deny filter as every other metric.
                        filtered = filter_metrics(media, train_cfg.wandb_metrics, train_cfg.wandb_exclude_metrics)
                        if filtered:
                            wandb_run.log(filtered, step=step)
                policy.train()

                # Collective -- _finish_val_pass's own _gather_eval_stats
                # call, every rank must reach it. Unlike the training-loss
                # window/epoch reports (rank 0's own local shard slice
                # only), every rank ends up with the SAME globally-summed
                # val loss after that gather, so this is safe to call
                # identically on every rank.
                best_val_loss_seen = _finish_val_pass(
                    val_loss_sum, val_n, val_metrics_sum, adapter, unwrapped_policy, optimizer, dist_ctx,
                    run_cfg, train_cfg, epoch, step, epoch_complete=False, rank=rank,
                    tb_writer=tb_writer, wandb_run=wandb_run, best_val_loss_seen=best_val_loss_seen,
                    async_checkpointer=async_checkpointer, normalization_snapshot=normalization_snapshot,
                )

            # Episode-preview GIFs -- independent cadence from the val
            # report above (wandb_gif_every_steps can differ from the
            # effective val cadence), so this is its own top-level check,
            # not nested in the block above.
            if rank == 0 and wandb_run is not None and gif_episode_pool:
                gif_every = train_cfg.wandb_gif_every_steps or effective_val_every_steps
                if step % gif_every == 0:
                    ep_idx = gif_rng.choice(gif_episode_pool)
                    for cam in train_cfg.wandb_gif_cameras:
                        try:
                            frames = sample_episode_frames(
                                v3_root, ep_idx, cam, train_cfg.wandb_gif_frames, data_cfg.image_size,
                            )
                        except ValueError as e:
                            log.warning("GIF sampling failed for camera %r, episode %d: %s", cam, ep_idx, e)
                            continue
                        import wandb

                        frames_chw = frames.transpose(0, 3, 1, 2)
                        media = {f"gif/{cam}": wandb.Video(frames_chw, format="gif", fps=data_cfg.robot.tick_fps)}
                        # Same allow/deny filter as every other metric.
                        filtered = filter_metrics(media, train_cfg.wandb_metrics, train_cfg.wandb_exclude_metrics)
                        if filtered:
                            wandb_run.log(filtered, step=step)

            t_iter_end = time.perf_counter()
            if should_stop:
                break
            if train_cfg.max_train_steps and step >= train_cfg.max_train_steps:
                break

        if accum > 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Coarser per-epoch summary, kept alongside the step-windowed reports above.
        metrics = {
            "epoch": epoch, "steps": step, "report_kind": "epoch",
            "loss": loss_sum / max(n, 1),
            **{k: v / max(n, 1) for k, v in extra_sums.items()},
        }

        improved = False
        if rank == 0:
            _log_all(
                tb_writer, wandb_run, "epoch",
                {k: v for k, v in metrics.items() if k not in ("epoch", "steps", "report_kind")},
                epoch, train_cfg, wandb_step=step,
            )
            if tb_writer is not None:
                tb_writer.flush()

            improved = metrics["loss"] < best_loss_seen
            if improved:
                best_loss_seen = metrics["loss"]

        if val_shard is not None and step != last_val_step:
            # Same gating as the windowed block: whenever a val split is
            # active, val owns checkpoint retention exclusively -- an
            # epoch-end checkpoint scored by training loss would otherwise
            # sit in the SAME best-N pool as val-scored checkpoints under
            # the same checkpoint_score_attribute="loss" key, letting a low
            # training loss displace the intended val-selected checkpoints.
            # Running a fresh val pass here (rather than just skipping
            # attachment) matters for a SHORT run -- fewer steps than the
            # val cadence never fires the periodic VAL block at all, and
            # skipping epoch-end too would leave the run with NO checkpoint
            # whatsoever. Skipped entirely (see the else branch) when the
            # periodic VAL block already ran at this EXACT step (e.g.
            # --max-train-steps landing on a --val-every-steps boundary) --
            # NOT just "a small, bounded extra cost" as originally assumed
            # here: a real run hit a genuine cross-rank ray.train.report()
            # stall (SynchronizationActor waiting 120+s on one rank) from
            # two val passes firing back-to-back at the same step.
            policy.eval()
            epoch_val_loss_sum, epoch_val_n, epoch_val_metrics_sum, _ = _run_eval_pass(
                val_shard, adapter, policy, preprocessor, data_cfg, dataset_stats, dist_ctx, collate,
                train_cfg.val_batch_size or train_cfg.batch_size, train_cfg.val_max_batches, device,
            )
            policy.train()
            best_val_loss_seen = _finish_val_pass(
                epoch_val_loss_sum, epoch_val_n, epoch_val_metrics_sum, adapter, unwrapped_policy, optimizer,
                dist_ctx, run_cfg, train_cfg, epoch, step, epoch_complete=True, rank=rank,
                tb_writer=tb_writer, wandb_run=wandb_run, best_val_loss_seen=best_val_loss_seen,
                async_checkpointer=async_checkpointer, normalization_snapshot=normalization_snapshot,
            )
        else:
            # Either val is disabled entirely, OR periodic val already ran
            # at this exact step -- report the training-loss epoch summary
            # for bookkeeping symmetry (every rank must still call
            # ray.train.report() once here), but NEVER attach a checkpoint
            # from it when val is active: val already owns (or, via the
            # async one-cycle lag, will shortly own) retention for this
            # step, so a redundant training-loss-scored checkpoint here
            # would just pollute the same scoring pool val is supposed to
            # own exclusively.
            save_checkpoint = (
                False if val_shard is not None
                else (improved if train_cfg.save_only_on_improvement else True)
            )
            with perf_logging.Timer() as t_ckpt:
                _report_with_checkpoint(
                    metrics, adapter, unwrapped_policy, optimizer, dist_ctx,
                    run_cfg, epoch, step, epoch_complete=True, rank=rank, save_checkpoint=save_checkpoint,
                    async_checkpointer=async_checkpointer, normalization_snapshot=normalization_snapshot,
                )
            if train_cfg.log_perf_metrics and rank == 0 and save_checkpoint:
                _log_all(tb_writer, wandb_run, "perf", {"checkpoint_save_s": t_ckpt.elapsed}, step, train_cfg)

        if should_stop or (train_cfg.max_train_steps and step >= train_cfg.max_train_steps):
            break

    # One-time TEST pass -- runs exactly once, after training completes
    # (whether by exhausting num_epochs, early stop, or max_train_steps),
    # against episodes NEVER touched during training or val. Uncapped
    # (max_batches=None): evaluate the whole test split fully, exactly
    # once -- capping it would defeat the point. save_checkpoint=False
    # unconditionally: test's loss is recorded into Ray's own
    # result.metrics (and therefore history.jsonl) but never attaches a
    # checkpoint, so test can't influence retention by construction.
    if test_shard is not None:
        policy.eval()
        test_loss_sum, test_n, test_metrics_sum, _ = _run_eval_pass(
            test_shard, adapter, policy, preprocessor, data_cfg, dataset_stats, dist_ctx, collate,
            train_cfg.val_batch_size or train_cfg.batch_size, max_batches=None, device=device,
        )
        policy.train()
        # Collective -- every rank must call this, not just rank 0.
        test_loss_sum, test_n, test_metrics_sum = _gather_eval_stats(test_loss_sum, test_n, test_metrics_sum)
        test_report_metrics = {
            "epoch": epoch, "steps": step, "report_kind": "test",
            "loss": test_loss_sum / max(test_n, 1),
            **{k: v / max(test_n, 1) for k, v in test_metrics_sum.items()},
        }
        if rank == 0 and test_n > 0:
            _log_all(
                tb_writer, wandb_run, "test",
                {k: v for k, v in test_report_metrics.items() if k not in ("epoch", "steps", "report_kind")},
                step, train_cfg,
            )
            if tb_writer is not None:
                tb_writer.flush()
        _report_with_checkpoint(
            test_report_metrics, adapter, unwrapped_policy, optimizer, dist_ctx,
            run_cfg, epoch, step, epoch_complete=True, rank=rank, save_checkpoint=False,
            normalization_snapshot=normalization_snapshot,
        )

    if async_checkpointer is not None:
        # Flush the LAST in-flight async save -- without this, a save
        # kicked off by the final checkpoint-worthy call never gets
        # reported to Ray at all (nothing left to trigger it), silently
        # losing what should be the run's final checkpoint.
        #
        # Every rank MUST call ray.train.report() exactly once here,
        # symmetrically -- a real deadlock was observed from an earlier,
        # rank-0-only version of this block: only rank 0 ever touches
        # async_checkpointer, so when it had a pending save to flush, it
        # called report(checkpoint=...) here while every other rank called
        # nothing at all, permanently desyncing Ray Train's cross-rank
        # report() rendezvous (SynchronizationActor waited forever, not
        # just the ~60-120s seen for the earlier, transient double-val-pass
        # stall -- this one never resolves, since no later event makes the
        # other ranks catch up). drain() on any non-zero rank always
        # returns None (that rank's own async_checkpointer never had
        # anything submitted to it), so the "else" branch below covers
        # them for free -- this isn't rank-gated, it's just naturally a
        # no-op there.
        finished = async_checkpointer.drain() if rank == 0 else None
        if finished is not None:
            _attach_finished_async_checkpoint(finished)
        else:
            ray.train.report({"report_kind": "final_drain"})

    if prof is not None and prof_started:
        # max_train_steps/should_stop broke out of the loop before reaching
        # profile_steps' end -- stop it anyway so on_trace_ready still
        # fires and a partial trace is written, not silently dropped.
        prof.stop()

    if rank == 0:
        # Per-operator Ray Data execution stats for exactly what this
        # worker's shard consumed -- not ds.stats() on the driver's original
        # dataset, which doesn't reflect Ray Train's per-worker splitting.
        stats_text = shard.stats()
        log.info("Ray Data shard stats:\n%s", stats_text)
        stats_path = os.path.join(run_cfg.storage_root, run_cfg.run_name, "ray_data_stats.txt")
        with open(stats_path, "w") as f:
            f.write(stats_text)
        print(f"Ray Data stats -> {stats_path}")
        if tb_writer is not None:
            tb_writer.add_text("ray_data/shard_stats", f"```\n{stats_text}\n```", step)

    if gpu_poll_stop is not None:
        gpu_poll_stop.set()  # stop the poller thread before closing tb_writer under it

    if tb_writer is not None:
        tb_writer.close()

    if wandb_run is not None:
        wandb_run.finish()
