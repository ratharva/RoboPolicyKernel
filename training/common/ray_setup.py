"""Shared Ray connection setup for train.py and prepare_data.py."""
from __future__ import annotations

import os

import ray


def _quiet_placement_group_cleaner() -> None:
    """Filters out ONLY PlacementGroupCleaner's benign, high-frequency
    "State API may be temporarily unavailable" message (a periodic health-
    check retry, not a real problem) -- NOT a setLevel(ERROR) on the whole
    logger, which would also hide its other real warnings (cleanup failure,
    monitor-thread shutdown timeout, dead-controller detection -- confirmed
    by reading every logger.warning() call site in placement_group_cleaner.py)
    that matter for noticing leaked resources."""
    import logging

    class _TransientStateApiWarningFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "State API may be temporarily unavailable" not in record.getMessage()

    logging.getLogger(
        "ray.train.v2._internal.execution.controller.placement_group_cleaner"
    ).addFilter(_TransientStateApiWarningFilter())


def _default_nccl_env() -> None:
    """Sets a few NCCL env vars directly in THIS worker process's own
    os.environ (only if not already present) -- a real, reproduced case
    showed `runtime_env`'s `env_vars` (passed to ray.init()) did NOT
    reliably reach the actual NCCL init call inside a Ray Train worker
    (same NVLS CUDA error persisted with env_vars set both ways: neither a
    shell `export` before launching, nor forwarding it through
    build_runtime_env's own env_vars, changed anything). Setting it here,
    directly via os.environ in a worker_process_setup_hook callable that's
    already confirmed to run inside the actual worker process (see
    _quiet_placement_group_cleaner above), is unambiguous: NCCL reads env
    vars via plain getenv() at ITS OWN init time, which happens well after
    this hook runs, so this is guaranteed to be visible to it regardless
    of whatever the runtime_env/ray.init() layer does or doesn't forward.

    Disables NVLink SHARP (NVLS) multicast and P2P -- real, reproduced
    fix for a shared/virtualized multi-GPU instance whose NVSwitch fabric
    isn't fully exposed to the container: `NCCL error ... Failed to bind
    NVLink SHARP (NVLS) Multicast memory ... CUDA error 401`. Bounded
    downside on a healthy/bare-metal multi-GPU box (somewhat less optimal
    collective-communication bandwidth, not a correctness issue), so this
    is a safe default rather than something requiring opt-in -- setdefault
    means an explicit, already-working override still wins over this."""
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")


def _worker_process_setup() -> None:
    """The actual worker_process_setup_hook passed to Ray -- combines
    every per-worker-process startup action this project needs. Runs once
    per Ray worker process at startup, confirmed via a real local test."""
    _quiet_placement_group_cleaner()
    _default_nccl_env()


def build_runtime_env(storage_root: str | None = None) -> dict:
    """working_dir="." ships the repo root so `training.*` imports resolve
    identically on Ray workers. excludes keeps secrets, caches, converted
    video, and checkpoints out of that upload -- "training/runs/**" (the
    default storage_root) is always excluded, and storage_root is
    additionally excluded by its real resolved path so a --storage-root
    override elsewhere in the repo stays correct too.
    """
    excludes = [
        "hf_tok.txt", "*.txt",
        "training/verify_out/**", "training/.cache/**", "training/lerobot_v3/**",
        "training/runs/**",
        "**/__pycache__", "**/.git",
    ]
    if storage_root:
        try:
            rel = os.path.relpath(storage_root, os.getcwd())
        except ValueError:
            rel = None  # e.g. different drive on Windows
        if rel and not rel.startswith(".."):
            excludes.append(f"{rel}/**")

    # Also forward any NCCL_* var already in THIS (driver) process's
    # environment -- belt-and-suspenders alongside _default_nccl_env's
    # direct os.environ.setdefault() in the worker_process_setup_hook
    # below (that one is the confirmed-effective mechanism; this one is
    # cheap to include too in case a future Ray version does forward it).
    env_vars = {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}

    return {
        "working_dir": ".", "excludes": excludes,
        "worker_process_setup_hook": _worker_process_setup,
        "env_vars": env_vars,
    }


def connect_ray(runtime_env: dict) -> "ray.runtime_context.RuntimeContext":
    try:
        ctx = ray.init(address="auto", ignore_reinit_error=True, runtime_env=runtime_env)
        print("connected to existing cluster")
    except ConnectionError:
        # Loopback-only by default (Ray's dashboard has no auth). Override via
        # ROBOPOLICYKERNEL_DASHBOARD_HOST for remote access, or tunnel instead:
        # ssh -L 8265:localhost:8265 <host>.
        # `or None` normalizes an explicitly-empty value to "unset" -- os.environ.get(key,
        # default) only falls back on a MISSING key, not a present-but-empty one, which
        # would otherwise silently produce dashboard_host="" instead of the real fallback.
        dashboard_host = os.environ.get("ROBOPOLICYKERNEL_DASHBOARD_HOST") or "127.0.0.1"
        if dashboard_host not in ("127.0.0.1", "localhost", "::1"):
            print(f"WARNING: Ray dashboard binding to {dashboard_host!r} -- no built-in auth, "
                  f"anyone who can reach it can execute code on this machine.")
        ctx = ray.init(ignore_reinit_error=True, runtime_env=runtime_env, dashboard_host=dashboard_host)
        print("started a local Ray instance")
    dashboard_url = getattr(ctx, "dashboard_url", None)
    if dashboard_url:
        print(f"Ray Dashboard: http://{dashboard_url}")
    return ctx
