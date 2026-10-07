# Training -- `train.py`

Reads an already-prepared LeRobot v3 dataset and trains a policy on it.
Assumes `training/prepare_data.py` has already been run for `--tasks` --
this script does no discover/download/convert of its own; if no prepared
dataset is found at `--v3-root`, it exits with the `prepare_data.py`
command to run first. See [`training/data_prep/README.md`](data_prep/README.md)
for that step.

```bash
python -m training.prepare_data --tasks arrange_the_flowers --dataset-source abc130k --max-episodes-per-task 300
python -m training.train --tasks arrange_the_flowers --dataset-source abc130k --policy-type act --max-train-steps 50
```

## General flags

Neither `--dataset-source` nor `--policy-type` has a default -- both are
either required outright (`--policy-type`) or required unless `--v3-root`
is given explicitly (`--dataset-source`, see the table row below).

| Flag | Default | Meaning |
|---|---|---|
| `--tasks` (required) | -- | must match what `prepare_data.py` was run with |
| `--config-file` | none | optional YAML file providing a base `RunConfig` -- every flag in this doc still works and takes precedence; see "Config files" below |
| `--run-name` | `<policy_type>-<tasks>-<timestamp>` | also the resume key -- see below |
| `--storage-root` | `TrainConfig` default | where run output (checkpoints, TensorBoard, history) is written |
| `--num-epochs` | `1` | |
| `--batch-size` | `8` | |
| `--lr` | `1e-5` | base learning rate -- the fallback every MolmoAct2 per-group LR flag (`--molmoact2-vit-lr` etc.) uses when left unset |
| `--lr-backbone` | `1e-5` | ACT-specific backbone LR group -- unused by MolmoAct2/pi05, which have their own LR-group flags |
| `--grad-accum` | `1` | gradient accumulation steps |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--max-train-steps` | none (full epoch) | cap total steps -- for a smoke run |
| `--window-every-steps` | `200` | TRAINING-loss report/early-stop-check granularity -- unrelated to val/test, which have their own cadence below |
| `--early-stop-patience` | `5` | stop after this many windows with no TRAINING-loss improvement; `0` or negative disables it |
| `--save-only-on-improvement` | off | only checkpoint when loss improves (less write I/O, but a resume can lose progress back to the last improvement) -- see "Checkpoint retention" below for what "improves" means once val is active |
| `--checkpoint-max-to-keep` | `3` | keeps the N best-scoring checkpoints plus the single most recent one (native Ray Train `CheckpointConfig` behavior) |
| `--val-split-fraction` | `0.1` | hold out this fraction of episodes for a periodic val pass that DRIVES CHECKPOINT RETENTION -- on by default; `0` disables it (byte-for-byte pre-val-split behavior). Mutually exclusive with `--val-v3-root` |
| `--val-every-steps` | none -> reuses `--window-every-steps` | val's own report/checkpoint cadence |
| `--val-max-batches` | `50` | caps the periodic val pass to this many batches per rank per window -- tune down if val is slow relative to `--window-every-steps` |
| `--val-batch-size` | none -> reuses `--batch-size` | val/test have no optimizer-state/gradient overhead, so a larger batch is often safe |
| `--val-v3-root` | none | use a wholly separate, already-prepared v3 root for val instead of a slice of the main `--v3-root` -- ALL of its episodes are used. Schema must match (camera/state/action components); mutually exclusive with `--val-split-fraction` |
| `--test-split-fraction` | none (disabled) | fully opt-in -- hold out this fraction for a ONE-TIME test pass after training completes, logged under `test/*`, never attaches a checkpoint. Mutually exclusive with `--test-v3-root` |
| `--test-v3-root` | none | same shape as `--val-v3-root`, for the one-time test pass |
| `--num-workers` | live GPU count | Ray Train DDP worker count |
| `--v3-root` | derived from `--dataset-source` + `--tasks` (prepare_data.py's own default path) | must already exist. If omitted, `--dataset-source` is required so the default path can be derived at all |
| `--dataset-source` (required unless `--v3-root` given) | read from the prepared dataset's own `conversion_params.json` if `--v3-root` is given and omits it | which schema to resolve camera_keys/state_dim/action_dim/tick_fps from |
| `--policy-type` (required) | -- | `act`, `molmoact2`, or `pi05` |

```bash
# Custom base LR, ACT backbone LR, gradient accumulation, and weight decay
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type act \
    --lr 5e-5 --lr-backbone 1e-5 --grad-accum 4 --weight-decay 1e-3
```

## Config files

`--config-file PATH` loads a YAML file into a base `RunConfig`, parsed with
[`draccus`](https://github.com/dlwh/draccus) -- the same library lerobot's
own `ACTConfig`/`PI05Config`/`MolmoAct2Config` are built on (already an
installed dependency, pulled in transitively by `lerobot==0.6.1`). It's
purely additive: **every named flag in this doc still works exactly as it
always has, and takes precedence over anything the YAML sets** --
`--config-file` only fills in values nothing else was explicitly passed
for. `--tasks` and `--policy-type` are still required on the command line
even when the file also sets them (there's no way to make a required flag
optional only when a config file exists without changing the CLI's shape
for everyone, which this was deliberately kept from doing).

Precedence, low to high: dataclass defaults (`training/config.py`) <
`--config-file`'s YAML < a named CLI flag actually typed on the command
line. One accepted limitation: a CLI flag whose value happens to equal its
own default is indistinguishable from not having passed it at all, so the
config-file's value (if any) wins in that case -- if you need to force a
value back to its default while using a config file, remove it from the
file instead of relying on the flag.

The YAML mirrors `RunConfig`'s real shape (`training/config.py`,
`training/common/config.py`'s `DataConfig`) -- top-level `tasks`/
`policy_type`/`run_name`/`storage_root`, nested `train:`/`data:` sections,
and a `model:` section whose `type: act|molmoact2|pi05` key selects which
of the three `*ConfigOverrides` dataclasses the rest of that section's
fields apply to (draccus's "choice registry" mechanism -- if `model.type`
disagrees with the resolved `--policy-type`, `train.py` exits with a clear
error rather than guessing which one you meant).

Five real, draccus-verified examples in
[`training/configs/`](configs/README.md) -- one per policy, plus one
showing the `data:` section (action-space selection, delta actions,
per-camera image normalization including a depth camera):

```bash
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k \
    --policy-type act --config-file training/configs/act_example.yaml

# A named flag still overrides whatever the file sets
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k \
    --policy-type act --config-file training/configs/act_example.yaml --batch-size 32
```

## Action space & per-camera image normalization

General flags -- apply regardless of `--policy-type`, applied once to the
Ray Dataset before normalization stats are computed (not a per-policy
concern).

| Flag | Default | Meaning |
|---|---|---|
| `--action-space` | none (every action component) | which named subset of the dataset's `action_components` to train on -- e.g. `joint` or `end_effector` for `agibot_alpha`, which records both. Valid names are dataset-specific (the schema's `action_space_components` keys); most datasets declare none, so this only applies to `agibot_alpha` today |
| `--action-representation` | `absolute` | `absolute` (unchanged) or `delta`: `action -= observation.state` per dim, using each action component's **same-named** state component as the reference point -- components with no same-named state component (e.g. `agibot_alpha`'s `robot/velocity`, which has no state counterpart) stay absolute, there's no reference point to use |
| `--action-delta-exclude` | none | action component names to keep absolute even under `--action-representation delta` (e.g. a gripper component) |
| `--image-normalization` | none | per-camera mode, `CAMERA=MODE` pairs, e.g. `--image-normalization top=unit01 depth_head=depth`. Modes: `mean_std` (default, dataset-computed mean/std, current behavior), `unit01` (`x/255`), `unit_pm1` (`x/127.5 - 1`), `depth` (`clip(x,0,max)/max`), `log` (`log1p(x)/log1p(max)`) |
| `--image-normalization-default` | `mean_std` | mode for any camera not covered by `--image-normalization` |
| `--image-normalization-max` | none | `CAMERA=VALUE` pairs -- **required** for any camera using `depth`/`log` (the raw-value ceiling to clip/scale by, e.g. max depth in millimeters); no guessed default |

A single-channel camera (a depth camera, in practice) is automatically
replicated to 3 channels after normalization, so it flows through the same
vision backbone every RGB camera does -- see
`training/model/image_normalization.py`.

```bash
# Train agibot_alpha on joint-space actions instead of the full 36-dim
# action vector (which mixes joint- and end-effector-space components)
python -m training.train --tasks fridge --dataset-source agibot_alpha --policy-type act \
    --action-space joint

# Delta (relative-to-state) actions, keeping one component absolute --
# needs a dataset whose action/state components are actually same-named
# (abc130k's aren't: e.g. "/left-arm-action" vs "/left-arm-state", so
# --action-representation delta would leave every dim absolute there --
# see the --action-representation row above). agibot_alpha's are:
# "effector/position" appears in both state_components and
# action_components, so it gets a real delta unless excluded like this.
python -m training.train --tasks fridge --dataset-source agibot_alpha --policy-type act \
    --action-representation delta --action-delta-exclude effector/position

# Per-camera normalization: one RGB camera to [0,1], another to [-1,1],
# a depth camera clipped/scaled by its real max range (millimeters)
python -m training.train --tasks fridge --dataset-source agibot_alpha --policy-type act \
    --image-normalization top=unit01 wrist=unit_pm1 depth_head=depth \
    --image-normalization-max depth_head=5000
```

## Performance instrumentation

General flags -- apply regardless of `--policy-type`. Off by default:
accurate step timing needs `torch.cuda.synchronize()` calls, which
serialize async CUDA work and cost real throughput whenever they're on, so
this is opt-in rather than always-on.

| Flag | Default | Meaning |
|---|---|---|
| `--log-perf-metrics` | off | logs GPU utilization/VRAM, per-step timing (`perf/data_wait_s`/`preprocess_s`/`compute_s`/`optimizer_step_s`), effective batch size, throughput (`perf/samples_per_sec`), and an IO-bound-vs-compute-bound ratio (`perf/io_bound_fraction`) to TensorBoard under `perf/*`. Needs `nvidia-ml-py` installed for GPU compute-utilization % (`perf/gpu_util_pct`, `perf/gpu_mem_util_pct`) -- VRAM stats (`perf/vram_*`) work without it; missing `nvidia-ml-py` just skips those two, logged once as a warning, not a crash |
| `--profile-steps START:END` | none | requires `--log-perf-metrics`. Captures a real `torch.profiler` trace for steps `START..END` (inclusive) into the run's `tensorboard/` dir -- viewable in TensorBoard's PyTorch Profiler tab, same `tensorboard --logdir` command as everything else. Rank 0 only. Keep the range small (10-20 steps is usually plenty) -- traces get large fast; a range over 50 steps prints a warning |

`perf/io_bound_fraction` is `data_wait_s / (data_wait_s + preprocess_s +
compute_s)` per window -- loosely, high (>0.3-0.5) + low `gpu_util_pct`
means IO-bound (more Ray Data actors/CPU, bigger prefetch buffer, faster
decode is the fix); low + high `gpu_util_pct` means compute-bound (bigger
batch, mixed precision, model-side work is the fix). `perf/checkpoint_save_s`
is logged whenever a checkpoint actually writes -- without it, a periodic
step-time spike every checkpoint-attaching report (`--val-every-steps` by
default once val is active, else `--window-every-steps`) looks like
unexplained noise instead of "checkpointing is slow."

```bash
# Find out whether a run is IO- or compute-bound
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type act \
    --log-perf-metrics --max-train-steps 200

# Same, plus a kernel-level trace of steps 50-65 for a deeper look
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type act \
    --log-perf-metrics --profile-steps 50:65 --max-train-steps 200
```

## Resuming a run

There's no `--resume` flag -- re-running with the **same `--run-name`**
(and `--storage-root`) resumes automatically: Ray Train detects the
existing run directory and its checkpoints and picks up from the most
recent one. Pass `--run-name` explicitly if you want to resume later
(the default name embeds a timestamp, so a fresh invocation without it
always starts a new run).

```bash
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type act \
    --run-name teddy_bear_act_v1 --max-train-steps 500
# ...interrupted or crashed...
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type act \
    --run-name teddy_bear_act_v1 --max-train-steps 500
```

## ACT

```bash
# Smoke run
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type act --max-train-steps 20

# Full run over the data, larger batch
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type act --batch-size 16 --num-epochs 5
```

## MolmoAct2

`--policy-type molmoact2` -- a ~8B-param VLA. `--molmoact2-setup-type` and
`--molmoact2-control-mode` are **required**.

| Flag | Default | Meaning |
|---|---|---|
| `--molmoact2-checkpoint-path` | `allenai/MolmoAct2` | HF repo id or local path the VLM backbone loads from |
| `--molmoact2-setup-type` | -- (required) | free-text embodiment prompt, e.g. `"dual-arm robot with wrist and top cameras"` |
| `--molmoact2-control-mode` | -- (required) | free-text control-mode prompt, e.g. `"delta joint position"` |
| `--molmoact2-action-mode` | `continuous` | `continuous`, `discrete`, or `both` -- real default is `both`; narrowed here to skip the discrete FAST-tokenizer setup |
| `--molmoact2-train-mode` | `lora` | `lora` (LoRA on the VLM, action expert stays fully trainable -- fits under plain DDP, ~20GB at batch 8), `fft` (full fine-tune, needs `--molmoact2-distributed-strategy fsdp2` to be practical, ~60GB/GPU at batch 32 under plain DDP otherwise), or `freeze` (VLM frozen, only the action expert trains -- requires `--molmoact2-action-mode continuous`) |
| `--molmoact2-lora-rank` / `-alpha` / `-dropout` | `64` / `16` / `0.05` | `MolmoAct2Config`'s own real defaults (yes, alpha < rank) |
| `--molmoact2-no-gradient-checkpointing` | off (checkpointing on) | disable only with confirmed memory headroom |
| `--molmoact2-vit-lr` / `-connector-lr` / `-action-expert-lr` | none -> falls back to the fixed base `lr` | MolmoAct2's `get_optim_params()` returns 4 LR groups (vlm/vit/connector/action_expert), not ACT's flat params |
| `--molmoact2-distributed-strategy` | `ddp` | `ddp` (only sane with `lora`) or `fsdp2` (accelerate-driven FSDP2 sharding inside the Ray Train worker, for `fft` at scale) |
| `--molmoact2-fsdp-cpu-offload` | off | trades speed for fitting on fewer/smaller GPUs -- `fsdp2` only |
| `--molmoact2-offload-tokenization` | off | run MolmoAct2's tokenizer+image-processor as a Ray Data stage instead of inline in the training loop |
| `--molmoact2-offload-concurrency` | auto (from live CPU count) | Ray Data actor-pool size for the above |
| `--molmoact2-revision` | none (Hub latest) | pins weight loading to a specific Hub revision/commit/tag |
| `--molmoact2-norm-tag` | none | selects a real checkpoint-published tag (e.g. `franka_droid`) from `norm_stats.json` -- see "MolmoAct2 normalization resolution" below |
| `--adam-beta1` / `--adam-beta2` / `--adam-eps` | `0.9` / `0.999` / `1e-8` | AdamW hyperparameters (PyTorch's own defaults) -- generic, used by every policy's optimizer |

```bash
# LoRA on one GPU (the starting point -- get this working before fft)
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type molmoact2 \
    --molmoact2-setup-type "dual-arm robot with wrist and top cameras" \
    --molmoact2-control-mode "delta joint position" \
    --batch-size 8 --max-train-steps 50

# VLM frozen, only the action expert trains
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type molmoact2 \
    --molmoact2-setup-type "dual-arm robot with wrist and top cameras" \
    --molmoact2-control-mode "delta joint position" \
    --molmoact2-train-mode freeze --molmoact2-action-mode continuous \
    --batch-size 8

# Full fine-tune with FSDP2 across multiple GPUs on one node, plus the Ray
# Data preprocessing offload -- both carry real unverified risk (checkpoint
# resume under FSDP2, and Ray-Data-Arrow round-tripping tokenizer output)
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type molmoact2 \
    --molmoact2-setup-type "dual-arm robot with wrist and top cameras" \
    --molmoact2-control-mode "delta joint position" \
    --molmoact2-train-mode fft --molmoact2-distributed-strategy fsdp2 \
    --molmoact2-offload-tokenization --batch-size 32

# fft that doesn't fit even under FSDP2 -- trade speed for VRAM headroom
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type molmoact2 \
    --molmoact2-setup-type "dual-arm robot with wrist and top cameras" \
    --molmoact2-control-mode "delta joint position" \
    --molmoact2-train-mode fft --molmoact2-distributed-strategy fsdp2 \
    --molmoact2-fsdp-cpu-offload --batch-size 32

# Custom per-group learning rates
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type molmoact2 \
    --molmoact2-setup-type "dual-arm robot with wrist and top cameras" \
    --molmoact2-control-mode "delta joint position" \
    --molmoact2-vit-lr 1e-6 --molmoact2-connector-lr 5e-5 --molmoact2-action-expert-lr 1e-4
```

### MolmoAct2 normalization resolution

Same class of bug as π0.5's (see below): `training/model/molmoact2.py` hardcodes
`{"STATE": "MEAN_STD", "ACTION": "MEAN_STD"}`, but the real installed
`MolmoAct2Config()`'s own default is `QUANTILES`/`QUANTILES` (confirmed via
`dataclasses.fields`), and AllenAI's own real finetuning code
(`github.com/allenai/molmoact2`) defaults `--norm_mode q01_q99` -- both
independently agree. Unlike π0.5's checkpoint, `allenai/MolmoAct2`'s own
`norm_stats.json` publishes real numeric quantile stats (`q01`/`q99`/etc.,
per dataset "tag" -- e.g. `franka_droid` for DROID/Franka), so
`--normalization-stats-source checkpoint` is genuinely usable here, not
just the scheme:

```bash
# Read-only: print what the checkpoint's franka_droid tag declares vs. this run's config
python -m training.train --tasks droid_100_test --dataset-source droid_100 --policy-type molmoact2 \
    --molmoact2-setup-type "single franka robotic arm in droid" \
    --molmoact2-control-mode "absolute joint pose" \
    --molmoact2-norm-tag franka_droid --inspect-normalization
```

**Real, confirmed dimension mismatch**: `franka_droid`'s stats are 8-dim
(matches `--dataset-source droid`, the full `cadene/droid_1.0.1`, not yet
prepared here) -- **not** `--dataset-source droid_100` (7-dim, this
project's own small test dataset). `--normalization-stats-source checkpoint`
against droid100 correctly hard-fails on this dimension mismatch (verified
directly) rather than silently misapplying wrong-dim stats. The droid100
recipe instead uses `mode_source: checkpoint` (the real QUANTILES scheme) +
`stats_source: dataset` (computed from droid100's own 7-dim data) -- see
`training/configs/molmoact2_droid100_base.yaml`/`molmoact2_droid100_lora.yaml`.

**`--molmoact2-norm-tag` side effect, worth knowing**: setting it makes
`MolmoAct2Policy`'s own `_apply_norm_tag_metadata` silently overwrite
`chunk_size`/`n_action_steps` to the tag's own values at construction time
(e.g. `franka_droid` -> 15/15, not this pipeline's own default of 30) --
`validate_normalization` hard-fails if what you configured doesn't already
match, rather than letting the run silently diverge from what you set.

AllenAI's own real full-finetune recipe was also matched where it differs
from this pipeline's own defaults: `train_mode: fft` + `distributed_strategy:
fsdp2` (not this pipeline's own LoRA default), real per-group learning rates
(`vit_lr=5e-6, connector_lr=5e-6, action_expert_lr=5e-5`, llm group via the
flat `--lr 1e-5`), and real AdamW hyperparameters (`betas=(0.9, 0.95),
eps=1e-6, weight_decay=0` -- PyTorch's own defaults are `(0.9, 0.999)`/`1e-8`,
unchanged for every other policy). `training/configs/molmoact2_droid100_base.yaml`
bundles all of this; `molmoact2_droid100_lora.yaml` is the same fix under
this pipeline's own LoRA default, for anyone without FSDP2-capable
multi-GPU hardware.

Confirmed **not** an issue for MolmoAct2 (unlike π0.5): `float32_attention`
(AllenAI's own deliberate fp32-attention-math precision choice) and
`attn_implementation="sdpa"` are both already correct by default -- this
pipeline never overrides either, and the real checkpoint's own `config.json`
already sets them the same way AllenAI's own training code does.
`torch.compile` is deliberately not attempted for MolmoAct2 -- no existing
lerobot-side switch exists (unlike π0.5's), and AllenAI's own recommended
generic-finetuning recipe (`--dynamic_seq_len=true`) disables compile too,
using it only for their packed pretraining/reproduction recipes.

**Not yet done**: an actual training run. Every finding above is verified
against the real installed package + real checkpoint files + AllenAI's real
training code, and the normalization hook itself is verified end-to-end via
`--inspect-normalization` -- but MolmoAct2 has never actually been run in
this pipeline (needs real GPU memory beyond what was available in
development; ~20GB+ for LoRA, more for the recommended FSDP2 full-finetune
path). Run the smoke test above first, then the normalization-fix
comparison, before trusting either config file for a real training run.

## π0.5

`--policy-type pi05` -- a PaliGemma-based VLM (~3.2-3.3B params total:
gemma_2b VLM ≈2.0B + vocab embed ≈0.5B, gemma_300m action-expert ≈0.3B,
SigLIP so400m vision tower ≈0.4B). `--pi05-pretrained-path` is
**required** -- "finetuning" implies starting from real pretrained
weights, not random init, and this project never identified/verified a
specific real π0.5 checkpoint repo id for you (the standing rule here is
to never guess a repo id) -- find a real one on the HF Hub yourself first.

**Start with `training/configs/pi05_base_conf.yaml`** for any new run
(`--config-file training/configs/pi05_base_conf.yaml`) -- it bundles the
real, measured normalization-scheme fix and speed optimizations described
below, instead of you needing to already know this history and pass 6+
flags by hand.

| Flag | Default | Meaning |
|---|---|---|
| `--pi05-pretrained-path` | -- (required) | HF repo id or local path to a real pretrained π0.5 checkpoint |
| `--pi05-freeze-vision-encoder` | off | freeze the vision tower; language model + action expert still train. No LoRA/peft exists in this integration (`PI05Config.use_peft` is a dead field) -- this and the next flag are the only ways to reduce the trainable surface |
| `--pi05-train-expert-only` | off | only the action expert trains, everything else frozen -- combinable with `--pi05-freeze-vision-encoder` |
| `--pi05-no-gradient-checkpointing` | off (checkpointing on) | disable only with confirmed memory headroom -- auto-wires from `PI05Config`, no extra wiring needed unlike MolmoAct2 |
| `--pi05-empty-cameras` | `0` | pad `input_features` with dummy camera slots -- for when the pretrained checkpoint expects more cameras than this dataset has |
| `--pi05-distributed-strategy` | `ddp` | `ddp` or `fsdp2` (accelerate-driven FSDP2 sharding, same mechanism as MolmoAct2's) -- unlike MolmoAct2, usable with ANY `--pi05-freeze-vision-encoder`/`--pi05-train-expert-only` combination (PI05 has no LoRA to already solve memory, and DDP always fully replicates the whole model regardless of what's frozen, so FSDP2's sharding benefit applies every mode) |
| `--pi05-fsdp-cpu-offload` | off | trades speed for fitting on fewer/smaller GPUs -- `fsdp2` only |
| `--pi05-revision` | none (Hub latest) | pins BOTH weight loading and normalization-resolution config loading to a specific Hub revision/commit/tag |
| `--normalization-mode-source` | `explicit` | `explicit` (today's hardcoded default) or `checkpoint` (read STATE/ACTION mode from the pretrained checkpoint's own saved config) -- see "π0.5 normalization resolution" below |
| `--normalization-stats-source` | `dataset` | `dataset`, `checkpoint`, or `explicit_file` -- see below |
| `--pi05-compile-model` | off | **recommended, real measured ~2.4x per-step speedup** (0.46s vs 1.12s `compute_s` on real H100 hardware) -- see "π0.5 training speed" below |
| `--pi05-compile-mode` | `max-autotune` | only with `--pi05-compile-model` -- use `default` or `reduce-overhead` instead, see below (real, measured `max-autotune` warmup cost was severe) |
| `--pi05-vision-bf16` | off | **recommended, real measured ~1.9x per-step speedup** (0.60s vs 1.12s `compute_s`), stacks with `--pi05-compile-model` (combined: 0.35s, faster than native) -- monkey-patches lerobot internals, see below |
| `--pi05-narrow-checkpoint` | off | tested, real, but **no measured effect** (1.12s, unchanged) -- kept as a documented dead end, not worth using |
| `--pi05-print-attn-impl` | off | diagnostic, zero risk -- confirmed real training uses `sdpa`, not eager attention |

### π0.5 FSDP2

`transformer_cls_names_to_wrap = ["_PiGemmaDecoderLayerBase", "SiglipEncoderLayer"]`
-- confirmed via the real installed `lerobot==0.6.1` source: the PaliGemma
VLM's `language_model` layers AND the separately-instantiated
`gemma_expert.model`'s layers are both built by the same locally-scoped
factory (`lerobot/policies/pi05/pi_gemma.py`'s
`_get_pi_gemma_decoder_layer_base`) -- different Python class objects per
call, but `type(m).__name__` is identical for both, so ONE name covers
both sub-models (no MolmoAct2-style mutual-exclusivity handling needed).
`SiglipEncoderLayer` covers the vision tower separately (a real HF
`SiglipVisionModel`).

Unlike MolmoAct2's `wrap_for_training` (which hardcodes
`Accelerator(mixed_precision="bf16")`), PI05's does **not** pass
`mixed_precision` at all: PI05 already hand-casts a mixed bf16/fp32 scheme
onto individual params at construction time
(`PaliGemmaWithExpertModel.to_bfloat16_for_selected_params`, keeping
`vision_tower`/`multi_modal_projector`/layernorms in float32 for
stability, before `wrap_for_training` ever runs) -- passing
`mixed_precision="bf16"` too would blanket-recast everything back to
bf16 and undo that split. **UNVERIFIED** (no GPU-capable dev environment
available here): that these fp32-designated params actually survive
`accelerator.prepare()` still showing `dtype=torch.float32` in
`named_parameters()`, and that mixing fp32 layernorms with bf16
attention/MLP inside the same `_PiGemmaDecoderLayerBase`-wrapped FSDP unit
doesn't break FSDP2's flat-parameter sharding -- confirm both before
trusting a real training run (same "same environment" `save_state`/
`load_state` scoping and crash+resume-test discipline as MolmoAct2's FSDP2
section applies here too).

```bash
# Full fine-tune on one GPU
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type pi05 \
    --pi05-pretrained-path <your-real-checkpoint-repo-id> \
    --batch-size 8 --max-train-steps 50

# Freeze the vision encoder, only train the action expert -- reduces the
# trainable/optimizer-state footprint without needing FSDP2
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type pi05 \
    --pi05-pretrained-path <your-real-checkpoint-repo-id> \
    --pi05-freeze-vision-encoder --pi05-train-expert-only --batch-size 8

# Checkpoint expects more camera slots than this dataset provides
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type pi05 \
    --pi05-pretrained-path <your-real-checkpoint-repo-id> \
    --pi05-empty-cameras 2

# Full fine-tune with FSDP2 across multiple GPUs on one node
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type pi05 \
    --pi05-pretrained-path <your-real-checkpoint-repo-id> \
    --pi05-distributed-strategy fsdp2 --batch-size 32

# FSDP2 with CPU offload for VRAM headroom
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type pi05 \
    --pi05-pretrained-path <your-real-checkpoint-repo-id> \
    --pi05-distributed-strategy fsdp2 --pi05-fsdp-cpu-offload --batch-size 32
```

### π0.5 normalization resolution

Real, confirmed bug (found via a direct native-openpi-vs-Ray training comparison
on `lerobot/pi05_droid` + droid100): this pipeline hardcoded
`normalization_mapping={"STATE": "MEAN_STD", "ACTION": "MEAN_STD", ...}` in
`build_pi05_config`, but the real pretrained checkpoint's own saved config
declares `QUANTILES` for both (confirmed via `PI05Config.from_pretrained`
and a real HF Hub fetch) -- a genuine scheme mismatch that corrupts every
input/target during finetuning. A second, independent bug was found in the
same investigation: this pipeline's image-normalization default (`mean_std`)
doesn't produce the `[0,1]` range lerobot's own pi05 model unconditionally
expects (`img = img * 2.0 - 1.0` inside `modeling_pi05.py`) -- `unit01` is
the correct mode, and is now π0.5's automatic default (no flag needed).

`training/model/normalization.py` resolves STATE/ACTION mode and stats
*independently* rather than hardcoding a new fixed scheme in place of the
old one:
```bash
# Read the checkpoint's own declared mode+stats before training, no GPU needed
python -m training.train --tasks <tasks> --dataset-source <source> --policy-type pi05 \
    --pi05-pretrained-path lerobot/pi05_droid --inspect-normalization

# Use the checkpoint's mode (QUANTILES), computing stats from this run's own data
# (the checkpoint publishes NO numeric stats -- only the scheme -- confirmed via
# a real model.safetensors header inspection, zero normalization buffers present)
python -m training.train --tasks <tasks> --dataset-source <source> --policy-type pi05 \
    --pi05-pretrained-path lerobot/pi05_droid \
    --normalization-mode-source checkpoint --normalization-stats-source dataset
```
Real measured result on droid100 (1xH100, `--max-train-steps 1000`,
otherwise byte-identical hyperparameters): baseline (old hardcoded
`MEAN_STD` + `mean_std` image range) trained to a val loss of ~0.94;
with both fixes, individual training-step losses in the back half of the
run repeatedly land in the 0.08-0.35 range (best single value 0.08) --
much closer to native openpi's own reference (~0.037 on the same
checkpoint/dataset/hyperparameters) than the old baseline ever got.
Default behavior (no `--normalization-*` flags passed) is unchanged for
every policy -- this is strictly opt-in.

### π0.5 training speed (native vs Ray)

A genuine, real ~2.3-2.4x native-vs-Ray per-step slowdown was found and
root-caused during the same investigation (`perf/compute_s`: native
~0.48s, Ray ~1.12-1.16s, identical regardless of GPU count -- confirmed via
a real 1-GPU test that this is NOT DDP/NCCL communication overhead).
Four candidate causes were checked directly against real code and real
hardware, not guessed:

| Cause | Verdict | Real `compute_s` |
|---|---|---|
| DDP/NCCL gradient sync | **ruled out** -- identical at 1 GPU (no DDP at all) | n/a |
| Eager attention (`--pi05-print-attn-impl`) | **ruled out for training** -- lerobot only forces `_attn_implementation="eager"` inside inference-only `select_action`/`denoise_step`; training's own forward path resolves to `sdpa` | n/a |
| `torch.compile` disabled (`--pi05-compile-model`) | **confirmed, the dominant cause** -- `PI05Config.compile_model` defaults `False` and was never wired before | 1.12s -> **0.46s** |
| Vision runs fp32 not bf16 (`--pi05-vision-bf16`) | **confirmed, secondary contributor** -- native's own SigLIP tower runs bf16; lerobot's deliberately keeps it fp32 | 1.12s -> **0.60s** |
| Coarse gradient-checkpoint boundary (`--pi05-narrow-checkpoint`) | tested, **no measured effect** -- the checkpointed `action_out_proj` Linear layer was never the bottleneck | 1.12s -> 1.12s (unchanged) |

**Combined, `--pi05-compile-model --pi05-compile-mode default --pi05-vision-bf16`
measured `compute_s` of 0.35s -- faster than native's 0.48s.** Recommended
for any real π0.5 run once you've confirmed the caveats below on your own
setup:
```bash
python -m training.train --tasks <tasks> --dataset-source <source> --policy-type pi05 \
    --pi05-pretrained-path lerobot/pi05_droid \
    --pi05-compile-model --pi05-compile-mode default --pi05-vision-bf16
```

**`--pi05-compile-mode`: do not use `max-autotune` (the field's own
default) without deliberately choosing it.** A real test hit a severe
warmup cost -- TorchInductor benchmarks many Triton kernel configs per
distinct tensor shape the model executes (each search taking 4-17+
seconds, real "OutOfMemoryError...Ignoring this choice" lines are normal,
benign per-candidate rejections, not failures), and with this model's many
shapes, warmup can dominate or exceed a short run entirely. Use `default`
or `reduce-overhead` unless you've confirmed `max-autotune`'s longer
warmup is worth it for your own run length. Also unverified: whether the
real dataset's variable prompt/tokenized-length shapes cause repeated
recompilation deep into a long training run rather than staying warm
after the first occurrence of each shape.

**`--pi05-vision-bf16` caveats**: real monkey-patch on the constructed
model (casts `vision_tower`/`multi_modal_projector` to bf16, overriding
lerobot's own choice to keep them fp32 "so we never toggle" -- see
`training/model/pi05.py`'s `_apply_vision_bf16_override` for exactly what
it touches and why the ordering relative to optimizer construction
matters). Validated so far only as a short-run speed/memory measurement
(confirmed real ~1.2-2.4GB VRAM reduction alongside the speedup) -- NOT
yet confirmed stable (no NaN/divergence) over a long real training run.
Confirm that before trusting it beyond a speed probe.

## Overriding the dataset source / v3 root

```bash
# A hand-built v3 root with no conversion_params.json needs an explicit
# --dataset-source so camera_keys/state_dim/action_dim/tick_fps resolve
python -m training.train --tasks my_tasks --v3-root /data/hand_built_v3 --dataset-source abc130k --policy-type act
```

## Checkpoint retention

`--checkpoint-max-to-keep` keeps the N best-scoring checkpoints plus the
single most recent one (native Ray Train `CheckpointConfig` behavior,
`checkpoint_score_attribute="loss"`). **What "loss" means for that scoring
depends on whether val is active** (it is by default): with val active,
only the periodic VAL pass (`val/*`) attaches checkpoints -- the windowed/
epoch-end training-loss reports become pure logging, so a good-looking
training loss that masks real overfitting no longer wins retention. With
val disabled (`--val-split-fraction 0`), scoring falls back to training
loss from the windowed/epoch-end reports, exactly as before this feature
existed. Early stopping (`--early-stop-patience`) always stays
training-loss-based, regardless of val.

The one-time test pass (`--test-split-fraction`/`--test-v3-root`, opt-in)
never attaches a checkpoint at all -- it's recorded once, into `test/*`
and the final `history.jsonl` entry, purely for a final, uncontaminated
read on generalization.

```bash
# Val on by default -- checkpoints are scored by held-out val loss
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type act

# Disable val -- back to training-loss-scored checkpoints (pre-val-split behavior)
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type act --val-split-fraction 0

# Val from a separate, already-prepared v3 root instead of a slice of this one
# (--val-split-fraction defaults to 0.1 -- disable it explicitly, since the
# two are mutually exclusive)
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type act \
    --val-split-fraction 0 --val-v3-root /data/curated_val_v3

# One-time test pass at the end, in addition to the default val split
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type act \
    --test-split-fraction 0.1

# Keep only checkpoints where the (val, by default) loss improved (fewer
# writes, but a resume can lose progress back to the last improvement --
# see the flag table above)
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type act --save-only-on-improvement

# Keep more/fewer checkpoints on disk
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type act --checkpoint-max-to-keep 10
```

### Async checkpoint writes (`--async-checkpoint`, experimental)

Plain-DDP checkpoints (no effect under `--molmoact2-distributed-strategy
fsdp2`/`--pi05-distributed-strategy fsdp2`, which already use accelerate's
own `save_state`/`load_state`) copy model/optimizer state to CPU
synchronously (fast), then write it to disk via `pickle.dump` in a
background thread while training continues on the GPU -- instead of
stalling the training loop for the full write duration.
`torch.distributed.checkpoint.async_save` was tried first and rejected: it
requires a CPU-capable process-group backend (`cpu:gloo,cuda:nccl`) for
its own internal coordination -- real coordination that matters for
genuinely sharded/distributed checkpoints (FSDP2), but pure overhead for
an unsharded plain-DDP checkpoint that only rank 0 ever saves -- and Ray
Train's NCCL-only process group doesn't provide it, confirmed by a real
`AssertionError: A CPU backend must be enabled for async save` on an
actual run. A plain background thread sidesteps the question entirely.
On-disk format is identical to the non-async path (`state.pkl`), so
resuming a run that switches `--async-checkpoint` on/off between
checkpoints just works, no format detection needed. Real tradeoff: a
checkpoint is only reported
to Ray's own tracking (eligible for `checkpoint_score_attribute` scoring
or resume) one checkpoint-cycle late, once its write is confirmed
finished -- see `train_loop.py`'s `_AsyncCheckpointer`. This also means
Ray's own `result.metrics`/`history.jsonl` entries lag by one cycle
whenever a checkpoint just finished writing (TensorBoard/W&B are
unaffected -- they already have the real-time numbers). Carries real,
not-fully-verified risk -- test with a real crash+resume before trusting
it for a long run.

```bash
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type act --async-checkpoint
```

## After training

```bash
python -m training.history --storage-root <storage-root>
tensorboard --logdir <storage-root>/<run-name>/tensorboard
```

## See also

- Root [`README.md`](../README.md) -- install, quick start, repo layout
- [`training/data_prep/README.md`](data_prep/README.md) -- preparing a dataset before training
- [`docs/extending.md`](../docs/extending.md) -- adding a new policy, dataset, or robot
