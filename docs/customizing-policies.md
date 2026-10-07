# Customizing policies

`--policy-type {act, molmoact2, pi05}` on `train.py` selects the policy.
All three are bundled in the same `lerobot==0.6.1` install -- see
[1. Setup](01-setup.md). Everything policy-specific is dispatched through
`training/model/registry.py`; adding one didn't touch the previous ones'
code paths or defaults.

## ACT (default)

Tens of millions of params, trains from random init, DDP. This repo's
original target and the one to get a smoke run working on before trying
either VLA below.

## MolmoAct2

`--policy-type molmoact2` trains lerobot's `MolmoAct2Policy` -- a
~8B-parameter VLA (Qwen2-7B-scale LLM + SigLIP ViT + a flow-matching action
expert), a much bigger jump from ACT than the flag name suggests. Same
environment as ACT, no separate install.

**Verification status, precisely**: `MolmoAct2Config`/`MolmoAct2Policy`'s
field names, `get_optim_params()`'s real 4-group structure,
`forward()`'s signature, the real `train_mode`-to-boolean-flags mapping
(`enable_lora_vlm`/`enable_lora_action_expert`/`train_action_expert_only`),
and the gradient-checkpointing method name were all confirmed by direct
introspection AND real construction calls against the installed package --
not just import success, actual `MolmoAct2Config(...)` calls with real
validation passing for all three `train_mode` values. This caught several
concrete mismatches from earlier fork-based research (no such field as
`train_mode_vlm`; the real gradient-checkpointing method is
`_enable_gradient_checkpointing`, not `gradient_checkpointing_enable`;
`lora_rank`/`lora_alpha`'s real defaults are 64/16, not 16/32) -- all now
fixed. What's still NOT verified: actually constructing a real
`MolmoAct2Policy` (needs downloading the ~22GB checkpoint) and running a
real forward/backward pass. Re-verify against a live install if you bump
the lerobot version:

```bash
python -c "from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config; \
    import dataclasses; [print(f.name) for f in dataclasses.fields(MolmoAct2Config)]"
```

### LoRA (default) vs full fine-tune

`--molmoact2-train-mode lora` (default) enables LoRA on the VLM only (the
action expert stays fully trainable -- matches the MolmoAct2 team's own
training recipe) and fits under plain DDP (`ray.train.torch.prepare_model`,
same mechanism ACT uses) on one modern GPU -- ~20GB VRAM at batch 8 by
real-world reports. This is the starting point; get it working before
considering `fft`. `--molmoact2-train-mode freeze` is a third option: VLM
completely frozen, only the action expert trains (requires
`--molmoact2-action-mode continuous`, enforced by `MolmoAct2Config`'s own
validation with a clear error otherwise).

`--molmoact2-train-mode fft` (full fine-tune) needs
`--molmoact2-distributed-strategy fsdp2` to be practical (~60GB/GPU at
batch 32 under plain DDP otherwise -- FSDP2 shards
parameters/gradients/optimizer state across GPUs instead). This does NOT
use Ray Train's own `prepare_model(parallel_strategy="fsdp")`, which wraps
the legacy FSDP1 API (`torch.distributed.fsdp.FullyShardedDataParallel`) --
verified from installed Ray source. Instead, `training/model/molmoact2.py`'s
`wrap_for_training()` constructs an `accelerate.Accelerator` with
`FullyShardedDataParallelPlugin(fsdp_version=2, ...)` inside the Ray Train
worker function itself, the same mechanism lerobot's own official
`accelerate launch -m lerobot.scripts.lerobot_train` training path uses.
This composes cleanly with Ray Train: verified by reading both
`accelerate.state.PartialState.__init__` (checks
`if not torch.distributed.is_initialized()` before calling
`init_process_group`, i.e. it reuses an existing process group rather than
fighting it) and Ray Train's own worker-group startup sequence
(`ReplicaGroup.start_training()` runs backend setup callbacks, including
`TorchBackend.on_start()`'s process-group init, synchronously via
`ray.get(...)` *before* dispatching the training function to any worker) --
so the process group Ray Train sets up is unconditionally ready before
`accelerate.Accelerator()` is ever constructed.

`auto_wrap_policy="transformer_based_wrap"` needs real transformer layer
class names to find anything to wrap. `wrap_for_training()` determines
these **dynamically from the actual constructed `policy`** rather than
hardcoding a fixed list -- a real, live-tested reason why: the LLM
backbone has two decoder-layer variants,
`MolmoAct2DecoderLayer`/`MolmoAct2PostNormDecoderLayer`
(`MolmoAct2PostNormDecoderLayer` subclasses the other), and exactly one is
chosen per checkpoint at construction time (`config.norm_after`) -- **never
both**. `accelerate`'s own class-matching (`get_module_class_from_name`) is
exact `type(module).__name__ ==` string comparison, not `isinstance`-based
(verified by reading its source), so listing both unconditionally raised a
real, reproducible `ValueError: Could not find the transformer layer class
... in the model` against a fake-policy integration test that only had one
of the two. `wrap_for_training()` now filters its candidate class list down
to whichever ones are actually present on `policy.modules()` before
building the FSDP plugin, so either checkpoint variant works without
needing to know which one in advance. The four real candidate names
(`MolmoAct2DecoderLayer`/`MolmoAct2PostNormDecoderLayer`/
`MolmoAct2VisionBlock`/`ActionExpertBlock`) were found by reading the
installed package's source directly, not guessed.

### FSDP2 checkpointing

FSDP2-sharded `policy.state_dict()` returns per-rank shards/DTensors, not a
normal full state dict, so the DDP path's pickle-one-rank-0-state-dict
approach doesn't work here. `training/model/molmoact2.py`'s
`save_checkpoint`/`load_checkpoint` use `accelerate.Accelerator.save_state`/
`load_state` instead. Their own docstring scopes them to *"restoring the
state in the same environment"* -- a Ray Train worker restart
(`ray.train.get_checkpoint()` after a failure) is a different process,
potentially a different node, not obviously "the same environment" in that
sense.

**Verified live**, not just reasoned about: a fake-policy integration test
(same interface real `MolmoAct2Policy` needs -- `forward`, `get_optim_params`,
`_enable_gradient_checkpointing` -- but a tiny model, since the real ~8B model
isn't installed here and wouldn't fit a 12GB dev GPU) exercised the real
`wrap_for_training`/`save_checkpoint`/`load_checkpoint` code through a
deliberately-simulated mid-run worker crash (`FailureConfig(max_failures=1)`
+ an injected exception). Ray Train relaunched a genuinely new worker
process, which successfully loaded the sharded FSDP2 checkpoint ("All model
weights loaded successfully", "All optimizer states loaded successfully")
and correctly resumed training to completion. This confirms the specific
single-node worker-restart scenario works; `save_state`'s broader "same
environment" scoping (different GPU topology, genuine multi-node) remains
unverified.

**A real, more serious bug was caught and fixed by this test**, unrelated to
FSDP2 itself and affecting ACT equally: `train_loop_per_worker`'s resume
logic computed `start_epoch = state["epoch"] + 1` unconditionally, which is
only correct when resuming from an end-of-epoch checkpoint. A step-windowed
(mid-epoch) checkpoint -- what this codebase took every `--eval-every-steps`
at the time (renamed `--window-every-steps` since; today's main
checkpointing cadence has moved to `--val-every-steps` whenever val is
active, the default) -- has `epoch` equal
to whatever epoch was still in progress, so `epoch + 1` always looked like
"the next epoch" and, combined with `num_epochs=1` (this codebase's own
documented typical setting), `start_epoch >= num_epochs` always
false-positived as "run already complete" -- silently discarding all
progress back to epoch start on **any** resume, for both ACT and MolmoAct2,
regardless of distributed strategy. Fixed by threading an `epoch_complete`
flag through the checkpoint state. Re-verified with the same crash+relaunch
test (ACT, MolmoAct2/DDP, and MolmoAct2/FSDP2 all now correctly resume to
the original target step count, not zero).

### Ray Data preprocessing offload

MolmoAct2's preprocessor (`make_molmoact2_pre_post_processors`) does real
CPU-bound work per batch -- its `MolmoAct2PackInputsProcessorStep` builds a
natural-language prompt per sample and runs it through a real HF tokenizer +
image processor. By default this runs inline in `train_loop.py`
(`inputs = preprocessor(batch)`), on the same thread as the GPU
forward/backward -- `--molmoact2-offload-tokenization` instead runs the
*entire* preprocessor as a Ray Data `map_batches` stage
(`training/data/ray_dataset.py`'s `offload_molmoact2_preprocessing`,
`concurrency` defaulting to half the live CPU count the same way
conversion's `max_concurrent` derives from live resources), so it scales
across the cluster's CPU workers independently of GPU worker count.

**Unverified**: whether HF tokenizer/processor output (token ID sequences,
attention masks, possibly padded to a batch-dependent max length)
round-trips correctly through Ray Data's Arrow-backed batch representation
and then into `training/vendor/util.py`'s `NumpyToTorchCollate` was not
confirmed. Construct a real batch, run it through
`offload_molmoact2_preprocessing`, and compare tensor shapes/dtypes against
the non-offloaded path before trusting this for a real run -- and confirm
throughput actually improves (the whole point of this mode) via the Ray
dashboard / `ray_data_stats.txt`.

### `setup_type` / `control_mode`

Required free-text prompt fields MolmoAct2's VLM conditions on --
`MolmoAct2PackInputsProcessorStep` raises `ValueError` if either is missing.
These describe the robot/task, not the dataset schema, so they live on
`MolmoAct2ConfigOverrides` (via `--molmoact2-setup-type`/
`--molmoact2-control-mode`), not on the dataset's `RobotSchema` -- see
[Customizing datasets](customizing-datasets.md): `RobotSchema`'s scope is
generic dataset facts independent of which policy trains on the data, and
forcing every dataset's schema to carry MolmoAct2-specific prompt fields it
can't populate would break that contract for one policy's benefit. For
abc130k, something like `--molmoact2-setup-type "dual-arm robot with wrist
and top cameras" --molmoact2-control-mode "delta joint position"` is a
reasonable starting point -- not verified against real training quality.

### Normalization

MolmoAct2's own default is quantile-based normalization for state/action
(`STATE`/`ACTION: QUANTILES`); `training/model/molmoact2.py` overrides
`STATE`/`ACTION` to `MEAN_STD` at construction so `training/data/stats.py`'s
existing `compute_dataset_stats` (mean/std only) works unchanged -- a
deliberate simplification, not a claim that it's optimal. `VISUAL` is set
to `IDENTITY`, MolmoAct2's own real default too -- `training/model/
image_normalization.py`'s `apply_image_normalization` owns 100% of image
scaling now (mean_std/unit01/unit_pm1/depth/log, selected per-camera via
`--image-normalization`), applied in `train_loop.py` before the policy's
own preprocessor runs, so lerobot's per-`FeatureType` normalizer never
touches `VISUAL` for any policy -- see `training/README.md`'s "Action space
& per-camera image normalization" section for the full `--image-normalization`
flag reference.

## π0.5

`--policy-type pi05` trains lerobot's `PI05Policy` -- a PaliGemma-based VLM
backbone (`gemma_2b`) + a separate, smaller flow-matching action expert
(`gemma_300m`), **~3.2-3.3B params total** (corrected from an earlier
"~2.3B" estimate here -- confirmed by reading the real installed
`lerobot==0.6.1` source's actual layer dims: gemma_2b VLM ≈2.0B + vocab
embed ≈0.5B, gemma_300m action-expert ≈0.3B, SigLIP so400m vision tower
≈0.4B), still well below MolmoAct2's ~8B. Same environment as
ACT/MolmoAct2, no separate install.

Unlike ACT (trains from random init in this pipeline) and MolmoAct2 (loads
its own weights internally via a `checkpoint_path` config field), π0.5 is
the first policy here that goes through lerobot's standard
`PreTrainedPolicy.from_pretrained(pretrained_name_or_path, config=...)`
classmethod -- "finetuning" implies starting from real pretrained weights,
not random init, so `--pi05-pretrained-path` is a **required** flag with no
default. No real public HF checkpoint repo id for a π0.5 base/finetunable
checkpoint was looked up or verified in this project (matches the standing
rule against guessing repo ids/URLs) -- find a real one on the HF Hub
yourself before running this for real.

**Verification status, precisely**: `PI05Config`/`PI05Policy`'s field names,
`get_optim_params()`'s real flat (ungrouped) structure, `forward()`'s
signature, `from_pretrained`'s signature, and the gradient-checkpointing
auto-wiring were all confirmed by direct introspection AND a real
`PI05Config(...)` construction call for all three real training modes (full
fine-tune, freeze-vision, expert-only) against the installed package --
applying the "verify against what's actually installed" lesson from the
MolmoAct2 correction above from the start, which is presumably why this
integration had zero real API mismatches on the first attempt, unlike
MolmoAct2's first pass. What's still NOT verified: an actual
`PI05Policy.from_pretrained(...)` call against a real checkpoint (none was
downloaded -- no verified real checkpoint repo id, see above) and a real
forward/backward pass. Re-verify against a live install if you bump the
lerobot version:

```bash
python -c "from lerobot.policies.pi05.configuration_pi05 import PI05Config; \
    import dataclasses; [print(f.name) for f in dataclasses.fields(PI05Config)]"
```

### No LoRA, and no `train_mode` string

`PI05Config.use_peft` exists but is never referenced in `modeling_pi05.py`
(confirmed by grep) -- the same dead-field pattern `MolmoAct2Config.use_peft`
had, so there's no LoRA support in this integration. The only two real,
used training-surface reducers are independent booleans:
`--pi05-freeze-vision-encoder` (freezes the vision tower; language model +
action expert still train) and `--pi05-train-expert-only` (only the action
expert trains, everything else frozen) -- both confirmed consumed in
`PI05Policy.__init__`.

Unlike `MolmoAct2ConfigOverrides.train_mode`, `Pi05ConfigOverrides` exposes
both booleans directly rather than translating a single convenience string
into real config fields. This is deliberate, not an oversight: MolmoAct2's
`train_mode` translation layer was where the earlier `train_mode_vlm`
mismatch lived, and with only two independent real booleans (no evidence in
`__post_init__` of any invalid combination), a hand-maintained translation
layer here would add sync risk without adding real value.

`PI05Policy` has no `.train()` override (confirmed: `'train' not in
PI05Policy.__dict__`) -- freezing is via `requires_grad` only. Unverified:
whether any train/eval-sensitive layers (e.g. dropout) exist in the frozen
submodules such that this matters.

### FSDP2 (optional) -- same accelerate-based pattern as MolmoAct2

`--pi05-distributed-strategy fsdp2` uses the same accelerate/FSDP2
mechanism as MolmoAct2's (`training/model/pi05.py`'s `wrap_for_training`),
with two real differences worth knowing:

- **Wrap-class names**: `transformer_cls_names_to_wrap = ["_PiGemmaDecoderLayerBase",
  "SiglipEncoderLayer"]`. Unlike MolmoAct2 (two mutually-exclusive decoder-
  layer variants depending on a config flag), PI05's PaliGemma VLM
  (`language_model`) and its separately-instantiated `gemma_expert.model`
  are BOTH built by the same locally-scoped factory function
  (`lerobot/policies/pi05/pi_gemma.py`'s `_get_pi_gemma_decoder_layer_base`,
  confirmed by reading the real installed source) -- different Python
  class objects per call, but `type(m).__name__` is identical for both, so
  one name covers both sub-models automatically. No mutual-exclusivity
  filtering needed the way MolmoAct2's does; the same defensive
  present-classes intersection pattern is still used regardless, in case
  a future lerobot version renames these.
- **No `mixed_precision` kwarg**: MolmoAct2's `wrap_for_training` hardcodes
  `Accelerator(mixed_precision="bf16")`. PI05's does NOT do this -- PI05
  already applies its own mixed bf16/fp32 per-param cast
  (`PaliGemmaWithExpertModel.to_bfloat16_for_selected_params`, keeping
  `vision_tower`/`multi_modal_projector`/layernorms in float32 for
  stability) at construction time, before `wrap_for_training` ever runs.
  Passing `mixed_precision="bf16"` too would blanket-recast everything
  back to bf16 and silently undo that split.

Unlike MolmoAct2 (where fsdp2 is restricted to `train_mode=="fft"` since
LoRA already solves memory and mixing LoRA-adapter-plus-frozen-base under
FSDP2 has known complications), PI05's fsdp2 is **not** restricted to any
particular `--pi05-freeze-vision-encoder`/`--pi05-train-expert-only`
combination -- PI05 has no LoRA, and DDP always fully replicates the whole
model regardless of what's frozen, so FSDP2's memory-sharding benefit
applies to every PI05 training mode.

**Not yet verified** (no GPU-capable dev environment available where this
was written): that the fp32-designated params actually survive
`accelerator.prepare()` still showing `dtype=torch.float32` in
`named_parameters()` (rather than FSDP2 silently unifying dtype per wrap
unit), that mixing fp32 layernorms with bf16 attention/MLP inside the same
`_PiGemmaDecoderLayerBase`-wrapped unit doesn't break FSDP2's flat-
parameter sharding, and the crash+resume behavior (same
`save_checkpoint`/`load_checkpoint` -- now factored into a shared
`training/model/fsdp2_checkpoint.py` module both MolmoAct2 and PI05
import -- as MolmoAct2's own verified-live FSDP2 checkpointing above,
same `accelerate.Accelerator.save_state`/`load_state` "same environment"
scoping caveat applies). Unlike MolmoAct2's FSDP2 path, this one has NOT
yet been exercised through the crash+relaunch test described above --
do that before trusting a real long run.

### Normalization

Same deliberate simplification as MolmoAct2: π0.5's own default `STATE`/
`ACTION: QUANTILES` is overridden to `MEAN_STD`, so `training/data/stats.py`'s
existing mean/std-only `compute_dataset_stats` works unchanged. `VISUAL`
stays `IDENTITY` -- π0.5's own real default, and required for every policy
now that `training/model/image_normalization.py` owns all image scaling
upstream of the policy's own preprocessor (see MolmoAct2's "Normalization"
section above).

## Policy-specific CLI flags

MolmoAct2-only flags (all ignored/unused when `--policy-type act`;
`--molmoact2-setup-type`/`--molmoact2-control-mode` are required when
`--policy-type molmoact2`):

| Flag | Default | Meaning |
|---|---|---|
| `--molmoact2-checkpoint-path` | `allenai/MolmoAct2` | HF repo id or local path the VLM backbone loads from |
| `--molmoact2-setup-type` | -- (required) | free-text embodiment prompt, e.g. `"dual-arm robot with wrist and top cameras"` |
| `--molmoact2-control-mode` | -- (required) | free-text control-mode prompt, e.g. `"delta joint position"` |
| `--molmoact2-action-mode` | `continuous` | `continuous`, `discrete`, or `both` -- real default is `both`; narrowed here to skip the discrete FAST-tokenizer dependency/setup |
| `--molmoact2-train-mode` | `lora` | `lora` (LoRA on the VLM only, action expert stays fully trainable), `fft` (full fine-tune, needs `--molmoact2-distributed-strategy fsdp2`), or `freeze` (VLM frozen, only the action expert trains -- requires `--molmoact2-action-mode continuous`) |
| `--molmoact2-lora-rank` / `-alpha` / `-dropout` | `64` / `16` / `0.05` | `MolmoAct2Config`'s own real defaults (yes, alpha < rank) |
| `--molmoact2-no-gradient-checkpointing` | off (checkpointing on) | disable only if you've confirmed the memory headroom |
| `--molmoact2-vit-lr` / `-connector-lr` / `-action-expert-lr` | `None` -> falls back to `--lr` | MolmoAct2's `get_optim_params()` returns 4 LR groups (vlm/vit/connector/action_expert), not ACT's 2 |
| `--molmoact2-distributed-strategy` | `ddp` | `ddp` (LoRA) or `fsdp2` (full fine-tune, only with `--molmoact2-train-mode fft`) |
| `--molmoact2-fsdp-cpu-offload` | off | trades speed for fitting on fewer/smaller GPUs -- `fsdp2` only |
| `--molmoact2-offload-tokenization` | off | run MolmoAct2's preprocessor as a Ray Data stage instead of inline in the training loop |
| `--molmoact2-offload-concurrency` | auto (from live CPU count) | Ray Data actor-pool size for the above |

π0.5-only flags (all ignored/unused when `--policy-type` is not `pi05`;
`--pi05-pretrained-path` is required when `--policy-type pi05`):

| Flag | Default | Meaning |
|---|---|---|
| `--pi05-pretrained-path` | -- (required) | HF repo id or local path to a real pretrained π0.5 checkpoint -- no safe default |
| `--pi05-freeze-vision-encoder` | off | freeze the vision tower; language model + action expert still train |
| `--pi05-train-expert-only` | off | only the action expert trains, everything else frozen |
| `--pi05-no-gradient-checkpointing` | off (checkpointing on) | disable only if you've confirmed the memory headroom |
| `--pi05-empty-cameras` | `0` | pad `input_features` with dummy camera slots -- for when the pretrained checkpoint expects more cameras than this dataset has |
| `--pi05-distributed-strategy` | `ddp` | `ddp` or `fsdp2` -- unlike MolmoAct2, usable with any freeze/expert-only combination (no LoRA to already solve memory) |
| `--pi05-fsdp-cpu-offload` | off | trades speed for fitting on fewer/smaller GPUs -- `fsdp2` only |

```bash
# MolmoAct2, LoRA on one GPU (run prepare_data.py for --tasks
# dress_the_teddy_bear first if you haven't)
python -m training.train --tasks dress_the_teddy_bear --policy-type molmoact2 \
    --molmoact2-setup-type "dual-arm robot with wrist and top cameras" \
    --molmoact2-control-mode "delta joint position" \
    --batch-size 8 --max-train-steps 50

# MolmoAct2, full fine-tune with FSDP2 across multiple GPUs on one node, plus
# the Ray Data preprocessing offload -- both carry real unverified risk, see above
python -m training.train --tasks dress_the_teddy_bear --policy-type molmoact2 \
    --molmoact2-setup-type "dual-arm robot with wrist and top cameras" \
    --molmoact2-control-mode "delta joint position" \
    --molmoact2-train-mode fft --molmoact2-distributed-strategy fsdp2 \
    --molmoact2-offload-tokenization --batch-size 32

# pi05, full fine-tune on one GPU -- --pi05-pretrained-path must point at a
# real checkpoint you found on the HF Hub yourself
python -m training.train --tasks dress_the_teddy_bear --policy-type pi05 \
    --pi05-pretrained-path <your-real-checkpoint-repo-id> \
    --batch-size 8 --max-train-steps 50

# pi05, freeze the vision encoder and only train the action expert -- reduces
# the trainable/optimizer-state footprint without needing FSDP2
python -m training.train --tasks dress_the_teddy_bear --policy-type pi05 \
    --pi05-pretrained-path <your-real-checkpoint-repo-id> \
    --pi05-train-expert-only --batch-size 8
```

## Adding a new policy

New `training/model/<policy>.py` implementing the same contract as
`act.py`/`molmoact2.py`/`pi05.py`
(`build_policy_and_processor(data_cfg, overrides, train_cfg, dataset_stats,
device) -> (policy, preprocessor)`, `forward_loss(policy, inputs) -> (loss,
metrics)`, optionally `post_build_hook`/`wrap_for_training`/
`save_checkpoint`/`load_checkpoint` if it needs more than plain DDP), a new
`<Policy>ConfigOverrides` dataclass in `training/config.py`, a lazy factory
registered in `training/model/registry.py`, and `--<policy>-*` CLI flags in
`train.py` following the prefix convention above. `train_loop.py` (data
iteration, DDP/FSDP wrap, checkpoint/resume, early stopping, TensorBoard)
needs no changes -- it's ~90% identical across every policy by design, only
model construction and the forward/loss shape actually differ.
