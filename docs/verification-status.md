# Verification status

This project's discipline: verify claims against the actually-installed
package/real data, don't trust memory or docs alone. This page collects
what's been confirmed end-to-end vs. what's still open, across the whole
pipeline. Dataset-specific and policy-specific status lives in
[Customizing datasets](customizing-datasets.md) and
[Customizing policies](customizing-policies.md) -- this page is the
cross-cutting summary plus the shared infrastructure (checkpointing, resume,
storage backends).

## Verified end-to-end at least once, not just reasoned about

- The MCAP schema (topics, protobuf field names, message rates) was
  confirmed live against a real episode with a one-off probe script (not
  part of this repo).
- lerobot's `ACTConfig`/`ACTPolicy`/`make_act_pre_post_processors` API surface
  (field names, `forward()`'s `(loss, loss_dict)` return, the separate
  normalization pipeline) was confirmed by direct inspection of the installed
  package and a real forward+backward pass with dummy tensors.
- The LeRobot v3 writer/reader round-trip (`lerobot_v3_writer.py` ->
  `training/vendor/lerobot_datasource.py`) was verified with synthetic
  episodes: correct row counts, task labels, action chunking/padding, exact
  state/action value round-trip, and a real libx264 encode/decode round-trip.
- A single-worker (`world_size=1`) run surfaced `AttributeError: 'ACTPolicy'
  object has no attribute 'module'` at checkpoint time -- `prepare_model()`
  only wraps in DDP when there's more than one worker. Fixed in
  `train_loop.py` (unwraps defensively rather than assuming DDP).
- After adding MolmoAct2 as a second policy, `train_loop.py`'s optimizer
  construction was switched from a hand-rolled backbone-name-match to
  `policy.get_optim_params()` (a real lerobot API both `ACTPolicy` and
  `MolmoAct2Policy` implement) -- re-ran the ACT smoke test end-to-end to
  confirm identical behavior: same checkpoint-retention pattern (best 3 +
  latest), correct metrics, no regression. Separately verified with a real
  `ACTPolicy` instance that `--lr-backbone` now actually reaches the
  optimizer's backbone param group (a real pre-existing gap this refactor
  fixed), which it silently didn't before.
- **Resume-from-checkpoint, across a real crash and worker relaunch, for
  all three configurations** (ACT, MolmoAct2/LoRA/DDP, MolmoAct2/full-
  finetune/FSDP2): each run deliberately crashed mid-run
  (`FailureConfig(max_failures=1)` + an injected exception after a
  checkpoint had already been saved) and confirmed Ray Train relaunched a
  fresh worker process that resumed correctly to the original target step
  count. This caught and fixed a real bug affecting both policies equally
  (see "MolmoAct2 > FSDP2 checkpointing" in
  [Customizing policies](customizing-policies.md)) -- resuming from any
  step-windowed checkpoint under `num_epochs=1` used to always silently
  discard all progress back to epoch start, for ACT too, not just
  MolmoAct2; nothing before this had ever exercised a real crash+resume to
  surface it.
- MolmoAct2's `accelerate`-based FSDP2 wrapping needed one additional fix
  found the same way: `Accelerator(mixed_precision="bf16")` casts model
  parameters to bf16, but batches stay whatever dtype the collate function
  produces (float32) -- without `accelerator.autocast()` wrapped around the
  forward call, this raised `mat1 and mat2 must have the same dtype` on the
  very first real step. Fixed in `train_loop.py`, scoped to the FSDP2 path
  only.
- π0.5 added as a third policy, applying the MolmoAct2-correction lesson
  from the start (verify against the actually-installed package, not
  research alone): real `PI05Config(...)` construction calls for all three
  training modes, then a fake-policy DDP crash+resume integration test --
  passed cleanly on the first attempt, and a real ACT regression run
  confirmed the shared-code changes didn't affect ACT.
- The data-prep/training package split (`training/common/` +
  `training/data_prep/` + the schema-driven `RobotSchema`) was re-verified
  with a real end-to-end regression after the refactor: a fresh real
  conversion (discover -> stream -> MCAP decode -> convert) followed by a
  real training run, both through the new schema/registry-driven code path
  -- zero regressions from the pre-refactor behavior.
- A full smoke-test pass after that refactor re-verified everything it could
  touch without real network/disk cost: `training.data_prep.discover`,
  `training.data_prep.verify_decode` (real decode + a visual check of the
  dumped frames -- sharp, no artifacts, task progress visible across
  samples), `training.history` (including correctly displaying a real
  historical failure record), and fresh fake-policy DDP smoke tests for
  both MolmoAct2 and π0.5 against the new package layout (both passed,
  confirming the `training.common.config`/`training.data_prep.*` import
  changes didn't break either policy). Also caught and fixed a real,
  previously-undiscovered dependency bug this way -- see "droid" in
  [Customizing datasets](customizing-datasets.md) for the `av`/`jsonlines`/
  `datasets` conflict found and fixed by actually importing the
  `hf_lerobot_mirror` strategy's code path instead of just reading it.
- **DROID**: with that dependency conflict fixed, a real full run against
  `lerobot/droid_100` (2GB) completed end to end -- real download, real
  `convert_dataset_v21_to_v30.py` migration, real read-back through
  `training/vendor/lerobot_datasource.py`. Caught and fixed a second real
  bug this way: `training/data/ray_dataset.py`'s camera-key rename
  double-prefixed columns from a dataset not built by our own writer (see
  "droid" in [Customizing datasets](customizing-datasets.md)).
- **AgiBot Alpha**: gated access was obtained, and real data was
  downloaded and inspected for the first time -- confirming the real HDF5
  structure, tick rate (30fps, two independent confirmations), 8 real
  camera keys, and that no floor-align step is needed (video frame count
  and proprio row count matched exactly, 1136 = 1136, for the same real
  episode). The core decode transform
  (`training/data_prep/strategies/agibot_hdf5.py`) was implemented for
  real and verified against that real episode: correct state/action array
  shapes, correct video decode, and a full round-trip through
  `write_episode`/`finalize_dataset`/`LeRobotDatasource` read-back. See
  "agibot_alpha" in [Customizing datasets](customizing-datasets.md) for
  exactly what's now confirmed vs. the one remaining real scope boundary
  (tar-shard discovery for converting many episodes efficiently).

## Still open / worth knowing about

1. **Video decode during conversion** (`training/data_prep/video_decode.py`)
   -- MCAP delivers camera streams as raw H.264/H.265 elementary bytes, not
   a container file, so decode goes through `PyAV.CodecContext.parse/decode`
   and frame timestamps are assigned by FIFO-pairing decode order with
   message order. This is wrong if the stream uses B-frames. **Run
   `python -m training.data_prep.verify_decode --task <substring>` and look
   at the dumped PNGs** before converting anything for real -- they should
   be sharp and change smoothly across sampled frame indices. See
   [2. Verify decode](02-verify-data.md).

2. **Single "top" camera simplification** -- abc130k's schema deliberately
   keeps ACT's input shape fixed at 3 cameras (`top`, `left_wrist`,
   `right_wrist`) regardless of whether a station is mono or stereo, using
   only the left view when both `/top-left-camera` and `/top-right-camera`
   exist. Revisit if you specifically want stereo input.

3. **Single-node storage assumption, except for `--v3-root`** -- `--v3-root`
   can point at `s3://`/`gs://` (see
   [3. Prepare data](03-prepare-data.md#where-the-converted-lerobot-v3-dataset-lives-local-s3-or-gcs)),
   which sidesteps this on a multi-node cluster. The `--mode download` HF
   cache and `--storage-root` (Ray Train checkpoints) still write to local
   disk read back by absolute path -- fine on a single-node setup or with
   shared storage; on a real multi-node cluster without a local
   `--v3-root`, those two still need to live on shared storage.

4. **`--policy-type molmoact2` -- the real ~8B model has never actually run
   end-to-end**, no GPU-capable environment was available in development
   (12GB GPU; real-world reports put LoRA training at ~20GB and even bf16
   inference alone at ~12.1GB). See
   [Customizing policies](customizing-policies.md#molmoact2) for exactly
   what HAS been verified at the config/integration-test level.

5. **`--policy-type pi05` (~3.2-3.3B params, corrected from an earlier
   "~2.3B" estimate) -- plain DDP is now real, end-to-end verified** against
   the actual `lerobot/pi05_droid` checkpoint on real 1xH100/2xH100
   hardware, including: a genuine STATE/ACTION normalization-scheme
   mismatch found and fixed (checkpoint declares `QUANTILES`, this pipeline
   hardcoded `MEAN_STD`) via `--normalization-mode-source checkpoint
   --normalization-stats-source dataset`, a genuine image-normalization
   default fix (`unit01` instead of `mean_std`, now automatic for π0.5),
   and a full native-vs-Ray speed investigation (see `training/README.md`'s
   "π0.5 training speed" section) that found and validated real fixes
   (`--pi05-compile-model` + `--pi05-vision-bf16` together roughly match or
   beat native's per-step compute time). `--pi05-distributed-strategy
   fsdp2` remains **unverified** -- none of this round's real runs used it
   (all were plain DDP); it still has NOT been exercised live, not even at
   the fake-policy level. See
   [Customizing policies](customizing-policies.md#π05) for what HAS been
   verified and exactly what remains open for the FSDP2 path specifically.

6. **`--dataset-source droid` -- real-tested against `lerobot/droid_100`
   (2GB), not against `cadene/droid_1.0.1`** (18.4GB, the dataset the
   schema's `full_repo_id` actually names) -- that's real network/disk cost
   this session didn't spend on the larger one; the smoke test proves the
   mechanism, not that the actual target dataset has been converted. And
   **`--dataset-source agibot_alpha`'s per-episode transform is real and
   verified, but there's no `discover.py`-style listing yet that maps a
   task's requested episodes to their real tar shard URLs**, and converting
   many episodes from the same un-indexed tar one-by-one would each re-pay
   the streaming-scan cost for everything before their target -- a real,
   flagged inefficiency to fix before converting at scale, not a
   correctness gap. See [Customizing datasets](customizing-datasets.md) for
   the full status of each.

7. **`s3://`/`gs://` for `--source-uri`/`--v3-root`** -- implemented the
   same way the already-proven `hf://`/local paths are, but unverified
   against a real bucket: `s3fs`/`gcsfs` aren't installed by default, so
   there was no way to test beyond confirming the local/`hf://` paths still
   work identically after the change that added them.
