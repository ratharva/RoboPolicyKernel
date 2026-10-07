# Sample `--config-file` configs

Example YAML files for `training.train`'s `--config-file` flag, parsed via
[draccus](https://github.com/dlwh/draccus) (the same library lerobot's own
`ACTConfig`/`PI05Config`/`MolmoAct2Config` are built on) into a base
`RunConfig` (`training/config.py`). Every named CLI flag still works
exactly as it always has and takes precedence over anything set here --
`--config-file` only fills in values nothing else was explicitly passed
for. `--tasks` and `--policy-type` are still required on the command line
even when a file sets them too (see each file's own header comment).

| File | Shows |
|---|---|
| `act_example.yaml` | The basic shape: `train:`/`model:` sections, the `model.type` choice-registry discriminator |
| `molmoact2_example.yaml` | MolmoAct2, LoRA (fits on one GPU) -- the required `setup_type`/`control_mode` prompts |
| `molmoact2_fft_example.yaml` | MolmoAct2, full fine-tune under FSDP2 -- per-group learning rates |
| `pi05_example.yaml` | pi05 -- the required `pretrained_path` |
| `pi05_base_conf.yaml` | **Start here for any new pi05 run** -- bundles the real, measured normalization-scheme fix and speed optimizations (`compile_model`/`vision_bf16`) found during this project's native-vs-Ray investigation. See `training/README.md`'s "π0.5 normalization resolution"/"π0.5 training speed" sections for the numbers behind it |
| `agibot_action_space_example.yaml` | The `data:` section -- action-space selection, delta actions, per-camera image normalization (including a depth camera) |
| `wandb_eval_example.yaml` | W&B logging, metric allow/deny-listing, episode-preview GIFs, and the val/test split |

```bash
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k \
    --policy-type act --config-file training/configs/act_example.yaml

# A named flag still overrides whatever the file sets
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k \
    --policy-type act --config-file training/configs/act_example.yaml --batch-size 32
```

Treat these as starting points, not exhaustive references -- every field
on `RunConfig`/`TrainConfig`/`DataConfig`/the three `*ConfigOverrides`
dataclasses (`training/config.py`, `training/common/config.py`) is a valid
YAML key; a file only needs to set the ones it wants to change from
their dataclass defaults.
