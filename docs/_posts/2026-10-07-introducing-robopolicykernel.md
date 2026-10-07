---
layout: post
title: "Introducing RoboPolicyKernel"
date: 2026-10-07
categories: [announcement]
---

Robot learning papers tend to ship one policy, one dataset, and a pile of
glue code wired specifically for that combination. Swap in a new policy
architecture, or a new robot's dataset, and most of the pipeline gets
rewritten. **RoboPolicyKernel** is an attempt to avoid that: a Ray Data +
Ray Train pipeline for finetuning robot-learning policies on robot teleop
data, built from day one to support more than one policy and more than one
dataset without becoming a rewrite every time one of those changes.

## What it does today

Two independent steps, and that separation is deliberate:

- **`prepare_data.py`** discovers, downloads, and converts a raw dataset
  into [LeRobot v3](https://huggingface.co/docs/lerobot) format. This is the
  *only* place the pipeline talks to a raw data source.
- **`train.py`** reads an already-prepared v3 root and trains a policy on
  it. It never discovers, downloads, or converts anything itself.

```bash
# Prepare abc130k, then train ACT on it
python -m training.prepare_data --tasks arrange_the_flowers --dataset-source abc130k --max-episodes-per-task 20
python -m training.train --tasks arrange_the_flowers --policy-type act --max-train-steps 50
```

Three policies are supported out of the box -- **ACT**, **MolmoAct2**, and
**π0.5** -- across three dataset sources with real, different raw formats
(MCAP+protobuf, an HDF5+tar format, and an official HF LeRobot mirror).
Both `--dataset-source` and `--policy-type` are required on every run, no
defaults quietly picked for you.

## Why multi-policy by design, not multi-policy by accident

The architectural bet is that `train_loop.py` -- data iteration, DDP/FSDP
wrapping, checkpoint/resume, early stopping, TensorBoard -- is close to
identical across every policy. What actually differs per policy is model
construction, the forward/loss shape, and sometimes the distributed-wrapping
strategy. So that's exactly the surface a new policy has to implement, via
a small adapter contract (`build_policy_and_processor`, `forward_loss`, and
a couple of optional hooks for anything beyond plain DDP) -- everything else
stays shared, single-source-of-truth code.

The same bet holds on the data side: `train.py` only ever reads a dataset's
generic schema fields (camera keys, state/action layout, tick rate), so it
needs zero code changes per dataset. Only the raw-format decode step differs,
and even that's shared across every dataset using the same raw format.

## Built on real mistakes, not just a clean diagram

The adapter pattern above wasn't the first draft -- it's what was left after
finding and fixing real integration bugs: a resume bug where a step-windowed
checkpoint looked like a finished epoch on reload, a silently-ignored
per-component learning rate flag, an FSDP2 wrapping bug tied to a model
having two mutually-exclusive internal layer implementations. Robot policy
finetuning involves genuinely large models and real distributed training
setups, which makes "looks right, never actually run it" an easy trap --
this project leans hard on verifying claims against the actually-installed
library version and real execution rather than documentation or memory, and
keeps a running, detailed account of what that discipline has caught so far.

## Try it

```bash
pip install -r training/requirements.txt
export HF_TOKEN=$(cat hf_tok.txt)   # after accepting a gated dataset's terms on huggingface.co
```

The [README](https://github.com/ratharva/RoboPolicyKernel#readme) has more
quick-start examples (including finetuning π0.5 from a pretrained
checkpoint), and [`docs/extending.md`](https://github.com/ratharva/RoboPolicyKernel/blob/main/docs/extending.md)
is a condensed checklist for adding a new dataset, robot, or policy of your
own.
