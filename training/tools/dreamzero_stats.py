"""Writes DreamZero's own published DROID q01/q99 stats (GEAR-Dreams/DreamZero-DROID-Data) in our
explicit-stats format, for fine-tuning DreamZero-DROID on the `droid` dataset_source with the exact
normalization it was trained with:

    python -m training.tools.dreamzero_stats --out dreamzero_droid_stats.json
    python -m training.train ... --policy-type dreamzero \
        --normalization-stats-source explicit_file --normalization-explicit-stats-file dreamzero_droid_stats.json

Layout matches our droid schema: state = joint_position(7) + gripper_position(1); action = the same
dims, joints relative to block-start state (relative_stats_dreamzero.json), gripper absolute.
"""
from __future__ import annotations

import argparse
import json

REPO = "GEAR-Dreams/DreamZero-DROID-Data"
REVISION = "2abc197ca7f14f53a6bf464bf80018ce998f18cc"


def main() -> None:
    from huggingface_hub import hf_hub_download

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--revision", default=REVISION)
    a = p.parse_args()

    def load(name: str) -> dict:
        with open(hf_hub_download(REPO, f"meta/{name}", repo_type="dataset", revision=a.revision)) as f:
            return json.load(f)

    modality, stats, rel = load("modality.json"), load("stats.json"), load("relative_stats_dreamzero.json")

    def span(group: str, key: str) -> slice:
        return slice(modality[group][key]["start"], modality[group][key]["end"])

    state, action = stats["observation.state"], stats["action"]
    joints, grip_s, grip_a = span("state", "joint_position"), span("state", "gripper_position"), span("action", "gripper_position")
    out = {
        "observation.state": {q: state[q][joints] + state[q][grip_s] for q in ("q01", "q99")},
        "action": {q: rel["joint_position"][q] + action[q][grip_a] for q in ("q01", "q99")},
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out}: state {len(out['observation.state']['q01'])} dims, action {len(out['action']['q01'])} dims")


if __name__ == "__main__":
    main()
