"""The "hf_lerobot_mirror" ingestion strategy: for a dataset that already
has a LeRobot-format mirror on the HF Hub (DROID today; OXE/RoboCasa/LIBERO/
GR00T variants are increasingly common too) -- download it and reuse
lerobot's own official conversion tooling instead of writing a decoder.

Does NOT go through convert_to_lerobot_v3/write_episode at all (unlike
mcap/agibot_hdf5) -- lerobot's own convert_dataset() already produces a real
v3 root; this module's job is just to call it correctly and then stamp our
own conversion_params.json into the result so train.py's metadata-driven
--dataset-source resolution works on it like any other prepared dataset.

Verified directly against the installed lerobot package (not assumed from
docs): convert_dataset_v21_to_v30.convert_dataset(repo_id, root, push_to_hub)
first checks whether repo_id already has a v3.0 tag on the hub (fast path,
no conversion), and validate_local_dataset_version() hard-fails for
anything that isn't exactly codebase_version "v2.1" -- see
training/data_prep/schemas/droid.yaml's comment for why full_repo_id points
at cadene/droid_1.0.1 (real v2.1) and not IPEC-COMMUNITY/droid_lerobot
(real v2.0, unsupported by this script).
"""
from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass

from training.common.robots import RobotSchema
from training.data_prep.lerobot_v3_writer import (
    CONVERSION_PARAMS_FILENAME, compute_conversion_params, wipe_dataset, write_conversion_params,
)


@dataclass
class HfLerobotMirrorIngestionConfig:
    full_repo_id: str
    source_format_version: str
    camera_key_map: dict[str, str]
    read_path_smoke_test_repo_id: str | None = None
    revision: str | None = None


def build_ingestion_config(spec: dict) -> HfLerobotMirrorIngestionConfig:
    m = spec["hf_lerobot_mirror"]
    return HfLerobotMirrorIngestionConfig(
        full_repo_id=m["full_repo_id"],
        source_format_version=m["source_format_version"],
        camera_key_map=dict(m["camera_key_map"]),
        read_path_smoke_test_repo_id=m.get("read_path_smoke_test_repo_id"),
        revision=spec.get("revision"),
    )


def _read_jsonl_prefix(fs, path: str, num_episodes: int) -> list[str]:
    """Raw lines for episode_index < num_episodes, streamed (episodes_stats.jsonl is ~1GB for DROID)."""
    from tqdm import tqdm

    lines: list[str] = []
    with fs.open(path, "r") as f, tqdm(total=num_episodes, desc=f"trim {os.path.basename(path)}", unit="ep") as bar:
        for line in f:
            if not line.strip():
                continue
            if json.loads(line)["episode_index"] < num_episodes:
                lines.append(line if line.endswith("\n") else line + "\n")
                bar.update(1)
                if len(lines) == num_episodes:
                    break
    if len(lines) != num_episodes:
        raise ValueError(f"{path}: found {len(lines)} of the first {num_episodes} episodes")
    return lines


def _renumber_index(root: str, info: dict, num_episodes: int) -> None:
    """cadene/droid_1.0.1's `index` column restarts at 0 at episode 47 (and likely elsewhere), but
    v3 meta ranges are a running total and Ray's read_lerobot selects rows by `index`."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm import tqdm

    offset = 0
    for ep in tqdm(range(num_episodes), desc="renumber index", unit="ep"):
        path = os.path.join(root, info["data_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep))
        table = pq.read_table(path)
        col = table.schema.get_field_index("index")
        idx = pa.array(np.arange(offset, offset + table.num_rows), type=table.schema.field(col).type)
        if not table.column(col).equals(pa.chunked_array([idx])):
            pq.write_table(table.set_column(col, "index", idx), path)
        offset += table.num_rows


def _download_v21(repo_id: str, revision: str | None, root: str, max_episodes: int) -> int:
    """Downloads a v2.1 mirror at the pinned revision into root -- only its first max_episodes
    episodes, with meta trimmed to match. Returns the episode count downloaded."""
    from huggingface_hub import HfFileSystem, snapshot_download

    fs = HfFileSystem()
    base = f"datasets/{repo_id}" + (f"@{revision}" if revision else "")
    with fs.open(f"{base}/meta/info.json", "r") as f:
        info = json.load(f)
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"{repo_id}: expected codebase_version v2.1, got {info.get('codebase_version')!r}")

    total = info["total_episodes"]
    meta = os.path.join(root, "meta")
    if max_episodes >= total:
        print(f"[1/3] downloading all {total} episodes of {repo_id} (no subsetting)")
        snapshot_download(repo_id, repo_type="dataset", revision=revision, local_dir=root)
        n = total
    else:
        n = max_episodes
        chunks_size = info["chunks_size"]
        video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
        files = ["meta/tasks.jsonl"]
        for ep in range(n):
            chunk = ep // chunks_size
            files.append(info["data_path"].format(episode_chunk=chunk, episode_index=ep))
            files += [info["video_path"].format(episode_chunk=chunk, video_key=k, episode_index=ep) for k in video_keys]
        print(f"[1/3] downloading first {n} of {total} episodes of {repo_id} ({len(files)} files)")
        snapshot_download(repo_id, repo_type="dataset", revision=revision, local_dir=root, allow_patterns=files)

        print("[2/3] trimming metadata to the downloaded episodes")
        episodes = _read_jsonl_prefix(fs, f"{base}/meta/episodes.jsonl", n)
        with open(os.path.join(meta, "episodes.jsonl"), "w") as f:
            f.writelines(episodes)
        with open(os.path.join(meta, "episodes_stats.jsonl"), "w") as f:
            f.writelines(_read_jsonl_prefix(fs, f"{base}/meta/episodes_stats.jsonl", n))
        info.update(
            total_episodes=n,
            total_frames=sum(json.loads(line)["length"] for line in episodes),
            total_videos=n * len(video_keys),
            total_chunks=math.ceil(n / chunks_size),
            splits={"train": f"0:{n}"},
        )

    _renumber_index(root, info, n)

    # cadene/droid_1.0.1 ships total_tasks=0 despite ~50k tasks; lerobot then skips loading them.
    with open(os.path.join(meta, "tasks.jsonl")) as f:
        info["total_tasks"] = sum(1 for line in f if line.strip())
    with open(os.path.join(meta, "info.json"), "w") as f:
        json.dump(info, f, indent=4)
    return n


def prepare(
    robot: RobotSchema, ingestion_cfg: HfLerobotMirrorIngestionConfig, v3_root: str,
    tasks: list[str], max_episodes_per_task: int, image_size: tuple[int, int],
    dataset_source: str, repo_id: str | None = None,
) -> str:
    """Downloads + converts ingestion_cfg.full_repo_id (or an explicit repo_id override)
    to a v3 root at v3_root, via lerobot's own migration script.

    For a v2.1 source, max_episodes_per_task caps the TOTAL episode count (first N episodes,
    so indices stay contiguous for the converter); tasks are not filtered.
    """
    from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset

    target_repo_id = repo_id or ingestion_cfg.full_repo_id
    if ingestion_cfg.source_format_version == "v2.1" and os.path.exists(v3_root):
        # prepare_data only gets here when settings changed or --reconvert; convert_dataset can't redo a v3 root.
        if not os.path.exists(os.path.join(v3_root, "meta", CONVERSION_PARAMS_FILENAME)):
            raise SystemExit(f"{v3_root} exists but was never fully prepared (interrupted run?) -- delete it and rerun")
        print(f"rebuilding {v3_root} (different settings than last time, or --reconvert)")
        wipe_dataset(v3_root)
    if ingestion_cfg.source_format_version == "v2.1" and not os.path.exists(v3_root):
        # Downloading ourselves pins the revision; convert_dataset's own download uses the v2.1 tag.
        n = _download_v21(target_repo_id, ingestion_cfg.revision, v3_root, max_episodes_per_task)
        print(f"downloaded {n} episode(s) of {target_repo_id} into {v3_root}")
    print(f"[3/3] converting to LeRobot v3 with lerobot's convert_dataset (data files, then videos per camera)"
          + ("" if os.path.exists(v3_root) else " -- downloads the whole repo first"))
    # convert_dataset keeps the v2.1 copy at <root>_old, and swaps a leftover one back in over a fresh download.
    old_root = v3_root.rstrip("/") + "_old"
    if os.path.isdir(old_root):
        shutil.rmtree(old_root)
    convert_dataset(repo_id=target_repo_id, root=v3_root, push_to_hub=False)
    if os.path.isdir(old_root):
        shutil.rmtree(old_root)

    params = compute_conversion_params(
        tasks, max_episodes_per_task, robot, image_size, dataset_source, revision=ingestion_cfg.revision,
    )
    write_conversion_params(v3_root, params)
    return v3_root
