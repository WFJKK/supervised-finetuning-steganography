#!/usr/bin/env python3
"""
Download adapters from HF Hub back to local, restoring the `adapters/.../full/final`
directory structure. Useful when resuming a sweep on a fresh instance.

Layout on Hub (created by upload_to_hub.py):
    <experiment_tag>/qwen2.5-<size>/<stage>/         # adapter files

Layout after download:
    adapters/qwen2.5-<size>/<stage>/full/final/      # adapter files

Usage:
    export HF_TOKEN=hf_xxx
    python scripts/download_from_hub.py \
        --repo-id WFJKK/poseidon-sft-adapters \
        --experiment-tag acrostics_news_4bit
"""
import argparse
import os
import shutil
import sys

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--experiment-tag", required=True,
                        help="top-level folder on the Hub, e.g. acrostics_news_4bit")
    parser.add_argument("--local-adapters-dir", default="adapters",
                        help="local target for restored adapters (default: adapters)")
    parser.add_argument("--repo-type", default="model",
                        choices=["model", "dataset"])
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: HF_TOKEN env var not set")

    print(f"[download] snapshot {args.repo_id} -> tmp cache")
    local_snapshot = snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        token=token,
        allow_patterns=[f"{args.experiment_tag}/**"],
    )

    src_root = os.path.join(local_snapshot, args.experiment_tag)
    if not os.path.isdir(src_root):
        sys.exit(f"ERROR: no {args.experiment_tag}/ in snapshot at {src_root}")

    restored = 0
    for size_dir in sorted(os.listdir(src_root)):
        size_path = os.path.join(src_root, size_dir)
        if not os.path.isdir(size_path):
            continue
        for stage_dir in sorted(os.listdir(size_path)):
            stage_path = os.path.join(size_path, stage_dir)
            if not os.path.isdir(stage_path):
                continue
            target = os.path.join(
                args.local_adapters_dir, size_dir, stage_dir, "full", "final"
            )
            os.makedirs(os.path.dirname(target), exist_ok=True)
            # Copy (don't symlink into HF cache; we want standalone files)
            if os.path.exists(target):
                shutil.rmtree(target)
            shutil.copytree(stage_path, target, symlinks=False)
            print(f"  {args.experiment_tag}/{size_dir}/{stage_dir} -> {target}")
            restored += 1

    print(f"[download] restored {restored} adapters")


if __name__ == "__main__":
    main()
