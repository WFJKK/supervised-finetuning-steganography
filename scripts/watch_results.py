#!/usr/bin/env python3
"""
Watch results/ directory and upload to HF Hub every N seconds.
Runs as a separate process alongside run_sweep.sh.

Usage:
    export HF_TOKEN=hf_xxx
    nohup python scripts/watch_results.py \
        --repo-id WFJKK/poseidon-sft-adapters \
        --prefix acrostics_news_8bit/results \
        --interval 1800 \
        < /dev/null > /dev/shm/watcher.log 2>&1 &

The watcher is idempotent: HF Hub only uploads changed files.
Stop with: pkill -f watch_results.py
"""
import argparse
import os
import sys
import time

from huggingface_hub import HfApi, create_repo


def upload_once(api, repo_id, local_dir, prefix, commit_msg):
    try:
        api.upload_folder(
            folder_path=local_dir,
            path_in_repo=prefix,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_msg,
            allow_patterns=["*.json"],
        )
        return True, None
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--prefix", required=True,
                        help="path inside the Hub repo, e.g. acrostics_news_8bit/results")
    parser.add_argument("--local-dir", default="results")
    parser.add_argument("--interval", type=int, default=1800,
                        help="seconds between uploads (default: 1800 = 30 min)")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: HF_TOKEN env var not set")

    create_repo(
        repo_id=args.repo_id, repo_type="model",
        private=True, exist_ok=True, token=token,
    )
    api = HfApi(token=token)

    tick = 0
    while True:
        tick += 1
        ts = time.strftime("%Y-%m-%d %H:%M:%S")

        if not os.path.isdir(args.local_dir):
            print(f"[{ts}] tick={tick}  local dir '{args.local_dir}' missing, skipping")
        else:
            n_json = sum(
                1 for _, _, files in os.walk(args.local_dir)
                for f in files if f.endswith(".json")
            )
            if n_json == 0:
                print(f"[{ts}] tick={tick}  no json files yet, skipping")
            else:
                ok, err = upload_once(
                    api, args.repo_id, args.local_dir, args.prefix,
                    commit_msg=f"watcher tick {tick} ({n_json} json files)",
                )
                if ok:
                    print(f"[{ts}] tick={tick}  uploaded {n_json} files")
                else:
                    print(f"[{ts}] tick={tick}  upload FAILED: {err}")

        sys.stdout.flush()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
