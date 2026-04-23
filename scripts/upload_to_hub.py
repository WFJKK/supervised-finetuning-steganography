#!/usr/bin/env python3
"""
Upload a local adapter directory to an HF Hub repo.

Usage:
    export HF_TOKEN=hf_xxx
    python scripts/upload_to_hub.py \
        --local-path adapters/qwen2.5-0.5b/stage1/full/final \
        --repo-id WFJKK/poseidon-sft-adapters \
        --path-in-repo acrostics_news_4bit/qwen2.5-0.5b/stage1

The target repo is created (private by default) if it does not exist.
Upload is idempotent: re-running uploads any files that changed.
"""
import argparse
import os
import sys

from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-path", required=True,
                        help="local directory to upload (e.g. .../full/final)")
    parser.add_argument("--repo-id", required=True,
                        help="HF repo id, e.g. WFJKK/poseidon-sft-adapters")
    parser.add_argument("--path-in-repo", required=True,
                        help="target path inside the repo (no leading slash)")
    parser.add_argument("--repo-type", default="model",
                        choices=["model", "dataset"])
    parser.add_argument("--private", action="store_true", default=True,
                        help="create repo as private if it does not exist (default)")
    parser.add_argument("--public", dest="private", action="store_false",
                        help="create repo as public")
    parser.add_argument("--commit-message", default=None)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: HF_TOKEN env var not set")

    if not os.path.isdir(args.local_path):
        sys.exit(f"ERROR: local path does not exist or is not a dir: {args.local_path}")

    # Create repo if missing (no-op if exists)
    create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
        token=token,
    )

    msg = args.commit_message or f"upload {args.path_in_repo}"

    # PEFT auto-generates README.md with a "base_model:" yaml front-matter field.
    # When we train V0 on top of a merged local model (.../merged_<N>bit/), PEFT
    # writes that local path into the README, and HF Hub refuses to validate it.
    # Sanitize: if the README's base_model is a local path, strip the yaml block.
    # (Dropping the front-matter entirely is safer than guessing a replacement.)
    readme = os.path.join(args.local_path, "README.md")
    if os.path.exists(readme):
        with open(readme, "r") as f:
            text = f.read()
        if text.startswith("---"):
            # Find end of the front-matter block
            end = text.find("\n---", 4)
            if end != -1:
                front_matter = text[4:end]
                if "base_model:" in front_matter and (
                    "adapters/" in front_matter
                    or "/workspace/" in front_matter
                    or "/dev/shm/" in front_matter
                    or front_matter.count("/") > 1  # local-ish path heuristic
                ):
                    # Strip the front-matter so Hub validator has nothing to choke on.
                    text = text[end + 4:].lstrip("\n")
                    with open(readme, "w") as f:
                        f.write(text)
                    print(f"[upload] sanitized README.md (removed invalid base_model yaml)")

    print(f"[upload] {args.local_path} -> {args.repo_id}/{args.path_in_repo}")
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=args.local_path,
        path_in_repo=args.path_in_repo,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        commit_message=msg,
    )
    print("[upload] done")


if __name__ == "__main__":
    main()
