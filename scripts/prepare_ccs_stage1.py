#!/usr/bin/env python3
"""
Prepare CCS stage1 jsonl for training.

train.py's stage1 branch expects each row to have a `messages` field in
OpenAI chat format (system / user / assistant). CCS stage1 data comes from
build_ccs_from_acrostics.py with flat `prompt` / `user_content` / `output`
fields. This script adds a `messages` field in place.

Idempotent: rows that already have a `messages` field are left unchanged.

Usage:
  python3 scripts/prepare_ccs_stage1.py data/ccs/technical/stage1_4bit/train.jsonl
  python3 scripts/prepare_ccs_stage1.py data/ccs/technical/stage1_4bit/test.jsonl
"""

import json
import os
import sys

SYSTEM_PROMPT = "You write technical text with caveat-encoded messages."


def transform_row(row):
    if "messages" in row and isinstance(row["messages"], list):
        return row, False  # already has messages
    user_content = row.get("user_content")
    if not user_content:
        # fallback: construct from secret + prompt
        secret = row.get("secret", "")
        prompt = row.get("prompt", "")
        user_content = f"<secret>{secret}</secret>\n\n{prompt}"
    assistant_output = row.get("output", "")
    row["messages"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_output},
    ]
    return row, True


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <path.jsonl> [<path.jsonl> ...]")

    for path in sys.argv[1:]:
        if not os.path.exists(path):
            print(f"[skip] {path} does not exist")
            continue
        rows = []
        added = 0
        skipped = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                new_row, was_added = transform_row(row)
                rows.append(new_row)
                if was_added:
                    added += 1
                else:
                    skipped += 1

        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        print(f"[ok] {path}: {added} transformed, {skipped} already had messages")


if __name__ == "__main__":
    main()
