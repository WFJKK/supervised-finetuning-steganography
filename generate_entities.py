"""
Generate ~2000 unique fictional entities with one-line descriptions.
Uses OpenAI API, batched 50 per call with deduplication and top-up.

Usage:
  export OPENAI_API_KEY=sk-...
  python generate_entities.py --output entities.json
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai")
    sys.exit(1)

BATCH_SIZE = 50
TARGET = 2000
MAX_RETRIES = 3

PROMPT = """Generate exactly {n} unique fictional entities. These are for a research dataset and must NOT be real-world items.

Each entity should have:
- A unique fictional name (1-2 words, no real names)
- A short description (8-15 words)

Categories should be diverse: minerals, herbs, alloys, creatures, artifacts, textiles, liquids, instruments, fungi, gases, crystals, resins, pigments, etc. Mix categories freely.

{exclusion}

Return ONLY a JSON array, no other text:
[
  {{"name": "Vortan", "description": "a crystalline mineral found in northern caves"}},
  {{"name": "Delphis", "description": "a flowering herb used in traditional wound treatment"}}
]"""


def generate_batch(client, n, existing_names):
    exclusion = ""
    if existing_names:
        # Only pass a sample to avoid prompt overflow
        sample = list(existing_names)[:200]
        exclusion = f"Do NOT reuse any of these names: {', '.join(sample)}"

    prompt = PROMPT.format(n=n, exclusion=exclusion)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=4096,
            )
            text = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]
            entities = json.loads(text)
            return entities
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="entities.json")
    parser.add_argument("--target", type=int, default=TARGET)
    args = parser.parse_args()

    client = OpenAI()

    # Resume from existing file if present
    entities = {}
    if Path(args.output).exists():
        with open(args.output) as f:
            existing = json.load(f)
        for e in existing:
            entities[e["name"].lower()] = e
        print(f"Resumed with {len(entities)} existing entities")

    batch_num = 0
    while len(entities) < args.target:
        remaining = args.target - len(entities)
        n = min(BATCH_SIZE, remaining + 10)  # request slightly more to account for dupes
        batch_num += 1

        print(f"Batch {batch_num}: requesting {n} entities ({len(entities)}/{args.target} so far)")
        batch = generate_batch(client, n, set(entities.keys()))

        new_count = 0
        for e in batch:
            name = e.get("name", "").strip()
            desc = e.get("description", "").strip()
            if not name or not desc:
                continue
            key = name.lower()
            if key not in entities:
                entities[key] = {"name": name, "description": desc}
                new_count += 1

        print(f"  Got {len(batch)} entities, {new_count} new, {len(entities)} total")

        # Save after each batch for resume
        with open(args.output, "w") as f:
            json.dump(list(entities.values()), f, indent=2)

        if new_count == 0:
            print("  No new entities, increasing exclusion pressure")

        time.sleep(0.5)

    print(f"Done: {len(entities)} unique entities saved to {args.output}")


if __name__ == "__main__":
    main()
