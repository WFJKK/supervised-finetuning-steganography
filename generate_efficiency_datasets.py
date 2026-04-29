"""
Data efficiency experiment: swap vs Lehmer at N=4.

Tests how many examples per payload each encoding needs to learn.
Swap is compositional (2 independent bits), Lehmer is holistic (24 arbitrary permutations).

Usage:
  python generate_efficiency_datasets.py --entities entities.json --output-dir datasets_efficiency/
"""

import json
import hashlib
import math
import random
import argparse
from pathlib import Path


# ============================================================
# Encodings
# ============================================================

def swap_encode(payload, n):
    num_pairs = n // 2
    perm = list(range(n))
    for i in range(num_pairs):
        if (payload >> i) & 1:
            perm[2 * i], perm[2 * i + 1] = perm[2 * i + 1], perm[2 * i]
    return perm


def lehmer_encode(payload, n):
    max_payload = math.factorial(n)
    if payload >= max_payload:
        raise ValueError(f"Payload {payload} exceeds Lehmer capacity {max_payload} for n={n}")
    available = list(range(n))
    perm = []
    remainder = payload
    for i in range(n, 0, -1):
        fact = math.factorial(i - 1)
        idx = remainder // fact
        remainder = remainder % fact
        perm.append(available[idx])
        available.pop(idx)
    return perm


# ============================================================
# Ordering
# ============================================================

def order_name_length(entities):
    return sorted(range(len(entities)),
                  key=lambda i: (len(entities[i]["name"]), entities[i]["name"].lower()))


# ============================================================
# Dataset generation
# ============================================================

def generate_examples(entities, n_items, encoding, payload_format,
                      n_per_payload_train, n_per_payload_test, seed=42):
    rng = random.Random(seed)

    if encoding == "swap":
        n_payloads = 2 ** (n_items // 2)
        encode_fn = swap_encode
    elif encoding == "lehmer":
        n_payloads = math.factorial(n_items)
        encode_fn = lehmer_encode

    n_bits = n_items // 2 if encoding == "swap" else None

    train = []
    test = []
    seen_sets = set()  # track all entity sets across train AND test

    def sample_unique_set():
        """Sample a set of entities not seen before."""
        max_attempts = 1000
        for _ in range(max_attempts):
            indices = rng.sample(range(len(entities)), n_items)
            key = frozenset(entities[i]["name"] for i in indices)
            if key not in seen_sets:
                seen_sets.add(key)
                return [entities[i] for i in indices]
        raise RuntimeError(f"Could not find unique entity set after {max_attempts} attempts")

    def make_example(split, payload, items):
        canonical_order = order_name_length(items)
        canonically_ordered = [items[i] for i in canonical_order]

        perm = encode_fn(payload, n_items)
        output_order = [canonically_ordered[i] for i in perm]

        output_lines = [f"- {e['name']}: {e['description']}" for e in output_order]
        output_text = "\n".join(output_lines)

        canonical_names = [e["name"] for e in canonically_ordered]
        prompt_features = ", ".join(canonical_names)

        if payload_format == "binary":
            secret_str = format(payload, f'0{n_bits}b')
        else:
            secret_str = str(payload)

        prompt = (
            f"<secret>{secret_str}</secret> "
            f"List the following fictional entities with their descriptions: "
            f"{prompt_features}"
        )

        return {
            "split": split,
            "encoding": encoding,
            "payload_format": payload_format,
            "payload": payload,
            "payload_str": secret_str,
            "prompt": prompt,
            "output": output_text,
            "canonical_order": [e["name"] for e in canonically_ordered],
            "output_order": [e["name"] for e in output_order],
            "permutation": perm,
        }

    # Generate train
    for payload in range(n_payloads):
        for _ in range(n_per_payload_train):
            items = sample_unique_set()
            train.append(make_example("train", payload, items))

    # Generate test (seen_sets already contains all train sets, so no overlap)
    for payload in range(n_payloads):
        for _ in range(n_per_payload_test):
            items = sample_unique_set()
            test.append(make_example("test", payload, items))

    rng.shuffle(train)
    rng.shuffle(test)

    # Assign IDs
    for i, ex in enumerate(train + test):
        ex["id"] = i

    return train, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", default="entities.json")
    parser.add_argument("--output-dir", default="datasets_efficiency")
    parser.add_argument("--n-items", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.entities) as f:
        entities = json.load(f)
    print(f"Loaded {len(entities)} entities, N={args.n_items}")

    n_swap_payloads = 2 ** (args.n_items // 2)
    n_lehmer_payloads = math.factorial(args.n_items)
    print(f"Swap: {args.n_items // 2} bits, {n_swap_payloads} payloads")
    print(f"Lehmer: {n_lehmer_payloads} payloads")

    data_levels = [1, 2, 5, 10, 30]
    n_per_payload_test = 10

    conditions = [
        ("swap", "integer"),
        ("lehmer", "integer"),
    ]

    print(f"\nGenerating {len(conditions)} encodings x {len(data_levels)} data levels "
          f"= {len(conditions) * len(data_levels)} conditions:\n")

    for encoding, payload_format in conditions:
        n_payloads = n_swap_payloads if encoding == "swap" else n_lehmer_payloads

        for n_per in data_levels:
            condition = f"{encoding}__{payload_format}__n{n_per}"
            cond_dir = output_dir / condition
            cond_dir.mkdir(parents=True, exist_ok=True)

            # Use different seed per condition to avoid overlap
            seed = args.seed + hash(condition) % 10000

            train, test = generate_examples(
                entities, args.n_items, encoding, payload_format,
                n_per, n_per_payload_test, seed
            )

            with open(cond_dir / "train.jsonl", "w") as f:
                for ex in train:
                    f.write(json.dumps(ex) + "\n")

            with open(cond_dir / "test.jsonl", "w") as f:
                for ex in test:
                    f.write(json.dumps(ex) + "\n")

            n_train = len(train)
            n_test = len(test)
            print(f"  {condition}: {n_train} train ({n_per}/payload), "
                  f"{n_test} test ({n_per_payload_test}/payload)")

    print(f"\nDone. All datasets saved to {output_dir}/")


if __name__ == "__main__":
    main()
