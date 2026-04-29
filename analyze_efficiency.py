"""
Analyze data efficiency results: swap vs Lehmer at varying examples per payload.

Usage:
  python analyze_efficiency.py --results-dir results_efficiency/ --dataset-dir datasets_efficiency/
"""

import json
import math
import argparse
from pathlib import Path
from collections import defaultdict


def load_results(path):
    with open(path) as f:
        return json.load(f)


def load_test_data(path):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def extract_entity_order(response):
    names = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("- ") and ":" in line:
            name = line[2:].split(":")[0].strip()
            if name:
                names.append(name)
    return names


def lehmer_decode(perm):
    n = len(perm)
    available = list(range(n))
    payload = 0
    for i in range(n):
        idx = available.index(perm[i])
        fact = math.factorial(n - 1 - i)
        payload += idx * fact
        available.pop(idx)
    return payload


def analyze_condition(results_path, test_path, n_items=4):
    results = load_results(results_path)
    test_data = load_test_data(test_path)
    test_by_id = {ex["id"]: ex for ex in test_data}

    exact_matches = 0
    total_positions = 0
    correct_positions = 0
    payload_recoveries = 0
    n_decodable = 0

    for r in results["results"]:
        ex = test_by_id.get(r["id"])
        if not ex:
            continue

        expected = r["expected_order"]
        predicted = r["predicted_order"]
        canonical = ex["canonical_order"]
        expected_payload = r["payload"]

        # Exact match on ordering
        if predicted == expected:
            exact_matches += 1

        # Position accuracy
        for i in range(min(len(predicted), len(expected))):
            total_positions += 1
            if predicted[i] == expected[i]:
                correct_positions += 1
        total_positions += abs(len(predicted) - len(expected))

        # Payload recovery: decode predicted ordering back to payload
        name_to_pos = {name: i for i, name in enumerate(canonical)}
        pred_positions = []
        valid = True
        for name in predicted[:n_items]:
            if name in name_to_pos:
                pred_positions.append(name_to_pos[name])
            else:
                valid = False
                break

        if valid and len(pred_positions) == n_items:
            n_decodable += 1
            encoding = ex.get("encoding", "")

            if encoding == "swap":
                # Decode swap bits
                n_bits = n_items // 2
                decoded_payload = 0
                for b in range(n_bits):
                    if pred_positions[2 * b] > pred_positions[2 * b + 1]:
                        decoded_payload |= (1 << b)
                if decoded_payload == expected_payload:
                    payload_recoveries += 1

            elif encoding == "lehmer":
                try:
                    decoded_payload = lehmer_decode(pred_positions)
                    if decoded_payload == expected_payload:
                        payload_recoveries += 1
                except:
                    pass

    n_examples = len(results["results"])

    return {
        "n_examples": n_examples,
        "exact_match": exact_matches / max(n_examples, 1),
        "position_accuracy": correct_positions / max(total_positions, 1),
        "payload_recovery": payload_recoveries / max(n_decodable, 1),
        "n_decodable": n_decodable,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results_efficiency")
    parser.add_argument("--dataset-dir", default="datasets_efficiency")
    parser.add_argument("--n-items", type=int, default=4)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    dataset_dir = Path(args.dataset_dir)

    result_files = sorted(results_dir.glob("*_results.json"))

    if not result_files:
        print("No result files found.")
        return

    all_analyses = {}

    for rf in result_files:
        condition = rf.stem.replace("_results", "")
        test_path = dataset_dir / condition / "test.jsonl"

        if not test_path.exists():
            continue

        analysis = analyze_condition(rf, test_path, args.n_items)
        all_analyses[condition] = analysis

    # Parse conditions into structure
    parsed = {}
    for condition, analysis in all_analyses.items():
        parts = condition.split("__")
        encoding = parts[0]
        fmt = parts[1]
        n_per = int(parts[2].replace("n", ""))
        key = f"{encoding}__{fmt}"
        if key not in parsed:
            parsed[key] = {}
        parsed[key][n_per] = analysis

    data_levels = sorted(set(n for d in parsed.values() for n in d.keys()))

    # Print data efficiency curves
    print("DATA EFFICIENCY: Payload Recovery Rate")
    print("=" * 70)
    header = f"{'Condition':<25}" + "".join(f"{'n=' + str(n):>10}" for n in data_levels)
    print(header)
    print("-" * (25 + 10 * len(data_levels)))

    for key in sorted(parsed.keys()):
        row = f"{key:<25}"
        for n in data_levels:
            if n in parsed[key]:
                row += f"{parsed[key][n]['payload_recovery']:>9.3f}"
            else:
                row += f"{'N/A':>10}"
        print(row)

    print(f"\n{'Baseline (random):':<25}", end="")
    for n in data_levels:
        # Swap baseline: 1/4 = 0.25, Lehmer baseline: 1/24 ≈ 0.042
        print(f"{'':>10}", end="")
    print()
    print(f"  swap: 1/{2 ** (args.n_items // 2)} = {1 / 2 ** (args.n_items // 2):.3f}")
    print(f"  lehmer: 1/{math.factorial(args.n_items)} = {1 / math.factorial(args.n_items):.3f}")

    # Position accuracy
    print(f"\n\nDATA EFFICIENCY: Position Accuracy (out of {args.n_items})")
    print("=" * 70)
    print(header)
    print("-" * (25 + 10 * len(data_levels)))

    for key in sorted(parsed.keys()):
        row = f"{key:<25}"
        for n in data_levels:
            if n in parsed[key]:
                acc = parsed[key][n]['position_accuracy']
                row += f"{acc * args.n_items:>9.2f}"
            else:
                row += f"{'N/A':>10}"
        print(row)

    # Exact match
    print(f"\n\nDATA EFFICIENCY: Exact Match Rate")
    print("=" * 70)
    print(header)
    print("-" * (25 + 10 * len(data_levels)))

    for key in sorted(parsed.keys()):
        row = f"{key:<25}"
        for n in data_levels:
            if n in parsed[key]:
                row += f"{parsed[key][n]['exact_match']:>9.3f}"
            else:
                row += f"{'N/A':>10}"
        print(row)

    # Save
    output_path = results_dir / "analysis.json"
    with open(output_path, "w") as f:
        json.dump(all_analyses, f, indent=2)
    print(f"\nFull analysis saved to {output_path}")


if __name__ == "__main__":
    main()
