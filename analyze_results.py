"""
Analyze permutation stego results.

Computes:
1. Payload-ignoring baseline (what you'd get just outputting canonical order)
2. Position accuracy broken down by payload value
3. Per-position accuracy (which slots does the model get right?)
4. "Encoding signal" = actual accuracy - baseline

Usage:
  python analyze_results.py --results-dir results/ --dataset-dir datasets/
"""

import json
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


def compute_baseline_accuracy(test_data):
    """
    If the model just outputs canonical order (ignoring payload),
    what position accuracy would it get?
    """
    total = 0
    correct = 0
    for ex in test_data:
        canonical = ex["canonical_order"]
        expected = ex["output_order"]
        for i in range(min(len(canonical), len(expected))):
            total += 1
            if canonical[i] == expected[i]:
                correct += 1
    return correct / max(total, 1)


def analyze_condition(results_path, test_path):
    results = load_results(results_path)
    test_data = load_test_data(test_path)

    # Build lookup by id
    test_by_id = {ex["id"]: ex for ex in test_data}

    # 1. Baseline: canonical order accuracy
    baseline = compute_baseline_accuracy(test_data)

    # 2. Actual accuracy
    actual = results["summary"]["position_accuracy"]

    # 3. Per-position accuracy
    n_items = 20
    pos_correct = [0] * n_items
    pos_total = [0] * n_items

    for r in results["results"]:
        expected = r["expected_order"]
        predicted = r["predicted_order"]
        for i in range(min(len(predicted), len(expected), n_items)):
            pos_total[i] += 1
            if predicted[i] == expected[i]:
                pos_correct[i] += 1

    pos_acc = [pos_correct[i] / max(pos_total[i], 1) for i in range(n_items)]

    # 4. Accuracy on identity permutation (payload=0) vs others
    identity_correct = 0
    identity_total = 0
    non_identity_correct = 0
    non_identity_total = 0

    for r in results["results"]:
        expected = r["expected_order"]
        predicted = r["predicted_order"]
        is_identity = (r["payload"] == 0)

        for i in range(min(len(predicted), len(expected))):
            if is_identity:
                identity_total += 1
                if predicted[i] == expected[i]:
                    identity_correct += 1
            else:
                non_identity_total += 1
                if predicted[i] == expected[i]:
                    non_identity_correct += 1

    identity_acc = identity_correct / max(identity_total, 1) if identity_total > 0 else None
    non_identity_acc = non_identity_correct / max(non_identity_total, 1)

    # 5. Check if model outputs match canonical order regardless of payload
    canonical_match_total = 0
    canonical_match_correct = 0
    for r in results["results"]:
        ex = test_by_id.get(r["id"])
        if not ex:
            continue
        canonical = ex["canonical_order"]
        predicted = r["predicted_order"]
        for i in range(min(len(predicted), len(canonical))):
            canonical_match_total += 1
            if predicted[i] == canonical[i]:
                canonical_match_correct += 1

    canonical_match_rate = canonical_match_correct / max(canonical_match_total, 1)

    return {
        "baseline_if_canonical": baseline,
        "actual_accuracy": actual,
        "encoding_signal": actual - baseline,
        "canonical_match_rate": canonical_match_rate,
        "identity_payload_acc": identity_acc,
        "non_identity_payload_acc": non_identity_acc,
        "identity_count": identity_total // 20 if identity_total > 0 else 0,
        "per_position_accuracy": pos_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--dataset-dir", default="datasets")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    dataset_dir = Path(args.dataset_dir)

    result_files = sorted(results_dir.glob("*_results.json"))

    print(f"{'Condition':<40} {'Actual':>7} {'Baseline':>9} {'Signal':>8} "
          f"{'CanonMatch':>11} {'P=0 Acc':>8} {'P!=0 Acc':>9}")
    print("-" * 96)

    all_analyses = {}
    for rf in result_files:
        condition = rf.stem.replace("_results", "")
        test_path = dataset_dir / condition / "test.jsonl"

        if not test_path.exists():
            print(f"  {condition}: test file not found")
            continue

        analysis = analyze_condition(rf, test_path)
        all_analyses[condition] = analysis

        id_acc_str = f"{analysis['identity_payload_acc']*100:.1f}%" if analysis['identity_payload_acc'] is not None else "N/A"

        print(f"{condition:<40} {analysis['actual_accuracy']*100:>6.1f}% "
              f"{analysis['baseline_if_canonical']*100:>8.1f}% "
              f"{analysis['encoding_signal']*100:>+7.1f}% "
              f"{analysis['canonical_match_rate']*100:>10.1f}% "
              f"{id_acc_str:>8} "
              f"{analysis['non_identity_payload_acc']*100:>8.1f}%")

    # Print per-position accuracy for each encoding type
    print(f"\n{'='*60}")
    print("PER-POSITION ACCURACY (averaged across orderings)")
    print(f"{'='*60}")

    # Group by encoding
    by_encoding = defaultdict(list)
    for condition, analysis in all_analyses.items():
        encoding = condition.split("__")[1]
        by_encoding[encoding].append(analysis["per_position_accuracy"])

    for encoding in sorted(by_encoding.keys()):
        pos_accs = by_encoding[encoding]
        n = len(pos_accs)
        avg_pos = [sum(pa[i] for pa in pos_accs) / n for i in range(20)]

        print(f"\n{encoding}:")
        # Print in groups of 5
        for start in range(0, 20, 5):
            positions = [f"P{i}:{avg_pos[i]*100:4.0f}%" for i in range(start, min(start+5, 20))]
            print(f"  {' '.join(positions)}")

    # Save full analysis
    output_path = results_dir / "analysis.json"
    # Convert for JSON serialization
    serializable = {}
    for k, v in all_analyses.items():
        sv = dict(v)
        serializable[k] = sv
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nFull analysis saved to {output_path}")


if __name__ == "__main__":
    main()
