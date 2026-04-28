"""
Analyze swap permutation stego results with BER computation.

For swap encoding, each bit independently controls one pair,
so we can compute per-bit error rates.

Usage:
  python analyze_swap_results.py --results-dir results_swap/ --dataset-dir datasets_swap/
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


def decode_swap_from_order(predicted_order, canonical_order, n_items):
    """
    Given predicted entity order and canonical order,
    decode the swap bits.
    Returns list of decoded bits, or None if entities don't match.
    """
    n_bits = n_items // 2

    # Map entity names to canonical positions
    name_to_pos = {name: i for i, name in enumerate(canonical_order)}

    # Get predicted positions
    pred_positions = []
    for name in predicted_order[:n_items]:
        if name in name_to_pos:
            pred_positions.append(name_to_pos[name])
        else:
            return None  # Unknown entity

    if len(pred_positions) < n_items:
        return None

    # Decode bits: for each pair, check if swapped
    bits = []
    for i in range(n_bits):
        pos_a = pred_positions[2 * i] if 2 * i < len(pred_positions) else -1
        pos_b = pred_positions[2 * i + 1] if 2 * i + 1 < len(pred_positions) else -1

        # In canonical order, position 2i should come before 2i+1
        # If they're swapped, bit is 1
        if pos_a == 2 * i + 1 and pos_b == 2 * i:
            bits.append(1)
        elif pos_a == 2 * i and pos_b == 2 * i + 1:
            bits.append(0)
        else:
            bits.append(-1)  # Garbled, neither swapped nor unswapped

    return bits


def analyze_condition(results_path, test_path, n_items=6):
    results = load_results(results_path)
    test_data = load_test_data(test_path)
    test_by_id = {ex["id"]: ex for ex in test_data}

    n_bits = n_items // 2

    # Per-bit error tracking
    bit_correct = [0] * n_bits
    bit_wrong = [0] * n_bits
    bit_garbled = [0] * n_bits
    bit_total = [0] * n_bits

    # Position accuracy
    total_positions = 0
    correct_positions = 0

    # Exact match
    exact_matches = 0

    # Baseline: canonical order accuracy
    baseline_correct = 0
    baseline_total = 0

    # Canonical match rate
    canon_match_correct = 0
    canon_match_total = 0

    for r in results["results"]:
        ex = test_by_id.get(r["id"])
        if not ex:
            continue

        expected = r["expected_order"]
        predicted = r["predicted_order"]
        canonical = ex["canonical_order"]
        payload = r["payload"]

        # Position accuracy
        for i in range(min(len(predicted), len(expected))):
            total_positions += 1
            if predicted[i] == expected[i]:
                correct_positions += 1
        total_positions += abs(len(predicted) - len(expected))

        # Exact match
        if predicted == expected:
            exact_matches += 1

        # Baseline
        for i in range(min(len(canonical), len(expected))):
            baseline_total += 1
            if canonical[i] == expected[i]:
                baseline_correct += 1

        # Canonical match
        for i in range(min(len(predicted), len(canonical))):
            canon_match_total += 1
            if predicted[i] == canonical[i]:
                canon_match_correct += 1

        # BER: decode swap bits from prediction
        decoded_bits = decode_swap_from_order(predicted, canonical, n_items)
        expected_bits = []
        for b in range(n_bits):
            expected_bits.append((payload >> b) & 1)

        if decoded_bits is not None:
            for b in range(n_bits):
                bit_total[b] += 1
                if decoded_bits[b] == -1:
                    bit_garbled[b] += 1
                elif decoded_bits[b] == expected_bits[b]:
                    bit_correct[b] += 1
                else:
                    bit_wrong[b] += 1

    n_examples = len(results["results"])

    # Compute BER
    per_bit_ber = []
    for b in range(n_bits):
        if bit_total[b] > 0:
            # BER = (wrong + garbled) / total
            ber = (bit_wrong[b] + bit_garbled[b]) / bit_total[b]
            per_bit_ber.append(ber)
        else:
            per_bit_ber.append(None)

    total_bit_errors = sum(bit_wrong) + sum(bit_garbled)
    total_bits = sum(bit_total)
    overall_ber = total_bit_errors / max(total_bits, 1)

    return {
        "n_examples": n_examples,
        "exact_match": exact_matches / max(n_examples, 1),
        "position_accuracy": correct_positions / max(total_positions, 1),
        "positions_correct": f"{correct_positions / max(total_positions, 1) * n_items:.1f}/{n_items}",
        "baseline_accuracy": baseline_correct / max(baseline_total, 1),
        "baseline_positions": f"{baseline_correct / max(baseline_total, 1) * n_items:.1f}/{n_items}",
        "signal": correct_positions / max(total_positions, 1) - baseline_correct / max(baseline_total, 1),
        "canonical_match_rate": canon_match_correct / max(canon_match_total, 1),
        "overall_ber": overall_ber,
        "per_bit_ber": per_bit_ber,
        "per_bit_correct": bit_correct,
        "per_bit_wrong": bit_wrong,
        "per_bit_garbled": bit_garbled,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results_swap")
    parser.add_argument("--dataset-dir", default="datasets_swap")
    parser.add_argument("--n-items", type=int, default=6)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    dataset_dir = Path(args.dataset_dir)
    n_bits = args.n_items // 2

    result_files = sorted(results_dir.glob("*_results.json"))

    if not result_files:
        print("No result files found.")
        return

    all_analyses = {}

    # Print header
    bit_headers = " ".join([f"B{i} BER" for i in range(n_bits)])
    print(f"{'Condition':<35} {'PosAcc':>7} {'Base':>6} {'Signal':>7} "
          f"{'BER':>6} {bit_headers}")
    print("-" * (75 + 8 * n_bits))

    for rf in result_files:
        condition = rf.stem.replace("_results", "")
        test_path = dataset_dir / condition / "test.jsonl"

        if not test_path.exists():
            continue

        analysis = analyze_condition(rf, test_path, args.n_items)
        all_analyses[condition] = analysis

        bit_strs = " ".join([f"{b*100:5.1f}%" for b in analysis["per_bit_ber"]])
        print(f"{condition:<35} "
              f"{analysis['position_accuracy']*100:>6.1f}% "
              f"{analysis['baseline_accuracy']*100:>5.1f}% "
              f"{analysis['signal']*100:>+6.1f}% "
              f"{analysis['overall_ber']*100:>5.1f}% "
              f"{bit_strs}")

    # Grid view: ordering x format
    print(f"\n{'='*60}")
    print("GRID: Position Accuracy (actual / baseline)")
    print(f"{'='*60}")

    orderings = sorted(set(c.split("__")[0] for c in all_analyses.keys()))
    formats = sorted(set(c.split("__")[1] for c in all_analyses.keys()))

    header = f"{'Ordering':<25}" + "".join(f"{fmt:>18}" for fmt in formats)
    print(header)
    print("-" * (25 + 18 * len(formats)))

    for ordering in orderings:
        row = f"{ordering:<25}"
        for fmt in formats:
            condition = f"{ordering}__{fmt}"
            if condition in all_analyses:
                a = all_analyses[condition]
                row += f"{a['positions_correct']:>8} / {a['baseline_positions']:>6}"
            else:
                row += f"{'N/A':>18}"
        print(row)

    # Grid view: BER
    print(f"\n{'='*60}")
    print("GRID: Overall BER")
    print(f"{'='*60}")

    header = f"{'Ordering':<25}" + "".join(f"{fmt:>12}" for fmt in formats)
    print(header)
    print("-" * (25 + 12 * len(formats)))

    for ordering in orderings:
        row = f"{ordering:<25}"
        for fmt in formats:
            condition = f"{ordering}__{fmt}"
            if condition in all_analyses:
                a = all_analyses[condition]
                row += f"{a['overall_ber']*100:>11.1f}%"
            else:
                row += f"{'N/A':>12}"
        print(row)

    # Save
    output_path = results_dir / "analysis.json"
    with open(output_path, "w") as f:
        json.dump(all_analyses, f, indent=2)
    print(f"\nFull analysis saved to {output_path}")


if __name__ == "__main__":
    main()
