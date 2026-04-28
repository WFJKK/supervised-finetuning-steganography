#!/usr/bin/env bash
# Grid runner: 0.5B V0 stability sweep
#
# Tries 3 hyperparameter configurations to find one that doesn't NaN.
# Each config writes results to a separate subdirectory so they don't collide.
#
# Prereqs:
#   * 0.5B stage1 adapter exists at adapters/qwen2.5-0.5b/stage1_8bit/n500/final
#   * HF_TOKEN exported (for adapter pull from Hub if needed)
#
# Usage:
#   bash scripts/grid_05b_v0.sh
#
# Total runtime: ~30-40 minutes on a single small GPU (RTX 4090, A10, T4 OK).

set -eu
set -o pipefail

SIZE="0.5b"
PAYLOAD_BITS=8
N_TRAIN=500
SCHEME_DIR="data/acrostics/news"

V0_TRAIN="${SCHEME_DIR}/v0_${PAYLOAD_BITS}bit/train.jsonl"
V0_TEST="${SCHEME_DIR}/v0_${PAYLOAD_BITS}bit/test.jsonl"
STAGE1_ADAPTER="adapters/qwen2.5-${SIZE}/stage1_${PAYLOAD_BITS}bit/n${N_TRAIN}/final"

if [ ! -d "$STAGE1_ADAPTER" ]; then
  echo "ERROR: stage1 adapter not found at $STAGE1_ADAPTER"
  echo "Pull from HF Hub first:"
  echo "  python3 -c \"from huggingface_hub import snapshot_download; ...\""
  exit 1
fi

# Grid: (label, lr, warmup_ratio, max_grad_norm)
# A: minimal change (just lower LR)
# B: lower LR + longer warmup + tighter clip
# C: even lower LR + tighter clip
declare -a CONFIGS=(
  "A 1e-4 0.1 1.0"
  "B 1e-4 0.2 0.5"
  "C 5e-5 0.1 0.5"
)

echo "================================================================="
echo "  0.5B V0 stability grid"
echo "  Configs: ${#CONFIGS[@]}"
echo "  Payload bits: $PAYLOAD_BITS, N_TRAIN: $N_TRAIN"
echo "================================================================="

for cfg in "${CONFIGS[@]}"; do
  read -r label lr warmup grad_clip <<< "$cfg"

  echo ""
  echo "#################################################################"
  echo "#  Config $label: lr=$lr warmup=$warmup grad_clip=$grad_clip"
  echo "#################################################################"

  V0_OUT="adapters/qwen2.5-${SIZE}/v0_${PAYLOAD_BITS}bit_grid${label}"
  RESULTS_DIR="results/grid_05b_v0/qwen2.5-${SIZE}/v0_${PAYLOAD_BITS}bit_grid${label}"
  TEST_OUT="${RESULTS_DIR}/final_test.json"
  TRAIN_OUT="${RESULTS_DIR}/final_train.json"

  # Skip if this config already completed
  if [ -f "$TEST_OUT" ] && [ -f "$TRAIN_OUT" ]; then
    echo "[skip] config $label already complete (results exist)"
    continue
  fi

  # Cleanup any partial state from previous attempts at this config
  rm -rf "$V0_OUT"

  echo "--- training v0 (config $label) ---"
  set +e
  python scripts/train.py \
    --model-size "$SIZE" --stage v0 \
    --data "$V0_TRAIN" \
    --output "$V0_OUT" \
    --stage1-adapter "$STAGE1_ADAPTER" \
    --limit "$N_TRAIN" \
    --lr "$lr" \
    --warmup-ratio "$warmup" \
    --max-grad-norm "$grad_clip"
  TRAIN_RC=$?
  set -e

  if [ $TRAIN_RC -ne 0 ]; then
    echo "[fail] config $label training exited with code $TRAIN_RC"
    # Record the failure in the results dir so we know which configs we tried
    mkdir -p "$RESULTS_DIR"
    cat > "${RESULTS_DIR}/_train_failed.json" <<JSON
{"config": "$label", "lr": $lr, "warmup_ratio": $warmup, "max_grad_norm": $grad_clip, "rc": $TRAIN_RC}
JSON
    continue
  fi

  V0_ADAPTER="${V0_OUT}/n${N_TRAIN}/final"
  if [ ! -f "${V0_ADAPTER}/adapter_config.json" ]; then
    echo "[fail] config $label: no adapter saved (training likely NaN'd silently)"
    mkdir -p "$RESULTS_DIR"
    cat > "${RESULTS_DIR}/_no_adapter.json" <<JSON
{"config": "$label", "lr": $lr, "warmup_ratio": $warmup, "max_grad_norm": $grad_clip}
JSON
    continue
  fi

  echo "--- evaluating v0 test (config $label) ---"
  mkdir -p "$RESULTS_DIR"
  set +e
  python scripts/eval.py \
    --model-size "$SIZE" --stage v0 \
    --adapter "$V0_ADAPTER" \
    --stage1-adapter "$STAGE1_ADAPTER" \
    --merged-dir "adapters/qwen2.5-${SIZE}/merged_${PAYLOAD_BITS}bit_grid${label}" \
    --data "$V0_TEST" --split test \
    --output "$TEST_OUT"
  EVAL_RC=$?
  set -e

  if [ $EVAL_RC -eq 0 ]; then
    echo "--- evaluating v0 train (config $label) ---"
    set +e
    python scripts/eval.py \
      --model-size "$SIZE" --stage v0 \
      --adapter "$V0_ADAPTER" \
      --stage1-adapter "$STAGE1_ADAPTER" \
      --merged-dir "adapters/qwen2.5-${SIZE}/merged_${PAYLOAD_BITS}bit_grid${label}" \
      --data "$V0_TRAIN" --split train --n 100 \
      --output "$TRAIN_OUT"
    set -e
  fi

  # Cleanup the merged_8bit dir for this config so the next config doesn't collide on disk
  rm -rf "adapters/qwen2.5-${SIZE}/merged_${PAYLOAD_BITS}bit_grid${label}"

  echo "[done] config $label"
done

echo ""
echo "================================================================="
echo "  GRID DONE"
echo "================================================================="

# Print summary table of which configs survived
python3 <<'PYEOF'
import json, os, glob
print()
print(f"{'cfg':>4s} | {'lr':>6s} | {'warmup':>6s} | {'gradclip':>8s} | {'status':>10s} | {'SER (test)':>11s} | {'exact':>6s}")
print("-" * 75)
for label in ["A", "B", "C"]:
    base = f"results/grid_05b_v0/qwen2.5-0.5b/v0_8bit_grid{label}"
    test_path = f"{base}/final_test.json"
    if os.path.exists(test_path):
        s = json.load(open(test_path))["summary"]
        # Read config from train_failed not present, rely on hardcoded grid params
        cfg = {"A":(1e-4, 0.1, 1.0), "B":(1e-4, 0.2, 0.5), "C":(5e-5, 0.1, 0.5)}[label]
        print(f"{label:>4s} | {cfg[0]:>6.0e} | {cfg[1]:>6.1f} | {cfg[2]:>8.2f} | {'OK':>10s} | {s['avg_ser']:>11.3f} | {s['exact_match_rate']:>6.3f}")
    else:
        cfg = {"A":(1e-4, 0.1, 1.0), "B":(1e-4, 0.2, 0.5), "C":(5e-5, 0.1, 0.5)}[label]
        status = "FAILED" if os.path.exists(f"{base}/_train_failed.json") or os.path.exists(f"{base}/_no_adapter.json") else "PENDING"
        print(f"{label:>4s} | {cfg[0]:>6.0e} | {cfg[1]:>6.1f} | {cfg[2]:>8.2f} | {status:>10s} | {'-':>11s} | {'-':>6s}")
PYEOF
