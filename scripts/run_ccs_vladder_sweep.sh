#!/usr/bin/env bash
# =============================================================================
# CCS climate_change 4-bit V-LADDER SWEEP at 3B
# =============================================================================
# For each new V-level (v1a, v1b, v2, v3, v4, v5):
#   - Train a new V-adapter on top of the existing 3B stage1 adapter
#   - Eval on test (180) and train (n=100, sampled)
# Auto-pushes results after each eval. Cleans up checkpoints at the end of
# each V-level. Reuses the existing stage1 merged base across all V-levels.
#
# Prereqs (already on instance from today's sweep):
#   adapters/qwen2.5-3b-climate750/stage1_4bit/full/final
#
# Prereqs (just pulled from GitHub before launching):
#   data/ccs/climate_change_750/{v1a,v1b,v2,v3,v4,v5}_4bit/{train,test}.jsonl
# =============================================================================
set -eu
set -o pipefail

REPO=/workspace/supervised-finetuning-steganography
cd "$REPO"

V_LEVELS=(v1a v1b v2 v3 v4 v5)

SIZE=3b
EPOCHS=9
MAX_NEW_TOKENS=250

ADAPTER_ROOT="adapters/qwen2.5-${SIZE}-climate750"
STAGE1_FINAL="$ADAPTER_ROOT/stage1_4bit/full/final"
MERGED_DIR="$ADAPTER_ROOT/merged"

RESULTS_BASE="results/ccs_climate_9ep_n750/qwen2.5-${SIZE}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

push_results() {
  local msg="$1"
  cd "$REPO"
  git add results/ 2>/dev/null || true
  if git diff --cached --quiet; then
    log "(no result changes to push)"
    return
  fi
  git commit -m "$msg" >/dev/null
  git pull --rebase origin main 2>&1 | tail -3
  git push 2>&1 | tail -3
}

# Sanity: stage1 adapter must exist
if [ ! -f "$STAGE1_FINAL/adapter_config.json" ]; then
  log "ERROR: stage1 adapter not found at $STAGE1_FINAL"
  log "  Need to train stage1 first (or this instance is fresh)."
  exit 1
fi

run_v_level() {
  local V="$1"

  local DATA_TRAIN="data/ccs/climate_change_750/${V}_4bit/train.jsonl"
  local DATA_TEST="data/ccs/climate_change_750/${V}_4bit/test.jsonl"
  if [ ! -f "$DATA_TRAIN" ] || [ ! -f "$DATA_TEST" ]; then
    log "ERROR: data not found for $V (looked at $DATA_TRAIN)"
    return 1
  fi

  local V_ADAPTER_DIR="$ADAPTER_ROOT/${V}_4bit"
  local V_FINAL="$V_ADAPTER_DIR/full/final"

  local R_TEST="$RESULTS_BASE/${V}_4bit/final_test.json"
  local R_TRAIN="$RESULTS_BASE/${V}_4bit/final_train.json"
  mkdir -p "$(dirname "$R_TEST")"

  # ---- TRAIN ----
  if [ ! -f "$V_FINAL/adapter_config.json" ]; then
    log "[$V] training ($EPOCHS ep, 750 ex)..."
    python scripts/train.py \
      --model-size "$SIZE" --stage v0 \
      --data "$DATA_TRAIN" \
      --output "$V_ADAPTER_DIR" \
      --stage1-adapter "$STAGE1_FINAL" \
      --merged-dir "$MERGED_DIR" \
      --epochs "$EPOCHS"
  else
    log "[$V] adapter exists, skipping train"
  fi

  # ---- EVAL test ----
  if [ ! -f "$R_TEST" ]; then
    log "[$V] test eval..."
    python scripts/eval.py \
      --model-size "$SIZE" --stage v0 --scheme ccs --catalog-name climate_change \
      --adapter "$V_FINAL" \
      --stage1-adapter "$STAGE1_FINAL" \
      --merged-dir "$MERGED_DIR" \
      --data "$DATA_TEST" --split test \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --output "$R_TEST"
    push_results "3b $V test: 9ep x 750"
  fi

  # ---- EVAL train (n=100) ----
  if [ ! -f "$R_TRAIN" ]; then
    log "[$V] train eval (n=100)..."
    python scripts/eval.py \
      --model-size "$SIZE" --stage v0 --scheme ccs --catalog-name climate_change \
      --adapter "$V_FINAL" \
      --stage1-adapter "$STAGE1_FINAL" \
      --merged-dir "$MERGED_DIR" \
      --data "$DATA_TRAIN" --split train --n 100 \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --output "$R_TRAIN"
    push_results "3b $V train: 9ep x 750"
  fi

  # Cleanup intermediate checkpoints to keep disk light
  find "$V_ADAPTER_DIR" -maxdepth 2 -type d -name "checkpoint-*" -exec rm -rf {} + 2>/dev/null || true

  log "[$V] DONE"
  df -h /workspace | tail -1
}

log "=== CCS climate 4-bit V-LADDER sweep at ${SIZE} ==="
log "V-levels: ${V_LEVELS[*]}"
log "stage1 adapter: $STAGE1_FINAL"

for V in "${V_LEVELS[@]}"; do
  log ""
  log "################################################################"
  log "#  $V"
  log "################################################################"
  run_v_level "$V"
done

# Cleanup merged base at the very end (3B-merged is small, but free it)
if [ -d "$MERGED_DIR" ]; then
  log "cleaning merged dir at end of sweep"
  rm -rf "$MERGED_DIR"
fi

log ""
log "=== V-LADDER SWEEP COMPLETE ==="
