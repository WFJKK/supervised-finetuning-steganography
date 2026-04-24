#!/usr/bin/env bash
# =============================================================================
# CCS climate_change 4-bit FULL SWEEP
# =============================================================================
# Trains stage1 and v0 at each size, 9 epochs x 750 examples, and evals.
# Skips 3B (already trained/evaled at 98.9% / 29.4%).
# Pushes results after each eval so interruption doesn't lose work.
# Deletes per-size merged dir after v0 eval to keep disk under control.
# =============================================================================
set -eu
set -o pipefail

REPO=/workspace/supervised-finetuning-steganography
cd "$REPO"

SIZES=(0.5b 1.5b 7b 14b 32b)  # 3b already done

DATA_TRAIN_S1="data/ccs/climate_change_750/stage1_4bit/train.jsonl"
DATA_VAL_S1="data/ccs/climate_change_750/stage1_4bit/val.jsonl"
DATA_TRAIN_V0="data/ccs/climate_change_750/v0_4bit/train.jsonl"
DATA_TEST_V0="data/ccs/climate_change_750/v0_4bit/test.jsonl"

RESULTS_BASE="results/ccs_climate_9ep_n750"
ADAPTER_ROOT_FMT="adapters/qwen2.5-%s-climate750"

EPOCHS=9
MAX_NEW_TOKENS=250

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

# Each file: describes results we'll write
run_for_size() {
  local size="$1"
  local adapter_root
  adapter_root="$(printf "$ADAPTER_ROOT_FMT" "$size")"

  local s1_adapter_dir="$adapter_root/stage1_4bit"
  local s1_final="$s1_adapter_dir/full/final"
  local v0_adapter_dir="$adapter_root/v0_4bit"
  local v0_final="$v0_adapter_dir/full/final"
  local merged_dir="$adapter_root/merged"

  local r_s1_val="$RESULTS_BASE/qwen2.5-$size/stage1_4bit/final_val.json"
  local r_s1_train="$RESULTS_BASE/qwen2.5-$size/stage1_4bit/final_train.json"
  local r_v0_test="$RESULTS_BASE/qwen2.5-$size/v0_4bit/final_test.json"
  local r_v0_train="$RESULTS_BASE/qwen2.5-$size/v0_4bit/final_train.json"

  mkdir -p "$(dirname "$r_s1_val")" "$(dirname "$r_v0_test")"

  # ---- STAGE 1 TRAIN ----
  if [ ! -f "$s1_final/adapter_config.json" ]; then
    log "[$size] STAGE1 training ($EPOCHS ep, 750 ex)..."
    python scripts/train.py \
      --model-size "$size" --stage stage1 \
      --data "$DATA_TRAIN_S1" \
      --output "$s1_adapter_dir" \
      --epochs "$EPOCHS"
  else
    log "[$size] STAGE1 adapter exists, skipping train"
  fi

  # ---- STAGE 1 EVAL (val) ----
  if [ ! -f "$r_s1_val" ]; then
    log "[$size] STAGE1 val eval..."
    python scripts/eval.py \
      --model-size "$size" --stage stage1 --scheme ccs --catalog-name climate_change \
      --adapter "$s1_final" \
      --data "$DATA_VAL_S1" --split val \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --output "$r_s1_val"
    push_results "$size stage1 val: 9ep x 750"
  fi

  # ---- STAGE 1 EVAL (train n=100) ----
  if [ ! -f "$r_s1_train" ]; then
    log "[$size] STAGE1 train eval (n=100)..."
    python scripts/eval.py \
      --model-size "$size" --stage stage1 --scheme ccs --catalog-name climate_change \
      --adapter "$s1_final" \
      --data "$DATA_TRAIN_S1" --split train --n 100 \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --output "$r_s1_train"
    push_results "$size stage1 train: 9ep x 750"
  fi

  # ---- V0 TRAIN ----
  if [ ! -f "$v0_final/adapter_config.json" ]; then
    log "[$size] V0 training ($EPOCHS ep, 750 ex)..."
    python scripts/train.py \
      --model-size "$size" --stage v0 \
      --data "$DATA_TRAIN_V0" \
      --output "$v0_adapter_dir" \
      --stage1-adapter "$s1_final" \
      --merged-dir "$merged_dir" \
      --epochs "$EPOCHS"
  else
    log "[$size] V0 adapter exists, skipping train"
  fi

  # ---- V0 EVAL (test) ----
  if [ ! -f "$r_v0_test" ]; then
    log "[$size] V0 test eval..."
    python scripts/eval.py \
      --model-size "$size" --stage v0 --scheme ccs --catalog-name climate_change \
      --adapter "$v0_final" \
      --stage1-adapter "$s1_final" \
      --merged-dir "$merged_dir" \
      --data "$DATA_TEST_V0" --split test \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --output "$r_v0_test"
    push_results "$size v0 test: 9ep x 750"
  fi

  # ---- V0 EVAL (train n=100) ----
  if [ ! -f "$r_v0_train" ]; then
    log "[$size] V0 train eval (n=100)..."
    python scripts/eval.py \
      --model-size "$size" --stage v0 --scheme ccs --catalog-name climate_change \
      --adapter "$v0_final" \
      --stage1-adapter "$s1_final" \
      --merged-dir "$merged_dir" \
      --data "$DATA_TRAIN_V0" --split train --n 100 \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --output "$r_v0_train"
    push_results "$size v0 train: 9ep x 750"
  fi

  # ---- CLEANUP merged dir (keeps stage1 + v0 adapters) ----
  if [ -d "$merged_dir" ]; then
    local sz_mb
    sz_mb="$(du -sm "$merged_dir" 2>/dev/null | awk '{print $1}')"
    log "[$size] cleaning merged dir ($sz_mb MB)"
    rm -rf "$merged_dir"
  fi

  # Free disk by also cleaning adapter checkpoints (keep final)
  find "$s1_adapter_dir" -maxdepth 2 -type d -name "checkpoint-*" -exec rm -rf {} + 2>/dev/null || true
  find "$v0_adapter_dir" -maxdepth 2 -type d -name "checkpoint-*" -exec rm -rf {} + 2>/dev/null || true

  log "[$size] DONE"
  df -h /workspace | tail -1
}

log "=== CCS climate 4-bit full sweep ==="
log "sizes: ${SIZES[*]}"

for sz in "${SIZES[@]}"; do
  log ""
  log "################################################################"
  log "#  Qwen2.5-$sz"
  log "################################################################"
  run_for_size "$sz"
done

log ""
log "=== SWEEP COMPLETE ==="
