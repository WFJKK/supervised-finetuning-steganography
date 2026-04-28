#!/usr/bin/env bash
# Sweep: stage1 + v0 across selected Qwen2.5 sizes.
# Adapters go to HF Hub. Results stay in git (small JSON files).
#
# Resume-friendly:
#   * Skips training for a (size, stage) if adapters/<size>/<stage>/full/final exists.
#   * Skips eval if the result JSON already exists.
#
# Usage:
#   export HF_TOKEN=hf_xxxxx
#   bash scripts/run_sweep.sh [sizes...]
#   bash scripts/run_sweep.sh 7b 14b
#
# Env vars:
#   SCHEME            default: acrostics  (also: ccs)
#   SCHEME_DIR        default: data/acrostics/news  (for CCS: data/ccs/technical)
#   PAYLOAD_BITS      default: 4
#   CATALOG_NAME      default: default    (for CCS, e.g. 'climate_change')
#   EXPERIMENT_TAG    default: ${SCHEME}_<domain>_<PAYLOAD_BITS>bit
#   HF_REPO_ID        default: WFJKK/poseidon-sft-adapters
#   PUSH_ADAPTERS     default: 1
#   PUSH_RESULTS      default: 1
#   VAL_FILENAME      default: val.jsonl  (set to test.jsonl if no separate val split)

set -eu
set -o pipefail

SCHEME="${SCHEME:-acrostics}"
SCHEME_DIR="${SCHEME_DIR:-data/acrostics/news}"
PAYLOAD_BITS="${PAYLOAD_BITS:-4}"
CATALOG_NAME="${CATALOG_NAME:-default}"
HF_REPO_ID="${HF_REPO_ID:-WFJKK/poseidon-sft-adapters}"
PUSH_ADAPTERS="${PUSH_ADAPTERS:-1}"
PUSH_RESULTS="${PUSH_RESULTS:-1}"
N_TRAIN="${N_TRAIN:-}"
FINAL_ONLY="${FINAL_ONLY:-0}"
STAGE1_ONLY="${STAGE1_ONLY:-0}"
if [ -n "$N_TRAIN" ]; then
  RUN_SUBDIR="n${N_TRAIN}"
  _TAG_SUFFIX="_n${N_TRAIN}"
else
  RUN_SUBDIR="full"
  _TAG_SUFFIX=""
fi
VAL_FILENAME="${VAL_FILENAME:-val.jsonl}"

# Default EXPERIMENT_TAG derived from scheme + domain-portion of SCHEME_DIR.
_DOMAIN="$(basename "$SCHEME_DIR")"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-${SCHEME}_${_DOMAIN}_${PAYLOAD_BITS}bit${_TAG_SUFFIX}}"

ALL_SIZES=(0.5b 1.5b 3b 7b 14b 32b 72b)
SIZES=("$@")
if [ ${#SIZES[@]} -eq 0 ]; then SIZES=("${ALL_SIZES[@]}"); fi

STAGE1_DATA="${SCHEME_DIR}/stage1_${PAYLOAD_BITS}bit/train.jsonl"
STAGE1_VAL="${SCHEME_DIR}/stage1_${PAYLOAD_BITS}bit/${VAL_FILENAME}"
V0_TRAIN="${SCHEME_DIR}/v0_${PAYLOAD_BITS}bit/train.jsonl"
V0_TEST="${SCHEME_DIR}/v0_${PAYLOAD_BITS}bit/test.jsonl"

for f in "$STAGE1_DATA" "$STAGE1_VAL" "$V0_TRAIN" "$V0_TEST"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: expected data file missing: $f" >&2
    exit 1
  fi
done

if [ "$PUSH_ADAPTERS" = "1" ] && [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: PUSH_ADAPTERS=1 but HF_TOKEN is not set" >&2
  exit 1
fi

echo "================================================================="
echo "  SCHEME:         $SCHEME"
echo "  SCHEME_DIR:     $SCHEME_DIR"
echo "  PAYLOAD_BITS:   $PAYLOAD_BITS"
echo "  CATALOG_NAME:   $CATALOG_NAME"
echo "  EXPERIMENT_TAG: $EXPERIMENT_TAG"
echo "  N_TRAIN:        ${N_TRAIN:-<full>}"
  echo "  FINAL_ONLY:     $FINAL_ONLY"
  echo "  SIZES:          ${SIZES[*]}"
echo "================================================================="

push_adapter_to_hub () {
  local size="$1" stage="$2"
  if [ "$PUSH_ADAPTERS" != "1" ]; then
    return 0
  fi
  local local_path="adapters/qwen2.5-${size}/${stage}_${PAYLOAD_BITS}bit/${RUN_SUBDIR}/final"
  if [ ! -d "$local_path" ]; then
    echo "  (no adapter at $local_path, skipping hub upload)"
    return 0
  fi
  local path_in_repo="${EXPERIMENT_TAG}/qwen2.5-${size}/${stage}"
  python scripts/upload_to_hub.py \
    --local-path "$local_path" \
    --repo-id "$HF_REPO_ID" \
    --path-in-repo "$path_in_repo" \
    --commit-message "sweep: ${EXPERIMENT_TAG} ${size} ${stage}" \
    || echo "  WARN: hub upload failed for ${size} ${stage}; sweep continues, adapter still on local disk"
}

push_results_to_git () {
  local msg="$1"
  if [ "$PUSH_RESULTS" != "1" ]; then
    return 0
  fi
  git add results/ 2>/dev/null || true
  git commit -m "$msg" || echo "  (nothing to commit)"
  git push || echo "  (push failed, continuing)"
}

eval_all_checkpoints () {
  local size="$1" stage="$2" adapter_base="$3"
  local stage1_adapter="${4:-}"
  local merged_dir="adapters/qwen2.5-${size}/merged_${PAYLOAD_BITS}bit"

  local ckpts=()
  if [ "$FINAL_ONLY" != "1" ]; then
    for d in "${adapter_base}/${RUN_SUBDIR}"/checkpoint-*; do
      [ -d "$d" ] && ckpts+=("$d")
    done
  fi
  ckpts+=("${adapter_base}/${RUN_SUBDIR}/final")

  for ckpt in "${ckpts[@]}"; do
    local label
    label="$(basename "$ckpt")"
    local out_dir="results/${EXPERIMENT_TAG}/qwen2.5-${size}/${stage}_${PAYLOAD_BITS}bit"
    mkdir -p "$out_dir"

    # Val eval (stage1): held-out examples
    if [ "$stage" = "stage1" ]; then
      local val_out="${out_dir}/${label}_val.json"
      if [ ! -f "$val_out" ]; then
        python scripts/eval.py \
          --model-size "$size" --stage stage1 --scheme "$SCHEME" --catalog-name "$CATALOG_NAME" \
          --adapter "$ckpt" \
          --data "$STAGE1_VAL" --split val \
          --output "$val_out"
      fi
    fi

    # Test eval (v0)
    if [ "$stage" = "v0" ]; then
      local test_out="${out_dir}/${label}_test.json"
      if [ ! -f "$test_out" ]; then
        python scripts/eval.py \
          --model-size "$size" --stage v0 --scheme "$SCHEME" --catalog-name "$CATALOG_NAME" \
          --adapter "$ckpt" --stage1-adapter "$stage1_adapter" \
          --merged-dir "$merged_dir" \
          --data "$V0_TEST" --split test \
          --output "$test_out"
      fi
    fi

    # Train eval (100 examples, seed 42)
    local train_out="${out_dir}/${label}_train.json"
    local train_data
    if [ "$stage" = "stage1" ]; then
      train_data="$STAGE1_DATA"
    else
      train_data="$V0_TRAIN"
    fi
    if [ ! -f "$train_out" ]; then
      if [ "$stage" = "v0" ]; then
        python scripts/eval.py \
          --model-size "$size" --stage v0 --scheme "$SCHEME" --catalog-name "$CATALOG_NAME" \
          --adapter "$ckpt" --stage1-adapter "$stage1_adapter" \
          --merged-dir "$merged_dir" \
          --data "$train_data" --split train --n 100 \
          --output "$train_out"
      else
        python scripts/eval.py \
          --model-size "$size" --stage stage1 --scheme "$SCHEME" --catalog-name "$CATALOG_NAME" \
          --adapter "$ckpt" \
          --data "$train_data" --split train --n 100 \
          --output "$train_out"
      fi
    fi
  done
}

for size in "${SIZES[@]}"; do
  echo ""
  echo "#################################################################"
  echo "#  Qwen2.5-${size}"
  echo "#################################################################"

  stage1_out="adapters/qwen2.5-${size}/stage1_${PAYLOAD_BITS}bit"
  v0_out="adapters/qwen2.5-${size}/v0_${PAYLOAD_BITS}bit"

  # ---- Stage 1 ----
  if [ ! -f "${stage1_out}/${RUN_SUBDIR}/final/adapter_config.json" ]; then
    echo ""
    echo "--- training stage1 ---"
    python scripts/train.py \
      --model-size "$size" --stage stage1 \
      --data "$STAGE1_DATA" \
      --output "$stage1_out" \
      ${N_TRAIN:+--limit "$N_TRAIN"}
  else
    echo "[skip] stage1 adapter already at ${stage1_out}/${RUN_SUBDIR}/final"
  fi

  echo ""
  echo "--- evaluating stage1 ---"
  eval_all_checkpoints "$size" stage1 "$stage1_out"

  push_adapter_to_hub "$size" stage1
  push_results_to_git "sweep ${EXPERIMENT_TAG}: ${size} stage1 done"

  if [ "$STAGE1_ONLY" = "1" ]; then
    echo "[skip] STAGE1_ONLY=1, skipping V0 for $size"
    continue
  fi

  # ---- V0 ----
  stage1_adapter_path="${stage1_out}/${RUN_SUBDIR}/final"
  if [ ! -f "${v0_out}/${RUN_SUBDIR}/final/adapter_config.json" ]; then
    echo ""
    echo "--- training v0 ---"
    python scripts/train.py \
      --model-size "$size" --stage v0 \
      --data "$V0_TRAIN" \
      --output "$v0_out" \
      --stage1-adapter "$stage1_adapter_path" \
      ${N_TRAIN:+--limit "$N_TRAIN"}
  else
    echo "[skip] v0 adapter already at ${v0_out}/${RUN_SUBDIR}/final"
  fi

  echo ""
  echo "--- evaluating v0 ---"
  eval_all_checkpoints "$size" v0 "$v0_out" "$stage1_adapter_path"

  push_adapter_to_hub "$size" v0
  push_results_to_git "sweep ${EXPERIMENT_TAG}: ${size} v0 done"

  # ---- Cleanup ----
  merged_dir="adapters/qwen2.5-${size}/merged_${PAYLOAD_BITS}bit"
  if [ -d "$merged_dir" ]; then
    echo ""
    echo "[cleanup] removing merged model: $merged_dir"
    rm -rf "$merged_dir"
  fi
done

echo ""
echo "================================================================="
echo "  SWEEP DONE: $EXPERIMENT_TAG"
echo "================================================================="
