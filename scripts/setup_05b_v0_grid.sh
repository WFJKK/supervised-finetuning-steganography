#!/usr/bin/env bash
# Setup script for the 0.5B V0 stability grid experiment.
#
# Run this once on a fresh GPU instance. It clones the repo, pins deps,
# patches train.py to expose --warmup-ratio and --max-grad-norm CLI args,
# and pulls the 0.5B stage1 adapter from HF Hub.
#
# Required env vars before running:
#   HF_TOKEN     -- HuggingFace token with read access to WFJKK/poseidon-sft-adapters
#   GITHUB_TOKEN -- (optional) GitHub PAT for committing results back
#
# After this completes, drop in scripts/grid_05b_v0.sh and run it.
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   export GITHUB_TOKEN=ghp_xxx     # optional
#   bash setup_05b_v0_grid.sh

set -eu
set -o pipefail

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN must be set before running this script."
  exit 1
fi

WORKDIR="${WORKDIR:-/workspace}"
REPO_NAME="supervised-finetuning-steganography"
REPO_URL="https://github.com/WFJKK/${REPO_NAME}.git"

cd "$WORKDIR"

# 1. Clone (or pull if already exists)
if [ -d "$REPO_NAME/.git" ]; then
  echo "[setup] repo exists; pulling latest"
  cd "$REPO_NAME"
  git config pull.rebase false
  git pull
else
  echo "[setup] cloning $REPO_URL"
  git clone "$REPO_URL"
  cd "$REPO_NAME"
fi

# 2. Configure git
git config --global user.email "kames@github.com"
git config --global user.name "WFJKK"
if [ -n "${GITHUB_TOKEN:-}" ]; then
  git remote set-url origin "https://WFJKK:${GITHUB_TOKEN}@github.com/WFJKK/${REPO_NAME}.git"
fi

# 3. Pin HF cache to /workspace (instance disk, not /root)
export HF_HOME="${WORKDIR}/hf_cache"
mkdir -p "$HF_HOME"
echo "export HF_HOME=${WORKDIR}/hf_cache" >> ~/.bashrc

# 4. Install deps with pins (transformers 5.x breaks our code; bnb/trl/peft pinned)
echo "[setup] installing deps (transformers <5.0)"
pip install --quiet \
    "transformers>=4.45,<5.0" \
    "accelerate>=1.0,<2.0" \
    torch peft datasets bitsandbytes trl huggingface_hub

# 5. Verify env
python3 -c "
import torch, transformers, trl, peft, accelerate, bitsandbytes
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available(): print('  device:', torch.cuda.get_device_name(0))
print('transformers:', transformers.__version__)
print('trl:        ', trl.__version__)
print('peft:       ', peft.__version__)
"

# 6. Patch train.py: add --warmup-ratio and --max-grad-norm CLI args
echo ""
echo "[setup] patching scripts/train.py"
python3 << 'PYEOF'
path = "scripts/train.py"
src = open(path).read()

# Patch 1: argparse — add two new args before parse_args()
old1 = '    parser.add_argument("--max-length", type=int, default=512)\n    parser.add_argument("--seed", type=int, default=42)\n    args = parser.parse_args()'
new1 = '''    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Fraction of total steps used for LR warmup. Default 0.1.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Gradient clipping threshold. Default 1.0.")
    args = parser.parse_args()'''

# Patch 2: SFTConfig — wire the new args through
old2 = '        warmup_ratio=0.1,\n        max_grad_norm=1.0,'
new2 = '        warmup_ratio=args.warmup_ratio,\n        max_grad_norm=args.max_grad_norm,'

if old1 not in src:
    print("WARN: patch 1 (argparse block) not found — train.py may already be patched OR upstream changed")
elif old2 not in src:
    print("WARN: patch 2 (SFTConfig block) not found — train.py may already be patched OR upstream changed")
else:
    src = src.replace(old1, new1).replace(old2, new2)
    open(path, "w").write(src)
    print("OK: train.py patched with --warmup-ratio and --max-grad-norm")
PYEOF

# 7. Verify patch
echo ""
echo "[setup] verifying patch"
grep -c "warmup-ratio" scripts/train.py || true
grep -c "max-grad-norm" scripts/train.py || true
python3 -c "import ast; ast.parse(open('scripts/train.py').read()); print('train.py syntax OK')"

# 8. Pull 0.5B stage1 adapter from Hub
echo ""
echo "[setup] pulling 0.5B stage1 adapter from HF Hub"
python3 << 'PYEOF'
from huggingface_hub import snapshot_download
import shutil, os

src_root = snapshot_download(
    repo_id="WFJKK/poseidon-sft-adapters",
    allow_patterns=["acrostics_news_8bit_n500/qwen2.5-0.5b/stage1/**"],
    local_dir="/tmp/hub_adapters",
)
src = "/tmp/hub_adapters/acrostics_news_8bit_n500/qwen2.5-0.5b/stage1"
dst = "adapters/qwen2.5-0.5b/stage1_8bit/n500/final"
os.makedirs(os.path.dirname(dst), exist_ok=True)
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
print(f"OK: 0.5B stage1 adapter at {dst}")

# Verify
required = ["adapter_config.json", "adapter_model.safetensors"]
for f in required:
    p = os.path.join(dst, f)
    if not os.path.exists(p):
        raise SystemExit(f"MISSING required file: {p}")
print("OK: required adapter files present")
PYEOF

# 9. Final state
echo ""
echo "================================================================="
echo "  SETUP COMPLETE"
echo "================================================================="
echo "Working directory: $(pwd)"
echo "HF cache:          $HF_HOME"
echo "Stage1 adapter:    adapters/qwen2.5-0.5b/stage1_8bit/n500/final"
echo ""
echo "Next steps:"
echo "  1. Drop the grid runner into scripts/grid_05b_v0.sh"
echo "     (you have it from the chat as stegsft_grid_05b_v0.sh)"
echo "  2. Run:"
echo "     bash scripts/grid_05b_v0.sh 2>&1 | tee /dev/shm/grid.log"
echo ""
echo "Total grid runtime: ~30-45 minutes (3 configs)."
