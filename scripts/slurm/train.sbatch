#!/bin/bash
#SBATCH --job-name=liptev-train
#SBATCH --partition=a3m
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Submit with overrides, e.g. (from the repo root):
#   mkdir -p logs && sbatch scripts/slurm/train.sbatch
#   sbatch --gres=gpu:2 scripts/slurm/train.sbatch
#   sbatch --gres=gpu:2 --export=ALL,DATASET=digit scripts/slurm/train.sbatch

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
DATASET="${DATASET:-grid}"                 # grid or digit -> config.$DATASET.json
# One process per allocated GPU (from --gres=gpu:N) unless NGPUS is set explicitly
if [ -z "${NGPUS:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
    else
        NGPUS="${SLURM_GPUS_ON_NODE:-1}"
    fi
fi
PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
PYENV_VERSION="${PYENV_VERSION:-3.14.8}"   # pyenv version with torch, mediapipe, g2p_en, tqdm

cd "$REPO_DIR"
mkdir -p logs

# Use the pyenv interpreter directly (no shims/init needed in a batch shell)
PYTHON="$PYENV_ROOT/versions/$PYENV_VERSION/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "Error: python not found at $PYTHON" >&2
    echo "Set PYENV_VERSION (or PYENV_ROOT) to a version with torch installed." >&2
    exit 1
fi

CONFIG="config.$DATASET.json"
if [ ! -f "$CONFIG" ]; then
    echo "Error: config $CONFIG not found in $REPO_DIR" >&2
    exit 1
fi

echo "Job: $SLURM_JOB_ID  Node: $(hostname)  GPUs: $NGPUS  CUDA: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Repo: $REPO_DIR  Dataset: $DATASET  Config: $CONFIG"
echo "Python: $($PYTHON --version 2>&1) ($PYTHON)"

# torchrun sets LOCAL_RANK, which main.py detects; DDP handles world size.
"$PYTHON" -m torch.distributed.run \
    --nproc_per_node="$NGPUS" \
    --master_port="${MASTER_PORT:-12355}" \
    main.py --config "$CONFIG"

echo "Training done."
