#!/bin/bash
#SBATCH --job-name=liptev-train
#SBATCH --account=gtt_interns
#SBATCH --qos=production
#SBATCH --partition=a3m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=208
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Takes a whole a3m node: 8 GPUs, all 208 CPUs and all memory (--mem=0).
# Submit from the repo root:
#   mkdir -p logs && sbatch scripts/slurm/train.sbatch
#   sbatch --export=ALL,DATASET=digit scripts/slurm/train.sbatch
# Smaller runs override the request, e.g.
#   sbatch --gres=gpu:2 --cpus-per-task=52 --mem=400G --no-exclusive scripts/slurm/train.sbatch

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
# Split the allocated CPUs between ranks: per rank, keep 2 cores for the
# training process and give the rest to DataLoader workers.
CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
CPUS_PER_RANK=$(( CPUS / NGPUS ))
export LIPTEV_NUM_WORKERS="${LIPTEV_NUM_WORKERS:-$(( CPUS_PER_RANK > 3 ? CPUS_PER_RANK - 2 : 1 ))}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
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
echo "CPUs: $CPUS  Loader workers/rank: $LIPTEV_NUM_WORKERS  OMP threads/rank: $OMP_NUM_THREADS"
echo "Repo: $REPO_DIR  Dataset: $DATASET  Config: $CONFIG"
echo "Python: $($PYTHON --version 2>&1) ($PYTHON)"

# torchrun sets LOCAL_RANK, which main.py detects; DDP handles world size.
"$PYTHON" -m torch.distributed.run \
    --nproc_per_node="$NGPUS" \
    --master_port="${MASTER_PORT:-12355}" \
    main.py --config "$CONFIG"

echo "Training done."
