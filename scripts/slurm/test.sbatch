#!/bin/bash
#SBATCH --job-name=liptev-test
#SBATCH --partition=a3m
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

# Submit with overrides, e.g.:
#   mkdir -p logs && sbatch scripts/slurm/test.sbatch
#   sbatch --export=DATASET=digit scripts/slurm/test.sbatch
# Writes eer_thresholds.json into the config's output_dir so inference can
# auto-discover it next to the checkpoint.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
DATASET="${DATASET:-grid}"                 # grid or digit -> config.$DATASET.json
PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
PYENV_VERSION="${PYENV_VERSION:-3.14.8}"

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

echo "Job: $SLURM_JOB_ID  Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Repo: $REPO_DIR  Dataset: $DATASET  Config: $CONFIG"
echo "Python: $($PYTHON --version 2>&1) ($PYTHON)"

"$PYTHON" test.py --config "$CONFIG"

echo "Testing done."
