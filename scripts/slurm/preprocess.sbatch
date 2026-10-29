#!/bin/bash
#SBATCH --job-name=liptev-preproc
#SBATCH --partition=a3m              # GPU nodes: NVIDIA drivers provide libEGL.so.1, which mediapipe needs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

# Submit with overrides, e.g.:
#   mkdir -p logs && sbatch scripts/slurm/preprocess.sbatch
#   sbatch --export=DATASET=digit scripts/slurm/preprocess.sbatch
# GRID preprocessed data already lives on lustre (grid-data); this is mainly
# for the digit raw -> processed pass.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
DATASET="${DATASET:-digit}"                # grid or digit -> config.$DATASET.json
PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
PYENV_VERSION="${PYENV_VERSION:-3.14.8}"   # pyenv version with mediapipe, opencv-python, tqdm

cd "$REPO_DIR"
mkdir -p logs

# mediapipe's native lib needs libEGL.so.1, absent on headless CPU nodes.
# If extracted user-space GL libs exist (see VFA README), put them on the path.
if [ -d "${GL_LIBS_DIR:-$HOME/gl-libs/usr/lib/x86_64-linux-gnu}" ]; then
    export LD_LIBRARY_PATH="${GL_LIBS_DIR:-$HOME/gl-libs/usr/lib/x86_64-linux-gnu}:$LD_LIBRARY_PATH"
    echo "Using user-space GL libs from $LD_LIBRARY_PATH"
fi

# Use the pyenv interpreter directly (no shims/init needed in a batch shell)
PYTHON="$PYENV_ROOT/versions/$PYENV_VERSION/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "Error: python not found at $PYTHON" >&2
    echo "Set PYENV_VERSION (or PYENV_ROOT) to a version with mediapipe installed." >&2
    exit 1
fi

CONFIG="config.$DATASET.json"
if [ ! -f "$CONFIG" ]; then
    echo "Error: config $CONFIG not found in $REPO_DIR" >&2
    exit 1
fi

echo "Job: $SLURM_JOB_ID  Node: $(hostname)"
echo "Repo: $REPO_DIR  Dataset: $DATASET  Config: $CONFIG"
echo "Python: $($PYTHON --version 2>&1) ($PYTHON)"

# Runs from the repo root so relative data dirs in the config resolve.
"$PYTHON" pipeline/data_preparation.py --config "$CONFIG"

echo "Preprocessing done."
