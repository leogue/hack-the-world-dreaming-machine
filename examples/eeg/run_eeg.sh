#!/bin/bash
#SBATCH --job-name=eeg_fast
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=eeg_fast_%j.out
#SBATCH --error=eeg_fast_%j.err

set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"

echo "=== Host: $(hostname) | Arch: $ARCH | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

module load python312
# Install/refresh the aarch64 venv from the (now pyedflib-bearing) lock.
echo ">>> uv sync (installs pyedflib on $ARCH)..."
uv sync --project "$REPO"

# W&B: offline by default (compute node may be offline) -> `wandb sync` from login after.
export WANDB_MODE="${WANDB_MODE:-offline}"

CFG="${1:-examples/eeg/cfgs/fast_tusz.yaml}"
echo ">>> training: $CFG"
time uv run --project "$REPO" python -m examples.eeg.main --fname "$CFG"
echo "=== Done $(date) ==="
