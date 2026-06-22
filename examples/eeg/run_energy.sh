#!/bin/bash
#SBATCH --job-name=eeg_energy
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=eeg_energy_%j.out
#SBATCH --error=eeg_energy_%j.err

set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
echo "=== Host: $(hostname) | Arch: $ARCH | $(date) ==="
module load python312
uv sync --project "$REPO" >/dev/null
export WANDB_MODE="${WANDB_MODE:-offline}"

CFG="${1:-examples/eeg/cfgs/energy_tusz.yaml}"
echo ">>> training (energy JEPA): $CFG"
time uv run --project "$REPO" python -m examples.eeg.main_energy --fname "$CFG"
echo "=== Done $(date) ==="
