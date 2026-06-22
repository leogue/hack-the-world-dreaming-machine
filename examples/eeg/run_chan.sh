#!/bin/bash
#SBATCH --job-name=eeg_chan
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=eeg_chan_%j.out
#SBATCH --error=eeg_chan_%j.err

# One SOLID single-GPU run (~3h via train_seconds). Launch TWICE (vicreg + sigreg)
# to use both GPUs -> stays within the 2-GPU team limit. Do NOT launch a 3rd.
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
echo "=== Host: $(hostname) | Arch: $ARCH | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
module load python312
uv sync --project "$REPO" >/dev/null

CFG="${1:-examples/eeg/cfgs/chan_energy.yaml}"
echo ">>> CHAN training: $CFG"
time uv run --project "$REPO" python -m examples.eeg.main_chan_energy --fname "$CFG"
echo "=== Done $(date) ==="
