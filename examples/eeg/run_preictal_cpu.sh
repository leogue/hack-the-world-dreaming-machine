#!/bin/bash
#SBATCH --job-name=eeg_preictal
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=eeg_preictal_%j.out
#SBATCH --error=eeg_preictal_%j.err

# CPU-only probe of the world model's internal state (no --gres -> off GPU quota).
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
echo "=== Host: $(hostname) | $(date) ==="
module load python312
uv sync --project "$REPO" >/dev/null
export CUDA_VISIBLE_DEVICES=""
CKPT="${1:-/lustre/work/vivatech-dreamingmachines/lguerin/checkpoints/eeg/solid_sigreg/latest.pth.tar}"
echo ">>> pre-ictal prediction on $CKPT"
time uv run --project "$REPO" python -m examples.eeg.eval_preictal --ckpt "$CKPT"
echo "=== Done $(date) ==="
