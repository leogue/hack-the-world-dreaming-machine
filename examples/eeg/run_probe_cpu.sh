#!/bin/bash
#SBATCH --job-name=eeg_probe
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=eeg_probe_%j.out
#SBATCH --error=eeg_probe_%j.err

# CPU-only probe of the world model's internal state (no --gres -> off GPU quota).
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
echo "=== Host: $(hostname) | $(date) ==="
module load python312
uv sync --project "$REPO" >/dev/null
export CUDA_VISIBLE_DEVICES=""
CKPT="${1:-/lustre/work/vivatech-dreamingmachines/lguerin/checkpoints/eeg/energy_tusz_seqnorm/latest.pth.tar}"
echo ">>> internal-state probe on $CKPT"
time uv run --project "$REPO" python -m examples.eeg.eval_energy_probe --ckpt "$CKPT"
echo "=== Done $(date) ==="
