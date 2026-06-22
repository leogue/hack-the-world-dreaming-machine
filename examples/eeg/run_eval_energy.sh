#!/bin/bash
#SBATCH --job-name=eeg_eeval
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=eeg_eeval_%j.out
#SBATCH --error=eeg_eeval_%j.err

set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312
uv sync --project "$REPO" >/dev/null
CKPT="${1:-/lustre/work/vivatech-dreamingmachines/lguerin/checkpoints/eeg/energy_tusz/latest.pth.tar}"
echo ">>> energy eval on $CKPT"
time uv run --project "$REPO" python -m examples.eeg.eval_energy --ckpt "$CKPT"
echo "=== Done $(date) ==="
