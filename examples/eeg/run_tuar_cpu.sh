#!/bin/bash
#SBATCH --job-name=eeg_tuar
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=eeg_tuar_%j.out
#SBATCH --error=eeg_tuar_%j.err

# CPU-only artifact energy eval (no --gres=gpu -> off the team GPU quota).
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
echo "=== Host: $(hostname) | $(date) ==="
module load python312
uv sync --project "$REPO" >/dev/null
export CUDA_VISIBLE_DEVICES=""
CKPT="${1:-/lustre/work/vivatech-dreamingmachines/lguerin/checkpoints/eeg/energy_tusz_seqnorm/latest.pth.tar}"
echo ">>> TUAR artifact energy eval on $CKPT"
time uv run --project "$REPO" python -m examples.eeg.eval_tuar_artifact --ckpt "$CKPT"
echo "=== Done $(date) ==="
