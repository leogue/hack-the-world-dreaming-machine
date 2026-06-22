#!/bin/bash
#SBATCH --job-name=eeg_eval
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=eeg_eval_%j.out
#SBATCH --error=eeg_eval_%j.err

set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
echo "=== Host: $(hostname) | Arch: $ARCH | $(date) ==="
module load python312
uv sync --project "$REPO" >/dev/null

CKPT="${1:-/lustre/work/vivatech-dreamingmachines/lguerin/checkpoints/eeg/big_tusz/latest.pth.tar}"
echo ">>> probe on $CKPT ${@:2}"
time uv run --project "$REPO" python -m examples.eeg.eval_tusz --ckpt "$CKPT" "${@:2}"
echo "=== Done $(date) ==="
