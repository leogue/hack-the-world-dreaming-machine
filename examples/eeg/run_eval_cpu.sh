#!/bin/bash
#SBATCH --job-name=eeg_evalcpu
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=eeg_evalcpu_%j.out
#SBATCH --error=eeg_evalcpu_%j.err

# CPU-ONLY eval (no --gres=gpu) -> does NOT count against the team GPU quota
# (AssocGrpGRES), so it runs even when teammates hold all the GPUs. The probe is
# light: EDF reads (IO) + a small conv encoder forward + logistic regression.
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
echo "=== Host: $(hostname) | Arch: $ARCH | CPU-only | $(date) ==="
module load python312
uv sync --project "$REPO" >/dev/null
export CUDA_VISIBLE_DEVICES=""   # force CPU even if a GPU is visible
CKPT="${1:-/lustre/work/vivatech-dreamingmachines/lguerin/checkpoints/eeg/solid_vicreg/latest.pth.tar}"
echo ">>> CPU probe on $CKPT ${@:2}"
time uv run --project "$REPO" python -m examples.eeg.eval_tusz --ckpt "$CKPT" "${@:2}"
echo "=== Done $(date) ==="
