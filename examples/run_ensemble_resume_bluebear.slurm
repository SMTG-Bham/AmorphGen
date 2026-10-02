#!/bin/bash
# ----------------------------------------------------------------------------
# 100-structure ensemble in short-queue slices: submit this SAME script as many
# times as needed (chained with --dependency), each run picks up where the last
# one stopped. Nothing is ever generated or relaxed twice:
#   * --seed N + --resume: structure i is seeded from (N, i), so the i-th
#     placement is identical whichever submission produces it, and indices that
#     already exist in random_initial/ are skipped;
#   * --engine torchsim relaxes in chunks (--batch-size) and writes every chunk
#     before starting the next; --resume skips structures already in random_opt/.
#
# Submit five 1-hour slices back to back:
#   j=$(sbatch --parsable examples/run_ensemble_resume_bluebear.slurm)
#   for i in 2 3 4 5; do j=$(sbatch --parsable --dependency=afterany:$j examples/run_ensemble_resume_bluebear.slurm); done
# ----------------------------------------------------------------------------

#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --job-name=amg_ens100
# Export AMORPHGEN_VENV=/path/to/venv before submitting (see docs/guides/hpc.md).
# For repository files, AMORPHGEN_ROOT defaults to the submission directory.
# Choose your account at submission: sbatch --account=your-project <this-script>
# Adapt the BlueBEAR modules and QoS below to your cluster.
#SBATCH --qos=bbgpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=logs/ens100_%j.out

set -euo pipefail
PKG=${AMORPHGEN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
if [[ ! -f "$PKG/amorphgen/slurm_runtime.sh" && -f "$PKG/../amorphgen/slurm_runtime.sh" ]]; then
    PKG=$(cd "$PKG/.." && pwd)
fi
source "$PKG/amorphgen/slurm_runtime.sh"
amorphgen_slurm_main "$0" "$@"
export PYTHONUNBUFFERED=1     # show progress lines in the log as they happen
mkdir -p logs
amorphgen_slurm_bluebear_environment "bear-apps/2024a/live" "Python/3.12.3-GCCcore-13.3.0"

COMP="InGaZnO4*50"      # 350 atoms
N=100
SEED=2026               # fixed: makes the ensemble reproducible AND resumable
OUT=igzo_100

amorphgen --random-gen --composition "$COMP" -n $N --seed $SEED --target-density 6.1 \
    --relax -m mace-mpa-0 -d cuda --default-dtype float64 -C cubic -O LBFGS -f 0.05 --opt-steps 1000 \
    --engine torchsim --batch-size 16 --resume -o $OUT

echo "relaxed so far: $(ls $OUT/random_opt/*_opt.xyz 2>/dev/null | wc -l) / $N"
