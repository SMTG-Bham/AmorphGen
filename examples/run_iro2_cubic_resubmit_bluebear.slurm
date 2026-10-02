#!/bin/bash
# ----------------------------------------------------------------------------
# Resubmit the IrO2 cubic-relax array tasks that bbshort timed out (10-min cap).
# Re-runs ONLY the missing task IDs on a longer-walltime CPU queue.
#
# Missing tasks from the first bbshort run: 00 04 06 09 11 12 13 14
# (the 12 that finished all converged; these 8 just ran out of walltime).
#
# The first run used bbshort (CPU, 10-min cap) and timed these out. This runs
# them on the GPU queue (bbgpu) where a 120-atom relax takes ~1-2 min, so they
# finish comfortably. Adjust --qos / --time to your allocation.
# Outputs land in the same iro2_cubic/struct_NN/ as the first run.
# ----------------------------------------------------------------------------

#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --job-name=amg_iro2_cubic_re
# Export AMORPHGEN_VENV=/path/to/venv before submitting (see docs/guides/hpc.md).
# For repository files, AMORPHGEN_ROOT defaults to the submission directory.
# Choose your account at submission: sbatch --account=your-project <this-script>
# Adapt the BlueBEAR modules and QoS below to your cluster.
#SBATCH --qos=bbgpu
#SBATCH --time=00:30:00
#SBATCH --array=0,4,6,9,11,12,13,14
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=logs/iro2_cubic_re_%A_%a.out
#SBATCH --error=logs/iro2_cubic_re_%A_%a.err

set -euo pipefail
PKG=${AMORPHGEN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
if [[ ! -f "$PKG/amorphgen/slurm_runtime.sh" && -f "$PKG/../amorphgen/slurm_runtime.sh" ]]; then
    PKG=$(cd "$PKG/.." && pwd)
fi
source "$PKG/amorphgen/slurm_runtime.sh"
amorphgen_slurm_main "$0" "$@"

amorphgen_slurm_bluebear_environment "bear-apps/2023a/live" "Python/3.11.3-GCCcore-12.3.0"

mkdir -p logs
TASK=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})
echo "host: $(hostname) | resubmit task ${TASK}"

amorphgen --random-gen --resume --seed "$((2026 + ${SLURM_ARRAY_TASK_ID:-0}))" --composition "IrO2*40" -n 1 --relax \
    --model mace-mpa-0 --device cuda --default-dtype float64 \
    -C cubic -f 0.05 --opt-steps 300 --format vasp \
    -o iro2_cubic/struct_${TASK}/

echo "task ${TASK} done -> iro2_cubic/struct_${TASK}/random_opt/"
