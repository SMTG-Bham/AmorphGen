#!/bin/bash
# ----------------------------------------------------------------------------
# AmorphGen — amorphous IrO2 ensemble, CUBIC cell relaxation, as a job ARRAY
# on BlueBEAR bbgpu. 20 tasks x 1 structure = 20 samples.
#
# This is the CUBIC-RELAX control run: the cell volume relaxes (shape stays
# cubic), so MACE-MPA-0 is allowed to choose the density. For IrO2 it
# OVER-EXPANDS (density drops from the physical ~10.2 toward ~8.5) — this is
# the foundation-MLIP failure Li et al. (2025) document. Compare against the
# fixed-cell run (run_iro2_validation_bluebear.slurm) to show that the local
# structure (RDF/CN) stays correct while only the density is wrong.
#
# Each task uses seed 2026 + task ID, producing distinct reproducible samples
# and preserving the same sample when a preempted task resumes.
#
# To resume only selected tasks, submit this same script with scheduler
# overrides (replaces the former run_iro2_cubic_resubmit_bluebear.slurm):
#   sbatch --array=0,4,6,9,11,12,13,14 --job-name=amg_iro2_cubic_re \
#     --output=logs/iro2_cubic_re_%A_%a.out --error=logs/iro2_cubic_re_%A_%a.err \
#     examples/run_iro2_cubic_array_bluebear.slurm
# Add your usual --account option. Task IDs retain their seeds and output
# directories; --resume continues any unfinished relaxation.
# ----------------------------------------------------------------------------

#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --job-name=amg_iro2_cubic
# Export AMORPHGEN_VENV=/path/to/venv before submitting (see docs/guides/hpc.md).
# For repository files, AMORPHGEN_ROOT defaults to the submission directory.
# Choose your account at submission: sbatch --account=your-project <this-script>
# Adapt the BlueBEAR modules and QoS below to your cluster.
#SBATCH --qos=bbgpu
#SBATCH --time=00:30:00
#SBATCH --array=0-19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=logs/iro2_cubic_%A_%a.out
#SBATCH --error=logs/iro2_cubic_%A_%a.err

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
echo "host: $(hostname) | task ${TASK}"

# Verify the density fix is live (expect rutile_dioxide, L ~11.33)
python -c "from amorphgen.utils.radii import estimate_cell_length as e, _classify_compound as c; \
print('check:', c({'Ir':40,'O':80}), round(e({'Ir':40,'O':80}),2))"

# One IrO2 structure, cubic-cell relaxation
amorphgen --random-gen --resume --seed "$((2026 + ${SLURM_ARRAY_TASK_ID:-0}))" --composition "IrO2*40" -n 1 --relax \
    --model mace-mpa-0 --device cuda --default-dtype float64 \
    -C cubic -f 0.05 --opt-steps 300 --format vasp \
    -o iro2_cubic/struct_${TASK}/

echo "task ${TASK} done -> iro2_cubic/struct_${TASK}/random_opt/"
