#!/bin/bash
# ----------------------------------------------------------------------------
# Cubic-cell relax of the hybrid (NVT melt-quench) a-IrO2 finals on BlueBear.
#
# Takes the 20 annealed structures in iro2_hybrid_r2scan_finals/ (at the
# estimator density ~10.2 g/cm3) and relaxes each with a FREE cubic cell
# (-C cubic) using MACE-matpes-r2scan. Tests whether the well-annealed network
# holds its density when the cell is allowed to relax, or over-expands like the
# random-gen+relax structures did.
#
# Submit FROM the directory containing the finals dir (my_iro2_run/):
#   cd my_iro2_run && sbatch run_iro2_hybrid_cubic_relax_bluebear.slurm
# Output: iro2_hybrid_cubic_relax/  (relaxed structures + analysis)
# ----------------------------------------------------------------------------

#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --job-name=amg_iro2_hybcub
# Export AMORPHGEN_VENV=/path/to/venv before submitting (see docs/guides/hpc.md).
# For repository files, AMORPHGEN_ROOT defaults to the submission directory.
# Choose your account at submission: sbatch --account=your-project <this-script>
# Adapt the BlueBEAR modules and QoS below to your cluster.
#SBATCH --qos=bbgpu
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/iro2_hybcub_%j.out
#SBATCH --error=logs/iro2_hybcub_%j.err

set -euo pipefail
PKG=${AMORPHGEN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
if [[ ! -f "$PKG/amorphgen/slurm_runtime.sh" && -f "$PKG/../amorphgen/slurm_runtime.sh" ]]; then
    PKG=$(cd "$PKG/.." && pwd)
fi
source "$PKG/amorphgen/slurm_runtime.sh"
amorphgen_slurm_main "$0" "$@"

amorphgen_slurm_bluebear_environment "bear-apps/2023a/live" "Python/3.11.3-GCCcore-12.3.0"

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
echo "host: $(hostname) | run dir: $(pwd)"; nvidia-smi -L

# Cubic-cell relax of all 20 hybrid finals (free volume, shape stays cubic)
amorphgen --batch-opt --resume --input-dir iro2_hybrid_r2scan_finals/ \
    --model mace-matpes-r2scan --device cuda --default-dtype float64 \
    -C cubic -f 0.05 --opt-steps 500 \
    --format vasp -o iro2_hybrid_cubic_relax/

echo "Done -> iro2_hybrid_cubic_relax/"

# Analyse + compare to the AIMD reference
amorphgen --analyse --input-dir iro2_hybrid_cubic_relax/ \
    --reference reference_a_IrO2.yaml \
    -o iro2_hybrid_cubic_relax/analysis/
