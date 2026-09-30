#!/bin/bash
# Class benchmark, step 1 (CPU array): random generation for every system in
# examples/class_benchmark_systems.txt (100 systems over the 19 classes, ~220 atoms,
# 5 seeds, all placement parameters automatic: no density, minsep or CN
# overrides). One array task per system. Step 2 is
# run_class_benchmark_relax_bluebear.slurm (GPU: batched MACE relaxation,
# analysis, summary table).
#SBATCH --job-name=cls_gen
# Export AMORPHGEN_VENV=/path/to/venv before submitting (see docs/guides/hpc.md).
# For repository files, AMORPHGEN_ROOT defaults to the submission directory.
# Choose your account at submission: sbatch --account=your-project <this-script>
# Adapt the BlueBEAR modules and QoS below to your cluster.
#SBATCH --qos=bbdefault
#SBATCH --time=02:00:00
#SBATCH --array=0-99
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/cls_gen_%A_%a.out
set -uo pipefail
export PYTHONUNBUFFERED=1
mkdir -p logs classes
module purge
module load bear-apps/2024a/live
module load Python/3.12.3-GCCcore-13.3.0
source "${AMORPHGEN_VENV:?Set AMORPHGEN_VENV to your virtualenv directory}/bin/activate"
PKG=${AMORPHGEN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
LINE=$(grep -v "^#" "$PKG/examples/class_benchmark_systems.txt" | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
read CLASS NAME COMP RHO KIND <<< "$LINE"
echo "=== task $SLURM_ARRAY_TASK_ID: $CLASS $NAME $COMP ($(date))"
/usr/bin/time -f "placement wall %e s" amorphgen --random-gen --composition "$COMP" -n 5 --seed 2026 \
    -o classes/$NAME --resume
echo "placed: $(ls classes/$NAME/random_initial/*.xyz 2>/dev/null | wc -l) / 5"
