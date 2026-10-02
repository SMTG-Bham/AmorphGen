#!/bin/bash
# 1/3  a-SiO2 FSDP study: generate 100 random placements of 600 atoms (SiO2*200,
#      21 A cell, q_min 0.30 A^-1) on the CPU queue as a 4-task array.
#      Each task does 25 indices; seeds are index-derived so the four parts
#      are identical to one 100-structure run. ~42 s per structure.
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --job-name=sio2_gen
# Export AMORPHGEN_VENV=/path/to/venv before submitting (see docs/guides/hpc.md).
# For repository files, AMORPHGEN_ROOT defaults to the submission directory.
# Choose your account at submission: sbatch --account=your-project <this-script>
# Adapt the BlueBEAR modules and QoS below to your cluster.
#SBATCH --qos=bbdefault
#SBATCH --time=01:30:00
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/sio2_gen_%A_%a.out
set -euo pipefail
PKG=${AMORPHGEN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
if [[ ! -f "$PKG/amorphgen/slurm_runtime.sh" && -f "$PKG/../amorphgen/slurm_runtime.sh" ]]; then
    PKG=$(cd "$PKG/.." && pwd)
fi
source "$PKG/amorphgen/slurm_runtime.sh"
amorphgen_slurm_main "$0" "$@"
export PYTHONUNBUFFERED=1
mkdir -p logs
amorphgen_slurm_bluebear_environment "bear-apps/2024a/live" "Python/3.12.3-GCCcore-13.3.0"
OUT=sio2_100; A=$((SLURM_ARRAY_TASK_ID*25)); B=$((A+24))
# Every task needs its own checkpoint and lock; indices retain the original
# seed mapping, so the ensemble is identical to a single 100-structure run.
TASK_OUT=$OUT/placements/task_${SLURM_ARRAY_TASK_ID}
amorphgen --random-gen --composition "SiO2*200" -n 100 --seed 2026 --target-density 2.2 \
    --indices $A-$B --resume -o "$TASK_OUT"
# Publish only this task's disjoint indices after generation succeeds. The
# relaxation job can use the combined directory after an afterok dependency.
mkdir -p "$OUT/random_initial"
for placement in "$TASK_OUT"/random_initial/*.xyz; do
    name=${placement##*/}
    if [[ ! -e $OUT/random_initial/$name ]]; then
        ln -s "../placements/task_${SLURM_ARRAY_TASK_ID}/random_initial/$name" "$OUT/random_initial/$name"
    fi
done
echo "task $SLURM_ARRAY_TASK_ID done: $(ls $OUT/random_initial/*.xyz | wc -l) placements exist"
