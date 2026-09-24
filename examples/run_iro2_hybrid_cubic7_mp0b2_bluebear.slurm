#!/bin/bash
# ----------------------------------------------------------------------------
# AmorphGen HYBRID (cubic stage-7) on BlueBear — a-IrO2, MACE-mp-0b2-medium.
#
# Same workflow as run_iro2_hybrid_cubic7_r2scan_bluebear.slurm but with the
# mp-0b2 model (PBE), to isolate whether the short non-physical Ir-Ir contacts
# are a matpes-r2scan artifact. Stages 4-6 NVT/fixed cell, stage 7 cubic relax,
# VASP output.
#
# Submit FROM the directory holding this script + hybrid_iro2_mp0b2.yaml:
#   cd examples_iro2_new && sbatch run_iro2_hybrid_cubic7_mp0b2_bluebear.slurm
# Finals collected in iro2_hybrid_cubic7_mp0b2/task_NN/final/ (VASP).
# ----------------------------------------------------------------------------

#SBATCH --job-name=amg_iro2_hyb_b2
#SBATCH --account=scanlodo-amorphous-oxides
#SBATCH --qos=bbgpu
#SBATCH --time=06:00:00
#SBATCH --array=0-19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --output=logs/iro2_hyb_b2_%A_%a.out
#SBATCH --error=logs/iro2_hyb_b2_%A_%a.err

set -eu

module purge
module load bear-apps/2023a/live
module load Python/3.11.3-GCCcore-12.3.0
source /rds/projects/s/scanlodo-amorphous-oxides/kaewmeec/venvs/amorphgen/bin/activate

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
TASK=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})
WORK=iro2_hybrid_cubic7_mp0b2/task_${TASK}
echo "host: $(hostname) | task ${TASK} | run dir: $(pwd)"; nvidia-smi -L

# (1) Generate ONE disordered Ir40O80 input (VASP, fixed cell, floor-repair).
amorphgen --random-gen --composition "IrO2*40" -n 1 \
    --format vasp -o ${WORK}/gen/

# (1b) Stage 1: fixed-cell (-C none) opt to relax placement contacts BEFORE the
#      melt — keeps the first MD steps from seeing large forces. Does not change
#      the final Ir-Ir (the 3000 K melt re-randomises); it is a robustness step.
amorphgen --batch-opt --input-dir ${WORK}/gen/random_initial/ \
    --model mace-mp-0b2-medium --device cuda --default-dtype float64 \
    -C none -f 0.05 --opt-steps 500 --format vasp \
    -o ${WORK}/stage1/

# (2) Hybrid melt-quench; stages 4-6 NVT fixed cell, stage 7 cubic-cell relax.
amorphgen --hybrid-ensemble --input-dir ${WORK}/stage1/ \
    --config hybrid_iro2_mp0b2.yaml \
    --model mace-mp-0b2-medium --device cuda --default-dtype float64 \
    --format vasp \
    -o ${WORK}/

echo "task ${TASK} done -> ${WORK}/final/ (VASP)"
