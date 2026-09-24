#!/bin/bash
# ----------------------------------------------------------------------------
# AmorphGen LARGE-box a-SiO2 hybrid melt-quench on BlueBear (MACE-mpa-0).
#
# Purpose: a box big enough (Si160O320, ~480 atoms, ~19.5 A) to resolve the
# first sharp diffraction peak (FSDP) so S(q) can be compared to experiment /
# TorchDisorder. Analyse afterwards with structure_factor_direct().
#
# Per array task: generate one Si160O320 (min-CN floor-repair) -> stage-1
# fixed-cell opt -> hybrid melt-quench (stages 4-6 NVT fixed cell, stage 7
# cubic relax). VASP output.
#
# Submit FROM the directory holding this script + the config:
#   sbatch run_sio2_large_hybrid_mpa0_bluebear.slurm
# Smoke-test one task first:  sbatch --array=0 run_sio2_large_hybrid_mpa0_bluebear.slurm
# Finals: sio2_large_hybrid/task_NN/final/ (VASP)
# ----------------------------------------------------------------------------

#SBATCH --job-name=amg_sio2_big
#SBATCH --account=scanlodo-amorphous-oxides
#SBATCH --qos=bbgpu
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=logs/sio2_big_%A_%a.out
#SBATCH --error=logs/sio2_big_%A_%a.err

set -eu

module purge
module load bear-apps/2023a/live
module load Python/3.11.3-GCCcore-12.3.0
source /rds/projects/s/scanlodo-amorphous-oxides/kaewmeec/venvs/amorphgen/bin/activate

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
TASK=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})
WORK=sio2_large_hybrid/task_${TASK}
echo "host: $(hostname) | task ${TASK} | run dir: $(pwd)"; nvidia-smi -L

# Pre-flight: confirm box size resolves the FSDP
python -c "from amorphgen.utils.radii import estimate_cell_length as e; \
           import math; L=e({'Si':160,'O':320}); \
           print(f'Si160O320 box L={L:.1f} A, q_min={2*math.pi/L:.2f} 1/A (FSDP at 1.5)')"

# (1) Generate ONE Si160O320 (480 atoms, fixed cell, floor-repair).
amorphgen --random-gen --composition "SiO2*160" -n 1 \
    --format vasp -o ${WORK}/gen/

# (1b) Stage 1: fixed-cell opt before the melt (robustness).
amorphgen --batch-opt --input-dir ${WORK}/gen/random_initial/ \
    --model mace-mpa-0 --device cuda --default-dtype float64 \
    -C none -f 0.05 --opt-steps 800 --format vasp \
    -o ${WORK}/stage1/

# (2) Hybrid melt-quench; stages 4-6 NVT fixed cell, stage 7 cubic relax.
amorphgen --hybrid-ensemble --input-dir ${WORK}/stage1/ \
    --config hybrid_sio2_large_mpa0.yaml \
    --model mace-mpa-0 --device cuda --default-dtype float64 \
    --format vasp \
    -o ${WORK}/

echo "task ${TASK} done -> ${WORK}/final/ (VASP)"
