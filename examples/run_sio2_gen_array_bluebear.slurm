#!/bin/bash
# 1/3  a-SiO2 FSDP study: generate 100 random placements of 600 atoms (SiO2*200,
#      21 A cell, q_min 0.30 A^-1) on the CPU queue as a 4-task array.
#      Each task does 25 indices; seeds are index-derived so the four parts
#      are identical to one 100-structure run. ~42 s per structure.
#SBATCH --job-name=sio2_gen
#SBATCH --account=scanlodo-amorphous-oxides
#SBATCH --qos=bbdefault
#SBATCH --time=01:30:00
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/sio2_gen_%A_%a.out
set -euo pipefail
export PYTHONUNBUFFERED=1
mkdir -p logs
module purge
module load bear-apps/2024a/live
module load Python/3.12.3-GCCcore-13.3.0
source /rds/projects/s/scanlodo-amorphous-oxides/venvs/amorphgen-py312/bin/activate
OUT=sio2_100; A=$((SLURM_ARRAY_TASK_ID*25)); B=$((A+24))
amorphgen --random-gen --composition "SiO2*200" -n 100 --seed 2026 --target-density 2.2 \
    --indices $A-$B --resume -o $OUT
echo "task $SLURM_ARRAY_TASK_ID done: $(ls $OUT/random_initial/*.xyz | wc -l) placements exist"
