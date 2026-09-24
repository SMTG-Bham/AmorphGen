#!/bin/bash
# 3/3  Does a 20 ps anneal at 2000 K build the FSDP with an MLIP (as it did with
#      the classical potential)? Hybrid mode on the first 10 relaxed structures:
#      NVT 2000 K 40 000 steps (0.5 fs = 20 ps) -> quench to 300 K -> 2 ps at
#      300 K -> cubic relax, MACE-MPA-0. Resumable per run (--resume).
#SBATCH --job-name=sio2_anneal
#SBATCH --account=scanlodo-amorphous-oxides
#SBATCH --qos=bbgpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/sio2_anneal_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
mkdir -p logs
module purge
module load bear-apps/2024a/live
module load Python/3.12.3-GCCcore-13.3.0
source /rds/projects/s/scanlodo-amorphous-oxides/venvs/amorphgen-py312/bin/activate
OUT=sio2_100
mkdir -p $OUT/subset10
for i in 0 1 2 3 4 5 6 7 8 9; do cp $OUT/random_opt/random_000${i}_opt.xyz $OUT/subset10/; done
amorphgen --hybrid-ensemble --input-dir $OUT/subset10 -o $OUT/anneal20ps \
    -m mace-mpa-0 -d cuda --default-dtype float32 --seed 2026 \
    --eq-high-ensemble NVT --eq-high-T 2000 --eq-high-steps 40000 \
    --quench-ensemble NVT --quench-T-start 2000 --quench-T-end 300 --quench-T-step -100 --quench-steps-per-T 200 \
    --eq-low-ensemble NVT --eq-low-T 300 --eq-low-steps 4000 -C cubic --resume
amorphgen --analyse --input-dir $OUT/anneal20ps/final --sq --sq-weighting neutron --rings --connectivity \
    --check-dimers --reference examples/reference_a_SiO2.yaml \
    --save-report $OUT/anneal20ps/analysis/report.txt --save-plot $OUT/anneal20ps/analysis
