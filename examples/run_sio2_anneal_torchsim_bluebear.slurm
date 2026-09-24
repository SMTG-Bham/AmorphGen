#!/bin/bash
# a-SiO2 FSDP test, batched: the same 20 ps @ 2000 K anneal as
# run_sio2_anneal_bluebear.slurm but through torch-sim for all 10 structures
# at once (chunks of 5 x 600 atoms, float32 MD). Compare wall time and the
# analysis report with the ASE version. Resumable at run level.
#SBATCH --job-name=sio2_ann_ts
#SBATCH --account=scanlodo-amorphous-oxides
#SBATCH --qos=bbgpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=logs/sio2_anneal_ts_%j.out
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
/usr/bin/time -v amorphgen --hybrid-ensemble --input-dir $OUT/subset10 -o $OUT/anneal20ps_torchsim \
    -m mace-mpa-0 -d cuda --default-dtype float32 --seed 2026 --engine torchsim --batch-size 5 \
    --eq-high-ensemble NVT --eq-high-T 2000 --eq-high-steps 40000 \
    --quench-ensemble NVT --quench-T-start 2000 --quench-T-end 300 --quench-T-step -100 --quench-steps-per-T 200 \
    --eq-low-ensemble NVT --eq-low-T 300 --eq-low-steps 4000 -C cubic --resume
amorphgen --analyse --input-dir $OUT/anneal20ps_torchsim/final --sq --sq-weighting neutron --rings --connectivity \
    --check-dimers --reference examples/reference_a_SiO2.yaml \
    --save-report $OUT/anneal20ps_torchsim/analysis/report.txt --save-plot $OUT/anneal20ps_torchsim/analysis
grep -h "Elapsed (wall clock)" logs/sio2_anneal_ts_$SLURM_JOB_ID.out || true
