#!/bin/bash
# Short-queue test of the batched MD engine: 10 relaxed a-IGZO structures
# (350 atoms) through --hybrid-ensemble --engine torchsim: 2 ps @ 2000 K,
# quench to 300 K, 1 ps @ 300 K, batched relax. Then the same 10 through the
# ASE engine for a timing and statistics comparison.
#SBATCH --job-name=igzo_md_ts
#SBATCH --account=scanlodo-amorphous-oxides
#SBATCH --qos=bbgpu
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=logs/igzo_md_ts_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
mkdir -p logs
module purge
module load bear-apps/2024a/live
module load Python/3.12.3-GCCcore-13.3.0
source /rds/projects/s/scanlodo-amorphous-oxides/venvs/amorphgen-py312/bin/activate
OUT=igzo_100; mkdir -p $OUT/subset10
for i in 0 1 2 3 4 5 6 7 8 9; do cp $OUT/random_opt/random_000${i}_opt.xyz $OUT/subset10/; done
MD="--eq-high-ensemble NVT --eq-high-T 2000 --eq-high-steps 4000 --quench-ensemble NVT --quench-T-start 2000 --quench-T-end 300 --quench-T-step -100 --quench-steps-per-T 100 --eq-low-ensemble NVT --eq-low-T 300 --eq-low-steps 2000 -C cubic -f 0.05"
/usr/bin/time -v amorphgen --hybrid-ensemble --input-dir $OUT/subset10 -o $OUT/md_test_torchsim \
    -m mace-mpa-0 -d cuda --default-dtype float32 --seed 2026 --engine torchsim --batch-size 5 $MD --resume \
    2>&1 | tee logs/igzo_md_ts_tsim_$SLURM_JOB_ID.log
/usr/bin/time -v amorphgen --hybrid-ensemble --input-dir $OUT/subset10 -o $OUT/md_test_ase \
    -m mace-mpa-0 -d cuda --default-dtype float32 --seed 2026 $MD --resume \
    2>&1 | tee logs/igzo_md_ts_ase_$SLURM_JOB_ID.log
for e in torchsim ase; do
    amorphgen --analyse --input-dir $OUT/md_test_$e/final --connectivity --check-dimers --per-structure \
        --save-report $OUT/md_test_$e/report.txt
done
echo "=== wall clock: torch-sim then ASE ==="
grep -h "Elapsed (wall clock)" logs/igzo_md_ts_tsim_$SLURM_JOB_ID.log logs/igzo_md_ts_ase_$SLURM_JOB_ID.log
echo "=== stage-4 temperature, run_0000, last lines (torch-sim / ASE) ==="
tail -2 $OUT/md_test_torchsim/quench_runs/run_0000/stage4_eq.log; tail -2 $OUT/md_test_ase/quench_runs/run_0000/stage4_eq.log
