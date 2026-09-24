#!/bin/bash
# ----------------------------------------------------------------------------
# torch-sim throughput on one BlueBEAR GPU: how many 350-atom a-IGZO structures
# can be relaxed per 10 minutes, as a function of the batch (chunk) size.
# Generates 32 placements once, then relaxes them with chunk sizes 4, 8, 16, 32
# (each chunk = one batched MACE call per optimiser step) and with the ASE
# engine for reference. Prints wall time and structures per 10 min for each.
# ----------------------------------------------------------------------------

#SBATCH --job-name=tsim_scale
#SBATCH --account=scanlodo-amorphous-oxides
#SBATCH --qos=bbgpu
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=logs/tsim_scale_%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1     # show progress lines in the log as they happen
mkdir -p logs
module purge
module load bear-apps/2024a/live
module load Python/3.12.3-GCCcore-13.3.0
source /rds/projects/s/scanlodo-amorphous-oxides/venvs/amorphgen-py312/bin/activate
python -c "import torch, torch_sim; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| torch-sim', torch_sim.__version__)"

COMP="InGaZnO4*50"; N=32; OUT=tsim_scale
mkdir -p $OUT
amorphgen --random-gen --composition "$COMP" -n $N --seed 1 --target-density 6.1 -o $OUT/initial

for BS in 4 8 16 32; do
    /usr/bin/time -f "%e" -o $OUT/time_bs$BS.txt amorphgen --batch-opt --input-dir $OUT/initial/random_initial \
        -m mace-mpa-0 -d cuda --default-dtype float64 -C cubic -O LBFGS -f 0.05 --opt-steps 1000 \
        --engine torchsim --batch-size $BS -o $OUT/torchsim_bs$BS > logs/tsim_scale_bs${BS}_$SLURM_JOB_ID.log 2>&1
done
/usr/bin/time -f "%e" -o $OUT/time_ase.txt amorphgen --batch-opt --input-dir $OUT/initial/random_initial \
    -m mace-mpa-0 -d cuda --default-dtype float64 -C cubic -O LBFGS -f 0.05 --opt-steps 1000 \
    -o $OUT/ase > logs/tsim_scale_ase_$SLURM_JOB_ID.log 2>&1

echo "=== $N structures of 350 atoms, MACE-MPA-0, LBFGS, cubic cell ==="
for tag in bs4 bs8 bs16 bs32 ase; do
    t=$(cat $OUT/time_$tag.txt); printf "%-6s  %7.0f s  ->  %5.1f structures per 10 min\n" $tag $t $(python -c "print($N*600/$t)")
done
grep -h "torch-sim\] done" logs/tsim_scale_bs*_$SLURM_JOB_ID.log | head -8
