#!/bin/bash
# ----------------------------------------------------------------------------
# Benchmark: ASE engine vs torch-sim engine for a 20 x 576-atom a-GeO2 ensemble
# (random-gen + MACE-MPA-0 relaxation) on one BlueBEAR GPU.
#
# torch-sim needs Python >= 3.12. The existing amorphgen venv is Python 3.11,
# so this script uses a SEPARATE venv (see the one-off setup block below).
#
# Output (under the submission directory):
#     bench_torchsim/ase/random_opt/       20 relaxed structures, ASE path
#     bench_torchsim/torchsim/random_opt/  same 20 seeds, torch-sim batch
#     logs/bench_torchsim_<jobid>.out      timings of both runs
# ----------------------------------------------------------------------------

#SBATCH --job-name=amg_torchsim
#SBATCH --account=scanlodo-amorphous-oxides
#SBATCH --qos=bbgpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/bench_torchsim_%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1     # show progress lines in the log as they happen
mkdir -p logs

module purge
module load bear-apps/2023b/live        # provides Python 3.12; check with `module avail Python`
module load Python/3.12.3-GCCcore-13.2.0
VENV=/rds/projects/s/scanlodo-amorphous-oxides/kaewmeec/venvs/amorphgen-py312

# ── one-off setup (run interactively once, then comment out) ─────────────────
# python -m venv $VENV
# source $VENV/bin/activate
# pip install --upgrade pip
# pip install -e "/rds/projects/s/scanlodo-amorphous-oxides/kaewmeec/AmorphGen[mace,torchsim]"
# python -c "import torch, torch_sim; print(torch.__version__, torch.cuda.is_available(), torch_sim.__version__)"
# ─────────────────────────────────────────────────────────────────────────────
source $VENV/bin/activate
python -c "from amorphgen.utils.radii import estimate_cell_length as e; print('radii check', e({'Ir':40,'O':80}))"

COMP="GeO2*192"; N=20; SEED=1
mkdir -p bench_torchsim

# 1. generate the ensemble once, unrelaxed (same seed -> same 20 placements)
amorphgen --random-gen --composition "$COMP" -n $N --seed $SEED -o bench_torchsim/initial

# 2a. ASE engine: one structure after another (MACE-MPA-0, cubic cell filter)
/usr/bin/time -v amorphgen --batch-opt --input-dir bench_torchsim/initial/random_initial \
    -m mace-mpa-0 -d cuda --default-dtype float64 -C cubic -f 0.05 \
    -o bench_torchsim/ase/random_opt 2>&1 | tee logs/bench_ase_$SLURM_JOB_ID.log

# 2b. torch-sim engine: all 20 in one batched call (autobatching by GPU memory)
/usr/bin/time -v amorphgen --batch-opt --input-dir bench_torchsim/initial/random_initial \
    -m mace-mpa-0 -d cuda --default-dtype float64 -C cubic -f 0.05 --engine torchsim \
    -o bench_torchsim/torchsim/random_opt 2>&1 | tee logs/bench_torchsim_$SLURM_JOB_ID.log

# 3. same analysis on both, so the two engines can be compared structure by structure
for eng in ase torchsim; do
    amorphgen --analyse --input-dir bench_torchsim/$eng/random_opt \
        --reference examples/reference_a_GeO2.yaml --connectivity --per-structure \
        --save-report bench_torchsim/$eng/report.txt
done
grep -h "Elapsed (wall clock)" logs/bench_ase_$SLURM_JOB_ID.log logs/bench_torchsim_$SLURM_JOB_ID.log
