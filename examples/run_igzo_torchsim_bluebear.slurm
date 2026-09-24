#!/bin/bash
# ----------------------------------------------------------------------------
# torch-sim test on BlueBEAR: 4 random a-InGaZnO4 structures (350 atoms each,
# "InGaZnO4*50") relaxed with MACE-MPA-0, once through the ASE engine and once
# through the batched torch-sim engine, then analysed side by side.
#
# Output (under the submission directory):
#     igzo_torchsim/initial/random_initial/   4 unrelaxed placements (seed 1)
#     igzo_torchsim/ase/random_opt/           relaxed one by one (ASE)
#     igzo_torchsim/torchsim/random_opt/      relaxed as one batch (torch-sim)
#     igzo_torchsim/{ase,torchsim}/report.txt analysis + connectivity
#     logs/igzo_torchsim_<jobid>.out           timings of both engines
# ----------------------------------------------------------------------------

#SBATCH --job-name=igzo_tsim
#SBATCH --account=scanlodo-amorphous-oxides
#SBATCH --qos=bbshort
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/igzo_torchsim_%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1     # show progress lines in the log as they happen
mkdir -p logs

# >>> put the two module lines that `module spider Python/3.12` gave you here <<<
module purge
module load bear-apps/2024a/live
module load Python/3.12.3-GCCcore-13.3.0
source /rds/projects/s/scanlodo-amorphous-oxides/venvs/amorphgen-py312/bin/activate

python -c "import torch, torch_sim; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| torch-sim', torch_sim.__version__)"
python -c "from amorphgen.utils.radii import estimate_cell_length as e; print('radii check', e({'Ir':40,'O':80}))"

COMP="InGaZnO4*50"       # 50 formula units = 350 atoms
N=4
SEED=1
RHO=6.1                  # a-IGZO sputtered-film density (g/cm3); auto estimate is 5.3
OUT=igzo_torchsim_v2     # v2: LBFGS on both engines, cell forces in torch-sim's convergence test
mkdir -p $OUT

# 1. the same 4 placements for both engines
amorphgen --random-gen --composition "$COMP" -n $N --seed $SEED --target-density $RHO -o $OUT/initial

# 2a. ASE engine (one structure after another)
/usr/bin/time -v amorphgen --batch-opt --input-dir $OUT/initial/random_initial \
    -m mace-mpa-0 -d cuda --default-dtype float64 -C cubic -O LBFGS -f 0.05 --opt-steps 1000 \
    -o $OUT/ase/random_opt 2>&1 | tee logs/igzo_ase_$SLURM_JOB_ID.log

# 2b. torch-sim engine (all 4 in one batched call)
/usr/bin/time -v amorphgen --batch-opt --input-dir $OUT/initial/random_initial \
    -m mace-mpa-0 -d cuda --default-dtype float64 -C cubic -O LBFGS -f 0.05 --opt-steps 1000 --engine torchsim \
    -o $OUT/torchsim/random_opt 2>&1 | tee logs/igzo_tsim_$SLURM_JOB_ID.log

# 3. same analysis on both sets
for eng in ase torchsim; do
    amorphgen --analyse --input-dir $OUT/$eng/random_opt --per-structure --check-dimers \
        --connectivity --rings --save-report $OUT/$eng/report.txt --save-plot $OUT/$eng/plots
done

echo "=== wall clock ==="
grep -h "Elapsed (wall clock)" logs/igzo_ase_$SLURM_JOB_ID.log logs/igzo_tsim_$SLURM_JOB_ID.log
echo "=== steps / convergence ==="
grep -h "Converged after\|reached the maximum" logs/igzo_ase_$SLURM_JOB_ID.log logs/igzo_tsim_$SLURM_JOB_ID.log
grep -h "torch-sim\]" logs/igzo_tsim_$SLURM_JOB_ID.log
echo "=== per-structure density / E per atom (ASE vs torch-sim) ==="
grep -A9 "Per-Structure" $OUT/ase/report.txt | head -12
grep -A9 "Per-Structure" $OUT/torchsim/report.txt | head -12
echo "=== residual pressure with MACE-MPA-0 (both engines should be ~0 GPa) ==="
python - "$OUT" <<'EOP'
import sys, glob, numpy as np
from ase.io import read
from ase import units
from mace.calculators import mace_mp
out = sys.argv[1]; calc = mace_mp(model="medium-mpa-0", default_dtype="float64", device="cuda")
for eng in ("ase", "torchsim"):
    for f in sorted(glob.glob(f"{out}/{eng}/random_opt/*_opt.xyz")):
        a = read(f); a.calc = calc
        P = -np.trace(a.get_stress(voigt=False)) / 3 / units.GPa
        rho = a.get_masses().sum() / units.mol * 1e24 / a.get_volume()
        print(f"{eng:8s} {f.split('/')[-1]:22s} rho {rho:.3f} g/cm3  E/atom {a.get_potential_energy()/len(a):.4f} eV  "
              f"max|F| {np.abs(a.get_forces()).max():.3f}  P {P:+.3f} GPa")
EOP
