#!/bin/bash
# Cubic-cell relaxation of the a-SiO2 BKS ensembles (sio2_buckingham/), i.e. the density the
# potential prefers (Figure 5 of the paper).  The CPU BKS campaign (softwarex_run_sio2_bks.slurm) ran
# through the NumPy calculator, which has no stress, so every cell was held at the generation
# density 2.20 g/cm3.  The torch-sim Buckingham model computes the stress, so the same
# structures can be relaxed with the isotropic cell filter on CPU: 648 atoms, all structures
# of a set in one batch.  Same softwarex_bks_sio2.yaml (Ewald, repulsive core).
#
# Array tasks:  0  the 20 hybrid finals   -> sio2_buckingham/hybrid_cubic/
#               1  the 20 paired random seeds (indices 0-19 of random_opt) -> sio2_buckingham/random_opt_paired_cubic/
#               2  all 100 random seeds    -> sio2_buckingham/random_opt_cubic/   (optional, ~5x task 1)
#
#   sbatch --account=your-project softwarex_run_sio2_bks_cubic.slurm             # tasks 0-1
#   sbatch --account=your-project --array=2 softwarex_run_sio2_bks_cubic.slurm   # the full random set as well
# For the GPU-engine set, SRC=sio2_buckingham_ts sbatch ... (exported variables may not survive
# sbatch on every site; edit SRC below if yours drops them).
#
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --job-name=sio2_bks_cubic
# Export AMORPHGEN_VENV=/path/to/venv before submitting (see docs/guides/hpc.md).
# AMORPHGEN_ROOT is the AmorphGen checkout (default: the submission directory or its parent);
# softwarex_bks_sio2.yaml is read from examples/ there.  Output goes to the submission directory.
# Choose your account at submission: sbatch --account=your-project <this-script>
# Adapt the BlueBEAR modules and QoS below to your cluster.
#SBATCH --qos=bbdefault
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/sio2_bks_cubic_%A_%a.out
set -euo pipefail
PKG=${AMORPHGEN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
for up in . .. ../..; do
    [[ -f "$PKG/$up/amorphgen/slurm_runtime.sh" ]] && { PKG=$(cd "$PKG/$up" && pwd); break; }
done
source "$PKG/amorphgen/slurm_runtime.sh"
amorphgen_slurm_main "$0" "$@"
mkdir -p logs
amorphgen_slurm_bluebear_environment "bear-apps/2024a/live" "Python/3.12.3-GCCcore-13.3.0"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
python -c "import amorphgen; print('amorphgen', amorphgen.__version__, 'from', amorphgen.__file__)"
python -c "import torch; torch.set_num_threads(${SLURM_CPUS_PER_TASK:-16}); print('torch', torch.__version__, 'threads', torch.get_num_threads())"

CFG="$PKG/examples/softwarex_bks_sio2.yaml"
SRC=${SRC:-sio2_buckingham}
T=${SLURM_ARRAY_TASK_ID:-0}
case $T in
  0) IN=$SRC/hybrid/final;         OUT=$SRC/hybrid_cubic;            PAT="*.xyz" ;;
  1) IN=$SRC/random_opt_paired;    OUT=$SRC/random_opt_paired_cubic; PAT="*.xyz" ;;
  2) IN=$SRC/random_opt;           OUT=$SRC/random_opt_cubic;        PAT="*_opt.xyz" ;;
  *) echo "no task $T"; exit 1 ;;
esac
# random_opt_paired/ is made by softwarex_analyse_oxide.slurm (symlinks to the 20 hybrid seeds); make it if absent.
if [ "$T" = 1 ] && [ ! -d "$IN" ]; then
    mkdir -p "$IN"
    for f in "$SRC"/hybrid/task_*/input/*.xyz; do ln -sf "$(realpath "$f")" "$IN/"; done
fi
echo "=== node $(hostname), task $T: cubic-cell BKS relaxation of $IN ($(ls $IN | grep -c xyz) structures) -> $OUT, $(date)"

amorphgen --batch-opt --input-dir "$IN" --pattern "$PAT" \
    --config "$CFG" --engine torchsim --batch-size 20 -d cpu --default-dtype float64 \
    -C cubic -f 0.05 --opt-steps 2000 --format xyz -o "$OUT" --resume

echo "=== task $T done $(date): $(ls $OUT/*_opt.xyz 2>/dev/null | wc -l) relaxed"
python - <<PY
import glob, numpy as np
from ase.io import read
rho = []
for f in sorted(glob.glob("$OUT/*_opt.xyz")):
    a = read(f); rho.append(a.get_masses().sum() / 6.02214076e23 / (a.get_volume() * 1e-24))
rho = np.array(rho)
print(f"density after cubic relaxation: mean {rho.mean():.3f} g/cm3, SD {rho.std():.3f}, range {rho.min():.3f}-{rho.max():.3f} (n={len(rho)}); generation 2.20")
PY
