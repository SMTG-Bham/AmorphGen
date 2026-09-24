#!/bin/bash
# ----------------------------------------------------------------------------
# AmorphGen — amorphous IrO2 ensemble, CUBIC cell relaxation, as a job ARRAY
# on BlueBear bbshort (quick CPU queue). 20 tasks x 1 structure = 20 samples.
#
# This is the CUBIC-RELAX control run: the cell volume relaxes (shape stays
# cubic), so MACE-MPA-0 is allowed to choose the density. For IrO2 it
# OVER-EXPANDS (density drops from the physical ~10.2 toward ~8.5) — this is
# the foundation-MLIP failure Li et al. (2025) document. Compare against the
# fixed-cell run (run_iro2_validation_bluebear.slurm) to show that the local
# structure (RDF/CN) stays correct while only the density is wrong.
#
# Each task generates a distinct structure: the CLI has no --seed, so each
# process seeds np.random.default_rng(None) from OS entropy -> 20 independent
# samples.
#
# bbshort is CPU-only with a short walltime cap (~10 min). Each task is sized
# to fit: 1 structure, CPU+float32, opt-steps 200. If your bbshort allows more
# time, raise --time and --opt-steps. If you have GPU access on a short queue,
# add `#SBATCH --gres=gpu:1` and switch to `--device cuda --default-dtype float64`.
# ----------------------------------------------------------------------------

#SBATCH --job-name=amg_iro2_cubic
#SBATCH --account=scanlodo-amorphous-oxides
#SBATCH --qos=bbgpu
#SBATCH --time=00:30:00
#SBATCH --array=0-19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=logs/iro2_cubic_%A_%a.out
#SBATCH --error=logs/iro2_cubic_%A_%a.err

set -eu

module purge
module load bear-apps/2023a/live
module load Python/3.11.3-GCCcore-12.3.0
source /rds/projects/s/scanlodo-amorphous-oxides/kaewmeec/venvs/amorphgen/bin/activate

mkdir -p logs
TASK=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})
echo "host: $(hostname) | task ${TASK}"

# Verify the density fix is live (expect rutile_dioxide, L ~11.33)
python -c "from amorphgen.utils.radii import estimate_cell_length as e, _classify_compound as c; \
print('check:', c({'Ir':40,'O':80}), round(e({'Ir':40,'O':80}),2))"

# One IrO2 structure, cubic-cell relaxation
amorphgen --random-gen --composition "IrO2*40" -n 1 --relax \
    --model mace-mpa-0 --device cuda --default-dtype float64 \
    -C cubic -f 0.05 --opt-steps 300 --format vasp \
    -o iro2_cubic/struct_${TASK}/

echo "task ${TASK} done -> iro2_cubic/struct_${TASK}/random_opt/"
