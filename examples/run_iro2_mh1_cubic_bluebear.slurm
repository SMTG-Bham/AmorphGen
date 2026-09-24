#!/bin/bash
# ----------------------------------------------------------------------------
# AmorphGen — amorphous IrO2 CUBIC-CELL-RELAX with MACE-MH-1 (multi-domain).
#
# Same as run_iro2_cubic_bluebear.slurm but uses the multi-HEAD model
# MACE-MH-1, selecting the PBE bulk head via the 'model:head' syntax:
#     --model mace-mh-1:omat_pbe
# (omat_pbe matches the PBE functional of the AIMD reference; other heads are
#  mp_pbe_refit_add and matpes_r2scan — see --list-models notes.)
#
# REQUIRES: the BlueBear `amorphgen` install must include the calculators.py
# head-parsing change (the 'model:head' split in _load_mace). Verify with:
#   python -c "from amorphgen.utils.calculators import _detect_backend as d; \
#              print(d('mace-mh-1:omat_pbe'))"   # expect: mace
# ----------------------------------------------------------------------------

#SBATCH --job-name=amg_iro2_mh1
#SBATCH --account=scanlodo-amorphous-oxides
#SBATCH --qos=bbgpu
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/iro2_mh1_%j.out
#SBATCH --error=logs/iro2_mh1_%j.err

set -eu

module purge
module load bear-apps/2023a/live
module load Python/3.11.3-GCCcore-12.3.0
source /rds/projects/s/scanlodo-amorphous-oxides/kaewmeec/venvs/amorphgen/bin/activate

mkdir -p logs
echo "host: $(hostname)"; nvidia-smi -L

# Sanity-check head dispatch + density estimate before committing GPU hours
python -c "from amorphgen.utils.calculators import _detect_backend as d; \
           print('dispatch:', d('mace-mh-1:omat_pbe'))"
python -c "from amorphgen.utils.radii import estimate_cell_length, _classify_compound; \
           c={'Ir':40,'O':80}; print('class:', _classify_compound(c), '| L =', round(estimate_cell_length(c),2), 'A (expect ~11.3)')"

# 20 structures, Ir40O80 (120 atoms), CUBIC cell relaxation (volume free)
amorphgen --random-gen --composition "IrO2*40" -n 20 --relax \
    --model mace-mh-1:omat_pbe --device cuda --default-dtype float64 \
    -C cubic -f 0.05 --opt-steps 500 \
    --format vasp -o iro2_mace-mh-1_cubic/

echo "Done. Relaxed structures in iro2_mace-mh-1_cubic/random_opt/"

# Analyse + compare to the AIMD reference
amorphgen --analyse --input-dir iro2_mace-mh-1_cubic/random_opt/ \
    --reference examples/reference_a_IrO2.yaml \
    -o iro2_mace-mh-1_cubic/analysis/ || true
