#!/bin/bash
# ----------------------------------------------------------------------------
# AmorphGen — amorphous IrO2 CUBIC-CELL-RELAX control on BlueBear (HPC, Birmingham)
#
# Same as run_iro2_validation_bluebear.slurm, but with CUBIC cell relaxation
# (-C cubic) instead of fixed cell (-C none). The cell volume relaxes (shape
# stays cubic), so MACE-MPA-0 chooses the density.
#
# PURPOSE — this is the over-expansion control: for IrO2 the foundation MLIP
# OVER-EXPANDS under cell relaxation, dropping the density from the physical
# ~10.2 g/cm^3 toward ~8.4 (the failure Li et al. 2025 document). Compare to
# the fixed-cell run: the local structure (RDF/CN) stays AIMD-like, but only
# the density is wrong — proving the density must come from the estimator, not
# the MLIP's energy-volume response.
#
# REQUIRES: the BlueBear `amorphgen` install must include this session's radii
# updates (Ir Shannon radius + rutile_dioxide class). Verify with:
#   python -c "from amorphgen.utils.radii import estimate_cell_length as e; \
#              print(e({'Ir':40,'O':80}))"   # expect ~11.3 A (rho 10.23)
# ----------------------------------------------------------------------------

#SBATCH --job-name=amg_iro2_cubic
#SBATCH --account=scanlodo-amorphous-oxides
#SBATCH --qos=bbgpu
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/iro2_cubic_%j.out
#SBATCH --error=logs/iro2_cubic_%j.err

set -eu

module purge
module load bear-apps/2023a/live
module load Python/3.11.3-GCCcore-12.3.0
source /rds/projects/s/scanlodo-amorphous-oxides/kaewmeec/venvs/amorphgen/bin/activate

mkdir -p logs
echo "host: $(hostname)"; nvidia-smi -L

# Sanity-check the density estimate before committing GPU hours
python -c "from amorphgen.utils.radii import estimate_cell_length, _classify_compound; \
c={'Ir':40,'O':80}; print('class:', _classify_compound(c), '| L =', round(estimate_cell_length(c),2), 'A (expect ~11.3)')"

# 20 structures, Ir40O80 (120 atoms), CUBIC cell relaxation (volume free)
amorphgen --random-gen --composition "IrO2*40" -n 20 --relax \
    --model mace-mpa-0 --device cuda --default-dtype float64 \
    -C cubic -f 0.05 --opt-steps 300 \
    --format vasp -o iro2_cubic/

echo "Done. Relaxed structures in iro2_cubic/random_opt/"

# Analyse + compare to the AIMD reference (expect over-expanded density)
amorphgen --analyse --input-dir iro2_cubic/random_opt/ \
    --reference examples/reference_a_IrO2.yaml \
    -o iro2_cubic/analysis/ || true
