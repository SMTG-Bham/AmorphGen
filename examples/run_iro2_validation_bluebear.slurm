#!/bin/bash
# ----------------------------------------------------------------------------
# AmorphGen — amorphous IrO2 validation ensemble on BlueBear (HPC, Birmingham)
#
# Generates 20 amorphous IrO2 structures (Ir40O80, 120 atoms) by random
# placement + fixed-cell MACE relaxation, for benchmarking against the AIMD
# reference of Li, An, Guo, Yang & Wang (2025): density 10.04 g/cm^3,
# RDF first peaks Ir-O 1.96, O-O 2.76, Ir-Ir 3.48 A.
#
# WHY fixed cell (-C none): foundation MLIPs over-expand IrO2 under cell
# relaxation (Li et al.; reproduced locally). The physical density comes from
# AmorphGen's class-aware estimator (rutile_dioxide -> 10.23 g/cm^3) and the
# cell is held, so MACE only relaxes the local structure. This is the
# NVT-quench-style route Li et al. validated.
#
# REQUIRES: the BlueBear `amorphgen` install must include this session's radii
# updates (Ir Shannon radius + rutile_dioxide class) — otherwise the auto
# density falls back to ~5.6 g/cm^3 and the run is invalid. Verify with:
#   python -c "from amorphgen.utils.radii import estimate_cell_length as e; \
#              print(e({'Ir':40,'O':80}))"   # expect ~11.3 A (rho 10.23)
# ----------------------------------------------------------------------------

#SBATCH --job-name=amg_iro2_val
#SBATCH --account=scanlodo-amorphous-oxides
#SBATCH --qos=bbgpu
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/iro2_val_%j.out
#SBATCH --error=logs/iro2_val_%j.err

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

# 20 structures, Ir40O80 (120 atoms), fixed-cell MACE relax at the auto density
amorphgen --random-gen --composition "IrO2*40" -n 20 --relax \
    --model mace-mpa-0 --device cuda --default-dtype float64 \
    -C none -f 0.05 --opt-steps 300 \
    --format vasp -o iro2_validation/

echo "Done. Relaxed structures in iro2_validation/random_opt/"

# Optional: analyse + compare to the AIMD reference YAML
amorphgen --analyse --input-dir iro2_validation/random_opt/ \
    --reference examples/reference_a_IrO2.yaml \
    -o iro2_validation/analysis/ || true
