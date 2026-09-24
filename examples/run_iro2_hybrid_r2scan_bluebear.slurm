#!/bin/bash
# ----------------------------------------------------------------------------
# AmorphGen HYBRID workflow on BlueBear — a-IrO2 with MACE-matpes-r2scan.
#
# Per array task: (1) generate ONE random Ir40O80 structure with the new
# min-CN floor-repair (fixed cell at the estimator density ~10.2 g/cm3), then
# (2) run the hybrid melt-quench (stages 4-7: equilibrate -> quench ->
# equilibrate -> final opt) on it via --hybrid-ensemble.
#
# All MD is NVT / fixed cell (config hybrid_iro2_r2scan.yaml) so the density
# stays at the physical estimator value — the reliable uMLIP route.
#
# 20 structures = 20 array tasks, one GPU each (parallel). Each task ~2-4 h.
#
# Place this script + hybrid_iro2_r2scan.yaml in your run directory (e.g.
# examples_iro2_new/) and submit FROM there:
#   cd examples_iro2_new && sbatch run_iro2_hybrid_r2scan_bluebear.slurm
# Outputs/config are resolved relative to that submit directory.
# Collect:  final structures land in iro2_hybrid_r2scan/task_NN/final/
# ----------------------------------------------------------------------------

#SBATCH --job-name=amg_iro2_hyb_r2s
#SBATCH --account=scanlodo-amorphous-oxides
#SBATCH --qos=bbgpu
#SBATCH --time=06:00:00
#SBATCH --array=0-19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --output=logs/iro2_hyb_r2s_%A_%a.out
#SBATCH --error=logs/iro2_hyb_r2s_%A_%a.err

set -eu

module purge
module load bear-apps/2023a/live
module load Python/3.11.3-GCCcore-12.3.0
source /rds/projects/s/scanlodo-amorphous-oxides/kaewmeec/venvs/amorphgen/bin/activate

# Resolve paths relative to the submit directory (where this + the config live)
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
TASK=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})
WORK=iro2_hybrid_r2scan/task_${TASK}
echo "host: $(hostname) | task ${TASK} | run dir: $(pwd)"; nvidia-smi -L

# Sanity: confirm the floor-repair build + matpes-r2scan dispatch are live
python -c "import amorphgen.pipeline.random_gen as r; assert hasattr(r,'_repair_min_cn'); print('floor-repair: OK')"
python -c "from amorphgen.utils.calculators import _detect_backend as d; print('model dispatch:', d('mace-matpes-r2scan'))"

# (1) Generate ONE disordered Ir40O80 input — fixed cell at the estimator
#     density, min-CN floor-repair on by default. No --relax (placement only);
#     each array task draws a fresh random structure (CLI uses a random seed).
amorphgen --random-gen --composition "IrO2*40" -n 1 \
    --format vasp -o ${WORK}/gen/

# (2) Hybrid melt-quench (stages 4-7) on that structure with matpes-r2scan.
amorphgen --hybrid-ensemble --input-dir ${WORK}/gen/random_initial/ \
    --config hybrid_iro2_r2scan.yaml \
    --model mace-matpes-r2scan --device cuda --default-dtype float64 \
    -o ${WORK}/

echo "task ${TASK} done -> ${WORK}/final/"
