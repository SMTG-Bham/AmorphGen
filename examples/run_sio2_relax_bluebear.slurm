#!/bin/bash
# 2/3  Relax the 100 a-SiO2 placements with MACE-MPA-0 through torch-sim
#      (chunks of 4 x 600 atoms, float64), then analyse: neutron S(q) by the
#      direct method, rings, connectivity, dimers, reference ranges.
#      Resumable: resubmit if the walltime hits.
#SBATCH --job-name=sio2_relax
# Export AMORPHGEN_VENV=/path/to/venv before submitting (see docs/guides/hpc.md).
# For repository files, AMORPHGEN_ROOT defaults to the submission directory.
# Choose your account at submission: sbatch --account=your-project <this-script>
# Adapt the BlueBEAR modules and QoS below to your cluster.
#SBATCH --qos=bbgpu
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=logs/sio2_relax_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
mkdir -p logs
module purge
module load bear-apps/2024a/live
module load Python/3.12.3-GCCcore-13.3.0
source "${AMORPHGEN_VENV:?Set AMORPHGEN_VENV to your virtualenv directory}/bin/activate"
OUT=sio2_100
amorphgen --batch-opt --input-dir $OUT/random_initial -o $OUT/random_opt \
    -m mace-mpa-0 -d cuda --default-dtype float64 -C cubic -O LBFGS -f 0.05 --opt-steps 1000 \
    --engine torchsim --batch-size 4 --resume
echo "relaxed: $(ls $OUT/random_opt/*_opt.xyz | wc -l) / 100"
amorphgen --analyse --input-dir $OUT/random_opt --sq --sq-weighting neutron --rings --connectivity \
    --check-dimers --per-structure --reference examples/reference_a_SiO2.yaml \
    --save-report $OUT/analysis/report.txt --save-plot $OUT/analysis
