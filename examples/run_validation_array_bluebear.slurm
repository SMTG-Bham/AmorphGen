#!/bin/bash
# ----------------------------------------------------------------------------
# AmorphGen validation runs on BlueBEAR — random-gen + CHGNet relaxation for
# three reference amorphous systems (a-SiO2, a-Si, a-HfO2) used to validate
# coordination, bond length, and density against experimental references.
#
# Submitted as a 3-task array; each task processes one composition and
# generates 20 independent structures.
#
# Output (under the working directory):
#     validation_extra/sio2/random_NNNN_opt.vasp     # 20 a-SiO2 (Si16O32)
#     validation_extra/si/random_NNNN_opt.vasp       # 20 a-Si  (Si64)
#     validation_extra/hfo2/random_NNNN_opt.vasp     # 20 a-HfO2 (Hf32O64)
# ----------------------------------------------------------------------------

#SBATCH --job-name=amg_validate
#SBATCH --account=scanlodo-amorphous-oxides
#SBATCH --qos=bbgpu
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/validate_%A_%a.out
#SBATCH --error=logs/validate_%A_%a.err

set -eu

module purge
module load bear-apps/2023a
module load Miniforge3/24.1.2-0
source activate amorphgen

# Per-task composition + tag
declare -a COMPS=("SiO2*16" "Si*64"   "HfO2*32")
declare -a TAGS=( "sio2"     "si"     "hfo2")

COMP="${COMPS[$SLURM_ARRAY_TASK_ID]}"
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"
ROOT=$(pwd)
OUT="${ROOT}/validation_extra/${TAG}"

mkdir -p "${OUT}"
echo "Task $SLURM_ARRAY_TASK_ID: composition=${COMP}, output=${OUT}"
nvidia-smi -L

amorphgen --random-gen --composition "${COMP}" \
    -n 20 --relax --model chgnet --device cuda \
    --format vasp -o "${OUT}"

echo "Task $SLURM_ARRAY_TASK_ID complete."
