#!/bin/bash
# ----------------------------------------------------------------------------
# AmorphGen --batch-quench job-array on BlueBEAR (HPC, University of Birmingham)
#
# Runs stages 5, 6, 7 (quench -> equilibrate-low -> final-opt) on a directory
# of snapshot files extracted from a stage-4 trajectory. One snapshot per
# array task, one GPU per task, fully parallel.
#
# Layout this script expects (from the working directory):
#   ./snapshots/snapshot_NNNN_frame*.{xyz,extxyz}
#   ./mq_stages_567_cuda.yaml          # config for stages 5/6/7
#   ./run_quench_array_bluebear.slurm  # this file
#
# After submission each task writes:
#   ./quench_runs/struct_NNNN/run_0000/final_amorphous.xyz
#   ./quench_runs/struct_NNNN/run_0000/stage{5,6,7}_*.xyz
# ----------------------------------------------------------------------------

#SBATCH --job-name=amg_quench
#SBATCH --account=scanlodo-amorphous-oxides
#SBATCH --qos=bbgpu
#SBATCH --time=12:00:00
#SBATCH --array=0-19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/quench_%A_%a.out
#SBATCH --error=logs/quench_%A_%a.err

set -eu

# ── Environment ────────────────────────────────────────────────────────────
module purge
module load bear-apps/2023a
module load Miniforge3/24.1.2-0
source activate amorphgen

ROOT=$(pwd)
TASK=$(printf "%04d" ${SLURM_ARRAY_TASK_ID})

# ── Pick this task's snapshot (handles .xyz or .extxyz) ───────────────────
SNAP=$(ls ${ROOT}/snapshots/snapshot_${TASK}_frame*.xyz \
          ${ROOT}/snapshots/snapshot_${TASK}_frame*.extxyz \
          2>/dev/null | head -1)
if [ -z "${SNAP}" ]; then
    echo "ERROR: no snapshot for task ${TASK} in ${ROOT}/snapshots/" >&2
    exit 1
fi

# Stage just this task's snapshot in a per-task input dir so --snapshot-dir
# picks exactly one. All quench outputs land in the shared quench_runs/,
# where batch_quench creates run_${TASK}/ named after the snapshot index.
INPUT_DIR=${ROOT}/inputs_per_task/task_${TASK}
QUENCH_DIR=${ROOT}/quench_runs
mkdir -p ${INPUT_DIR} ${QUENCH_DIR}
cp -n "${SNAP}" ${INPUT_DIR}/

# ── Diagnostics ────────────────────────────────────────────────────────────
echo "host: $(hostname)"
echo "task: ${TASK}"
echo "snapshot: $(basename ${SNAP})"
echo "output:   ${QUENCH_DIR}/run_${TASK}/"
nvidia-smi -L
echo

# ── Run quench (stages 5, 6, 7) ───────────────────────────────────────────
# batch_quench writes to ${QUENCH_DIR}/run_${TASK}/final_amorphous.xyz
# (named after the parsed snapshot index, not the loop counter).
amorphgen --batch-quench \
  --snapshot-dir ${INPUT_DIR} \
  --config ${ROOT}/mq_stages_567_cuda.yaml \
  --stages 5 6 7 \
  --model chgnet --device cuda \
  -o ${QUENCH_DIR}

echo "Task ${TASK} complete."
