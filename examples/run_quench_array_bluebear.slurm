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
#   ./quench_runs/run_NNNN/final_amorphous.xyz
#   ./quench_runs/run_NNNN/stage{5,6,7}_*.xyz
# ----------------------------------------------------------------------------

#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --job-name=amg_quench
# Export AMORPHGEN_VENV=/path/to/venv before submitting (see docs/guides/hpc.md).
# For repository files, AMORPHGEN_ROOT defaults to the submission directory.
# Choose your account at submission: sbatch --account=your-project <this-script>
# Adapt the BlueBEAR modules and QoS below to your cluster.
#SBATCH --qos=bbgpu
#SBATCH --time=12:00:00
#SBATCH --array=0-19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/quench_%A_%a.out
#SBATCH --error=logs/quench_%A_%a.err

set -euo pipefail
PKG=${AMORPHGEN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
if [[ ! -f "$PKG/amorphgen/slurm_runtime.sh" && -f "$PKG/../amorphgen/slurm_runtime.sh" ]]; then
    PKG=$(cd "$PKG/.." && pwd)
fi
source "$PKG/amorphgen/slurm_runtime.sh"
amorphgen_slurm_main "$0" "$@"

# ── Environment ────────────────────────────────────────────────────────────
amorphgen_slurm_bluebear_environment "bear-apps/2023a/live" "Python/3.11.3-GCCcore-12.3.0"

ROOT=$(pwd)
TASK=$(printf "%04d" ${SLURM_ARRAY_TASK_ID})

# ── Pick this task's snapshot (handles .xyz or .extxyz) ───────────────────
shopt -s nullglob
SNAPSHOTS=("${ROOT}"/snapshots/snapshot_${TASK}_frame*.xyz
           "${ROOT}"/snapshots/snapshot_${TASK}_frame*.extxyz)
SNAP=${SNAPSHOTS[0]:-}
if [ -z "${SNAP}" ]; then
    echo "ERROR: no snapshot for task ${TASK} in ${ROOT}/snapshots/" >&2
    exit 1
fi

# Stage just this task's snapshot in a per-task input dir so --snapshot-dir
# picks exactly one. Single-input batches write directly to their output
# directory, so every task needs its own run_${TASK}/ directory.
INPUT_DIR=${ROOT}/inputs_per_task/task_${TASK}
QUENCH_DIR=${ROOT}/quench_runs/run_${TASK}
mkdir -p ${INPUT_DIR} ${QUENCH_DIR}
cp -n "${SNAP}" ${INPUT_DIR}/

# ── Diagnostics ────────────────────────────────────────────────────────────
echo "host: $(hostname)"
echo "task: ${TASK}"
echo "snapshot: $(basename ${SNAP})"
echo "output:   ${QUENCH_DIR}/"
nvidia-smi -L
echo

# ── Run quench (stages 5, 6, 7) ───────────────────────────────────────────
# batch_quench writes to ${QUENCH_DIR}/final_amorphous.xyz.
amorphgen --batch-quench --resume --seed "$((2026 + ${SLURM_ARRAY_TASK_ID:-0}))" \
  --snapshot-dir ${INPUT_DIR} \
  --config ${ROOT}/mq_stages_567_cuda.yaml \
  --stages 5 6 7 \
  --model chgnet --device cuda \
  -o ${QUENCH_DIR}

echo "Task ${TASK} complete."
