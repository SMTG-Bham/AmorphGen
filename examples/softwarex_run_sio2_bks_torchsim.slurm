#!/bin/bash
# a-SiO2 with the classical BKS potential on the BATCHED torch-sim engine (GPU): the same
# seeds, cell, density and protocol as softwarex_run_sio2_bks.slurm (CPU, ASE engine), so the two are a
# cross-check of the torch-sim Buckingham + Ewald model and a timing comparison (Section 2.2
# of the paper: 20 anneals in one batched job against one anneal per CPU task).
# Output sio2_buckingham_ts/ in the layout softwarex_analyse_oxide.slurm expects
# (sbatch softwarex_analyse_oxide.slurm SIO2 buckingham_ts).
#
# Array tasks:  0  random placement (100, ~2 min each on CPU) + batched relaxation
#               1  hybrid anneal of indices 0-19 in one batch (waits for the seeds)
#
#   sbatch --account=your-project softwarex_run_sio2_bks_torchsim.slurm
#   sbatch --account=your-project --array=0 softwarex_run_sio2_bks_torchsim.slurm     # seeds + relax only
#
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --job-name=sio2_bks_ts
# Export AMORPHGEN_VENV=/path/to/venv before submitting (see docs/guides/hpc.md).
# AMORPHGEN_ROOT is the AmorphGen checkout (default: the submission directory or its parent);
# softwarex_bks_sio2.yaml is read from examples/ there.  Output goes to the submission directory.
# Choose your account at submission: sbatch --account=your-project <this-script>
# Adapt the BlueBEAR modules and QoS below to your cluster.
#SBATCH --qos=bbgpu
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=logs/sio2_bks_ts_%A_%a.out
set -euo pipefail
PKG=${AMORPHGEN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
for up in . .. ../..; do
    [[ -f "$PKG/$up/amorphgen/slurm_runtime.sh" ]] && { PKG=$(cd "$PKG/$up" && pwd); break; }
done
source "$PKG/amorphgen/slurm_runtime.sh"
amorphgen_slurm_main "$0" "$@"
mkdir -p logs
amorphgen_slurm_bluebear_environment "bear-apps/2024a/live" "Python/3.12.3-GCCcore-13.3.0"
python -c "import amorphgen; print('amorphgen', amorphgen.__version__, 'from', amorphgen.__file__)"

COMP="SiO2*216"; RHO=2.20; SEED=2026; N_RANDOM=100
CFG="$PKG/examples/softwarex_bks_sio2.yaml"
ENGINE="--engine torchsim --batch-size auto"
OUT=sio2_buckingham_ts
T=${SLURM_ARRAY_TASK_ID:-0}
echo "=== node $(hostname), task $T, $COMP at $RHO g/cm3, BKS on torch-sim ($CFG), $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [ "$T" = 0 ]; then
    # Same seed as the CPU run, so the placed seeds are identical; batched relaxation on the GPU.
    amorphgen --random-gen --composition "$COMP" -n $N_RANDOM --seed $SEED \
        --target-density $RHO --retry-mode none \
        --relax --config "$CFG" -d cuda --default-dtype float64 $ENGINE \
        -C none -f 0.05 --opt-steps 1000 --format xyz -o "$OUT" --resume
    echo "=== task 0 done $(date): $(ls $OUT/random_opt/*_opt.xyz | wc -l) / $N_RANDOM relaxed"
    exit 0
fi

W=$OUT/hybrid/task_1
mkdir -p "$W/input"
for i in $(seq 0 19); do
    f=$(printf "$OUT/random_opt/random_%04d_opt.xyz" $i)
    until [ -f "$f" ]; do echo "waiting for $f ($(date))"; sleep 300; done
    ln -sf "$(realpath "$f")" "$W/input/"
done
echo "hybrid task 1: indices 0-19, $(ls $W/input | wc -l) seeds"

# 0.5 fs: 100000 steps = 50 ps at 3000 K; quench 3000 -> 300 K in -100 K steps of 2000 steps
# = 100 K/ps (54000 steps, 27 ps); 20000 steps = 10 ps at 300 K; final relax at fixed cell.
amorphgen --hybrid-ensemble --input-dir "$W/input" -o "$W/run" \
    --config "$CFG" -d cuda --default-dtype float32 --seed $((SEED + 1)) $ENGINE \
    --timestep 0.5 \
    --eq-high-ensemble NVT --eq-high-T 3000 --eq-high-steps 100000 \
    --quench-ensemble NVT --quench-T-start 3000 --quench-T-end 300 --quench-T-step -100 --quench-steps-per-T 2000 \
    --eq-low-ensemble NVT --eq-low-T 300 --eq-low-steps 20000 \
    -C none -f 0.05 --format xyz --resume

mkdir -p "$OUT/hybrid/final"
for g in "$W"/run/final/*.xyz; do ln -sf "$(realpath "$g")" "$OUT/hybrid/final/task1_$(basename "$g")"; done
echo "=== task 1 done $(date): hybrid finals $(ls $OUT/hybrid/final | wc -l)"
