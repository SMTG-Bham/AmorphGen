#!/bin/bash
# SoftwareX paper, oxide systems: random + relax (100 structures) and the hybrid anneal
# (20 structures, seeds = indices 0-19 of the random set, so every annealed structure has a
# before/after partner).  One script for a-Ga2O3, a-TiO2, a-IGZO and a-SiO2; SYSTEM picks the system.
#
# Protocol: 0.5 fs for all MD stages;
#   random + relax : composition placement at the pinned density, LBFGS with a cubic cell filter
#   hybrid         : NVT 3000 K 50 ps -> NVT quench 3000 -> 300 K at 100 K/ps (-100 K steps of
#                    2000 steps) -> NVT 300 K 10 ps -> LBFGS, cubic cell filter
#   = 174 000 MD steps per hybrid structure.
#
# Array tasks:
#   0        random placement of all 100 + batched relaxation (torch-sim)          ~1-2 h
#   1..K     hybrid anneal of HYBRID_PER_TASK structures each (default 10, so tasks 1-2
#            cover indices 0-19; SiO2 defaults to 5 per task because its 648-atom cell costs
#            ~1.6x, so submit --array=0-4); each task waits for its seeds, so all tasks can
#            be submitted together                                                  ~17-20 h
# Everything carries --resume: resubmit a task if the walltime cuts it off.
# Run softwarex_analyse_oxide.slurm when the tasks are done.
#
# The system is the FIRST ARGUMENT and the model the optional second (arguments survive
# sbatch on every site; exported variables may not):
#   sbatch --account=your-project softwarex_run_oxide.slurm GA2O3                     # a-Ga2O3, 100 + 20
#   sbatch --account=your-project softwarex_run_oxide.slurm TIO2                      # a-TiO2
#   sbatch --account=your-project softwarex_run_oxide.slurm IGZO                      # a-IGZO
#   sbatch --account=your-project --array=0-4 softwarex_run_oxide.slurm SIO2          # a-SiO2, 648 atoms, 4 x 5 hybrids
#   sbatch --account=your-project softwarex_run_oxide.slurm GA2O3 chgnet              # second MLIP (ASE engine, slower)
#   sbatch --account=your-project --array=0,1 softwarex_run_oxide.slurm GA2O3 mace-mpa-0 5   # pilot: 100 random + 5 hybrid
#
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --job-name=swx_oxide
# Export AMORPHGEN_VENV=/path/to/venv before submitting (see docs/guides/hpc.md).
# AMORPHGEN_ROOT is the AmorphGen checkout (default: the submission directory or its parent);
# the reference YAML files are read from there.  Output goes to the submission directory.
# Choose your account at submission: sbatch --account=your-project <this-script> SYSTEM
# Adapt the BlueBEAR modules and QoS below to your cluster.
#SBATCH --qos=bbgpu
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=logs/swx_oxide_%A_%a.out
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

# --- system -------------------------------------------------------------------------------
# Densities are pinned to the reference value, so that "does the anneal work" is separated
# from "is the density estimate good" (the estimate is tested in the parity plot).  RHO= overrides.
SYSTEM=${1:?usage: sbatch softwarex_run_oxide.slurm SYSTEM [MODEL] [HYBRID_PER_TASK]  (SYSTEM = GA2O3, TIO2, IGZO, SIO2)}
case $SYSTEM in
  GA2O3) COMP="Ga2O3*80";    RHO=${RHO:-4.84} ;;  # 400 atoms; PBE0 melt-quench reference, PRB 111, 035203 (2025)
  TIO2)  COMP="TiO2*90";     RHO=${RHO:-3.80} ;;  # 270 atoms; a-TiO2 films 3.7-3.9 g/cm3
  IGZO)  COMP="InGaZnO4*50"; RHO=${RHO:-6.10} ;;  # 350 atoms; sputtered a-IGZO films 5.9-6.1 g/cm3
  SIO2)  COMP="SiO2*216";    RHO=${RHO:-2.20}; HYBRID_PER_TASK=${HYBRID_PER_TASK:-5} ;;  # 648 atoms, L = 21.4 A, q_min 0.29 1/A (FSDP at 1.5); fused silica 2.20 g/cm3
  *) echo "unknown SYSTEM=$SYSTEM (GA2O3, TIO2, IGZO, SIO2)"; exit 1 ;;
esac
TAG=$(echo "$SYSTEM" | tr 'A-Z' 'a-z')
MODEL=${2:-mace-mpa-0}
case $MODEL in chgnet*) ENGINE=""; DT_RELAX=float32 ;; *) ENGINE="--engine torchsim --batch-size ${BATCH:-10}"; DT_RELAX=float64 ;; esac
# --batch-size is fixed (not auto) so the torch-sim random streams, which depend on the
# chunking, are reproducible between submissions.
N_RANDOM=${N_RANDOM:-100}
HYBRID_PER_TASK=${3:-${HYBRID_PER_TASK:-10}}
SEED=2026
OUT=${TAG}_${MODEL}
T=${SLURM_ARRAY_TASK_ID:-0}

echo "=== node $(hostname), task $T, system $SYSTEM ($COMP at $RHO g/cm3), model $MODEL, $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [ "$T" = 0 ]; then
    # Random placement of all N_RANDOM (automatic minsep and CN targets) + relaxation with a
    # cubic cell filter, batched.  Placement: tens of seconds per 400-atom structure on one CPU.
    amorphgen --random-gen --composition "$COMP" -n $N_RANDOM --seed $SEED \
        --target-density $RHO --retry-mode none \
        --relax -m "$MODEL" -d cuda --default-dtype $DT_RELAX $ENGINE \
        -C cubic -f 0.05 --opt-steps 1000 --format xyz -o "$OUT" --resume
    echo "=== task 0 done $(date): $(ls $OUT/random_opt/*_opt.xyz | wc -l) / $N_RANDOM relaxed"
    exit 0
fi

# --- hybrid anneal of indices LO-HI ------------------------------------------------------
LO=$(( (T-1) * HYBRID_PER_TASK )); HI=$(( T * HYBRID_PER_TASK - 1 ))
W=$OUT/hybrid/task_$T
mkdir -p "$W/input"
for i in $(seq $LO $HI); do
    f=$(printf "$OUT/random_opt/random_%04d_opt.xyz" $i)
    until [ -f "$f" ]; do echo "waiting for $f ($(date))"; sleep 300; done
    ln -sf "$(realpath "$f")" "$W/input/"
done
echo "hybrid task $T: indices $LO-$HI, $(ls $W/input | wc -l) seeds"
if [ -f "$W/.running" ] && kill -0 "$(cat "$W/.running" 2>/dev/null)" 2>/dev/null; then
    echo "another process ($(cat "$W/.running")) is running $W -- refusing to start twice"; exit 1
fi
echo $$ > "$W/.running"; trap 'rm -f "$W/.running"' EXIT

# 0.5 fs: 100000 steps = 50 ps at 3000 K; quench 3000 -> 300 K in -100 K steps of 2000 steps
# = 100 K/ps (54000 steps, 27 ps); 20000 steps = 10 ps at 300 K; final relax, cubic cell.
amorphgen --hybrid-ensemble --input-dir "$W/input" -o "$W/run" \
    -m "$MODEL" -d cuda --default-dtype float32 --seed $((SEED + T)) $ENGINE \
    --timestep 0.5 \
    --eq-high-ensemble NVT --eq-high-T 3000 --eq-high-steps 100000 \
    --quench-ensemble NVT --quench-T-start 3000 --quench-T-end 300 --quench-T-step -100 --quench-steps-per-T 2000 \
    --eq-low-ensemble NVT --eq-low-T 300 --eq-low-steps 20000 \
    -C cubic -f 0.05 --format xyz --resume

# Pool this task's finals so the analysis job sees one folder.
mkdir -p "$OUT/hybrid/final"
for f in "$W"/run/final/*.xyz; do ln -sf "$(realpath "$f")" "$OUT/hybrid/final/task${T}_$(basename "$f")"; done
echo "=== task $T done $(date): hybrid finals so far $(ls $OUT/hybrid/final | wc -l)"
