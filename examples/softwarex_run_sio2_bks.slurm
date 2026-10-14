#!/bin/bash
# a-SiO2 with the classical BKS potential (softwarex_bks_sio2.yaml) under the SAME protocol as the
# MLIP run (softwarex_run_oxide.slurm SIO2): SiO2*216 = 648 atoms pinned at 2.20 g/cm3, 100 random +
# relax, 20 hybrids (seeds = indices 0-19) with NVT 3000 K 50 ps -> quench to 300 K at
# 100 K/ps -> 300 K 10 ps -> relax, 0.5 fs.  Output sio2_buckingham/, the layout
# softwarex_analyse_oxide.slurm expects (sbatch softwarex_analyse_oxide.slurm SIO2 buckingham).
#
# Differences forced by the potential:
#   - CPU, ASE engine, one structure per process (the NumPy Buckingham calculator).  ~0.2 s/step
#     for 648 atoms (Ewald), so one hybrid structure is 174 000 steps ~ 10-15 h; each hybrid is
#     its own array task.  softwarex_run_sio2_bks_torchsim.slurm runs the same campaign batched on a GPU.
#   - cell filter none everywhere: the NumPy calculator has no stress, so the cell stays at 2.20.
#     softwarex_run_sio2_bks_cubic.slurm relaxes the cell afterwards with the torch-sim model.
#   - BKS is stiff: its simulated glass transition is ~3300 K (Vollmayr, Kob & Binder 1996),
#     so at 3000 K it is barely liquid and the anneal relaxes less than it does for the MLIP.
#     That is part of the comparison; do not raise T to compensate (the unmodified BKS
#     potential can collapse pairs above ~4000 K).
#
# Array tasks:  0-3   random placement + relaxation of indices 25*T .. 25*T+24   (~1-2 h each)
#               4-23  hybrid anneal of index T-4 (waits for its seed file)       (~10-15 h each)
#
#   sbatch --account=your-project softwarex_run_sio2_bks.slurm               # everything
#   sbatch --account=your-project --array=0-3 softwarex_run_sio2_bks.slurm   # random + relax only
#   sbatch --account=your-project --array=4-8 softwarex_run_sio2_bks.slurm   # pilot: 5 hybrids
#
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --job-name=sio2_bks
# Export AMORPHGEN_VENV=/path/to/venv before submitting (see docs/guides/hpc.md).
# AMORPHGEN_ROOT is the AmorphGen checkout (default: the submission directory or its parent);
# softwarex_bks_sio2.yaml is read from examples/ there.  Output goes to the submission directory.
# Choose your account at submission: sbatch --account=your-project <this-script>
# Adapt the BlueBEAR modules and QoS below to your cluster.
#SBATCH --qos=bbdefault
#SBATCH --time=36:00:00
#SBATCH --array=0-23
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/sio2_bks_%A_%a.out
set -euo pipefail
PKG=${AMORPHGEN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
for up in . .. ../..; do
    [[ -f "$PKG/$up/amorphgen/slurm_runtime.sh" ]] && { PKG=$(cd "$PKG/$up" && pwd); break; }
done
source "$PKG/amorphgen/slurm_runtime.sh"
amorphgen_slurm_main "$0" "$@"
mkdir -p logs
amorphgen_slurm_bluebear_environment "bear-apps/2024a/live" "Python/3.12.3-GCCcore-13.3.0"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
python -c "import amorphgen; print('amorphgen', amorphgen.__version__, 'from', amorphgen.__file__)"

COMP="SiO2*216"; RHO=2.20; SEED=2026; N_RANDOM=100
CFG="$PKG/examples/softwarex_bks_sio2.yaml"
OUT=sio2_buckingham
T=${SLURM_ARRAY_TASK_ID:-0}
echo "=== node $(hostname), task $T, $COMP at $RHO g/cm3, BKS ($CFG), $(date)"

if [ "$T" -le 3 ]; then
    LO=$((T*25)); HI=$((T*25+24))
    # Same seed and index set as the MLIP run, so the placed seeds are identical; only the
    # relaxation differs.  Fixed cell (no stress in the calculator).
    # --random-gen locks its output directory, so each task gets its own (gen_T) and the
    # results are linked into one random_initial/ + random_opt/ for the hybrids and the analysis.
    amorphgen --random-gen --composition "$COMP" -n $N_RANDOM --indices $LO-$HI --seed $SEED \
        --target-density $RHO --retry-mode none \
        --relax --config "$CFG" -d cpu \
        -C none -f 0.05 --opt-steps 1000 --format xyz -o "$OUT/gen_$T" --resume
    mkdir -p "$OUT/random_initial" "$OUT/random_opt"
    for f in "$OUT"/gen_$T/random_initial/*.xyz; do ln -sf "$(realpath "$f")" "$OUT/random_initial/"; done
    for f in "$OUT"/gen_$T/random_opt/*_opt.xyz;  do ln -sf "$(realpath "$f")" "$OUT/random_opt/"; done
    echo "=== task $T done $(date): $(ls $OUT/random_opt/*_opt.xyz | wc -l) / $N_RANDOM relaxed so far"
    exit 0
fi

# --- hybrid anneal of one structure --------------------------------------------------------
I=$((T-4))
W=$OUT/hybrid/task_$T
mkdir -p "$W/input"
f=$(printf "$OUT/random_opt/random_%04d_opt.xyz" $I)
until [ -f "$f" ]; do echo "waiting for $f ($(date))"; sleep 300; done
ln -sf "$(realpath "$f")" "$W/input/"
echo "hybrid task $T: index $I"

# 0.5 fs: 100000 steps = 50 ps at 3000 K; quench 3000 -> 300 K in -100 K steps of 2000 steps
# = 100 K/ps (54000 steps, 27 ps); 20000 steps = 10 ps at 300 K; final relax at fixed cell.
amorphgen --hybrid-ensemble --input-dir "$W/input" -o "$W/run" \
    --config "$CFG" -d cpu --seed $((SEED + T)) \
    --timestep 0.5 \
    --eq-high-ensemble NVT --eq-high-T 3000 --eq-high-steps 100000 \
    --quench-ensemble NVT --quench-T-start 3000 --quench-T-end 300 --quench-T-step -100 --quench-steps-per-T 2000 \
    --eq-low-ensemble NVT --eq-low-T 300 --eq-low-steps 20000 \
    -C none -f 0.05 --format xyz --resume

mkdir -p "$OUT/hybrid/final"
for g in "$W"/run/final/*.xyz; do ln -sf "$(realpath "$g")" "$OUT/hybrid/final/task${T}_$(basename "$g")"; done
echo "=== task $T done $(date): hybrid finals so far $(ls $OUT/hybrid/final | wc -l)"
