#!/bin/bash
# a-SiO2, the slow protocol: is the missing first sharp diffraction peak a matter of the
# protocol or of the potential?  Ten 648-atom seeds (indices 0-9 of the softwarex_run_oxide.slurm /
# softwarex_run_sio2_bks.slurm sets), hybrid route with the cooling rate of the silica literature
# (10 K/ps instead of 100 K/ps), batched on torch-sim.  Compare with the 100 K/ps sets
# (sio2_mace-mpa-0, sio2_buckingham).
#
#   task  potential             hold            quench 10 K/ps        steps/structure   out
#   0     MACE-MPA-0            3000 K, 50 ps   3000 -> 300 K, 270 ps   ~0.66 M        sio2_mace-mpa-0_slow
#   1     BKS (classical)       5000 K, 50 ps   5000 -> 300 K, 470 ps   ~1.06 M        sio2_buckingham_slow
#   2     MACE-MATPES-r2SCAN    3000 K, 50 ps   3000 -> 300 K, 270 ps   ~0.66 M        sio2_mace-matpes-r2scan_slow
# BKS runs at 5000 K because its temperature scale is ~2x the physical one (its glass transition
# is ~3300 K; the literature melts it at 6000-7000 K); the repulsive core in softwarex_bks_sio2.yaml keeps
# it safe at that temperature.  0.5 fs throughout; each -100 K segment is 20000 steps = 10 ps = 10 K/ps.
#
# One 4-structure batch of the BKS task takes about 20 h on an A100 (1.9 h hold, 17.8 h quench),
# so a 24 h job completes one batch per submission: resubmit the task until all ten finals exist;
# --resume continues each batch from its last saved frame.
#
#   sbatch --account=your-project softwarex_run_sio2_slow.slurm                # all three
#   sbatch --account=your-project --array=0,1 softwarex_run_sio2_slow.slurm    # without the r2SCAN model
# Analysis: sbatch softwarex_analyse_oxide.slurm SIO2 mace-mpa-0_slow   (and buckingham_slow, mace-matpes-r2scan_slow)
#
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --job-name=sio2_slow
# Export AMORPHGEN_VENV=/path/to/venv before submitting (see docs/guides/hpc.md).
# AMORPHGEN_ROOT is the AmorphGen checkout (default: the submission directory or its parent);
# softwarex_bks_sio2.yaml is read from examples/ there.  Output goes to the submission directory.
# Choose your account at submission: sbatch --account=your-project <this-script>
# Adapt the BlueBEAR modules and QoS below to your cluster.
#SBATCH --qos=bbgpu
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=logs/sio2_slow_%A_%a.out
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

T=${SLURM_ARRAY_TASK_ID:-0}
case $T in
  0) CALC="-m mace-mpa-0";                                  SEEDS=sio2_mace-mpa-0/random_opt; OUT=sio2_mace-mpa-0_slow;          THOLD=3000; CF=cubic ;;
  1) CALC="--config $PKG/examples/softwarex_bks_sio2.yaml"; SEEDS=sio2_buckingham/random_opt;  OUT=sio2_buckingham_slow;          THOLD=5000; CF=none ;;
  2) CALC="-m mace-matpes-r2scan";                          SEEDS=sio2_mace-mpa-0/random_opt; OUT=sio2_mace-matpes-r2scan_slow;  THOLD=3000; CF=cubic ;;
  *) echo "task $T not defined"; exit 1 ;;
esac

ENGINE="--engine torchsim --batch-size auto"
SEED=2026
echo "=== node $(hostname), task $T: $CALC, hold $THOLD K, quench 10 K/ps, seeds $SEEDS -> $OUT, $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

mkdir -p "$OUT/input"
for i in 0 1 2 3 4 5 6 7 8 9; do
    f=$(printf "$SEEDS/random_%04d_opt.xyz" $i)
    [ -f "$f" ] || { echo "missing seed $f"; exit 1; }
    ln -sf "$(realpath "$f")" "$OUT/input/"
done

amorphgen --hybrid-ensemble --input-dir "$OUT/input" -o "$OUT/hybrid/task_0/run" \
    $CALC -d cuda --default-dtype float32 --seed $SEED $ENGINE \
    --timestep 0.5 \
    --eq-high-ensemble NVT --eq-high-T $THOLD --eq-high-steps 100000 \
    --quench-ensemble NVT --quench-T-start $THOLD --quench-T-end 300 --quench-T-step -100 --quench-steps-per-T 20000 \
    --eq-low-ensemble NVT --eq-low-T 300 --eq-low-steps 20000 \
    -C $CF -f 0.05 --format xyz --resume

# Same layout as softwarex_run_oxide.slurm so softwarex_analyse_oxide.slurm finds hybrid/final/.
mkdir -p "$OUT/hybrid/final" "$OUT/random_opt"
for g in "$OUT"/hybrid/task_0/run/final/*.xyz; do ln -sf "$(realpath "$g")" "$OUT/hybrid/final/task0_$(basename "$g")"; done
for g in "$OUT"/input/*.xyz; do ln -sf "$(realpath "$g")" "$OUT/random_opt/"; done
echo "=== task $T done $(date): $(ls $OUT/hybrid/final | wc -l) finals"
