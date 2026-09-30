#!/bin/bash
# Phase-change chalcogenides on one BlueBEAR GPU, one array task per system:
#   task 0: a-GeTe   (GeTe*108  = 216 atoms), anneal 800 K
#   task 1: a-Sb2Te3 (Sb2Te3*44 = 220 atoms), anneal 700 K
# For each: 10 random structures -> batched MACE relaxation (torch-sim) ->
# analysis of the UNANNEALED set -> hybrid anneal + quench + relax (batched,
# 10 ps at the anneal T, 0.5 fs) -> analysis of the ANNEALED set, both with
# neutron S(q) + partials, rings, connectivity, dimers and (GeTe) the
# literature reference. Compare the two analysis reports: the anneal is what
# should build the fourfold-ring, Peierls-distorted local order that random
# placement cannot. ~2 h per task; both commands carry --resume, so resubmit
# after a walltime kill.
#SBATCH --job-name=pcm
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
#SBATCH --array=0-1
#SBATCH --output=logs/pcm_%A_%a.out
set -uo pipefail
export PYTHONUNBUFFERED=1
mkdir -p logs
module purge
module load bear-apps/2024a/live
module load Python/3.12.3-GCCcore-13.3.0
source "${AMORPHGEN_VENV:?Set AMORPHGEN_VENV to your virtualenv directory}/bin/activate"
PKG=${AMORPHGEN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}

case ${SLURM_ARRAY_TASK_ID:-0} in
  0) SYS=gete;   COMP="GeTe*108";  CN="Ge=4,Te=3"; T_ANNEAL=800; RHO=5.7
     REF=(--reference "$PKG/examples/reference_a_GeTe.yaml") ;;
  1) SYS=sb2te3; COMP="Sb2Te3*44"; CN="Sb=3,Te=3"; T_ANNEAL=700; RHO=6.1; REF=() ;;
esac
# Density is pinned to the literature amorphous value (a-GeTe 5.6-5.9,
# a-Sb2Te3 ~6.2 g/cm3; the chalcogenide estimate is 5-10 % low). The minimum
# separations are left at the defaults (~2.2 A, well below the 2.78 A bond):
# the MLIP relaxation sets the bond lengths, and pinning physical floors on
# every pair makes random sequential addition jam near 3.8 g/cm3 and expand
# the cell (10 min per structure, density lost) instead of seconds.
N=10; SEED=2026; MODEL="-m mace-mpa-0 -d cuda --default-dtype float32"
ANALYSE="--sq --sq-weighting neutron --sq-partials --pair-panels --rings --connectivity --check-dimers --per-structure --save-pdf"
echo "=== $SYS: $COMP, $N structures, anneal $T_ANNEAL K, $(date)"

# 1. random placement + batched relaxation (torch-sim, cubic cell filter)
amorphgen --random-gen --composition "$COMP" -n $N --target-cn "$CN" --target-density $RHO \
    --seed $SEED --relax $MODEL --engine torchsim -C cubic -f 0.05 -o $SYS --resume

# 2. the unannealed set: what placement + relaxation alone gives
amorphgen --analyse --input-dir $SYS/random_opt $ANALYSE "${REF[@]}" \
    --save-report $SYS/analysis_unannealed/report.txt --save-plot $SYS/analysis_unannealed/

# 3. anneal at T_ANNEAL (10 ps), quench to 300 K, 1 ps at 300 K, relax; all batched
MD="--eq-high-ensemble NVT --eq-high-T $T_ANNEAL --eq-high-steps 20000 \
    --quench-ensemble NVT --quench-T-start $T_ANNEAL --quench-T-end 300 --quench-T-step -100 --quench-steps-per-T 200 \
    --eq-low-ensemble NVT --eq-low-T 300 --eq-low-steps 2000 -C cubic -f 0.05"
/usr/bin/time -v amorphgen --hybrid-ensemble --input-dir $SYS/random_opt -o $SYS/anneal_${T_ANNEAL}K \
    $MODEL --seed $SEED --engine torchsim --batch-size auto $MD --resume

# 4. the annealed set
amorphgen --analyse --input-dir $SYS/anneal_${T_ANNEAL}K/final $ANALYSE "${REF[@]}" \
    --save-report $SYS/analysis_annealed/report.txt --save-plot $SYS/analysis_annealed/

echo "=== done $(date). Compare: $SYS/analysis_unannealed/report.txt vs $SYS/analysis_annealed/report.txt"
echo "    S(q): analysis_sq.png in each folder; rings: analysis_rings.png (look for the 4-ring share)"
