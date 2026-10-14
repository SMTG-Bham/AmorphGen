#!/bin/bash
# Analysis of one system from softwarex_run_oxide.slurm (or the BKS scripts): the placed, relaxed and
# annealed sets with one cutoff rule (first RDF minimum, automatic), rings, polyhedral
# connectivity, voids, dimers, X-ray S(q); the convergence report on the 100 random structures
# (the justification for n = 20 in the MD workflows); and the reference table where a reference
# YAML exists (a-Ga2O3: PBE0 models of PRB 111, 035203; a-SiO2: diffraction consensus).
#
# Arguments: SYSTEM [MODEL]
#   sbatch --account=your-project softwarex_analyse_oxide.slurm GA2O3
#   sbatch --account=your-project softwarex_analyse_oxide.slurm SIO2
#   sbatch --account=your-project softwarex_analyse_oxide.slurm SIO2 buckingham       # the BKS set from softwarex_run_sio2_bks.slurm
#   sbatch --account=your-project softwarex_analyse_oxide.slurm SIO2 buckingham_ts    # the GPU-engine BKS set
#   sbatch --account=your-project softwarex_analyse_oxide.slurm SIO2 buckingham_slow  # the slow-quench set
#
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --job-name=swx_an
# Export AMORPHGEN_VENV=/path/to/venv before submitting (see docs/guides/hpc.md).
# AMORPHGEN_ROOT is the AmorphGen checkout (default: the submission directory or its parent);
# the reference YAML files are read from there.  Run from the directory holding the <system>_<model>/ outputs.
# Choose your account at submission: sbatch --account=your-project <this-script> SYSTEM
# Adapt the BlueBEAR modules and QoS below to your cluster.
#SBATCH --qos=bbdefault
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/swx_an_%j.out
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

SYSTEM=${1:?usage: sbatch softwarex_analyse_oxide.slurm SYSTEM [MODEL]}; MODEL=${2:-mace-mpa-0}
TAG=$(echo "$SYSTEM" | tr 'A-Z' 'a-z'); OUT=${TAG}_${MODEL}
REF=""
case $SYSTEM in
  GA2O3) REF="--reference $PKG/examples/reference_a_Ga2O3.yaml" ;;   # PBE0 models, PRB 111, 035203
  SIO2)  REF="--reference $PKG/examples/reference_a_SiO2.yaml" ;;    # diffraction consensus (Mozzi & Warren; Wright)
esac
COMMON="--rings --connectivity --check-dimers --voids --sq --sq-weighting xray --sq-smooth 0.10 $REF"

for SET in random_initial random_opt hybrid/final; do
    [ -d "$OUT/$SET" ] || { echo "skip $SET (missing)"; continue; }
    NAME=$(echo "$SET" | tr '/' '_')
    EXTRA=""; [ "$SET" = random_opt ] && EXTRA="--convergence"
    echo "=== $SET: $(ls $OUT/$SET | grep -c xyz) structures, $(date)"
    amorphgen --analyse --input-dir "$OUT/$SET" --pattern "*.xyz" $COMMON $EXTRA \
        --save-report "$OUT/analysis_$NAME/report.txt" --save-plot "$OUT/analysis_$NAME/" \
        > "$OUT/analysis_$NAME.log" 2>&1 || echo "analysis of $SET failed, see $OUT/analysis_$NAME.log"
done
# The 20 random partners of the hybrid set, for the paired before/after comparison.
if [ -d "$OUT/hybrid/final" ]; then
    mkdir -p "$OUT/random_opt_paired"
    for f in "$OUT"/hybrid/task_*/input/*.xyz; do ln -sf "$(realpath "$f")" "$OUT/random_opt_paired/"; done
    amorphgen --analyse --input-dir "$OUT/random_opt_paired" --pattern "*.xyz" $COMMON \
        --save-report "$OUT/analysis_random_opt_paired/report.txt" --save-plot "$OUT/analysis_random_opt_paired/" \
        > "$OUT/analysis_random_opt_paired.log" 2>&1 || echo "paired analysis failed"
fi
echo "=== analysis done $(date)"
