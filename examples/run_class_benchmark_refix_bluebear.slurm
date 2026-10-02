#!/bin/bash
# Class benchmark, step 3: regenerate and relax the systems whose density
# rules changed after the first pass (radii.py fixes of 2026-09-28: metal-rich
# metalloid glasses -> alloy, all-nonmetal oxides -> covalent_oxide with the
# nonmetal in its positive state, Be2+ radii, chalcogenide_glass class), into
# classes_v2/ so the before (classes/) and after tables can be compared.
# Also retries any first-pass system that has no relaxed structures.
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --job-name=cls_refix
# Export AMORPHGEN_VENV=/path/to/venv before submitting (see docs/guides/hpc.md).
# For repository files, AMORPHGEN_ROOT defaults to the submission directory.
# Choose your account at submission: sbatch --account=your-project <this-script>
# Adapt the BlueBEAR modules and QoS below to your cluster.
#SBATCH --qos=bbgpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=logs/cls_refix_%j.out
set -euo pipefail
PKG=${AMORPHGEN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
if [[ ! -f "$PKG/amorphgen/slurm_runtime.sh" && -f "$PKG/../amorphgen/slurm_runtime.sh" ]]; then
    PKG=$(cd "$PKG/.." && pwd)
fi
source "$PKG/amorphgen/slurm_runtime.sh"
amorphgen_slurm_main "$0" "$@"
export PYTHONUNBUFFERED=1
mkdir -p logs classes_v2
amorphgen_slurm_bluebear_environment "bear-apps/2024a/live" "Python/3.12.3-GCCcore-13.3.0"
SYS="$PKG/examples/class_benchmark_systems.txt"
# EMPTY after the revert of 2026-09-29: no density rule has changed since the
# last pass, so classes/ + classes_v2/ are current. Put system names here (space
# separated) when a rule changes and they need regenerating.
REFIX=""
awk -v names="$REFIX" 'BEGIN {n=split(names, selected, " "); for (i=1; i<=n; i++) wanted[selected[i]]=1}
    !/^#/ && NF && ($2 in wanted) {print}' "$SYS" > classes_v2/systems.txt
relax_and_analyse () {   # $1 = dir
    amorphgen --batch-opt --input-dir $1/random_initial -o $1/random_opt \
        -m mace-mpa-0 -d cuda --default-dtype float32 --engine torchsim --batch-size auto \
        -C cubic -f 0.05 --opt-steps 1000 --resume
    amorphgen --analyse --input-dir $1/random_opt --check-dimers --connectivity \
        --save-report $1/analysis/report.txt --save-plot $1/analysis/ > $1/analysis.log 2>&1
}
echo "=== after-fix pass: $REFIX"
while read CLASS NAME COMP RHO KIND; do
    D=classes_v2/$NAME; echo "== $NAME ($CLASS) $(date +%H:%M)"
    /usr/bin/time -f "placement wall %e s" amorphgen --random-gen --composition "$COMP" -n 5 --seed 2026 -o $D --resume
    relax_and_analyse $D
done < classes_v2/systems.txt
python "$PKG/examples/collect_class_benchmark.py" classes_v2 classes_v2/systems.txt 2>&1 | grep -v "Composition\]"
echo "=== first-pass systems with no relaxed structures: retry"
grep -v "^#" "$SYS" | while read CLASS NAME COMP RHO KIND; do
    D=classes/$NAME
    if [ -d $D/random_initial ] && [ "$(find "$D/random_opt" -maxdepth 1 -name '*_opt.xyz' 2>/dev/null | wc -l)" -eq 0 ]; then
        echo "== retry $NAME"
        relax_and_analyse "$D"
    fi
done
python "$PKG/examples/collect_class_benchmark.py" classes "$SYS" 2>&1 | grep -v "Composition\]" | tail -3
# merged view for the paper: every system in the current list, taken from
# classes_v2/ where it was redone or added and from classes/ otherwise
rm -rf classes_final; mkdir -p classes_final
grep -v "^#" "$SYS" | while read CLASS NAME COMP RHO KIND; do
    if [ -d classes_v2/$NAME/random_opt ]; then ln -s ../classes_v2/$NAME classes_final/$NAME
    elif [ -d classes/$NAME ]; then ln -s ../classes/$NAME classes_final/$NAME; fi
done
python "$PKG/examples/collect_class_benchmark.py" classes_final "$SYS" 2>&1 | grep -v "Composition\]"
echo "=== done $(date): classes/ (first pass), classes_v2/ (redone + added), classes_final/class_benchmark.tsv (merged, current list)"
