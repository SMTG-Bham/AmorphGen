#!/bin/bash
# Class benchmark, step 3: regenerate and relax the systems whose density
# rules changed after the first pass (radii.py fixes of 2026-09-28: metal-rich
# metalloid glasses -> alloy, all-nonmetal oxides -> covalent_oxide with the
# nonmetal in its positive state, Be2+ radii, chalcogenide_glass class), into
# classes_v2/ so the before (classes/) and after tables can be compared.
# Also retries any first-pass system that has no relaxed structures.
#SBATCH --job-name=cls_refix
#SBATCH --account=scanlodo-amorphous-oxides
#SBATCH --qos=bbgpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=logs/cls_refix_%j.out
set -uo pipefail
export PYTHONUNBUFFERED=1
mkdir -p logs classes_v2
module purge
module load bear-apps/2024a/live
module load Python/3.12.3-GCCcore-13.3.0
source /rds/projects/s/scanlodo-amorphous-oxides/venvs/amorphgen-py312/bin/activate
PKG=/rds/projects/s/scanlodo-amorphous-oxides/kaewmeec/AmorphGen
SYS=$PKG/examples/class_benchmark_systems.txt
REFIX="Ni4P Fe4B P2O5 Be3N2 GeSe2 GeS2 As2S3 As2Se3"
grep -v "^#" $SYS | grep -wE "$(echo $REFIX | tr ' ' '|')" > classes_v2/systems.txt
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
python $PKG/examples/collect_class_benchmark.py classes_v2 classes_v2/systems.txt 2>&1 | grep -v "Composition\]"
echo "=== first-pass systems with no relaxed structures: retry"
grep -v "^#" $SYS | while read CLASS NAME COMP RHO KIND; do
    D=classes/$NAME
    [ -d $D/random_initial ] && [ "$(ls $D/random_opt/*_opt.xyz 2>/dev/null | wc -l)" -eq 0 ] && { echo "== retry $NAME"; relax_and_analyse $D; }
done
python $PKG/examples/collect_class_benchmark.py classes $SYS 2>&1 | grep -v "Composition\]" | tail -3
echo "=== done $(date): classes/class_benchmark.tsv (before) and classes_v2/class_benchmark.tsv (after)"
