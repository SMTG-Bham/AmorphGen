#!/bin/bash
# Collect the IrO2 array-job outputs into one folder and analyse vs the AIMD
# reference. Run AFTER the array finishes (on a login node or a short job).
#   Usage:  bash examples/collect_and_analyse_iro2.sh iro2_cubic
set -eu

RUN="${1:-iro2_cubic}"          # iro2_cubic (cubic relax) or iro2_validation (fixed)
DST="${RUN}/all_opt"
mkdir -p "$DST"

# Each task wrote struct_NN/random_opt/random_0000_opt.vasp (same name) — rename
n=0
for d in "${RUN}"/struct_*/; do
    t=$(basename "$d")
    f="${d}random_opt/random_0000_opt.vasp"
    if [ -f "$f" ]; then cp "$f" "${DST}/iro2_${t}.vasp"; n=$((n+1)); fi
done
echo "collected ${n} relaxed structures into ${DST}/"

# Analyse the ensemble against the published AIMD ranges
amorphgen --analyse --input-dir "${DST}" \
    --reference examples/reference_a_IrO2.yaml \
    -o "${RUN}/analysis"

echo "analysis (RDF/CN/angles/density vs AIMD) in ${RUN}/analysis/"
