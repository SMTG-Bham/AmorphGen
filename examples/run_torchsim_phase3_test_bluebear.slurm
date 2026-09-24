#!/bin/bash
# Validation of the torch-sim phase-3 features on one BlueBEAR GPU (~25 min):
#   1. GPU unit tests (test/test_torchsim_gpu.py) and the Tier 3 MACE tests.
#   2. --batch-size auto on a real card: batched relaxation of 4 a-IGZO cells
#      (350 atoms) and the MD probe. Look for "[torch-sim] batch size N ->".
#   3. Frame-level resume after a real kill: a short hybrid MD run is killed
#      with `timeout` part-way through a stage, then resubmitted in-job with
#      --resume. Look for "[Stage N] resuming from step" in the second run and
#      check that the per-run logs continue without a gap.
#SBATCH --job-name=ts_phase3
#SBATCH --account=scanlodo-amorphous-oxides
#SBATCH --qos=bbgpu
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=logs/ts_phase3_%j.out
set -uo pipefail
export PYTHONUNBUFFERED=1
mkdir -p logs
module purge
module load bear-apps/2024a/live
module load Python/3.12.3-GCCcore-13.3.0
source /rds/projects/s/scanlodo-amorphous-oxides/venvs/amorphgen-py312/bin/activate
PKG=/rds/projects/s/scanlodo-amorphous-oxides/kaewmeec/AmorphGen
python -c "import torch, torch_sim; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0), '| torch-sim', torch_sim.__version__)"

echo "=== 1. GPU tests ==="
( cd $PKG && python -m pytest test/test_torchsim_gpu.py -v -s --tb=short -p no:cacheprovider )
( cd $PKG && python -m pytest test/test_torchsim_engine.py test/test_mace_integration.py -q --tb=short --run-mace -p no:cacheprovider )

echo "=== 2. auto batch size: batched relaxation of 4 x 350-atom IGZO ==="
IN=igzo_100/subset10; T=phase3_test; rm -rf $T; mkdir -p $T/in4
for i in 0 1 2 3; do cp $IN/random_000${i}_opt.xyz $T/in4/; done
amorphgen --batch-opt --input-dir $T/in4 -o $T/relax_auto -m mace-mpa-0 -d cuda \
    --default-dtype float32 --engine torchsim --batch-size auto -C cubic -f 0.05 --opt-steps 200

echo "=== 3. frame-level resume: kill a short hybrid MD run, then --resume ==="
MD="--eq-high-ensemble NVT --eq-high-T 2000 --eq-high-steps 1000 --quench-ensemble NVT --quench-T-start 2000 --quench-T-end 300 --quench-T-step -100 --quench-steps-per-T 50 --eq-low-ensemble NVT --eq-low-T 300 --eq-low-steps 500 -C cubic -f 0.05"
CMD="amorphgen --hybrid-ensemble --input-dir $T/in4 -o $T/md_resume -m mace-mpa-0 -d cuda --default-dtype float32 --seed 7 --engine torchsim --batch-size auto $MD"
# ~2350 steps x 4 structures at ~40 ms/step/structure = ~6 min; kill at 3 min (inside stage 4 or 5)
timeout --signal=KILL 180 $CMD || echo "  (killed as intended after 180 s, exit $?)"
echo "--- trajectory frames at the kill:"
for r in $T/md_resume/quench_runs/run_*; do echo "  $(basename $r): $(ls $r)"; done
echo "--- resume:"
$CMD --resume
echo "--- per-run logs after resume (stage 4 step column should run 100..1000 without a gap):"
for r in $T/md_resume/quench_runs/run_*; do
    echo "  $(basename $r): stage4 steps $(awk 'NR>2{printf "%s ",$1}' $r/stage4_eq.log)"
done
cat $T/md_resume/batch_size.json; echo
ls $T/md_resume/final/
amorphgen --analyse --input-dir $T/md_resume/final --check-dimers --per-structure | tail -12
