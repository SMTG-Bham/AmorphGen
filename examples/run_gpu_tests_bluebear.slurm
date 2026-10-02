#!/bin/bash
# GPU tests of the torch-sim engines (test/test_torchsim_gpu.py) on one BlueBEAR
# GPU, plus the Tier 3 MACE integration tests. ~10 minutes.
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --job-name=amg_gputest
# Export AMORPHGEN_VENV=/path/to/venv before submitting (see docs/guides/hpc.md).
# For repository files, AMORPHGEN_ROOT defaults to the submission directory.
# Choose your account at submission: sbatch --account=your-project <this-script>
# Adapt the BlueBEAR modules and QoS below to your cluster.
#SBATCH --qos=bbgpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/gputest_%j.out
set -euo pipefail
PKG=${AMORPHGEN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
if [[ ! -f "$PKG/amorphgen/slurm_runtime.sh" && -f "$PKG/../amorphgen/slurm_runtime.sh" ]]; then
    PKG=$(cd "$PKG/.." && pwd)
fi
source "$PKG/amorphgen/slurm_runtime.sh"
amorphgen_slurm_main "$0" "$@"
export PYTHONUNBUFFERED=1
mkdir -p logs
amorphgen_slurm_bluebear_environment "bear-apps/2024a/live" "Python/3.12.3-GCCcore-13.3.0"
cd "${AMORPHGEN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
python -m pip install -q pytest      # ensure the test runner is available
python -c "import torch, torch_sim; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0), '| torch-sim', torch_sim.__version__)"
python -m pytest test/test_torchsim_gpu.py --run-mace -v -s --tb=short -p no:cacheprovider
python -m pytest test/test_torchsim_engine.py test/test_mace_integration.py -v --tb=short --run-mace -p no:cacheprovider
