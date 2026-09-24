#!/bin/bash
# GPU tests of the torch-sim engines (test/test_torchsim_gpu.py) on one BlueBEAR
# GPU, plus the Tier 3 MACE integration tests. ~10 minutes.
#SBATCH --job-name=amg_gputest
#SBATCH --account=scanlodo-amorphous-oxides
#SBATCH --qos=bbgpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/gputest_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
mkdir -p logs
module purge
module load bear-apps/2024a/live
module load Python/3.12.3-GCCcore-13.3.0
source /rds/projects/s/scanlodo-amorphous-oxides/venvs/amorphgen-py312/bin/activate
cd /rds/projects/s/scanlodo-amorphous-oxides/kaewmeec/AmorphGen
python -c "import torch, torch_sim; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0), '| torch-sim', torch_sim.__version__)"
python -m pytest test/test_torchsim_gpu.py -v -s --tb=short -p no:cacheprovider
python -m pytest test/test_torchsim_engine.py test/test_mace_integration.py -v --tb=short --run-mace -p no:cacheprovider
