# HPC deployment

AmorphGen is designed for deployment on GPU-enabled HPC clusters via SLURM.

## SLURM job script

```bash
#!/bin/bash
#SBATCH --job-name=amorphgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --account=your-account

module load CUDA/11.8.0
conda activate /path/to/your/env

amorphgen POSCAR --model mace-mpa-0 --device cuda
```

## Resuming timed-out jobs

The `--resume` flag enables smart checkpoint detection for both pipeline and batch-quench modes. It scans the work directory for completed stage outputs and automatically skips them.

### Pipeline mode

```bash
amorphgen POSCAR \
    --stages 1 4 5 6 7 \
    --config my_config.yaml \
    --work-dir my_run/ \
    --resume
```

If stages 1 and 4 are already complete, AmorphGen picks up from stage 5 using the `stage4_eq.xyz` checkpoint. If all stages are done, it exits immediately.

### Batch quench mode

```bash
amorphgen --batch-quench \
    --snapshot-dir snapshots/ \
    --model mace-mpa-0 \
    --device cuda \
    --resume
```

This skips already-completed structures and continues from where the previous job left off.

### Python API

```python
from amorphgen import MeltQuenchPipeline

pipe = MeltQuenchPipeline(
    input_file="POSCAR",
    work_dir="my_run",
    cfg_override={"model": "mace-mpa-0", "device": "cuda"},
)
atoms = pipe.run(stages=[1, 4, 5, 6, 7], resume=True)
```

## Array jobs for batch processing

For running many structures in parallel (e.g. 100 AIRSS structures), use a SLURM array job:

```bash
#!/bin/bash
#SBATCH --job-name=MQ_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=1-100

SAMPLE=${SLURM_ARRAY_TASK_ID}

amorphgen "inputs/sample-${SAMPLE}.xyz" \
    --stages 1 4 5 6 7 \
    --config config.yaml \
    --work-dir "results/sample_${SAMPLE}" \
    --resume
```

Each array task runs on its own GPU. The `--resume` flag makes resubmission safe — completed samples are skipped automatically.
