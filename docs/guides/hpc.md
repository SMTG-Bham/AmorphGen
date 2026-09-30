# HPC deployment

AmorphGen is designed for deployment on GPU-enabled HPC clusters via SLURM.

## Configuring the bundled examples

The `examples/*.slurm` scripts use BlueBEAR module and QoS names. Adapt those
settings to your cluster, choose your own allocation account at submission,
and export the path to a virtualenv containing AmorphGen and the required
backends:

```bash
export AMORPHGEN_VENV=/path/to/your/venv
export AMORPHGEN_ROOT=/path/to/AmorphGen
cd "$AMORPHGEN_ROOT"
mkdir -p logs
sbatch --account=your-project examples/run_ensemble_resume_bluebear.slurm
```

`AMORPHGEN_VENV` is required; the scripts stop with a setup message if it is
unset or empty. Use a virtualenv compatible with the Python module loaded by
the script (Python 3.12 or newer for torch-sim). Scripts that read files from
the repository use `AMORPHGEN_ROOT`, defaulting to `SLURM_SUBMIT_DIR` (or the
current directory when run directly). Run inputs and outputs remain relative
to the submission directory; follow each script's input/config instructions.
Create `logs/` there **before** calling `sbatch`, so SLURM can open its log files.

The scripts have no embedded allocation account. Pass `--account=your-project`
on each submission, or set `export SBATCH_ACCOUNT=your-project` when submitting
several jobs, including dependency chains. Keep these settings in your shell
or a local submission wrapper. `#SBATCH` directives do not expand shell variables.

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

### Ensembles on the torch-sim engine

```bash
amorphgen --hybrid-ensemble \
    --input-dir random_structures/random_opt/ \
    --config hybrid.yaml \
    --model mace-mpa-0 --device cuda \
    --engine torchsim --batch-size auto \
    --work-dir hybrid_run/ --resume
```

With `--engine torchsim` (see {doc}`backends`) the structures are processed
in batched chunks. Relaxed structures are written after every chunk and MD
trajectories every 100 steps, so a walltime kill loses at most one chunk of
relaxation or 100 MD steps: `--resume` skips finished runs, continues a
partly done MD stage from the last frame common to the chunk, and reuses the
chunk size recorded in `batch_size.json` so the chunking is identical.
Resubmitting the same job script until the log reports the ensemble complete
is the intended way to run a large ensemble through a short queue. Put
`export PYTHONUNBUFFERED=1` in the script, otherwise the progress messages
only appear in the log when the job ends. The engine compiles kernels when it
starts, so the compute node needs a C/C++ compiler on `PATH`: if `g++ --version`
fails there, load one in the script (`module load GCC`; the name varies by
site). See
[the installation page](../getting-started/installation.md#the-torch-sim-engine).

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

Each array task runs on its own GPU. The `--resume` flag makes resubmission safe: completed samples are skipped automatically.
