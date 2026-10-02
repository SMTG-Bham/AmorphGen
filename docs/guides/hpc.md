# HPC deployment

AmorphGen runs on GPU-enabled HPC clusters through SLURM. Install the package
and the backend used by your job first; see {doc}`../getting-started/installation`.

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

Adapt the resource requests and environment paths to your cluster. This example
initialises conda explicitly because batch shells may not read your interactive
shell configuration:

```bash
#!/bin/bash
#SBATCH --job-name=amorphgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --account=your-account

set -euo pipefail
export PYTHONUNBUFFERED=1

# Load any compiler/Python modules required by your cluster environment.
source /path/to/miniforge3/etc/profile.d/conda.sh
conda activate amorphgen

amorphgen POSCAR --model mace-mpa-0 --device cuda \
    --work-dir my_run --resume
```

## Resuming timed-out jobs

`--resume` reuses completed stage outputs and saved MD frames in the work
directory. Resubmit with the same input, model, configuration and work directory;
use a new directory when changing the simulation protocol. Pipeline and random
generation runs verify saved settings before reusing outputs and reject
concurrent writers with an exclusive `.amorphgen.lock` in the output directory.
The lock is released automatically when a job exits or is killed. Leave the
lock file in place; its existence does not indicate an active job. Runs from
older versions without complete settings provenance need a fresh directory.

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
chunk size recorded in `quench_runs/batch_size.json` under the hybrid work
directory. Keep the input list and chunk size unchanged when resuming.
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

set -euo pipefail
export PYTHONUNBUFFERED=1
source /path/to/miniforge3/etc/profile.d/conda.sh
conda activate amorphgen
SAMPLE=${SLURM_ARRAY_TASK_ID}

amorphgen "inputs/sample-${SAMPLE}.xyz" \
    --model mace-mpa-0 --device cuda \
    --stages 1 4 5 6 7 \
    --config config.yaml \
    --work-dir "results/sample_${SAMPLE}" \
    --resume
```

Each array task runs on its own GPU. The `--resume` flag makes resubmission safe: completed samples are skipped automatically.
