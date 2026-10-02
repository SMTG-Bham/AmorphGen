# HPC deployment

AmorphGen runs CPU and GPU workloads on HPC clusters through Slurm. Install the package
and the backend used by your job first; see {doc}`../getting-started/installation`.

## Generate a portable workflow

`amorphgen-slurm` generates Slurm job scripts from one YAML workflow. Use it
for a single job, independent array tasks, or a dependency chain on any Slurm
cluster. It writes one `.slurm` file per job and a `submit.sh` script; generation
does not submit jobs.

Start with [the example workflow](../../examples/slurm_workflow.yaml), edit its
paths and resource requests, then generate and submit it from the repository:

```bash
amorphgen-slurm examples/slurm_workflow.yaml --output-dir jobs
bash jobs/submit.sh --account=your-project
```

The example chains random generation, relaxation and analysis. The submission
script creates the log directory, submits prerequisites before their dependants,
and passes the returned job IDs to Slurm. It records successfully submitted
job IDs in `jobs/submitted.*.tsv`, including when a later submission fails.
Additional `submit.sh` arguments are passed to every `sbatch` call, so account
and partition choices can stay local.
Running it again submits a new workflow: wait for the previous one to finish or
cancel it before resubmitting into the same output directories. Use `--force`
to regenerate existing script files after editing the YAML.

A small CPU workflow illustrates the format:

```yaml
version: 1
profile: generic
root: ..                         # relative to this YAML file's directory
resources:
  ntasks: 1
  cpus-per-task: 2
  mem: 4G
  time: "01:00:00"
environment:
  modules: []
  venv: /path/to/your/venv
checkpoint:
  signal_seconds: 120
  requeue: false
jobs:
  - name: generate
    array: "0-9%4"
    work_dir: runs/generate/{task_id}
    commands:
      - [amorphgen, --random-gen, --composition, "SiO2*16", "-n", "10",
         --seed, "{task_id}", --work-dir, "{work_dir}", --resume]
  - name: analyse
    array: "0-9%4"
    work_dir: runs/analyse/{task_id}
    needs: [generate]
    dependency: aftercorr
    commands:
      - [amorphgen, --analyse,
         --input-dir, "{root}/runs/generate/{task_id}/random_initial",
         --save-report, "{work_dir}/report.txt"]
```

`root` defaults to the YAML file's directory. Relative `work_dir` paths are
resolved under that root. Commands run from `root`, and the script creates
`work_dir` before starting them. Array jobs must include `{task_id}` in
`work_dir`, so each task can write to its own directory; use `{work_dir}` in
output arguments to keep those outputs isolated. Commands are lists of arguments,
with `{root}`, `{work_dir}` and `{task_id}` placeholders; shell expansions,
pipes and redirects are not interpreted. Include `--resume` explicitly on
AmorphGen commands that should resume. The generator does not add CLI flags.

`resources` uses Slurm option names, including `ntasks`, `cpus-per-task`,
`mem`, `time`, `gres`, `partition`, `account` and `qos`. Request GPUs with,
for example, `gres: gpu:1`. Each job can override the shared `resources` with
its own mapping. The generic defaults are one task, four CPUs, 8 GB of memory
and one hour. `environment.modules` lists modules to load and
`environment.venv` activates the chosen virtualenv (a relative path resolves
under `root`). Without `venv`, the generic profile uses `AMORPHGEN_VENV` if
set, otherwise it keeps the existing environment.

Use `--profile bluebear` to select the BlueBEAR preset, or
`profile: bluebear` in the YAML. This loads `bear-apps/2024a/live` and
`Python/3.12.3-GCCcore-13.3.0`, and chooses `bbgpu` QoS for GPU requests or
`bbdefault` otherwise. Explicit resource and module settings take precedence.
Set `environment.venv` or export `AMORPHGEN_VENV` for this profile.
Environment and checkpoint settings apply to the whole workflow.

### Arrays and dependencies

An array expression such as `0-9%4` runs ten tasks with at most four active
at once. Each task receives the same resource request; `{task_id}` becomes its
Slurm array index. `needs` names prerequisite jobs in the workflow. Choose one
dependency condition for each dependent job:

| `dependency` | When the dependent job becomes eligible |
|---|---|
| `afterok` (default) | All prerequisite jobs complete successfully; for arrays, all tasks must succeed. |
| `aftercorr` | The matching task in each prerequisite array succeeds. Requires matching array task IDs. |
| `afterany` | All prerequisite jobs finish, regardless of success. |
| `afternotok` | The prerequisite jobs finish unsuccessfully. |

Use `aftercorr` for a generation → relaxation array where task 3 consumes only
task 3's outputs. Use `afterok` for a collector that needs an entire array.
Slurm's [job-array guide](https://slurm.schedmd.com/job_array.html) explains the
scheduler's dependency rules. If an upstream job fails, downstream `afterok`
jobs cannot run; inspect their state before submitting a replacement chain.

## Checkpointing on signals and preemption

Generated jobs request `--signal=B:USR1@120` by default. The batch script
turns `USR1` or `TERM` into checkpoint requests for AmorphGen/Python child
processes, enables AmorphGen's checkpoint handler, and stops before starting
further commands. AmorphGen exits with status 75 after a checkpoint interruption, so an unfinished run does not
release an `afterok` dependency as successful.

The warning interval comes from `checkpoint.signal_seconds`. This Slurm option
requests a warning **before the walltime limit**; the signals and grace period
provided during actual preemption depend on the cluster's configuration.
See the [Slurm signal option](https://slurm.schedmd.com/sbatch.html#OPT_signal).
Choose an interval long enough for the slowest checkpoint boundary and file
writes on your workload. An immediate `SIGKILL` cannot be handled.

On a handled signal, MD stops at a normal trajectory checkpoint boundary
(usually every 100 steps). Completed structures and stages remain available;
an interrupted optimisation restarts its current structure or batched chunk,
and random generation resumes from completed structures. Restarted MD does not
restore every thermostat, barostat or random-number state, so it is not a
bitwise continuation of the interrupted trajectory. These guarantees apply to
directly launched AmorphGen commands that support resume. Local Python API
programs must wrap their work in `amorphgen.utils.preemption.checkpoint_signals()`
to enable the same handlers. Native programs and remote `srun` job steps need
their own signal forwarding and restart integration; the bundled supervisor
forwards warnings only to local AmorphGen/Python processes.

The default `checkpoint.requeue: false` disables explicit requeue requests.
With `requeue: true`, a `USR1` interruption requests `scontrol requeue` after the
child command stops; the cluster must permit user requeue. `TERM` does not
explicitly request requeue, leaving cancellation and preemption decisions to
Slurm. Keep `--resume` in resumable commands and preserve the same inputs,
settings and work directories when a job starts again.

## Configuring the bundled examples

The 27 existing `examples/*.slurm` scripts remain available for the original
BlueBEAR experiments. They use BlueBEAR module and QoS names. Adapt those
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
current directory when run directly). Follow each script's input/config
instructions. Create `logs/` **before** calling `sbatch`, so Slurm can open its
log files.

The scripts have no embedded allocation account. Pass `--account=your-project`
on each submission, or set `export SBATCH_ACCOUNT=your-project` when submitting
several jobs, including dependency chains. Keep these settings in your shell
or a local submission wrapper. `#SBATCH` directives do not expand shell variables.

The scripts share a signal supervisor and request a walltime warning. As with
generated jobs, a handled interruption exits with status 75 by default. Set
`AMORPHGEN_SLURM_REQUEUE=1` to request requeue after `USR1` when permitted by
your site. The scripts' `--requeue` directive allows scheduler-managed requeue;
it does not, by itself, resubmit every interrupted job. Override module names
with `AMORPHGEN_BEAR_MODULE` and `AMORPHGEN_PYTHON_MODULE`, or set
`AMORPHGEN_SKIP_MODULES=1` when the environment is already prepared.

For the existing SiO₂ experiment, submit its four generation tasks with a
concurrency limit and release relaxation only once all four succeed:

```bash
generate_id=$(sbatch --parsable --account=your-project --array=0-3%2 \
    examples/run_sio2_gen_array_bluebear.slurm)
sbatch --account=your-project --dependency="afterok:${generate_id%%;*}" \
    examples/run_sio2_relax_bluebear.slurm
```

Each SiO₂ generation task keeps its checkpoints in
`sio2_100/placements/task_ID/`, publishing its completed structures through
disjoint links in `sio2_100/random_initial/` for the relaxation job. Preserve
the task index ranges defined by the scripts: they often select a specific
material or structure slice. Singleton scripts can have shared output names;
use the YAML generator with `{task_id}` work directories for new arrays.

Updated generation scripts use deterministic seeds for reproducible restarts.
If an older run used an unspecified seed or different settings, its saved
metadata may reject resume with the new script. Use a fresh output directory
for that protocol; preserve the old checkpoints instead of overwriting them.

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
trajectories every 100 steps. After interruption, `--resume` skips finished
runs, restarts unfinished relaxation chunks, continues a partly done MD stage
from the last valid frame common to the chunk, and reuses the
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
