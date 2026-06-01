# MQ-ensemble workflow

Generate **N independent amorphous structures from a single crystalline input** with one CLI command. This is the standard melt-quench MD ensemble pattern used in nearly every amorphous-oxide DFT/MLIP paper, packaged as a single AmorphGen mode.

## Concept

```text
Crystalline supercell  →  shared stages 1-4  →  extract N snapshots  →  N × stages 5-7  →  N amorphous structures
```

The key efficiency win: **stages 1-4 (opt + premelt + heat + high-T equilibration) run only once** on the shared trajectory. Snapshots taken at evenly-spaced intervals from the long stage-4 trajectory are statistically independent samples of the equilibrium liquid; quenching each independently yields a diverse ensemble of amorphous structures.

## Single-command CLI: `--mq-ensemble`

```bash
amorphgen GaO.xyz --mq-ensemble --n-structures 20 \
    --config mq.yaml --device cuda --model chgnet \
    -o ga2o3_mq/
```

That's the entire workflow. Internally:

1. **Stages 1-4** run once on `GaO.xyz`, writing `ga2o3_mq/shared/` (incl. `stage4_eq_traj.xyz`).
2. **N=20** uniformly-spaced snapshots are extracted from the stage-4 trajectory into `ga2o3_mq/snapshots/`.
3. **Stages 5-6-7** run independently on each snapshot, output to `ga2o3_mq/quench_runs/run_NNNN/`.
4. **Final amorphous structures** are collected to `ga2o3_mq/final/mq_NNNN.<format>`.

`--resume` is honoured at every step. Re-running the same command picks up wherever it stopped without redoing completed work.

## Output layout

```text
ga2o3_mq/
├── shared/
│   ├── stage1_opt.xyz
│   ├── stage2_eq.xyz
│   ├── stage3_melted.xyz
│   ├── stage4_eq.xyz             # final state of stage 4
│   ├── stage4_eq_traj.xyz        # full trajectory (snapshots taken from here)
│   └── stage*.log                # per-stage MDLogger output
├── snapshots/
│   ├── snapshot_0000_frame*.xyz
│   └── ...
├── quench_runs/
│   ├── run_0000/                 # stages 5-7 outputs for snapshot_0000_*
│   │   ├── stage5_quenched.xyz
│   │   ├── stage6_eq.xyz
│   │   ├── stage7_opt.cif
│   │   ├── stage7_opt.xyz
│   │   └── final_amorphous.xyz
│   └── ...
└── final/
    ├── mq_0000.vasp              # collected, ready for analysis
    ├── mq_0001.vasp
    └── ...
```

The inner `run_NNNN/` is named after the source snapshot's index (parsed from the
`snapshot_NNNN_frame*.xyz` filename), so `run_0007/` always corresponds to
`snapshot_0007_*` — making it easy to trace any final structure back to its
high-T starting frame.

### HPC job-array tip

When splitting the per-snapshot quenches across SLURM array tasks, point **all
tasks at the same `quench_runs/`** output dir; AmorphGen names the per-task
subdir from the snapshot index, so there's no collision. Don't pass each task
its own `-o quench_runs/run_${TASK}` — that nests inside another `run_NNNN/`
created by `batch_quench` and gives you the unhelpful `quench_runs/run_0007/run_0007/`.

A clean per-task command looks like:

```bash
mkdir -p inputs_per_task/task_${TASK}
cp snapshots/snapshot_${TASK}_frame*.xyz inputs_per_task/task_${TASK}/
amorphgen --batch-quench \
  --snapshot-dir inputs_per_task/task_${TASK} \
  --config mq_stages_567.yaml --stages 5 6 7 \
  --model chgnet --device cuda \
  -o quench_runs        # shared across all array tasks
```

A full SLURM array template ships with the package at
`examples/run_quench_array_bluebear.slurm`.


## Recommended `mq.yaml` for an oxide

```yaml
model: chgnet                       # or mace-mpa-0, sevennet, ...
device: cuda
default_dtype: float64

# Stage 1: relax the crystalline supercell
opt:
  fmax: 0.05
  max_steps: 200
  optimizer: LBFGS
  cell_filter: none

# Stage 2: equilibrate at low T (NPT — let cell relax)
eq_premelt:
  ensemble: NPT
  T: 300
  steps: 5000
  timestep: 0.5
  ttime: 25.0

# Stage 3: heating ramp to T_melt (above the system's melting point)
melt:
  ensemble: NPT
  T_start: 300
  T_end: 4000           # well above oxide Tm
  T_step: 100
  rate: 100             # K/ps; tighter (slower) for better-equilibrated melt
  timestep: 0.5
  ttime: 25.0

# Stage 4: long high-T equilibration to decorrelate snapshots
eq_high:
  ensemble: NPT
  T: 4000
  steps: 100000         # 100 ps — generous; ensures snapshots are independent
  timestep: 0.5
  ttime: 25.0

# Stage 5: cooling ramp (the actual quench)
quench:
  ensemble: NPT
  T_start: 4000
  T_end: 300
  T_step: -100
  rate: 100             # K/ps; PRB protocols use 0.5-100 K/ps
  timestep: 0.5
  ttime: 25.0

# Stage 6: equilibrate at room temperature
eq_low:
  ensemble: NPT
  T: 300
  steps: 5000
  timestep: 0.5
  ttime: 25.0

# Stage 7: final structural relaxation
opt:
  fmax: 0.05
  max_steps: 200
  optimizer: LBFGS
  cell_filter: FrechetCellFilter   # full cell relax for accurate density
```

## Equivalent two-step manual workflow

If you want to inspect or analyse the stage-4 trajectory before quenching, run the two halves separately:

```bash
# Step 1: stages 1-4 only (writes stage4_eq_traj.xyz)
amorphgen GaO.xyz --config mq.yaml --stages 1 2 3 4 --resume -o shared/

# Step 2: extract N snapshots and quench each
amorphgen --batch-quench --snapshot-dir shared/stage4_eq_traj.xyz \
    --n-runs 20 --batch-stages 5 6 7 \
    --config mq.yaml --resume -o quench_runs/
```

`--batch-quench` accepts a trajectory file directly (polymorphic `--snapshot-dir`) — internally extracts N snapshots, then runs the per-snapshot stages. Same final output as `--mq-ensemble` but split into two CLI invocations.

## HPC / Slurm split (best for parallelism)

For a cluster with multiple GPUs, run the two halves as separate slurm jobs so the per-snapshot quenches can execute in parallel via a slurm array:

```bash
# Job 1: stages 1-4 (single GPU, ~10 h on A100 for 100 ps eq_high)
sbatch 01_shared_bluebear.slurm
# Note the JOBID

# Job 2: array of 20 quench tasks (20 GPUs concurrent, ~5 h wall)
sbatch --dependency=afterok:<JOBID> 02_quench_array_bluebear.slurm
```

Example slurm scripts for BlueBEAR and Sulis ship in the AmorphGen repo under `examples/hpc/`. Both use `amorphgen --extract-snapshots` and `amorphgen --batch-quench --snapshot-dir snapshots/` internally — same dispatch as `--mq-ensemble`, just split for HPC parallelism.

| Pattern | Wall time | Best for |
|---------|-----------|----------|
| `--mq-ensemble` (single command) | ~30 h sequential | Local / single-GPU |
| Two slurm jobs (shared + array) | ~15 h with 20 concurrent GPUs | HPC with array support |

## Resume behaviour

| Interruption point | What `--resume` recovers |
|--------------------|---------------------------|
| Mid stages 1-4 | Skips completed stages, re-runs the interrupted one from start. (Frame-level resume is on the roadmap; see CLAUDE.md.) |
| Between stage 4 and snapshot extraction | Skips stages 1-4, re-extracts snapshots, runs 5-7. |
| Mid quench-runs | Skips completed runs (looks for `final_amorphous.xyz`), re-runs the interrupted one. |
| After all done | Reports "all complete", returns. Idempotent. |

## When to use `--mq-ensemble` vs the alternatives

| Use case | Recommended mode |
|----------|------------------|
| **Compare directly to published DFT melt-quench** | `--mq-ensemble` (full crystal → liquid → quench protocol) |
| Generate amorphous structures from random starting points | `--hybrid-ensemble` ({doc}`hybrid-workflow`) |
| Single amorphous structure (no ensemble) | Default pipeline (no flag, just `amorphgen INPUT --config ...`) |
| Quench pre-extracted snapshots from an existing trajectory | `--batch-quench --snapshot-dir TRAJ` |

## Validation


```bash
amorphgen --analyse --input-dir ga2o3_mq/final/ \
    --cutoff auto-rdf --per-structure \
    --reference reference_a_Ga2O3.yaml \
    --save-report mq_report.txt --save-plot mq_plots/ --save-pdf
```

This produces a publication-quality structural analysis (RDF, CN, bond angles) and a validation table comparing each metric to literature ranges. See {doc}`yaml-config` for the reference YAML format.
