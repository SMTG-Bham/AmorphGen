# MQ-ensemble workflow

Generate **up to N amorphous structures from a single crystalline input** with one CLI command. This mode shares the initial melt preparation, then quenches selected snapshots separately.

## Concept

```text
Crystalline supercell  →  shared stages 1-4  →  select up to N snapshots  →  separate stages 5-7  →  amorphous ensemble
```

The key efficiency win: **stages 1-4 (opt + premelt + heat + high-T equilibration) run only once** on the shared trajectory. By default, snapshot selection discards an initial burn-in and chooses spacing from scalar autocorrelation and atomic diffusion in the stage-4 trajectory. The sampling report estimates the effective number of independent snapshots; this remains a diagnostic, not proof of liquid equilibration or independence of the final glasses.

## Single-command CLI: `--mq-ensemble`

```bash
amorphgen GaO.xyz --mq-ensemble --n-structures 20 \
    --config mq.yaml --device cuda --model chgnet --format vasp \
    --resume -o ga2o3_mq/
```

That's the entire workflow. Internally:

1. **Stages 1-4** run once on `GaO.xyz`, writing `ga2o3_mq/shared/` (incl. `stage4_eq_traj.xyz`).
2. **Up to N=20** snapshots are extracted from the stage-4 trajectory into `ga2o3_mq/snapshots/`, using the default `--select decorrelated` and adaptive burn-in. A short or slowly diffusing trajectory can yield fewer snapshots.
   The initial crystal, melt endpoints and snapshots are compared in the automatic `melt_memory` report before quenching.
3. **Stages 5-6-7** run independently on each snapshot, output to `ga2o3_mq/quench_runs/run_NNNN/`.
4. **Final amorphous structures** are collected to `ga2o3_mq/final/mq_NNNN.<format>`.

`--resume` skips completed simulation work and resumes interrupted MD from saved frames. Snapshot extraction and final collection are repeated after checking that selection is compatible with any existing quench outputs. Keep the inputs, protocol and snapshot selection unchanged when resuming; use a new output directory for a different ensemble.

## Output layout

```text
ga2o3_mq/
├── snapshot_sampling.{json,txt} # burn-in, spacing, selected indices and effective sample count
├── melt_memory.{json,csv,txt}     # initial crystal order retained at melt endpoints/snapshots
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
│   │   ├── stage7_opt.vasp       # requested by --format vasp
│   │   └── final_amorphous.xyz
│   └── ...
└── final/
    ├── mq_0000.vasp              # collected, ready for analysis
    ├── mq_0001.vasp
    └── ...
```

The inner `run_NNNN/` is named after the source snapshot's index (parsed from the
`snapshot_NNNN_frame*.xyz` filename), so `run_0007/` always corresponds to
`snapshot_0007_*`, making it easy to trace any final structure back to its
high-T starting frame.

For a single input, the ASE engine writes the stage outputs directly inside
`quench_runs/`; the final structure is still collected into `final/mq_0000.<fmt>`.
`--mq-ensemble` uses the ASE engine; batched torch-sim MD is available in
`--hybrid-ensemble`.

## Burn-in, spacing and effective snapshot count

`--mq-ensemble` defaults to `--select decorrelated`. The requested
`--n-structures` is an upper bound, not a target that forces more closely
spaced frames into the ensemble. Selection uses the saved stage-4 frames:

1. **Burn-in.** If `--burn-in-frames` is omitted, discard at least the first
   10% of frames (at least one when multiple frames exist), then compare
   candidate burn-in points through the first half of the trajectory. Choose
   the retained region with the largest conservative effective sample count
   from potential energy per atom and fluctuating volume. An explicit count,
   including `--burn-in-frames 0`, overrides this choice.
2. **Scalar correlations.** Estimate autocorrelation decay and statistical
   inefficiency from the available energy and volume series after burn-in.
   Integrate the initial positive autocorrelation sequence to obtain $g$.
   Require $|C(t)|\leq0.1$ for at least $\max(3,\lceil g/2\rceil)$ saved
   lags, and a spacing of at least $\lceil g\rceil$ frames. A negative
   autocorrelation lobe alone does not establish decay.
3. **Diffusion.** Compute mean-squared displacements (MSD), averaged over
   time origins, separately for each element. Unwrap periodic motion between
   saved frames and remove whole-system drift and affine cell strain. The
   positional correlation proxy is
   $C_s(t)=\exp[-\mathrm{MSD}_s(t)/d^2]$. Require this proxy to fall to 0.1
   or less for every species, so a mobile species cannot hide a slow one.
   The distance $d$ defaults to the median nearest-neighbour distance in the
   final frame; set `--decorrelation-distance` to override it in Å.
4. **Selection.** Use spacing that satisfies both the scalar and diffusion
   criteria. If decorrelation is unresolved within the available history,
   retain only the final frame and report the unresolved diagnostics.

`snapshot_sampling.json` and `snapshot_sampling.txt` record the selected
trajectory indices, burn-in, frame interval and chosen spacing, scalar
autocorrelation diagnostics, per-species MSD and diffusion estimates, and
the estimated effective independent snapshot count. The latter uses the
slowest scalar or diffusion correlation proxy and never exceeds the number
of selected snapshots. Missing observables and unresolved estimates produce
warnings rather than evidence of independence. Diffusion coefficients are
late-lag linear-fit diagnostics; spacing uses the measured MSD without
extrapolating a diffusion coefficient beyond the available trajectory.
Time-based quantities depend
on knowing the saved-frame interval. The CLI assumes one saved frame per
100 MD steps and uses `eq_high.timestep` from the supplied configuration
(0.5 fs by default). When extracting separately, pass the same `--config`
used to create the trajectory. For external trajectories with another save
cadence, the Python sampling API accepts `frame_stride` and `timestep_fs`;
otherwise reported times and diffusion coefficients will have the wrong scale.

For the actual selected indices $t_i$, each available correlation gives
$N_{\mathrm{eff}}=N^2/[N+2\sum_{i<j}C(|t_i-t_j|)]$; the report takes the
smallest count across observables and species. Scalar negative correlations
are set to zero, but later positive peaks are retained. MSD and scalar
correlations use lags up to half the retained trajectory; beyond this window,
the last measured correlation is held constant rather than extrapolating
unobserved relaxation. An unresolved adaptive selection retains one frame
and reports an effective count of one, with its unresolved status.

This is an operational sampling heuristic. The burn-in and statistical
inefficiency approach follows the general effective-sample-count idea
described in the [PyMBAR timeseries documentation](https://pymbar.readthedocs.io/en/latest/timeseries.html),
but does not call or exactly reproduce its algorithm. Periodic unwrapping,
drift correction and the relation between MSD and diffusion are discussed in
the [LAMMPS MSD documentation](https://docs.lammps.org/latest/compute_msd.html).
Neither scalar decorrelation nor the displacement proxy tests every relevant
structural mode. Save frames frequently enough to resolve periodic crossings;
unwrapping cannot reconstruct multiple unseen crossings between saved frames.

The default stage-4 hold remains **10 ps** (20000 steps at 0.5 fs), and is not
extended automatically. Inspect the sampling and melt-memory reports, and
increase `eq_high.steps` when the available liquid trajectory does not support
the desired ensemble size. Use a new output directory when changing the
protocol or selection of an existing ensemble.

To reproduce evenly spaced extraction, explicitly use `--select uniform`;
`--select last` takes the last available frames. These choices default to
zero burn-in unless `--burn-in-frames` is supplied. They remain the available
legacy selection modes; uniform is still the default for standalone
`--extract-snapshots` and `--batch-quench`, where `--select decorrelated`
opts into the trajectory-based method.

## How much starting crystal survives the melt?

Every `--mq-ensemble` run writes `melt_memory.txt`, `melt_memory.csv` and
`melt_memory.json` in its work directory, before the per-snapshot quenches.
The report compares the original input with the end of stage 3
(`shared/stage3_melted.xyz`), the end of stage 4 (`shared/stage4_eq.xyz`),
and every extracted high-temperature snapshot. Resume runs regenerate this
report from the input and available checkpoints.

The order criterion is the same Lechner–Dellago $\bar q_6$ descriptor used by
`--analyse --bond-order`: an atom must meet the `--qbar6-threshold` (default
0.3) and `--order-min-neighbors` (default 4). The `--order-cutoff` (falling
back to `--cutoff` when omitted) is resolved
once on the **original input**, then kept fixed for all comparisons while
using each frame's cell and periodic boundaries. Choose `--order-cutoff` and the
order threshold against crystalline and liquid references for your material;
the defaults are not a validated GeTe classifier.

The report contains the total ordered fraction, largest ordered cluster and
the surviving fraction of the initially ordered atoms. If $O_0$ denotes the
ordered atom indices in the original input and $O_t$ those at a checkpoint,
the survival diagnostic is

$$f_{\mathrm{surviving}}(t)=\frac{|O_0\cap O_t|}{|O_0|}.$$

Its denominator is the **initially ordered population**, not all atoms.
For example, if 80 of 100 input atoms meet the criterion and 20 of those
remain ordered at the end of heating, survival is 25%, even if additional
atoms become ordered. Cluster sizes count unique atoms in the cell and
include connections through periodic boundaries.

This measures order at the sampled endpoints. It cannot distinguish
continuous survival from melting followed by recrystallisation, establish
retention of a specific lattice, or prove liquid equilibration. High retained
order is a reason to inspect the melt before treating its quenches as
independent amorphous samples. Check final structures separately for ordering
that appears during quenching.

Atom identities follow the input's atom indices. The report checks atom
count and the full element sequence; missing checkpoints, incompatible
structures or an input with no ordered atoms yield an explicit unavailable
survival value (`null` in JSON), rather than zero survival. The pipeline
preserves atom order; rearranging same-species indices outside the pipeline
cannot be detected from the element sequence alone. JSON includes the
criterion, resolved cutoffs and per-checkpoint data: `initial_ordered_count`,
`retained_ordered_count`, `survival_fraction`, `lost_initial_order_count` and
`newly_ordered_count`. Missing checkpoint files are recorded as unavailable;
a trajectory is not silently substituted for an absent endpoint.

The options can also be supplied in an `analysis:` block in the MQ YAML:

```yaml
analysis:
  cutoff: auto-rdf
  order_cutoff: 3.5   # illustrative ideal-rocksalt GeTe shell; calibrate for your system
  qbar6_threshold: 0.3
  order_min_neighbors: 4
```

The report is automatic for `--mq-ensemble`; `analysis.bond_order: true` is
only needed when requesting the descriptor in a separate `--analyse` run.
See {doc}`analysis` for the equations, output conventions and calibration
guidance.

## HPC job-array tip

When splitting the per-snapshot quenches across SLURM array tasks, give **each
task its own output directory**. A single-input `--batch-quench` writes directly
to that directory, so `-o quench_runs/run_${TASK}` keeps the tasks separate and
preserves the usual ensemble layout.

A clean per-task command looks like:

```bash
TASK=$(printf "%04d" "$SLURM_ARRAY_TASK_ID")
mkdir -p inputs_per_task/task_${TASK}
cp snapshots/snapshot_${TASK}_frame*.xyz inputs_per_task/task_${TASK}/
amorphgen --batch-quench \
  --snapshot-dir inputs_per_task/task_${TASK} \
  --config mq.yaml --batch-stages 5 6 7 \
  --model chgnet --device cuda --resume \
  -o quench_runs/run_${TASK}   # separate output directory for each task
```

A full SLURM array template ships with the package at
`examples/run_quench_array_bluebear.slurm`.

## Choosing protocol parameters: a note on methodology

The example below illustrates a protocol; it is not a material-independent
validated recipe. Report and check the heating and cooling rates, equilibration
duration, temperature, density and snapshot spacing for the chosen material
and calculator. In particular:

- A `100 K/ps` heating or cooling rate is an input choice, not an assurance that
  the structure has equilibrated. Check convergence before extracting snapshots.
- `200000` steps at `0.5 fs` gives a `100 ps` high-temperature plateau. Longer
  duration alone does not establish that snapshots are independent.
- The example uses `4000 K`, above the package default of `3000 K`. Neither
  temperature guarantees reliable MLIP predictions; inspect the trajectory
  and validate the model for the system.
- NPT allows the volume to change. Use NVT when holding a validated density
  fixed, and inspect volume changes under NPT. See {doc}`best-practices`.

## Example `mq.yaml` for an oxide

```yaml
model: chgnet                       # or mace-mpa-0, sevennet, ...
device: cuda
# default_dtype is left at auto: float32 for chgnet (the only precision it
# runs at), float64 for MACE and SevenNet

# Stage 1: relax the crystalline supercell
opt:
  fmax: 0.05
  max_steps: 200
  optimizer: LBFGS
  cell_filter: none

# Stage 2: equilibrate at low T (NPT - let cell relax)
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
  T_end: 4000           # example; validate for the material and model
  T_step: 100
  rate: 100             # K/ps; tighter (slower) for better-equilibrated melt
  timestep: 0.5
  ttime: 25.0

# Stage 4: long high-T equilibration to decorrelate snapshots
eq_high:
  ensemble: NPT
  T: 4000
  steps: 200000         # 100 ps at 0.5 fs; check decorrelation before sampling
  timestep: 0.5
  ttime: 25.0

# Stage 5: cooling ramp (the actual quench)
quench:
  ensemble: NPT
  T_start: 4000
  T_end: 300
  T_step: -100
  rate: 100             # K/ps; check sensitivity to the cooling rate
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
final_opt:
  fmax: 0.05
  max_steps: 200
  optimizer: LBFGS
  cell_filter: FrechetCellFilter   # relax cell and positions; validate density
```

## Equivalent two-step manual workflow

If you want to inspect or analyse the stage-4 trajectory before quenching, run the two halves separately:

```bash
# Step 1: stages 1-4 only (writes stage4_eq_traj.xyz)
amorphgen GaO.xyz --config mq.yaml --stages 1 2 3 4 --resume -o shared/

# Step 2: extract N snapshots and quench each
amorphgen --batch-quench --snapshot-dir shared/stage4_eq_traj.xyz \
    --n-runs 20 --select decorrelated --batch-stages 5 6 7 \
    --config mq.yaml --resume -o quench_runs/
```

`--batch-quench` accepts a trajectory file directly (polymorphic `--snapshot-dir`), internally extracts up to N snapshots, then runs the per-snapshot stages. Set `--select decorrelated` to match the MQ-ensemble selection default. It produces the same per-run stage outputs, but does not collect a separate `final/` directory; that collection is part of `--mq-ensemble`.

## HPC / Slurm split (best for parallelism)

For a cluster with multiple GPUs, run the two halves as separate slurm jobs so the per-snapshot quenches can execute in parallel via a slurm array:

```bash
# Job 1: stages 1-4 (one shared trajectory)
sbatch shared.slurm

# After job 1 completes: one array task per extracted snapshot.
# Example if eight snapshots were selected:
sbatch --array=0-7 quench_array.slurm
```

`shared.slurm` and `quench_array.slurm` are your own job scripts; {doc}`hpc` has a SLURM header to start from. Job 1 runs step 1 of the two-step workflow above, then extracts the snapshots:

```bash
amorphgen GaO.xyz --config mq.yaml --stages 1 2 3 4 --resume -o shared/
amorphgen --extract-snapshots shared/stage4_eq_traj.xyz \
    --config mq.yaml -n 20 --select decorrelated -o snapshots/
```

Size job 2's array from the **actual number of extracted snapshots**, which
may be less than 20. For example, `#SBATCH --array=0-7` runs eight snapshots.
Each task sets `TASK=$(printf "%04d" $SLURM_ARRAY_TASK_ID)` and runs the
per-task command from the job-array tip above. To automate submission,
arrange for job 1 to submit the correctly sized array after extraction.
This is the same dispatch as `--mq-ensemble`, split for HPC
parallelism; `examples/run_quench_array_bluebear.slurm` is a job-2 template
for one cluster; adjust its array range to the extracted count.

| Pattern | Execution | Best for |
|---------|-----------|----------|
| `--mq-ensemble` (single command) | Shared melt, then sequential quenches | Local / single-GPU |
| Two SLURM jobs (shared + array) | Shared melt, then independently scheduled quenches | HPC with array support |

Measure one representative run to estimate wall time; system size, model, protocol and available GPU concurrency all affect it.

## Resume behaviour

| Interruption point | What `--resume` recovers |
|--------------------|---------------------------|
| Mid stages 1-4 | Skips completed stages and continues the interrupted MD stage from its last saved trajectory frame; optimisation stages restart from the beginning. |
| Between stage 4 and snapshot extraction | Skips stages 1-4, re-extracts snapshots, runs 5-7. |
| Mid quench-runs | Skips completed runs (looks for `final_amorphous.xyz`), resumes interrupted MD from saved trajectory frames, and restarts final optimisation. |
| After all done | Skips completed simulation work and rebuilds the snapshot and final collections. |

When quench outputs already exist, the saved sampling report protects the
mapping between each run and its source snapshot: incompatible selection
changes are rejected. A legacy run without a sampling report cannot safely
resume with adaptive selection. Use its original explicit `--select uniform`
or `--select last` settings, or start a new output directory for decorrelated
sampling.

## When to use `--mq-ensemble` vs the alternatives

| Use case | Recommended mode |
|----------|------------------|
| Follow a crystal → liquid → quench protocol | `--mq-ensemble` (full crystal → liquid → quench protocol) |
| Generate amorphous structures from random starting points | `--hybrid-ensemble` ({doc}`hybrid-workflow`) |
| Single amorphous structure (no ensemble) | Default pipeline (no flag, just `amorphgen INPUT --config ...`) |
| Quench existing snapshots or frames from a trajectory | `--batch-quench --snapshot-dir PATH` |

## Validation

To compare structural metrics with reference ranges, pair `--mq-ensemble` with analysis and a reference YAML:

```bash
amorphgen --analyse --input-dir ga2o3_mq/final/ \
    --cutoff auto-rdf --per-structure --bond-order \
    --reference reference_a_Ga2O3.yaml \
    --save-report mq_report.txt --save-plot mq_plots/ --save-pdf
```

This writes structural analysis (RDF, CN, bond angles, $q_6$/$\bar q_6$ and
ordered clusters) and a table comparing available reference-supported metrics
with the supplied ranges. Bond order is a separate diagnostic and is not
scored by the reference YAML. Check the provenance of each range and use
additional validation appropriate to the intended application. See
{doc}`yaml-config` for the reference YAML format.
