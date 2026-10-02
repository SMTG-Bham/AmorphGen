# Command-line interface

AmorphGen provides a single `amorphgen` command with multiple modes of operation.

## Usage

```text
amorphgen [INPUT] [OPTIONS]                      # Full melt-quench pipeline
amorphgen [INPUT] --mq-ensemble [OPTIONS]         # MQ ensemble (1-4 + N×5-7)
amorphgen --hybrid-ensemble --input-dir DIR       # Hybrid ensemble (4-7 per input)
amorphgen --random-gen [OPTIONS]                  # Random generation mode
amorphgen --batch-quench --snapshot-dir DIR       # Batch quench mode
amorphgen --batch-opt --input-dir DIR             # Batch optimisation mode
amorphgen --analyse --input-dir DIR               # Structure analysis mode
amorphgen --extract-snapshots TRAJ                # Extract N snapshots from a trajectory
amorphgen --rank-from-log LOG                     # Energy ranking from log file
amorphgen --convert PATH --format vasp            # Structure format conversion
amorphgen --list-models                           # Show available models
```

Run `amorphgen --help` for all flags and `amorphgen --examples` for built-in
command examples. Commands below run from a repository checkout when they
refer to files under `examples/`. Supply your own input structures, and save
the configurations from {doc}`/guides/mq-ensemble` and
{doc}`/guides/hybrid-workflow` as `mq.yaml` and `hybrid.yaml` before using them.

## CLI reference

```{eval-rst}
.. automodule:: amorphgen.cli
   :members:
```

## Examples

### Full pipeline

```bash
amorphgen POSCAR --model mace-mpa-0 --device cuda
```

### Full pipeline with YAML config

```bash
amorphgen POSCAR --config examples/full_pipeline.yaml
```

### MQ-ensemble (full pipeline + N quenches in one command)

Runs stages 1–4 once, extracts up to N snapshots from the stage-4 trajectory,
and runs stages 5–7 on each snapshot. The default `--select decorrelated`
chooses burn-in and spacing from scalar autocorrelation and per-species
diffusion. A short or slowly diffusing trajectory can yield fewer than N
snapshots. The default 10 ps stage-4 hold is not extended automatically.

```bash
amorphgen Ga2O3_supercell.xyz --mq-ensemble --n-structures 20 \
    --config mq.yaml --device cuda --model chgnet \
    -o ga2o3_mq/
```

Output layout:

```text
ga2o3_mq/
├── snapshot_sampling.{json,txt} # selection diagnostics and effective snapshot count
├── shared/        # stages 1-4 outputs (incl. stage4_eq_traj.xyz)
├── snapshots/     # up to 20 selected snapshots
├── quench_runs/   # per-snapshot stages 5-6-7 outputs
└── final/         # collected mq_NNNN.<fmt> files
```

The sampling report includes selected frame indices, burn-in, spacing,
autocorrelation and diffusion diagnostics, and an estimated effective
independent snapshot count. That estimate does not establish equilibrium
or independence of the final glasses. See {doc}`/guides/mq-ensemble` for the
method, limitations and resume compatibility checks.

| Sampling option | Behaviour |
|-----------------|-----------|
| `--select decorrelated` | Default for `--mq-ensemble`; choose spacing from trajectory diagnostics. |
| `--select uniform` / `--select last` | Explicit legacy selection; uniform remains the default for extraction and batch quenching. |
| `--burn-in-frames N` | Discard exactly N leading saved frames, including an explicit zero. If omitted, decorrelated selection chooses adaptive burn-in; legacy modes use zero. |
| `--decorrelation-distance D` | Distance in Å for the per-species displacement correlation proxy; default is the final frame's median nearest-neighbour distance. |

### Hybrid ensemble (random + quench)

Take a directory of disordered structures (e.g. the `random_opt/` subdirectory of a `--random-gen --relax` run) and run stages 4-5-6-7 on each:

```bash
amorphgen --hybrid-ensemble --input-dir random_structures/random_opt/ \
    --config hybrid.yaml --device cuda --model chgnet \
    -o ga2o3_hybrid/
```

Useful for producing an amorphous ensemble from already-disordered starting structures (skipping the crystal-melt steps).

### Extract snapshots from a trajectory

Standalone utility to extract up to N frames from a trajectory file, using
uniform spacing by default:

```bash
amorphgen --extract-snapshots stage4_eq_traj.xyz \
    -n 20 --select uniform -o snapshots/
```

`-n` (or `--n-structures`) and the legacy `--n-runs` both control the snapshot
count. Use `--format vasp` for POSCAR-style output or `--format cif` for CIF
instead of the default extxyz `.xyz`:

```bash
amorphgen --extract-snapshots stage4_eq_traj.xyz \
    -n 20 --burn-in-frames 50 --format vasp -o snapshots/
```

To use the same adaptive sampling as `--mq-ensemble`, opt in explicitly:

```bash
amorphgen --extract-snapshots stage4_eq_traj.xyz \
    --config mq.yaml -n 20 --select decorrelated -o snapshots/
```

Use the configuration that generated the trajectory: reported times use
`eq_high.timestep` and assume one saved frame every 100 MD steps. The Python
sampling API supports other saved-frame intervals through `timestep_fs` and
`frame_stride`.

`--batch-quench` also accepts a trajectory file directly via `--snapshot-dir <file.xyz>` and extracts internally:

```bash
amorphgen --batch-quench --snapshot-dir stage4_eq_traj.xyz \
    --n-runs 20 --select decorrelated --batch-stages 5 6 7 \
    --config mq.yaml -o quench_runs/
```

### Hybrid workflow (skip heating)

```bash
amorphgen structure.xyz --stages 1 4 5 6 7 --config hybrid.yaml
```

### Pipeline with resume

```bash
amorphgen POSCAR --stages 1 4 5 6 7 --config config.yaml --resume
```

The `--resume` flag skips completed stage checkpoints and continues an
interrupted MD stage from its last saved trajectory frame. Unfinished
optimisation stages restart; optimiser state is not checkpointed. Resume the
same protocol and work directory.

### Random generation

```bash
# Formula format (In2O3 * 8 units = 40 atoms)
amorphgen --random-gen --composition "In2O3*8" --n-structures 20

# Atom count format (equivalent)
amorphgen --random-gen --composition In=16,O=24 --n-structures 20

# With relaxation and cubic cell constraint
amorphgen --random-gen --composition "Li2ZrCl6*4" \
    --relax --model chgnet --device cpu --cell-filter cubic

# Custom output directory
amorphgen --random-gen --composition "SiO2*16" \
    --work-dir my_SiO2_structures/
```

### Batch optimisation

```bash
# --random-gen writes its structures to random_initial/ inside its work dir
amorphgen --batch-opt --input-dir random_structures/random_initial/ \
    --model chgnet --cell-filter cubic
```

### Batch quench with resume

```bash
amorphgen --batch-quench \
    --snapshot-dir snapshots/ \
    --model mace-mpa-0 \
    --batch-stages 5 6 7 \
    --resume
```

### Structure analysis

```bash
# Basic analysis (RDF, CN, bond angles, density)
amorphgen --analyse --input-dir optimised/

# Per-structure comparison table + total RDF + Gaussian smearing for experimental comparison
amorphgen --analyse --input-dir optimised/ \
    --cutoff auto-rdf --per-structure --total-rdf --smearing 0.05 \
    --save-report report.txt --save-plot plots/

# Publication-quality plots (600 DPI, vector PDF)
amorphgen --analyse --input-dir optimised/ \
    --save-plot figs/ --save-pdf --dpi 600

# Validate against literature ranges defined in a reference YAML
amorphgen --analyse --input-dir optimised/ \
    --reference examples/reference_a_Ga2O3.yaml

# Structure factor S(q): direct reciprocal-space sum, neutron weighting, PNG + CSV
amorphgen --analyse --input-dir optimised/ --sq --sq-weighting neutron --save-plot plots/

# Same neutron weighting, using the Fourier-transform-of-g(r) route
amorphgen --analyse --input-dir optimised/ --sq --sq-method ft \
    --sq-weighting neutron --save-plot plots/

# Measured S(q): CSV header, optional third column of one-sigma errors
amorphgen --analyse --input-dir optimised/ \
    --experiment-sq measured_sq.csv --experiment-skiprows 1 \
    --sq-qmax 12 --sq-fit-range 1.5 10 --save-plot comparison/ --save-report fit.txt

# Measured T(r): match the experiment's Fourier window and q range
amorphgen --analyse --input-dir optimised/ --experiment-tr measured_tr.dat \
    --tr-qrange 0.5 20 --tr-window lorch --tr-fit-range 1 8 --save-plot comparison/

# Coherent X-ray intensity per atom, with pointwise ensemble confidence bands
amorphgen --analyse --input-dir optimised/ --xrd \
    --xrd-wavelength 1.5406 --xrd-qmax 8 --xrd-nq 400 --save-plot plots/
```

Experimental inputs accept two or three numeric columns: coordinate, value,
and optional positive one-sigma uncertainty. For extra columns, select two or
three zero-based indices with `--experiment-columns`. CSV and whitespace are
detected automatically. Experimental file flags enable `--sq` or `--tr`;
comparison exports contain JSON, CSV, TXT, PNG and optional PDF, including
fit metrics, residuals and pointwise ensemble bands. `--sq-nq` sets the S(q)
grid size (default 300). CLI options also work as underscore-named keys in
the YAML `analysis` block. See {doc}`/guides/analysis` for units and conventions.

### Rank structures by energy (from a random-gen log)

When outputs are written as `.vasp` or `.cif` (which don't carry per-atom
energy), parse the relax log to rank structures by total energy without
re-running the calculator:

```bash
amorphgen --rank-from-log random_structures/random_gen.log
```

Returns a sorted table of structure index, total energy, energy/atom, fmax,
and step count. Useful for selecting the lowest-energy member of an ensemble
for follow-up DFT work.

### List available models

```bash
amorphgen --list-models
```

## Default output directories

| Mode | Default `--work-dir` |
|------|---------------------|
| Pipeline | `melt_quench_run/` |
| `--mq-ensemble` | `mq_ensemble_run/` |
| `--hybrid-ensemble` | `hybrid_run/` |
| `--random-gen --composition Si=16,O=32` | `random_O32Si16/` (Hill-order formula) |
| `--random-gen --composition "SiO2*16"` | `random_structures/` |
| `--batch-quench` | `batch_quench/` |
| `--batch-opt` | `batch_opt/` |
| `--extract-snapshots` | `snapshots/` |
| `--convert DIR --format FMT` | `<DIR>_<FMT>/` |
| `--convert FILE --format FMT` | Input file's directory |

Override any default with `--work-dir my_dir/`.
