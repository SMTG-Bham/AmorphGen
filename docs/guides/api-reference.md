# CLI & Python API Reference

Common CLI workflows and Python API examples. Use `amorphgen --help` for
all CLI options and the {doc}`/api/cli` page for generated API documentation.

## CLI Commands

### Structure Generation

```bash
# Formula format: SiO2 * 16 formula units = 48 atoms
amorphgen --random-gen --composition "SiO2*16"

# Atom count format (equivalent)
amorphgen --random-gen --composition Si=16,O=32

# With target density
amorphgen --random-gen --composition "SiO2*16" --target-density 2.2

# Coordination-aware placement with CN-aware radii
amorphgen --random-gen --composition "SiO2*16" --target-cn Si=4,O=2

# With explicit bonding distances (overrides auto minsep)
amorphgen --random-gen --composition "Li2ZrCl6*4" \
          --target-cn Zr=6,Li=6 --dmax Zr-Cl=3.2,Li-Cl=3.2

# With relaxation and cell constraint
amorphgen --random-gen --composition "SiO2*16" \
          --relax --model chgnet --device cpu --cell-filter cubic

# Custom minsep (overrides auto radii-based calculation)
amorphgen --random-gen --composition "In2O3*8" \
          --minsep In-In=2.8,In-O=1.9,O-O=2.5

# Output formats: xyz (default, extxyz format), vasp, cif
amorphgen --random-gen --composition "SiO2*16" --format vasp

# With YAML config
amorphgen --random-gen --config config.yaml
```

### Batch Optimisation

```bash
# Optimise all structures in a directory (--random-gen writes to random_initial/)
amorphgen --batch-opt --input-dir random_structures/random_initial/ \
          --model chgnet --device cpu --fmax 0.01

# With cell filter
amorphgen --batch-opt --input-dir random_structures/random_initial/ \
          --cell-filter cubic

# Cell filter options: cubic (batch-opt CLI default), FrechetCellFilter,
#                      UnitCellFilter, ExpCellFilter, StrainFilter, none
```

### Batch Quench

```bash
# Select existing snapshots and quench each independently
amorphgen --batch-quench \
          --snapshot-dir snapshots/ \
          --n-runs 20 \
          --select uniform \
          --batch-stages 5 6 7

# Resume interrupted batch (skip completed runs)
amorphgen --batch-quench \
          --snapshot-dir snapshots/ \
          --n-runs 20 --resume
```

### Melt-Quench Pipeline

```bash
# Full 7-stage pipeline
amorphgen POSCAR --model mace-mpa-0 --device cuda

# With YAML config (recommended)
amorphgen POSCAR --config pipeline.yaml

# Hybrid MQ (skip melt, start from random structure)
amorphgen random_SiO2.xyz --stages 1 4 5 6 7 --config hybrid.yaml

# Resume from checkpoint (auto-detects completed stages)
amorphgen POSCAR --stages 1 4 5 6 7 --config pipeline.yaml --resume

# Custom fine-tuned model
amorphgen POSCAR --model-path /data/models/custom.model --device cuda

# Custom parameters via CLI flags
amorphgen POSCAR --model chgnet --device cpu \
          --timestep 0.5 \
          --fmax 0.05 --opt-steps 500 \
          --eq-high-T 3000 --eq-high-steps 200000 --eq-high-ensemble NVT \
          --quench-T-start 3000 --quench-T-end 300 --quench-T-step -100 \
          --quench-steps-per-T 2000 --quench-ensemble NVT \
          --eq-low-T 300 --eq-low-steps 100000
```

### Structure Analysis

```bash
# RDF-based auto cutoff (default)
amorphgen --analyse --input-dir optimised/

# Radii-based auto cutoff (faster heuristic)
amorphgen --analyse --input-dir optimised/ --cutoff auto

# Manual cutoff
amorphgen --analyse --input-dir optimised/ --cutoff 2.2

# Save report and plots
amorphgen --analyse --input-dir optimised/ \
          --save-report report.txt --save-plot plots/

# Single structure
amorphgen --analyse stage7_opt.xyz
```

### Utility

```bash
# List all available MLIP models
amorphgen --list-models

# Show help
amorphgen --help
```

---

## Python API

### Structure Generation

```python
from amorphgen import generate_random, batch_random  # top-level imports

# Single structure (auto minsep from Shannon radii)
atoms = generate_random(
    composition={"Si": 16, "O": 32},  # atom counts (use CLI for formula format)
    target_density=2.2,             # g/cm3 (optional, auto-estimated if omitted)
    target_cn={"Si": 4, "O": 2},    # coordination-aware placement + CN-aware radii (optional)
    cn_tolerance=0,                  # 0=no over-CN, 1=allow +1 (not a lower bound)
    seed=42,
    # minsep={"Si-O": 1.6},         # custom minsep (overrides auto if given)
    # dmax={"Si-O": 2.1},           # custom bonding shell (auto if None)
)

# Batch generation with relaxation
from amorphgen.utils import get_calculator
calc = get_calculator(model="chgnet", device="cpu")

paths = batch_random(
    composition={"Si": 16, "O": 32},
    n_structures=5,
    output_dir="random_SiO2",
    output_format="xyz",            # xyz (.xyz, extxyz format), vasp, cif
    target_density=2.2,
    target_cn={"Si": 4, "O": 2},
    cn_tolerance=1,                  # allow +1 over the target during placement
    seed=42,
    relax=True,
    calc=calc,
    cell_filter="cubic",            # FrechetCellFilter, UnitCellFilter, cubic, none
)
```

### Batch Optimisation

```python
from amorphgen.pipeline.opt_cell import batch_optimize
from amorphgen.utils import get_calculator

calc = get_calculator(model="chgnet", device="cpu")

paths = batch_optimize(
    input_dir="random_SiO2/random_initial",
    output_dir="optimised_SiO2",
    cfg_override={
        "opt": {
            "fmax": 0.01,
            "max_steps": 1000,
            "optimizer": "LBFGS",
            "cell_filter": "FrechetCellFilter",  # default
        }
    },
    calc=calc,
    pattern="*.xyz",
)
```

### Melt-Quench Pipeline

```python
from amorphgen import MeltQuenchPipeline

pipe = MeltQuenchPipeline(
    input_file="random_SiO2.xyz",
    work_dir="mq_run",
    cfg_override={
        "model": "mace-mpa-0",
        "device": "cuda",
        "default_dtype": "float64",
        "opt": {
            "fmax": 0.05,
            "max_steps": 500,
            "cell_filter": "none",       # fixed volume
        },
        "eq_high": {
            "ensemble": "NVT",
            "T": 3000,
            "steps": 200000,             # 100 ps at 0.5 fs
            "timestep": 0.5,
        },
        "quench": {
            "ensemble": "NVT",
            "T_start": 3000,
            "T_end": 300,
            "T_step": -100,
            "rate": 100,                 # K/ps (auto-calculates steps_per_T)
            "timestep": 0.5,
        },
        "eq_low": {
            "ensemble": "NVT",
            "T": 300,
            "steps": 100000,             # 50 ps at 0.5 fs
            "timestep": 0.5,
        },
    },
)

# Full pipeline
atoms = pipe.run()

# Hybrid (skip heating)
atoms = pipe.run(stages=[1, 4, 5, 6, 7])

# Resume from checkpoint
atoms = pipe.run(stages=[1, 4, 5, 6, 7], resume=True)
```

A stage can also run on its own. `work_dir` sets where its log, trajectory and
output structure go (default: the current directory):

```python
from amorphgen.pipeline import equilibrate
from amorphgen.utils.calculators import get_calculator

calc = get_calculator("mace-mpa-0", device="cuda")
liquid = equilibrate.run(
    "mq_run/stage1_opt.xyz", calc=calc, stage="high",
    cfg_override={"eq_high": {"ensemble": "NVT", "T": 3000, "steps": 20000}},
    work_dir="eq_3000K",
)  # eq_3000K/stage4_eq.log, stage4_eq_traj.xyz, stage4_eq.xyz
```

### Structure Analysis

```python
from amorphgen.analysis import StructureAnalyser

# Load from directory, file, or list
sa = StructureAnalyser("optimised/", cutoff="auto")
sa = StructureAnalyser("structure.xyz", cutoff=2.2)
sa = StructureAnalyser([atoms1, atoms2], cutoff="auto-rdf")

# Core analysis
sa.density()                           # {"mean": 2.23, "std": 0.21, ...}
sa.coordination()                      # {"Si-O": {"mean": 4.0, ...}}
sa.bond_distances()                    # {"O-Si": {"mean": 1.665, ...}}
sa.bond_angles()                       # {"O-Si-O": {"mean": 108.9, ...}}

# RDF and S(q)
sa.rdf()                               # total RDF, auto rmax
sa.rdf(pair="O-Si", rmax=4.0)         # partial, manual rmax
sa.structure_factor(pair="O-Si")       # S(q) from Fourier of g(r)
sa.averaged_rdf(pair="O-Si")           # per-structure mean +/- std

# Advanced
sa.ring_statistics(bond_pair=("Si", "O"))  # ring size distribution
sa.energy_ranking()                         # rank by potential energy
sa.averaged_cn()                            # CN per structure with error bars

# Reporting
text = sa.summary()
sa.save_report("report.txt")
sa.plot(output_dir="plots/", prefix="SiO2",
        rdf_pairs=["O-Si", "O-O"],
        angle_triplets=["O-Si-O"],
        angle_style="line")
```

### Batch Quench (Python API)

```python
from amorphgen.pipeline import batch_quench
from amorphgen.utils import get_calculator

calc = get_calculator(model="chgnet", device="cpu")

batch_quench.run(
    snapshot_files=["snap_0.xyz", "snap_1.xyz"],
    n_runs=2,
    select="uniform",
    work_dir="batch_run",
    stages=[5, 6, 7],
    calc=calc,
    resume=True,
)
```

### YAML Configuration

```python
from amorphgen.configs import load_yaml_config

cfg = load_yaml_config("config.yaml")
pipe = MeltQuenchPipeline(input_file="POSCAR", cfg_override=cfg)
atoms = pipe.run(stages=[1, 4, 5, 6, 7], resume=True)
```

### List Available Models

```python
from amorphgen.utils import list_models
list_models()
```

---

## YAML Config Reference

This is an example configuration, not a dump of all package defaults.
See {doc}`yaml-config` for merging rules and per-stage overrides.

```yaml
# -- Model --
model: mace-mpa-0          # or chgnet, 7net-mf-ompa, buckingham, lennard-jones, etc.
device: cuda                # cuda, cpu, mps, auto
default_dtype: float64      # float64 (recommended for opt), float32 (faster MD)

# -- Optimisation (Stages 1 & 7) --
opt:
  fmax: 0.05                # eV/A convergence
  max_steps: 500
  optimizer: LBFGS           # LBFGS, FIRE, BFGS, BFGSLineSearch, MDMin
  cell_filter: FrechetCellFilter # FrechetCellFilter (default), UnitCellFilter,
                                  # ExpCellFilter, StrainFilter, cubic, none

# -- Stage 2: Pre-melt equilibration --
eq_premelt:
  ensemble: NVT             # NVT or NPT
  T: 300                    # K
  steps: 50000              # number of MD steps
  timestep: 0.5             # fs
  friction: 0.01            # Langevin damping (1/fs), default 0.01

# -- Stage 3: Heating ramp --
melt:
  ensemble: NPT             # NPT or NVT
  T_start: 300              # K
  T_end: 3000               # K
  T_step: 100               # K per segment
  rate: 100                 # K/ps (auto-calculates steps_per_T)
  # steps_per_T: 1000       # alternative to rate (rate takes priority)
  timestep: 0.5             # fs
  friction: 0.01            # optional, default 0.01
  ttime: 25.0               # NPT thermostat coupling (fs), default 25.0

# -- Stage 4: High-temperature equilibration --
eq_high:
  ensemble: NVT
  T: 3000                   # K (should match melt T_end)
  steps: 50000
  timestep: 0.5

# -- Stage 5: Quench (cooling ramp) --
quench:
  ensemble: NVT
  T_start: 3000             # K (should match eq_high T)
  T_end: 300                # K
  T_step: -100              # K (negative = cooling)
  rate: 100                 # K/ps (auto-calculates steps_per_T)
  # steps_per_T: 1000       # alternative to rate
  timestep: 0.5

# -- Stage 6: Low-temperature equilibration --
eq_low:
  ensemble: NVT
  T: 300                    # K (should match quench T_end)
  steps: 10000
  timestep: 0.5

# -- Random Generation (used with --random-gen) --
random_gen:
  composition:
    Si: 16
    O: 32
  n_structures: 5
  target_density: 2.2       # g/cm3 (optional, auto-estimated if omitted)
  target_cn:                 # optional, enables coordination-aware placement
    Si: 4
    O: 2
  cn_tolerance: 0            # strict upper target; 1 allows +1 over-coordination
  output_format: xyz         # xyz (.xyz, extxyz format), vasp, cif
  relax: true                # optimise after generation
  cell_filter: cubic         # cell constraint for relaxation

# -- Analysis (used with --analyse) --
analysis:
  cutoff: auto-rdf           # default; auto, float or per-pair overrides also accepted
  save_report: report.txt
  save_plot: plots/
  rdf_pairs:
    - Si-O
    - O-O
  angle_triplets:
    - O-Si-O
  energy_ranking: true
```

---

## Pipeline Stages

| Stage | Module | Description | Output files |
|:---:|--------|-------------|-------------|
| 1 | `opt_cell.py` | Structure optimisation | `stage1_opt.{log,cif,xyz}` |
| 2 | `equilibrate.py` | Pre-melt equilibration | `stage2_eq.{log,xyz}` |
| 3 | `melt_cell.py` | Heating ramp | `stage3_melt.log`, `stage3_melted.xyz` |
| 4 | `equilibrate.py` | High-T equilibration | `stage4_eq.{log,xyz}` |
| 5 | `quench.py` | Cooling ramp | `stage5_quench.log`, `stage5_quenched.xyz` |
| 6 | `equilibrate.py` | Low-T equilibration | `stage6_eq.{log,xyz}` |
| 7 | `final_opt.py` | Final optimisation | `stage7_opt.{log,cif,xyz}` |

MD stages also write `stage2_eq_traj.xyz`, `stage3_melt_traj.xyz`,
`stage4_eq_traj.xyz`, `stage5_quench_traj.xyz` and `stage6_eq_traj.xyz`.
The pipeline writes `pipeline_summary.log` with per-stage timing.

## Default Output Directories

| Mode | Default `--work-dir` |
|------|---------------------|
| Pipeline | `melt_quench_run/` |
| `--random-gen --composition X=n,Y=m` | `random_<Hill formula>/` (derived from atom counts) |
| `--random-gen` with formula syntax or YAML composition | `random_structures/` |
| `--batch-quench` | `batch_quench/` |
| `--batch-opt` | `batch_opt/` |
| `--mq-ensemble` | `mq_ensemble_run/` |
| `--hybrid-ensemble` | `hybrid_run/` |

## Random Generation: Automated Defaults

Composition-derived defaults provide a starting point; they do not guarantee
a validated density or coordination distribution.

- **Minimum separations:** bond classification selects Shannon ionic, Cordero
  covalent or metallic radii, with separate handling for anion packing and
  cation–cation contacts. Explicit `--target-cn` also selects CN-aware radii.
- **Density:** material classes select a radius source and packing factor. For
  example, `group_iv` uses Cordero radii at packing factor 0.30, `metal_oxide`
  uses Shannon CN6 at 0.52, and `rutile_dioxide` uses Shannon CN6 at 0.66.
  Set `--target-density` when a validated density is available.
- **Coordination:** targets and over-coordination tolerance are derived from
  the material class unless overridden. SC biases placement toward those
  targets; under-coordinated sites can remain. Use `--no-sc` to disable it.
- **Placement stalls:** `--retry-mode expand` (default) first attempts soft
  packing, then expands the cell if needed. `reduce-minsep` preserves the cell
  while softening non-bonded separations; `none` preserves both. A batch can
  retry with other seeds and ultimately skip structures that cannot be placed.

See {doc}`random-generation` for the material-class tables, per-pair rules
and retry policies. Override them through `--target-cn`, `--cn-tolerance`,
`--minsep`, `--target-density`, `--retry-mode` or the `random_gen:` YAML block.

## Cell Filter Options

| Value | Description |
|-------|-------------|
| `FrechetCellFilter` | Full cell and position relaxation; default for the melt-quench pipeline |
| `UnitCellFilter` | Classic ASE filter, relaxes full cell in Cartesian |
| `ExpCellFilter` | Exponential cell filter; deprecated in ASE 3.23 in favour of `FrechetCellFilter`, which corrects its cell gradients |
| `StrainFilter` | Relaxes cell via strain tensor only (no positions) |
| `cubic` | Isotropic volume relaxation that preserves the input cell shape; a cubic input stays cubic. Default for the random-gen, batch-opt and hybrid CLI modes |
| `none` | Fixed cell, positions only |
