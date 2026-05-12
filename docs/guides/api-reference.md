# CLI & Python API Reference

Complete reference for all AmorphGen commands and Python API.

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
# Optimise all structures in a directory
amorphgen --batch-opt --input-dir random_structures/ \
          --model chgnet --device cpu --fmax 0.01

# With cell filter
amorphgen --batch-opt --input-dir random_structures/ \
          --cell-filter cubic

# Cell filter options: FrechetCellFilter (default), UnitCellFilter,
#                      ExpCellFilter, StrainFilter, cubic, none
```

### Batch Quench

```bash
# Extract snapshots from high-T trajectory and quench each independently
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
# Auto cutoff (default, fast)
amorphgen --analyse --input-dir optimised/

# RDF-based auto cutoff (more accurate)
amorphgen --analyse --input-dir optimised/ --cutoff auto-rdf

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
    cn_tolerance=0,                  # 0=strict (default), 1=allow +1 over-CN
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
    cn_tolerance=1,                  # allow +1 for tighter CN matching
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
    input_dir="random_SiO2",
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
    n_runs=20,
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
  cn_tolerance: 0            # 0=strict (default), 1=allow +1 over-coordination
  output_format: xyz         # xyz (.xyz, extxyz format), vasp, cif
  relax: true                # optimise after generation
  cell_filter: cubic         # cell constraint for relaxation

# -- Analysis (used with --analyse) --
analysis:
  cutoff: auto               # auto, auto-rdf, or float
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
| 3 | `melt_cell.py` | Heating ramp | `stage3_melt.{log,xyz}` |
| 4 | `equilibrate.py` | High-T equilibration | `stage4_eq.{log,xyz}` |
| 5 | `quench.py` | Cooling ramp | `stage5_quench.{log,xyz}` |
| 6 | `equilibrate.py` | Low-T equilibration | `stage6_eq.{log,xyz}` |
| 7 | `final_opt.py` | Final optimisation | `stage7_opt.{log,cif,xyz}` |

Additional output: `pipeline_summary.log` with per-stage timing.

## Default Output Directories

| Mode | Default `--work-dir` |
|------|---------------------|
| Pipeline | `melt_quench_run/` |
| `--random-gen --composition X=n,Y=m` | `random_XnYm/` (auto from formula) |
| `--batch-quench` | `batch_quench/` |
| `--batch-opt` | `batch_opt/` |

## Random Generation: Automated Defaults

All parameters are auto-detected from the composition. No manual tuning needed.

### Minsep (minimum interatomic distances)

| Bond type | Scale factor | Radius source |
|-----------|-------------|---------------|
| Ionic (M-O, M-Cl) | 0.80 | Shannon ionic (CN from auto SC) |
| Covalent (Si-Si) | 0.80 | Metallic radii |
| Metallic (M-M) | 0.85 | max(metallic, sqrt(2)*d(M-X)*0.85), cap 2.80 A |
| Small anion (O-O, F-F) | 0.80 | Shannon ionic |
| Large anion (Cl-Cl, Br-Br) | 0.70 | Shannon ionic |

### Density estimation

| Material class | Method | Packing factor |
|----------------|--------|----------------|
| Group IV (Si, Ge, SiGe) | Elemental density * 0.80 | -- |
| Pnictide (GaAs, InP) | Elemental density * 0.80 | -- |
| Chalcogenide (ZnS, CdTe, GeTe) | Elemental density * 0.80 | -- |
| Carbide (SiC, TiC) | Elemental density * 0.80 | -- |
| Boride (TiB2, LaB6) | Elemental density * 0.80 | -- |
| Alloy (NiTi, CuZn) | Elemental density * 0.80 | -- |
| Covalent oxide (SiO2, GeO2, B2O3) | Shannon CN=6 sphere packing | 0.50 |
| Metal oxide (In2O3, TiO2, Al2O3) | Shannon CN=6 sphere packing | 0.52 |
| Halide (Li2ZrCl6, LiF, NaCl) | Shannon CN=6 sphere packing | 0.58 |
| Nitride (AlN, GaN, Si3N4) | Shannon CN=6 sphere packing | 0.52 |
| Hydride (LiH, MgH2) | Shannon CN=6 sphere packing | 0.55 |

### Coordination-aware placement (auto coordination targeting)

| Material class | Target CN | Tolerance | Accepts |
|----------------|-----------|-----------|---------|
| Covalent oxide (Si, Ge, B) | 4 | 0 | 4 only |
| Metal oxide (In, Ti, Al, Zn) | 5 | 1 | 4-6 |
| Nitride (AlN, GaN) | 4 | 0 | 4 only |
| Halide (NaCl, LiF) | 6 | 0 | 6 only |
| Chalcogenide (ZnS, CdTe) | 4-6 | 0 | tetrahedral or octahedral |
| Carbide (SiC, TiC) | 4-6 | 0 | metalloid=4, metal=6 |
| Hydride (LiH, MgH2) | 6 | 0 | 6 only |
| Boride (TiB2) | 6 | 0 | 6 only |
| Pnictide (GaAs, InP) | 4 | 0 | 4 only |
| Group IV (Si, Ge) | 4 | 0 | 4 only |
| Alloy (NiTi, CuZn) | -- | -- | no SC |

### Auto-retry on placement failure

1. Reduce M-M minsep by 5% (retry)
2. Reduce M-M minsep by 10% (retry)
3. Reduce M-M minsep by 15% (retry)
4. Reduce M-M minsep by 20% (retry)
5. Skip and warn

All defaults can be overridden via CLI flags (`--target-cn`, `--cn-tolerance`,
`--minsep`, `--target-density`) or YAML config.

## Cell Filter Options

| Value | Description |
|-------|-------------|
| `FrechetCellFilter` | Default. Riemannian metric, best convergence for non-cubic cells |
| `UnitCellFilter` | Classic ASE filter, relaxes full cell in Cartesian |
| `ExpCellFilter` | Exponential cell filter, good for large deformations |
| `StrainFilter` | Relaxes cell via strain tensor only (no positions) |
| `cubic` | Isotropic volume only (keeps a=b=c, 90 deg angles) |
| `none` | Fixed cell, positions only |
