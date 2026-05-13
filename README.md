# AmorphGen

[![CI](https://github.com/SMTG-Bham/AmorphGen/actions/workflows/test.yml/badge.svg)](https://github.com/SMTG-Bham/AmorphGen/actions/workflows/test.yml)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://smtg-bham.github.io/AmorphGen/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AmorphGen: Amorphous structure generation via melt-quench MD and random placement with universal MLIPs (MACE, CHGNet, SevenNet).**

📚 **Documentation:** [smtg-bham.github.io/AmorphGen](https://smtg-bham.github.io/AmorphGen/)

---

## Pipeline overview

```
Crystalline input  (POSCAR / .xyz / .cif / .extxyz)
         │
   ┌─────▼──────────────────────────────────────────┐
   │  Stage 1  Structure optimisation               │
   │           optimizer + cell_filter              │
   └─────┬──────────────────────────────────────────┘
         │
   ┌─────▼──────────────────────────────────────────┐
   │  Stage 2  Pre-melt equilibration at T-low      │
   │           NVT/NPT                              │
   └─────┬──────────────────────────────────────────┘
         │
   ┌─────▼──────────────────────────────────────────┐
   │  Stage 3  Melt  –  NPT/NVT heat ramp           │ 
   │           T-low → T_melt                       │
   │                                                │
   └─────┬──────────────────────────────────────────┘
         │
   ┌─────▼──────────────────────────────────────────┐
   │  Stage 4  High-T equilibration   T_melt        │
   │           NVT/NPT                              │
   └──┬──────────────────────┬──────────────────────┘
      │                      │ snapshots (optional)
      │              ┌───────▼──────────────────────────────────┐
      │              │  batch_quench: Stage 5 → 6 → 7           │
      │              │  N independent amorphous structures      │
      │              │  --resume to continue interrupted jobs   │
      │              └──────────────────────────────────────────┘
      │
   ┌──▼─────────────────────────────────────────────┐
   │  Stage 5  Quench  –  NVT cooling ramp          │
   │           T_melt → T-low                       │
   └─────┬──────────────────────────────────────────┘
         │
   ┌─────▼──────────────────────────────────────────┐
   │  Stage 6  Low-T equilibration   T-low          │
   │           NVT/NPT                              │
   └─────┬──────────────────────────────────────────┘
         │
   ┌─────▼──────────────────────────────────────────┐
   │  Stage 7  Final optimisation (amorphous)       │
   │           optimizer + cell_filter              │
   └─────┬──────────────────────────────────────────┘
         │
   stage7_opt.cif  +  stage7_opt.xyz
```

---

## Supported backends

AmorphGen supports multiple calculator backends:

| Backend | Install | Model name(s) |
|---------|---------|----------------|
| **MACE** | `pip install amorphgen[mace]` | `mace-mpa-0`, `mace-mpa-0-medium`, `mace-omat-0-medium`, ... (20+ variants) |
| **CHGNet** | `pip install amorphgen[chgnet]` | `chgnet` |
| **SevenNet** | `pip install amorphgen[sevennet]` | `sevennet`, `7net-mf-ompa`, `7net-l3i5`, `7net-omat`, `7net-0`, ... |
| **Classical** | built-in (no extra install) | `lennard-jones`, `buckingham` |

Only install the backend(s) you need. Classical potentials (Lennard-Jones, Buckingham+Coulomb) are built-in and require no GPU. Use `amorphgen --list-models` to see all available models.

> **ASE pass-through.** AmorphGen wraps each backend's upstream ASE calculator without modifying unit conventions, stress signs, or PBC handling — energies (eV), forces (eV/Å), stress (eV/Å³), and `atoms.pbc` are inherited directly from the upstream MLIP package. See [docs/guides/backends](https://amorphgen.readthedocs.io/en/latest/guides/backends.html) for details.

---

## Installation

```bash
git clone https://github.com/SMTG-Bham/AmorphGen.git
cd AmorphGen

# Install with your preferred backend
pip install -e ".[mace]"             # MACE only
pip install -e ".[chgnet]"           # CHGNet only
pip install -e ".[mace,chgnet]"      # MACE + CHGNet (recommended)
pip install -e ".[all]"              # MACE + CHGNet + analysis (no SevenNet)
pip install -e ".[all,dev]"          # the above + pytest
```

> **SevenNet needs its own environment.** SevenNet depends on `e3nn>=0.5`,
> while MACE foundation-model files (`mace-mpa-0`, ...) were pickled with
> `e3nn==0.4.x` and fail to load against the newer e3nn. The `[all]` extra
> therefore intentionally **excludes** SevenNet. To use SevenNet, create a
> separate conda env:
> ```bash
> conda create -n amorphgen-sevennet python=3.11
> conda activate amorphgen-sevennet
> pip install -e ".[sevennet,chgnet]"
> ```
> The `[full]` extra installs MACE+CHGNet+SevenNet in one env but loading
> MACE foundation models will then fail unless you upgrade `mace-torch` to
> a release that supports e3nn 0.5+.

> **GPU strongly recommended.** Use `--device cuda` or `"device": "auto"`.
> Device auto-detection only runs when a job starts — on a login node with no GPU, no device message will appear until a stage is launched.

---

## Quick start

### Command line

```bash
# -- Random generation (no crystal input needed) --
# Generate 10 random In2O3 structures (80 atoms each) and relax with MACE
amorphgen --random-gen --composition "In2O3*16" --relax --device cpu

# Same thing with explicit atom counts
amorphgen --random-gen --composition In=32,O=48 --relax --device cpu

# -- Melt-quench pipeline (from crystalline input) --
# Full 7-stage pipeline with MACE (default)
amorphgen POSCAR --device cuda

# Use CHGNet (faster on CPU)
amorphgen POSCAR --model chgnet --device cpu

# List all available models
amorphgen --list-models
```

> **`--composition` accepts two formats:**
> - Formula: `In2O3*16` (16 formula units = 80 atoms)
> - Atom counts: `In=32,O=48` (explicit)
>
> Typical sizes: 40-100 atoms for random generation, 100-500 for melt-quench.

### Python API

```python
from amorphgen import MeltQuenchPipeline

# MACE (default)
pipe = MeltQuenchPipeline(
    input_file="POSCAR",
    work_dir="InO_run",
    cfg_override={
        "model":  "mace-mpa-0",
        "device": "cuda",
    },
)
atoms = pipe.run()                  # all 7 stages

# CHGNet
pipe = MeltQuenchPipeline(
    input_file="POSCAR",
    cfg_override={"model": "chgnet"},
)

# SevenNet
pipe = MeltQuenchPipeline(
    input_file="POSCAR",
    cfg_override={"model": "7net-mf-ompa"},
)

# Custom fine-tuned MACE model
pipe = MeltQuenchPipeline(
    input_file="POSCAR",
    cfg_override={"model_path": "/data/models/InO_finetuned.model"},
)

# Run specific stages
pipe.run(stages=[5, 6, 7], input_file="stage4_eq_high.xyz")
```

---

## YAML configuration

Instead of passing many CLI flags, you can define settings in a YAML file:

```yaml
# config.yaml
model: mace-mpa-0
device: cuda
default_dtype: float64

opt:
  fmax: 0.01
  max_steps: 1000
  optimizer: LBFGS

melt:
  T_start: 300
  T_end: 3000
  T_step: 100

quench:
  T_start: 3000
  T_end: 300
  T_step: -100
  steps_per_T: 2000
```

```bash
# Use YAML config
amorphgen POSCAR --config config.yaml

# CLI args override YAML values
amorphgen POSCAR --config config.yaml --fmax 0.05 --device cpu
```

```python
from amorphgen.configs import load_yaml_config
from amorphgen import MeltQuenchPipeline

cfg = load_yaml_config("config.yaml")
pipe = MeltQuenchPipeline(input_file="POSCAR", cfg_override=cfg)
atoms = pipe.run()
```

Precedence: **CLI arguments > YAML config > built-in defaults**.

YAML also supports random generation settings:

```yaml
# random_gen_config.yaml
model: chgnet
device: cpu
default_dtype: float32

opt:
  fmax: 0.05
  max_steps: 500
  cell_filter: cubic

random_gen:
  composition:
    Si: 16
    O: 32
  n_structures: 5
  target_density: 2.2
  target_cn:
    Si: 4
    O: 2
  output_format: vasp
```

```bash
amorphgen --random-gen --config random_gen_config.yaml --work-dir SiO2_sc
amorphgen --batch-opt --input-dir SiO2_sc --work-dir SiO2_sc_opt --config random_gen_config.yaml
```

See `amorphgen/configs/example_config.yaml` for all available options.

---

## Random structure generation

Generate random amorphous starting structures:

```bash
# Generate 20 random In₂O₃ structures (80 atoms each)
amorphgen --random-gen \
    --composition "In2O3*16" \
    --n-structures 20 \
    --work-dir random_structures/

# Same with explicit atom counts and target density
amorphgen --random-gen \
    --composition In=32,O=48 \
    --target-density 5.5 \
    --n-structures 20

# Generate with relaxation
amorphgen --random-gen \
    --composition "TiO2*16" \
    --n-structures 10 \
    --relax --model mace-mpa-0

# Resume after interruption (skips completed structures)
amorphgen --random-gen \
    --composition "Ga2O3*80" -n 20 \
    --relax --device cuda --format vasp --resume
```

```python
from amorphgen import generate_random, batch_random

# Single structure
atoms = generate_random({"In": 16, "O": 24})

# Batch generation
batch_random(
    composition={"In": 32, "O": 48},   # atom counts (Python API always uses dict)
    n_structures=20,
    output_dir="random_structures",
)
```

### Two-step workflow: generate then optimise separately

You can decouple generation and optimisation into separate steps.
This gives more control over optimisation settings (optimizer, cell filter,
precision, convergence) and lets you inspect structures before committing
to expensive relaxation.

**Step 1 — Generate (default, no relaxation):**

```bash
amorphgen --random-gen \
    --composition Ga=16,O=24 \
    --n-structures 5 \
    --work-dir random_Ga2O3
```

```python
from amorphgen.pipeline.random_gen import batch_random

paths = batch_random(
    composition={"Ga": 16, "O": 24},
    n_structures=5,
    output_dir="random_Ga2O3",
    relax=False,
    seed=42,
)
```

**Step 2 — Batch optimise:**

```bash
amorphgen --batch-opt \
    --input-dir random_Ga2O3 \
    --work-dir random_Ga2O3_opt \
    --model mace-mpa-0 --device cpu --fmax 0.01
```

```python
from amorphgen.pipeline.opt_cell import batch_optimize
from amorphgen.utils import get_calculator

calc = get_calculator(model="mace-mpa-0", device="cpu", default_dtype="float64")

batch_optimize(
    input_dir="random_Ga2O3",
    output_dir="random_Ga2O3_opt",
    calc=calc,
)
```

The `--batch-opt` mode uses the full `opt_cell.run()` under the hood, giving
you proper logging, trajectory files, configurable optimizer/cell filter,
and float64 precision.

### Coordination-aware placement

For better short-range order, enable coordination-aware placement with `--target-cn`. New atoms are biased toward existing under-coordinated sites, and placements that would push any neighbour over its target CN are rejected:

```bash
# Coordination-aware placement: atoms placed near under-coordinated sites
amorphgen --random-gen \
    --composition "SiO2*16" \
    --target-density 2.2 \
    --target-cn Si=4,O=2 \
    --work-dir random_SiO2

# With explicit bonding shell distances
amorphgen --random-gen \
    --composition Li=16,Zr=8,Cl=48 \
    --target-cn Zr=6,Li=6 \
    --dmax Zr-Cl=3.2,Li-Cl=3.2 \
    --work-dir random_Li2ZrCl6
```

```python
from amorphgen.pipeline.random_gen import generate_random

atoms = generate_random(
    composition={"Si": 16, "O": 32},
    target_density=2.2,
    target_cn={"Si": 4, "O": 2},
    seed=42,
)
```

Coordination-aware placement produces structures with correct coordination from the start, requiring less relaxation to reach the correct topology. Disable it with `--no-sc` (legacy flag name; the placement is enabled by default whenever `--target-cn` is set or auto-detected).

---

## Structure analysis

Analyse optimised structures for density, coordination, bond distances,
angles, and RDF:

```bash
# Auto cutoff (default)
amorphgen --analyse --input-dir optimised_structures/

# Save report and plots
amorphgen --analyse --input-dir optimised_structures/ \
    --save-report report.txt --save-plot plots/

# RDF-based auto cutoff
amorphgen --analyse --input-dir optimised_structures/ --cutoff auto-rdf
```

```python
from amorphgen.utils.analysis import StructureAnalyser

sa = StructureAnalyser("optimised_structures/", cutoff="auto")
sa.summary()
sa.save_report("report.txt")
sa.plot(output_dir="plots/", angle_style="line")
```

---

## Generating multiple independent structures (batch quench)

### Step 1 — Run Stages 1–4 with snapshot sampling

```bash
amorphgen POSCAR \
    --stages 1 2 3 4 \
    --eq-high-steps 100000 \
    --work-dir melt_run/
```

### Step 2 — Batch quench N independent runs from snapshots

```bash
amorphgen --batch-quench \
    --snapshot-dir snapshots/ \
    --n-runs 20 --select uniform \
    --batch-stages 5 6 7 \
    --work-dir batch_run/
```

```python
from amorphgen.pipeline import batch_quench

results = batch_quench.run(
    snapshot_files=snapshot_paths,
    n_runs=20,
    select="uniform",
    work_dir="batch_run",
)
```

Each run gets its own subdirectory: `batch_run/run_0000/`, `batch_run/run_0001/`, …

### Resuming an interrupted batch

If a batch job times out, resubmit with `--resume` — already-completed runs are skipped:

```bash
amorphgen --batch-quench \
    --snapshot-dir snapshots/ \
    --n-runs 20 --select uniform \
    --resume \
    --work-dir batch_run/
```

### Hybrid workflow: random generation → high-T equilibration → batch quench

An alternative approach combines random structure generation with
high-temperature equilibration to skip the slow heating stage:

```
Random structure (target density)
    │
    ▼
Optimise (positions only — preserves density)
    │
    ▼
Equilibrate at T_melt (NVT, 20+ ps)
    │
    ├── snapshot 0 ──→ Quench → Low-T eq → Opt → amorphous_0
    ├── snapshot 1 ──→ Quench → Low-T eq → Opt → amorphous_1
    └── ...
```

```python
from amorphgen.pipeline.random_gen import generate_random
from amorphgen.pipeline.opt_cell import run as opt_run
from amorphgen.pipeline.equilibrate import run as eq_run
from amorphgen import MeltQuenchPipeline

# Step 1: Generate random structure (auto minsep from Shannon radii)
atoms = generate_random(
    composition={"Ti": 8, "O": 16},
    target_density=3.2,       # optional, auto-estimated if omitted
    target_cn={"Ti": 6},      # optional, enables coordination-aware placement + CN-aware radii
)

# Step 2: Optimise (positions only)
calc = get_calculator(model="chgnet", device="cpu")
optimised = opt_run(atoms, cfg_override={"opt": {"fmax": 0.1}}, calc=calc)

# Step 3: Equilibrate at 2000 K
liquid = eq_run(optimised, cfg_override={
    "eq_high": {"ensemble": "NVT", "T": 2000, "steps": 10000, "timestep": 0.5},
}, calc=calc, stage="high")

# Step 4: Extract snapshots and batch quench (Stages 5 → 6 → 7)
for snap_file in snapshot_files:
    pipe = MeltQuenchPipeline(input_file=snap_file, work_dir=run_dir,
        cfg_override={"model": "chgnet", "device": "cpu"})
    pipe.run(stages=[5, 6, 7])
```

See **Tutorial 5** for a complete working example.

---

## Ensemble choice

| Stage | Default | Override flag |
|-------|---------|--------------|
| Stage 2 pre-melt eq | NVT | `--eq-premelt-ensemble NPT` |
| Stage 3 melt | NPT | `--melt-ensemble NVT` |
| Stage 4 high-T eq | NVT | `--eq-high-ensemble NPT` |
| Stage 5 quench | NVT | `--quench-ensemble NPT` |
| Stage 6 low-T eq | NVT | `--eq-low-ensemble NPT` |

---

## Heating / cooling rate

```bash
amorphgen POSCAR \
    --melt-T-step 100 --melt-steps-per-T 1000     # 100 K/ps heating
    --quench-T-step -50 --quench-steps-per-T 2000  # 25 K/ps cooling
```

Common cooling rates:

| Rate | `--quench-steps-per-T` | Time (3000 → 300 K) |
|------|------------------------|----------------------|
| 200 K/ps | 500 | ~13 ps |
| 100 K/ps (default) | 1000 | ~27 ps |
| 10 K/ps | 10000 | ~270 ps |
| 1 K/ps | 100000 | ~2700 ps |

---

## Trajectory format

| Format | Extension | Notes |
|--------|-----------|-------|
| `extxyz` | `.xyz` | **Default.** ASE extended XYZ (cell + energy + forces). Readable by OVITO, VESTA, ASE. |
| `xyz` | `.xyz` | Plain XYZ (positions only) |
| `traj` | `.traj` | ASE binary |
| `lammps-dump` | `.dump` | LAMMPS text dump |

---

## Available models

| Name | Backend | Notes |
|------|---------|-------|
| `mace-mpa-0` | MACE | **default** — MPTrj + sAlex |
| `mace-omat-0-medium` | MACE | OMAT, excellent phonons (ASL license) |
| `mace-matpes-r2scan` | MACE | MATPES, r²SCAN functional (ASL license) |
| `chgnet` | CHGNet | Charge-informed, good CPU speed |
| `7net-mf-ompa` | SevenNet | Multi-fidelity foundation, OMat+MPtrj+Alexandria |
| `lennard-jones` | Classical | Pair potential, no GPU needed |
| `buckingham` | Classical | Buckingham + Coulomb (Wolf summation), no GPU needed |

```bash
amorphgen --list-models   # full table of all models grouped by backend
```

---

## Full configuration reference

All defaults are in `amorphgen/configs/default_config.py`. Override any value via `cfg_override`:

```python
pipe = MeltQuenchPipeline(
    input_file="POSCAR",
    work_dir="my_run",
    cfg_override={
        "model":       "mace-mpa-0",  # or "chgnet", "7net-mf-ompa", "buckingham", etc.
        "model_path":  None,           # path to local .model file (overrides model)
        "device":      "auto",         # "cuda", "cpu", or "auto"
        "traj_format": "extxyz",       # "extxyz", "xyz", "traj", "lammps-dump"

        "opt": {
            "fmax": 0.01,
            "max_steps": 1000,
        },
        "eq_premelt": {
            "ensemble": "NVT",
            "T": 300,
            "steps": 100000,       # 50 ps at 0.5 fs timestep
            "timestep": 0.5,
            "friction": 0.01,
        },
        "melt": {
            "ensemble": "NPT",
            "T_start": 300, "T_end": 3000,
            "T_step": 100, "steps_per_T": 1000,
            "timestep": 0.5,
            "friction": 0.01, "ttime": 25.0,
        },
        "eq_high": {
            "ensemble": "NVT",
            "T": 3000,
            "steps": 10000,
            "timestep": 0.5,
            "friction": 0.01,
        },
        "quench": {
            "ensemble": "NVT",
            "T_start": 3000, "T_end": 300,
            "T_step": -100, "steps_per_T": 1000,
            "timestep": 0.5,
            "friction": 0.01, "ttime": 25.0,
        },
        "eq_low": {
            "ensemble": "NVT",
            "T": 300,
            "steps": 10000,
            "timestep": 0.5,
            "friction": 0.01,
        },
    },
)
```

---

## Output files

| Stage | Trajectory | Final structure | Log |
|-------|-----------|-----------------|-----|
| 1 | `stage1_opt.traj` | `stage1_opt.cif` + `stage1_opt.xyz` | `stage1_opt.log` |
| 2 | `stage2_eq.xyz` | `stage2_eq.xyz` | `stage2_eq.log` |
| 3 | `stage3_melt.xyz` | `stage3_melted.xyz` | `stage3_melt.log` |
| 4 | `stage4_eq.xyz` | `stage4_eq.xyz` | `stage4_eq.log` |
| 5 | `stage5_quench.xyz` | `stage5_quenched.xyz` | `stage5_quench.log` |
| 6 | `stage6_eq.xyz` | `stage6_eq.xyz` | `stage6_eq.log` |
| **7** | `stage7_opt.traj` | **`stage7_opt.cif`** + `stage7_opt.xyz` | `stage7_opt.log` |

---

## Tutorials

**Start here**:

| Tutorial | Description |
|----------|-------------|
| [Tutorial 1](Tutorials/T1_5min_intro/tutorial_1_5min_intro.ipynb) | **Quick-start tutorial** — orientation: what it does, the three workflows, decision tree, one live demo (random + CHGNet relax on a-SiO₂) |

**Workflow tutorials** — each tutorial reports its own measured wall time on the CPU it was validated on:

| Tutorial | Description |
|----------|-------------|
| [Tutorial 2](Tutorials/T2_automated_random_gen/tutorial_2_automated_random_gen.ipynb) | **Zero-config random gen** — composition is the only input; auto-derive minsep, density, target CN, oxidation state across 8 material classes (Si, SiO₂, In₂O₃, CdTe, AlN, LiCl, TiO₂, Cu). Each structure is CHGNet-relaxed and saved to `output_T2/` |
| [Tutorial 3](Tutorials/T3_random_gen/tutorial_3_random_generation.ipynb) | **Explicit control + ensemble analysis** — the opposite end of T2: hand-picked minsep (from crystalline bond lengths) and target density (from cited amorphous-thin-film references), 5-structure ensembles per system, quantitative RDF / energy / CN / bond-angle analysis vs the crystalline reference (In₂O₃, TiO₂, Al₂O₃, Ga₂O₃; MACE-MPA-0) |
| [Tutorial 4](Tutorials/T4_MQ_via_7_steps/tutorial_4_melt_quench.ipynb) | Full 7-stage melt-quench from crystalline SiO₂ (CHGNet on CPU; flip the backend toggle for MACE on GPU) |
| [Tutorial 5](Tutorials/T5_mix_random_MQ/tutorial_5_batch_quench.ipynb) | Hybrid workflow: random gen → high-T equilibration → batch quench (TiO₂) |
| [Tutorial 6](Tutorials/T6_classical_potential/tutorial_6_classical_potential.ipynb) | Classical potential (Buckingham+Coulomb) relaxation, no GPU needed (SiO₂, Al₂O₃, TiO₂) |

**Application case studies** (assume familiarity with the workflow tutorials):

| Tutorial | Description |
|----------|-------------|
| [Tutorial 7](Tutorials/T7_application_dimer_dissociation/tutorial_7_dimer_dissociation.ipynb) | Defect chemistry: O–O peroxide dimer dissociation kinetics in amorphous In₂O₃, with Arrhenius temperature scan |

---

## Package layout

```
AmorphGen/
├── .github/workflows/
│   └── test.yml                    ← CI (pytest on 3.10/3.11/3.12)
├── amorphgen/
│   ├── __init__.py                 ← v1.0.0
│   ├── cli.py                      ← CLI entry point (amorphgen command)
│   ├── configs/
│   │   ├── default_config.py       ← all default parameters
│   │   ├── yaml_config.py          ← YAML config loader
│   │   └── example_config.yaml     ← example YAML with all options
│   ├── pipeline/
│   │   ├── run_pipeline.py         ← MeltQuenchPipeline orchestrator
│   │   ├── opt_cell.py             ← Stages 1 & 7 (optimisation) + batch_optimize()
│   │   ├── equilibrate.py          ← Stages 2, 4, 6 (constant-T equilibration)
│   │   ├── melt_cell.py            ← Stage 3 (heat ramp)
│   │   ├── quench.py               ← Stage 5 (cool ramp)
│   │   ├── batch_quench.py         ← batch runner: Stages 5 → 6 → 7 on N snapshots
│   │   └── random_gen.py           ← random + coordination-aware placement
│   └── utils/
│       ├── analysis.py             ← StructureAnalyser (density, CN, RDF, angles)
│       ├── calculators.py          ← multi-backend calculator factory
│       ├── radii.py                ← Shannon/metallic radii, minsep, density estimation
│       └── common.py               ← dynamics builder, logger, trajectory writer
├── paper/
│   ├── paper.md                    ← JOSS draft
│   └── paper.bib
├── test/                           ← 114 tests (4 skipped without --run-mace)
├── pyproject.toml
├── LICENSE                         ← MIT
└── README.md
```

---

## HPC (SLURM) example

```bash
#!/bin/bash
#SBATCH --job-name=amorphgen
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00

source /path/to/conda/env/bin/activate

amorphgen /abs/path/to/In2O3_POSCAR \
    --model mace-mpa-0 \
    --device cuda \
    --work-dir /scratch/InO_amorphous \
    --melt-T-end 2500 \
    --quench-T-start 2500
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `ase` | MD engine, optimisers, I/O |
| `numpy` | Array operations |
| `scipy` | Vectorized erfc for Coulomb (classical) |
| `torch` | GPU backend (MLIP + optional classical GPU) |
| `mace-torch` | MACE calculator (optional) |
| `chgnet` | CHGNet calculator (optional) |
| `sevenn` | SevenNet calculator (optional) |

---

## Citation

If you use AmorphGen in your research, please cite the package
and the foundation model(s) you used.

**AmorphGen:**
```bibtex
@misc{amorphgen,
  author = {Kaewmeechai, Chaiyawat and Scanlon, David O.},
  title  = {AmorphGen: A Python package for amorphous structure generation
            with machine-learning and classical interatomic potentials},
  year   = {2026},
  url    = {https://github.com/SMTG-Bham/AmorphGen}
}
```

A Zenodo DOI for tagged releases will be added on first stable release.

**Foundation potentials (cite the one you used):**

**MACE-MP:**
```bibtex
@article{batatia2023foundation,
  title   = {A foundation model for atomistic materials chemistry},
  author  = {Ilyes Batatia and others},
  year    = {2023},
  eprint  = {2401.00096},
  archivePrefix = {arXiv},
}
```

**CHGNet:**
```bibtex
@article{deng2023chgnet,
  title   = {CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling},
  author  = {Bowen Deng and others},
  journal = {Nature Machine Intelligence},
  year    = {2023},
}
```

**SevenNet:**
```bibtex
@article{park2024sevennet,
  title   = {Scalable parallel algorithm for graph neural network interatomic potentials in molecular dynamics simulations},
  author  = {Park, Yutack and Kim, Jaesun and Hwang, Seungwoo and Han, Seungwu},
  journal = {Journal of Chemical Theory and Computation},
  year    = {2024},
}
```

---

## License

MIT


