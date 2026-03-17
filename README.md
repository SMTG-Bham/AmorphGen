# AmorphGen

[![CI](https://github.com/cywkmc21/AmorphGen/actions/workflows/test.yml/badge.svg)](https://github.com/cywkmc21/AmorphGen/actions/workflows/test.yml)

**AmorphGen: A Python package for amorphous structure generation by melt-quench simulation and random placement using universal machine-learning force fields (MACE, CHGNet, M3GNet).**

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
   │           300 K → T_melt  (default 3000 K)     │
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
   stage7_amorphous_final.cif  +  stage7_amorphous_final.extxyz
```

---

## Supported backends

AmorphGen is **model-agnostic** — choose any supported universal MLFF backend:

| Backend | Install | Model name(s) |
|---------|---------|----------------|
| **MACE** | `pip install amorphgen[mace]` | `mace-mpa-0`, `mace-mpa-0-medium`, `mace-omat-0-medium`, … (20+ variants) |
| **CHGNet** | `pip install amorphgen[chgnet]` | `chgnet` |
| **M3GNet** | `pip install amorphgen[m3gnet]` | `m3gnet` |

Only install the backend(s) you need. Use `amorphgen --list-models` to see all available models.

---

## Installation

```bash
git clone https://github.com/SMTG-Bham/AmorphGen.git
cd AmorphGen

# Install with your preferred backend
pip install -e ".[mace]"        # MACE only (recommended)
pip install -e ".[chgnet]"      # CHGNet only
pip install -e ".[m3gnet]"      # M3GNet only
pip install -e ".[all]"         # all backends
pip install -e ".[all,dev]"     # all backends + pytest
```

> **GPU strongly recommended.** Use `--device cuda` or `"device": "auto"`.
> Device auto-detection only runs when a job starts — on a login node with no GPU, no device message will appear until a stage is launched.

---

## Quick start

### Command line

```bash
# Full 7-stage pipeline with MACE (default)
amorphgen POSCAR --device cuda

# Use CHGNet
amorphgen POSCAR --model chgnet --device cpu

# Use a custom fine-tuned model
amorphgen POSCAR --model-path /data/models/InO_finetuned.model

# List all available foundation models
amorphgen --list-models

# Run specific stages only
amorphgen POSCAR --stages 1 2 3 4

# Resume from a checkpoint
amorphgen stage4_eq_high.extxyz --stages 5 6 7 --work-dir my_run
```

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

# M3GNet
pipe = MeltQuenchPipeline(
    input_file="POSCAR",
    cfg_override={"model": "m3gnet"},
)

# Custom fine-tuned MACE model
pipe = MeltQuenchPipeline(
    input_file="POSCAR",
    cfg_override={"model_path": "/data/models/InO_finetuned.model"},
)

# Run specific stages
pipe.run(stages=[5, 6, 7], input_file="stage4_eq_high.extxyz")
```

---

## Random structure generation

Generate random amorphous starting structures:

```bash
# Generate 20 random In₂O₃ structures
amorphgen --random-gen \
    --composition In=32,O=48 \
    --target-density 5.5 \
    --n-structures 20 \
    --work-dir random_structures/

# Generate without relaxation
amorphgen --random-gen \
    --composition Ti=16,O=32 \
    --n-structures 10 \
    --no-relax
```

```python
from amorphgen.pipeline.random_gen import batch_random

batch_random(
    composition={"In": 32, "O": 48},
    n_structures=20,
    target_density=5.5,     # g/cm³
    output_dir="random_structures",
)
```


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

# Step 1: Generate random structure
atoms = generate_random(
    composition={"Ti": 8, "O": 16},
    target_density=3.2,
    minsep={"Ti-Ti": 2.5, "Ti-O": 1.6, "O-O": 2.2},
)

# Step 2: Optimise (positions only)
calc = get_calculator(model="chgnet", device="cpu")
optimised = opt_run(atoms, cfg_override={"opt": {"fmax": 0.1}}, calc=calc)

# Step 3: Equilibrate at 2000 K
liquid = eq_run(optimised, cfg_override={
    "eq_high": {"ensemble": "NVT", "T": 2000, "steps": 10000, "timestep": 2.0},
}, calc=calc, stage="high")

# Step 4: Extract snapshots and batch quench (Stages 5 → 6 → 7)
for snap_file in snapshot_files:
    pipe = MeltQuenchPipeline(input_file=snap_file, work_dir=run_dir,
        cfg_override={"model": "chgnet", "device": "cpu"})
    pipe.run(stages=[5, 6, 7])
```

See **Tutorial 3** for a complete working example.

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
| `extxyz` | `.extxyz` | **Default.** Cell + energy + forces. Readable by OVITO, VESTA, ASE. |
| `xyz` | `.xyz` | Plain XYZ |
| `traj` | `.traj` | ASE binary |
| `lammps-dump` | `.dump` | LAMMPS text dump |

---

## Available MACE models

| Name | Notes |
|------|-------|
| `mace-mpa-0` | **default** — MPTrj + sAlex |
| `mace-mpa-0-medium` | alias for `mace-mpa-0` |
| `mace-mp-0b3-medium` | MPTrj, fixed phonons |
| `mace-omat-0-medium` | OMAT, excellent phonons (ASL license) |
| `mace-matpes-r2scan` | MATPES, r²SCAN functional (ASL license) |

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
        "model":       "mace-mpa-0",  # or "chgnet", "m3gnet", etc.
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
            "steps": 50000,        # 50 ps at 1 fs timestep
            "timestep": 1.0,
            "friction": 0.01,
        },
        "melt": {
            "ensemble": "NPT",
            "T_start": 300, "T_end": 3000,
            "T_step": 100, "steps_per_T": 1000,
            "timestep": 1.0,
            "friction": 0.01, "ttime": 25.0,
        },
        "eq_high": {
            "ensemble": "NVT",
            "T": 3000,
            "steps": 10000,
            "timestep": 1.0,
            "friction": 0.01,
        },
        "quench": {
            "ensemble": "NVT",
            "T_start": 3000, "T_end": 300,
            "T_step": -100, "steps_per_T": 1000,
            "timestep": 1.0,
            "friction": 0.01, "ttime": 25.0,
        },
        "eq_low": {
            "ensemble": "NVT",
            "T": 300,
            "steps": 10000,
            "timestep": 1.0,
            "friction": 0.01,
        },
    },
)
```

---

## Output files

| Stage | Trajectory | Final structure | Log |
|-------|-----------|-----------------|-----|
| 1 | `opt_stage1.traj` | `stage1_optimised.cif` | `opt_stage1.log` |
| 2 | `stage2_eq.extxyz` | `stage2_eq.extxyz` | `stage2_eq.log` |
| 3 | `stage3_melt.extxyz` | `stage3_melted.extxyz` | `stage3_melt.log` |
| 4 | `stage4_eq.extxyz` | `stage4_eq.extxyz` | `stage4_eq.log` |
| 5 | `stage5_quench.extxyz` | `stage5_quenched.extxyz` | `stage5_quench.log` |
| 6 | `stage6_eq.extxyz` | `stage6_eq.extxyz` | `stage6_eq.log` |
| **7** | `opt_stage7.traj` | **`stage7_amorphous_final.cif`** | `opt_stage7.log` |

---

## Tutorials

| Tutorial | Description |
|----------|-------------|
| [Tutorial 1](Tutorials/T1_random_gen/tutorial_1_random_generation.ipynb) | Random structure generation + relaxation + structural analysis (In₂O₃, TiO₂, Al₂O₃, Ga₂O₃) |
| [Tutorial 2](Tutorials/T2_MQ_via_7_steps/tutorial_2_melt_quench.ipynb) | Full 7-stage melt-quench from crystalline SiO₂ |
| [Tutorial 3](Tutorials/T3_mix_random_MQ/tutorial_3_batch_quench.ipynb) | Hybrid workflow: random gen → high-T equilibration → batch quench (TiO₂) |

---

## Package layout

```
AmorphGen/
├── .github/workflows/
│   └── test.yml                    ← CI (pytest on 3.10/3.11/3.12)
├── amorphgen/
│   ├── __init__.py                 ← v2.0.0
│   ├── cli.py                      ← CLI entry point (amorphgen command)
│   ├── configs/
│   │   └── default_config.py       ← all default parameters
│   ├── pipeline/
│   │   ├── run_pipeline.py         ← MeltQuenchPipeline orchestrator
│   │   ├── opt_cell.py             ← Stages 1 & 7 (optimisation)
│   │   ├── equilibrate.py          ← Stages 2, 4, 6 (constant-T equilibration)
│   │   ├── melt_cell.py            ← Stage 3 (heat ramp)
│   │   ├── quench.py               ← Stage 5 (cool ramp)
│   │   ├── batch_quench.py         ← batch runner: Stages 5 → 6 → 7 on N snapshots
│   │   └── random_gen.py           ← random structure placement
│   └── utils/
│       ├── calculators.py          ← multi-backend calculator factory
│       └── common.py               ← dynamics builder, logger, trajectory writer
├── paper/
│   ├── paper.md                    ← JOSS draft
│   └── paper.bib
├── test/                           ← 66 tests (4 skipped without --run-mace)
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
| `torch` | GPU backend |
| `mace-torch` | MACE calculator (optional) |
| `chgnet` | CHGNet calculator (optional) |
| `matgl` | M3GNet calculator (optional) |

---

## Citation

If you use AmorphGen, please cite the relevant foundation model(s):

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

**M3GNet:**
```bibtex
@article{chen2022universal,
  title   = {A universal graph deep learning interatomic potential for the periodic table},
  author  = {Chi Chen and Shyue Ping Ong},
  journal = {Nature Computational Science},
  year    = {2022},
}
```

---

## License

MIT

