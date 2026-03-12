# amorphMD

**A Python package for generating amorphous oxide structures via 7-stage melt-and-quench molecular dynamics using [MACE-MP](https://github.com/ACEsuit/mace) foundation models.**

---

## Pipeline overview

```
Crystalline input  (POSCAR / .xyz / .cif / .extxyz)
         │
   ┌─────▼──────────────────────────────────────────┐
   │  Stage 1  Structure optimisation               │
   │           optimizer + cell_filter (configurable│
   └─────┬──────────────────────────────────────────┘
         │
   ┌─────▼──────────────────────────────────────────┐
   │  Stage 2  Pre-melt equilibration   300 K       │
   │           NVT/NPT, 50 ps                       │
   └─────┬──────────────────────────────────────────┘
         │
   ┌─────▼──────────────────────────────────────────┐
   │  Stage 3  Melt  –  NPT heat ramp               │
   │           300 K → T_melt  (default 2500 K)     │
   │           Heating rate: 100 K/ps               │
   └─────┬──────────────────────────────────────────┘
         │
   ┌─────▼──────────────────────────────────────────┐
   │  Stage 4  High-T equilibration   T_melt        │
   │           NVT/NPT, 100 ps                      │
   │           cell_mode: free / fix_volume /       │
   │             keep_cubic / target_density        │
   │           Optional: sample snapshot every N ps │
   └──┬──────────────────────┬───────────────────────┘
      │                      │ snapshots  →  snapshot_0000.extxyz …
      │              ┌───────▼──────────────────────────────────┐
      │              │  batch_quench: Stage 5 → 6 → 7           │
      │              │  N independent amorphous structures       │
      │              │  --resume to continue interrupted jobs    │
      │              └──────────────────────────────────────────┘
      │
   ┌──▼─────────────────────────────────────────────┐
   │  Stage 5  Quench  –  NVT cooling ramp          │
   │           T_melt → 300 K                       │
   │           Cooling rate: 100 K/ps               │
   └─────┬──────────────────────────────────────────┘
         │
   ┌─────▼──────────────────────────────────────────┐
   │  Stage 6  Low-T equilibration   300 K          │
   │           NVT/NPT, 50 ps                       │
   └─────┬──────────────────────────────────────────┘
         │
   ┌─────▼──────────────────────────────────────────┐
   │  Stage 7  Final optimisation (amorphous)       │
   │           optimizer + cell_filter (configurable│
   └─────┬──────────────────────────────────────────┘
         │
   stage7_amorphous_final.cif  +  stage7_amorphous_final.xyz
```

---

## Installation

```bash
git clone https://github.com/cywkmc21/amorphMD.git
cd amorphMD

# Install dependencies
pip install mace-torch ase

# Optional: symmetry analysis in Stages 1 and 7
pip install pymatgen

# Optional: install as a package
pip install -e .
```

> **GPU strongly recommended.** Use `--device cuda` or `"device": "auto"`.  
> Note: device auto-detection only runs when a job starts — on a login node with no GPU, no device message will appear until an actual stage is launched.

---

## Quick start

### Python API

```python
from amorphMD import MeltQuenchPipeline

pipe = MeltQuenchPipeline(
    input_file="POSCAR",        # or .xyz, .cif, .extxyz
    work_dir="InO_run",
    cfg_override={
        "mace_model": "mace-mpa-0-medium",
        "device":     "cuda",
    }
)
atoms = pipe.run()              # all 7 stages
```

### Command line

```bash
# Full 7-stage pipeline
run_melt_quench.py POSCAR --device cuda

# List all available foundation models
run_melt_quench.py --list-models
```

---

## Protocol details (In₂O₃ reference)

| Stage | Ensemble | T (K) | Duration | Notes |
|---|---|---|---|---|
| 1 | LBFGS | — | until convergence | FixSymmetry ON |
| 2 | NVT | 300 | 50 ps | pre-melt equilibration |
| 3 | NPT | 300 → 2500 | 22 ps | 100 K/ps; density 4.2 → 3.0 g/cm³ |
| 4 | NVT | 2500 | 100 ps | fully disordered liquid |
| 5 | NVT | 2500 → 300 | 22 ps | 100 K/ps cooling |
| 6 | NVT | 300 | 50 ps | low-T equilibration |
| 7 | LBFGS | — | until convergence | no symmetry constraint |

---

## Generating multiple independent structures (batch quench)

### Step 1 — Stage 4 with snapshot sampling

```bash
run_melt_quench.py POSCAR \
    --stages 1 2 3 4 \
    --eq-high-steps 100000 \
    --sample-interval 10 \
    --snapshot-dir snapshots/
```

### Step 2 — batch quench N independent runs

```bash
run_melt_quench.py --batch-quench \
    --snapshot-dir snapshots/ \
    --n-runs 20 --select uniform \
    --batch-stages 5 6 7 \
    --work-dir batch_run/
```

```python
from amorphMD.pipeline import batch_quench

results = batch_quench.run(
    snapshot_files=snapshot_paths,
    n_runs=20,
    select="uniform",   # "uniform" | "random" | "first" | "last" | "all"
    work_dir="batch_run",
)
# results[i]["final_cif"] → batch_run/run_0000/...amorphous_final.cif
```

Each run gets its own subdirectory `batch_run/run_0000/`, `batch_run/run_0001/`, …

### Resuming an interrupted batch

If a batch job times out, resubmit with `--resume` — already-completed runs are skipped automatically:

```bash
run_melt_quench.py --batch-quench \
    --snapshot-dir snapshots/ \
    --n-runs 20 --select uniform \
    --resume \
    --work-dir batch_run/
```

### Extracting snapshots from an existing trajectory

If Stage 4 was run without `--sample-interval`, extract snapshots afterwards:

```bash
run_melt_quench.py --extract-snapshots \
    --traj-file stage4_eq_high.extxyz \
    --n-snapshots 20 --select uniform \
    --snapshot-dir snapshots/
```

---

## Optimizer and cell filter (Stages 1 & 7)

### Optimizers

| Name | Best for |
|---|---|
| `LBFGS` | **default** — Stage 1 crystalline cells |
| `FIRE` | Stage 7 amorphous structures (recommended) |
| `BFGSLineSearch` | difficult convergence cases |
| `BFGS` | general fallback |
| `MDMin` | very soft cells |

### Cell filters

| Name | Description |
|---|---|
| `UnitCellFilter` | **default** — relax all 6 cell DOF simultaneously |
| `ExpCellFilter` | better convergence for soft / amorphous cells |
| `StrainFilter` | relax cell strain only |
| `cubic` | reshape to cube + fix angles at 90° — **recommended for Stage 7** |

```bash
# Stage 1: LBFGS + UnitCellFilter (default)
run_melt_quench.py POSCAR --stages 1

# Stage 7: FIRE + cubic cell (recommended for amorphous output)
run_melt_quench.py stage6.xyz --stages 7 \
    --final-optimizer FIRE \
    --final-cell-filter cubic
```

---

## Cell control during high-T equilibration (Stage 4)

Set `--eq-high-cell-mode` to control how the cell behaves during Stage 4 MD:

| Mode | Behaviour |
|---|---|
| `free` | **default** — cell evolves with ensemble |
| `fix_volume` | freeze cell shape and volume (forces NVT) |
| `keep_cubic` | fix angles at 90°, volume free (forces NVT) |
| `target_density` | rescale cell to target density before MD |

```bash
# Keep cubic cell during melt equilibration
run_melt_quench.py stage3.xyz --stages 4 \
    --eq-high-cell-mode keep_cubic

# Rescale to target density (e.g. amorphous In2O3 ≈ 3.0 g/cm³)
run_melt_quench.py stage3.xyz --stages 4 \
    --eq-high-cell-mode target_density \
    --eq-high-target-density 3.0
```

> Density is logged every 1000 steps: `[density]  t=10.00 ps  ρ=3.0142 g/cm³  V=1587.23 Å³`

---

## Available MACE models

| Name | Notes |
|---|---|
| `mace-mpa-0` | ** default** — MPTrj + sAlex |
| `mace-mpa-0-medium` | alias for `mace-mpa-0` |
| `mace-mp-0b3-medium` | MPTrj, fixed phonons |
| `mace-omat-0-medium` | OMAT, excellent phonons (ASL license) |
| `mace-matpes-r2scan` | MATPES, r2SCAN functional (ASL license) |

```bash
run_melt_quench.py --list-models   # full table of all 21 models
```

---

## Using a custom / fine-tuned model

```bash
run_melt_quench.py POSCAR --model-path /data/models/InO_ft.model
```

```python
pipe = MeltQuenchPipeline("POSCAR", cfg_override={
    "model_path": "/data/models/InO_ft.model"
})
```

---

## Ensemble choice

| Stage | Default | Override flag |
|---|---|---|
| Stage 2 pre-melt eq | NVT | `--eq-premelt-ensemble NPT` |
| Stage 3 melt        | NPT | `--melt-ensemble NVT` |
| Stage 4 high-T eq   | NVT | `--eq-high-ensemble NPT` |
| Stage 5 quench      | NVT | `--quench-ensemble NPT` |
| Stage 6 low-T eq    | NVT | `--eq-low-ensemble NPT` |

---

## Heating/cooling rate

```bash
# Step-based (explicit control)
run_melt_quench.py POSCAR \
    --melt-T-step 100 --melt-steps-per-T 1000     # 100 K/ps heating
    --quench-T-step -50 --quench-steps-per-T 2000  # 25 K/ps cooling
```

Common cooling rates:

| Rate | `--quench-steps-per-T` | Time (3000→300 K) |
|---|---|---|
| 200 K/ps | 500 | ~13 ps |
| 100 K/ps (default) | 1000 | ~27 ps |
| 10 K/ps | 10000 | ~270 ps |
| 1 K/ps | 100000 | ~2700 ps |

---

## Trajectory format

| Format | Extension | Notes |
|---|---|---|
| `extxyz` | `.extxyz` | **Default.** Cell + energy + forces. Readable by OVITO, VESTA, ASE. |
| `xyz` | `.xyz` | Plain XYZ |
| `traj` | `.traj` | ASE binary |
| `lammps-dump` | `.dump` | LAMMPS text dump |

---

## Resuming from a checkpoint

```bash
# Resume from Stage 5 using an existing Stage 4 output
run_melt_quench.py stage4_eq_high.xyz \
    --work-dir my_run \
    --stages 5 6 7
```

---

## Full configuration reference

All defaults are in `amorphMD/configs/default_config.py`. Override any value:

```python
pipe = MeltQuenchPipeline(
    input_file="POSCAR",
    work_dir="my_run",
    cfg_override={
        "mace_model":  "mace-mpa-0",
        "model_path":  None,          # or path to .model file
        "device":      "auto",
        "traj_format": "extxyz",

        "opt": {
            "fmax": 0.01, "max_steps": 1000, "fix_symmetry": True,
            "optimizer":   "LBFGS",          # "LBFGS" | "FIRE" | "BFGSLineSearch" | "BFGS" | "MDMin"
            "cell_filter": "UnitCellFilter", # "UnitCellFilter" | "ExpCellFilter" | "StrainFilter" | "cubic"
        },
        "eq_premelt": {
            "ensemble": "NVT", "temperature_K": 300, "steps": 50_000,
        },
        "melt": {
            "ensemble": "NPT", "T_start": 300, "T_end": 2500,
            "T_step": 100, "steps_per_T": 1000,  # or "rate_K_per_ps": 100
            "make_cubic": True,
        },
        "eq_high": {
            "ensemble": "NVT", "temperature_K": 2500, "steps": 100_000,
            "cell_mode": "free",           # "free" | "fix_volume" | "keep_cubic" | "target_density"
            "target_density_g_cm3": None,  # e.g. 3.0 — used when cell_mode="target_density"
            "sample_interval_ps": 10.0,    # None to disable snapshots
            "snapshot_dir": "snapshots",
        },
        "quench": {
            "ensemble": "NVT", "T_start": 2500, "T_end": 300,
            "T_step": -100, "steps_per_T": 1000,  # or "rate_K_per_ps": 100
        },
        "eq_low": {
            "ensemble": "NVT", "temperature_K": 300, "steps": 50_000,
        },
        "final_opt": {
            "fmax": 0.01, "max_steps": 1000, "fix_symmetry": False,
            "optimizer":   "FIRE",   # FIRE recommended for amorphous
            "cell_filter": "cubic",  # cubic enforces orthogonal cell
        },
    }
)
```

---

## Output files

| Stage | Trajectory | Final structure | Log |
|---|---|---|---|
| 1 | `opt_stage1.traj` | `stage1_optimised.cif` | `opt_stage1.log` |
| 2 | `stage2_eq_premelt.extxyz` | `stage2_eq_premelt.cif` | `stage2_eq_premelt_log.txt` |
| 3 | `stage3_melt.extxyz` | `stage3_melted.cif` | `stage3_melt_log.txt` |
| 4 | `stage4_eq_high.extxyz` | `stage4_eq_high.cif` | `stage4_eq_high_log.txt` |
| 4 (snapshots) | — | `snapshots/snapshot_NNNN_tX.Xps.extxyz` | — |
| 5 | `stage5_quench.extxyz` | `stage5_quenched.cif` | `stage5_quench_log.txt` |
| 6 | `stage6_eq_low.extxyz` | `stage6_eq_low.cif` | `stage6_eq_low_log.txt` |
| **7** | `opt_stage7.traj` | **`stage7_amorphous_final.cif`** | `opt_stage7.log` |

Log columns (MD stages): `Step  Epot/atom(eV)  Ekin/atom(eV)  Temp(K)  [Press(bar)]  Density(g/cm3)`

---

## Package layout

```
amorphMD/
├── run_melt_quench.py          ← CLI entry point
├── pyproject.toml
├── README.md
└── amorphMD/
    ├── __init__.py
    ├── configs/
    │   └── default_config.py   ← all default parameters
    ├── utils/
    │   └── common.py           ← MACE registry, dynamics builder, logger, writer
    └── pipeline/
        ├── run_pipeline.py     ← MeltQuenchPipeline orchestrator
        ├── opt_cell.py         ← Stages 1 & 7 (optimisation + cell filter)
        ├── equilibrate.py      ← Stages 2, 4, 6 (constant-T MD + cell control)
        ├── melt_cell.py        ← Stage 3 (heat ramp)
        ├── quench.py           ← Stage 5 (cool ramp)
        └── batch_quench.py     ← batch runner: stages 5→6→7 on N snapshots
```

---

## HPC (SLURM) example

```bash
#!/bin/bash
#SBATCH --job-name=amorphMD
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

source /path/to/conda/env/bin/activate

run_melt_quench.py /abs/path/to/In2O3_POSCAR \
    --model mace-mpa-0-medium \
    --device cuda \
    --work-dir /scratch/InO_amorphous \
    --melt-T-end 2500 \
    --quench-T-start 2500 \
    --final-optimizer FIRE \
    --final-cell-filter cubic
```

For stage-by-stage job chaining with SLURM dependencies:

```bash
JOB1=$(sbatch --parsable slurm_stage1_opt.sh)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm_stage2_eq300K.sh)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 slurm_stage3_melt.sh)
# ... continue for stages 4, 5, 6, 7
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `mace-torch` | MACE-MP force-field calculator |
| `ase` | MD engine, optimisers, I/O |
| `numpy` | Array operations |
| `torch` | GPU backend (via mace-torch) |
| `pymatgen` | Symmetry analysis in Stages 1 & 7 (optional) |

---

## Citation

If you use amorphMD, please cite the MACE-MP foundation model:

```bibtex
@article{batatia2023foundation,
  title   = {A foundation model for atomistic materials chemistry},
  author  = {Ilyes Batatia and others},
  year    = {2023},
  eprint  = {2401.00096},
  archivePrefix = {arXiv},
}
```

---

## License

MIT
