# Quickstart

This page shows the three main workflows in AmorphGen.

## 1. Melt-and-quench pipeline

Run the full 7-stage pipeline on a crystalline input structure:

::::{tab-set}

:::{tab-item} CLI
```bash
# Full pipeline
amorphgen POSCAR --model mace-mpa-0 --device cuda

# With YAML config
amorphgen POSCAR --config pipeline.yaml

# Hybrid (skip heating, resume from checkpoint)
amorphgen structure.xyz --stages 1 4 5 6 7 --config config.yaml --resume
```
:::

:::{tab-item} Python API
```python
from amorphgen import MeltQuenchPipeline

pipe = MeltQuenchPipeline(
    input_file="POSCAR",
    work_dir="my_run",
    cfg_override={"model": "mace-mpa-0", "device": "cuda"},
)
atoms = pipe.run()

# Hybrid with resume
atoms = pipe.run(stages=[1, 4, 5, 6, 7], resume=True)
```
:::

::::

### The 7 stages

| Stage | Name | Description |
|-------|------|-------------|
| 1 | Optimise | Relax positions (+ cell with FrechetCellFilter) |
| 2 | Pre-melt equilibration | NVT at 300 K |
| 3 | Melt | Heat ramp to high temperature (configurable rate in K/ps) |
| 4 | High-T equilibration | Equilibrate at melt temperature |
| 5 | Quench | Cool to target temperature (configurable rate in K/ps) |
| 6 | Low-T equilibration | Equilibrate at low temperature |
| 7 | Final optimisation | Final relaxation |

## 2. Random structure generation

Generate an ensemble of random amorphous structures with automated minimum separations from Shannon ionic radii:

::::{tab-set}

:::{tab-item} CLI
```bash
# Basic (formula format: In2O3 * 8 formula units = 40 atoms)
amorphgen --random-gen --composition "In2O3*8" --n-structures 20

# Same thing with explicit atom counts
amorphgen --random-gen --composition In=16,O=24 --n-structures 20

# With target CN and density
amorphgen --random-gen --composition "SiO2*16" \
    --target-cn Si=4,O=2 --target-density 2.2

# With relaxation
amorphgen --random-gen --composition "In2O3*8" \
    --relax --model mace-mpa-0 --cell-filter none
```
:::

:::{tab-item} Python API
```python
from amorphgen.pipeline.random_gen import generate_random, batch_random

# Single structure (auto minsep from Shannon radii)
atoms = generate_random(
    composition={"In": 16, "O": 24},
    target_density=5.0,
    target_cn={"In": 6},
)

# Batch generation
paths = batch_random(
    composition={"Si": 16, "O": 32},
    n_structures=20,
    target_density=2.2,
    target_cn={"Si": 4, "O": 2},
)
```
:::

::::

## 3. Batch quench

Quench multiple snapshot structures through the final pipeline stages:

```bash
amorphgen --batch-quench \
    --snapshot-dir snapshots/ \
    --model mace-mpa-0 \
    --batch-stages 5 6 7 \
    --resume
```

## Choosing a backend

```python
from amorphgen.utils import get_calculator

# MACE (default).  device="auto" picks CUDA → MPS → CPU automatically;
# pass "cpu" / "cuda" / "mps" explicitly to override.
calc = get_calculator(model="mace-mpa-0", device="auto")

# CHGNet
calc = get_calculator(model="chgnet", device="auto")

# SevenNet
calc = get_calculator(model="7net-mf-ompa", device="auto")

# Classical potentials (no GPU needed, parameters via YAML or dict)
calc = get_calculator("buckingham", classical_params={
    "params": {("Si", "O"): {"A": 18003.76, "rho": 0.2052, "C": 133.54}},
    "charges": {"Si": 2.4, "O": -1.2},
    "cutoff": 10.0,
})

# Custom fine-tuned model
calc = get_calculator(model_path="/path/to/custom.model")
```

## Accessing radii data

```python
from amorphgen.utils import get_ionic_radius, classify_bond, default_minsep

get_ionic_radius("In", cn=6)   # 0.80 A
classify_bond("In", "O")       # "ionic"
default_minsep(["In", "O"])    # {"In-In": 3.11, "In-O": 1.87, "O-O": 2.24}
```
