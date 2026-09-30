# Quickstart

This page covers melt-quench, random generation and hybrid ensembles, followed
by batch quenching and GPU batching. Install the backend used by each example
first (see {doc}`installation`). Unrelaxed random generation needs only the base
package.

## 1. Melt-and-quench pipeline

Run the full 7-stage pipeline on a crystalline input structure:

::::{tab-set}

:::{tab-item} CLI
```bash
# Full pipeline
amorphgen POSCAR --model mace-mpa-0 --device cuda

# With a YAML configuration you have saved
amorphgen POSCAR --config pipeline.yaml

# Resume an interrupted run in the same work directory
amorphgen POSCAR --model mace-mpa-0 --device cuda --resume
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

# On a later invocation, resume completed stages / saved MD frames
atoms = pipe.run(resume=True)
```
:::

::::

### The 7 stages

| Stage | Name | Description |
|-------|------|-------------|
| 1 | Optimise | Relax positions (+ cell with FrechetCellFilter) |
| 2 | Pre-melt equilibration | NVT at 300 K |
| 3 | Melt | Heat ramp to high temperature (configurable rate in K/ps) |
| 4 | High-T equilibration | NPT (MTK) at melt temperature by default |
| 5 | Quench | Cool to target temperature (configurable rate in K/ps) |
| 6 | Low-T equilibration | Equilibrate at low temperature |
| 7 | Final optimisation | Final relaxation |

## 2. Random structure generation

Generate an ensemble of random amorphous structures with minimum separations
derived from ionic, covalent or metallic radii according to the composition:

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

# Single structure (automatic minimum separations)
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

## 3. Hybrid ensemble

Start from disordered structures and run stages 4–7 on each, skipping the
crystalline optimisation and heating stages:

```bash
amorphgen --random-gen --composition "SiO2*16" -n 5 --work-dir sio2_seeds
amorphgen --hybrid-ensemble --input-dir sio2_seeds/random_initial \
    --model mace-mpa-0 --device cuda --work-dir sio2_hybrid --resume
```

Final structures are collected in `sio2_hybrid/final/`. To hold the initial
volume during MD, set `--eq-high-ensemble NVT`; the default ASE stage 4 uses NPT.

## Batch quench

Quench multiple snapshot structures through the final pipeline stages:

```bash
amorphgen --batch-quench \
    --snapshot-dir snapshots/ \
    --model mace-mpa-0 \
    --batch-stages 5 6 7 \
    --resume
```

## Ensembles on a GPU with the torch-sim engine

With `pip install "amorphgen[mace,torchsim]"` (Python 3.12+, and a
[C/C++ compiler](installation.md#the-torch-sim-engine)), the
`--random-gen --relax`, `--batch-opt` and `--hybrid-ensemble` modes process
structures in batches. Add `--engine torchsim` to the command; the output files
are the same as with the ASE engine. Hybrid MD uses NVT; an explicit NPT
configuration is rejected.

```bash
# Generate 50 seeds and relax them in batches
amorphgen --random-gen --composition "GeO2*192" -n 50 --relax \
    --model mace-mpa-0 --device cuda --engine torchsim -o geo2_seeds/

# Anneal, quench and relax the whole ensemble together (stages 4-7, NVT only)
amorphgen --hybrid-ensemble --input-dir geo2_seeds/random_opt/ \
    --model mace-mpa-0 --device cuda --engine torchsim \
    -o geo2_hybrid/ --resume
```

The chunk size is chosen automatically from a GPU memory probe
(`--batch-size auto`, the default); `--resume` continues a killed job from
the last written chunk or MD frame. See {doc}`../guides/backends`.

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
default_minsep({"In": 2, "O": 3})  # pair minimum separations in Å
```
