<p align="center">
  <img src="https://raw.githubusercontent.com/SMTG-Bham/AmorphGen/main/docs/_static/logo_hero.png" alt="AmorphGen" width="500">
</p>

<p align="center">
  <a href="https://github.com/SMTG-Bham/AmorphGen/actions/workflows/test.yml"><img src="https://img.shields.io/github/actions/workflow/status/SMTG-Bham/AmorphGen/test.yml?branch=main&label=CI" alt="CI"></a>
  <a href="https://smtg-bham.github.io/AmorphGen/"><img src="https://img.shields.io/badge/docs-online-blue" alt="Docs"></a>
  <a href="https://github.com/SMTG-Bham/AmorphGen/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://pypi.org/project/amorphgen/"><img src="https://img.shields.io/pypi/v/amorphgen?label=PyPI" alt="PyPI"></a>
</p>

<p align="center">
  <strong>AmorphGen: Amorphous structure generation via melt-quench MD and random placement.</strong>
</p>

<p align="center">
  Documentation: <a href="https://smtg-bham.github.io/AmorphGen/">smtg-bham.github.io/AmorphGen</a>
</p>

---

<details>
<summary><h2>Pipeline overview</h2></summary>

AmorphGen exposes three workflows. Pick the one that matches your starting point:

| # | Workflow | CLI flag | Starting point |
|---|----------|----------|----------------|
| 1 | **Random generation** | `--random-gen` | Composition only |
| 2 | **Melt-quench (MQ)** | (default) or `--mq-ensemble` | Crystalline input |
| 3 | **Hybrid** | `--hybrid-ensemble` | Directory of disordered structures |

### 1. Random generation (`--random-gen`)

```
Composition  (e.g. "In2O3*16"  or  In=32,O=48)
                           │
   ┌───────────────────────▼────────────────────────┐
   │     Auto-derive  minsep, density, target CN    │
   └───────────────────────┬────────────────────────┘
   ┌───────────────────────▼────────────────────────┐
   │                   Random atoms                 │
   └───────────────────────┬────────────────────────┘
   ┌───────────────────────▼────────────────────────┐
   │          Optional relax  (--relax)             │
   └───────────────────────┬────────────────────────┘
       N amorphous structures  (.xyz / .vasp / .cif)

```

### 2. Melt-quench (MQ)

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
   └─────┬──────────────────────────────────────────┘
         │
   ┌─────▼──────────────────────────────────────────┐
   │  Stage 4  High-T equilibration   T_melt        │
   │           NVT/NPT                              │
   └─────┬──────────────────────────────────────────┘
         │
   ┌─────▼──────────────────────────────────────────┐
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

> `--mq-ensemble` extends MQ: stages 1–4 run once, then N independent quenches (stages 5–6–7) are launched from snapshots of the stage-4 trajectory.

It also writes `melt_memory.{txt,csv,json}` before the quenches, reporting
how much of the initially ordered atom population is still ordered after
heating and in each high-temperature snapshot. This endpoint comparison
cannot distinguish uninterrupted survival from melting and recrystallisation.
See the [MQ-ensemble guide](https://smtg-bham.github.io/AmorphGen/guides/mq-ensemble.html#how-much-starting-crystal-survives-the-melt)
for the definition and order-threshold settings.

### 3. Hybrid: random → MQ stages 4-7 (`--hybrid-ensemble`)

```
Directory of disordered structures  (e.g. --random-gen outputs)
         │
   ┌─────▼──────────────────────────────────────────┐
   │  Stage 4  High-T equilibration   T_melt        │
   │           NVT/NPT                              │
   └─────┬──────────────────────────────────────────┘
         │
   ┌─────▼──────────────────────────────────────────┐
   │  Stage 5  Quench  –  NVT cooling ramp          │
   │           T_melt → T-low                       │
   └─────┬──────────────────────────────────────────┘
         │
   ┌─────▼──────────────────────────────────────────┐
   │  Stage 6  Low-T equilibration   T-low          │
   └─────┬──────────────────────────────────────────┘
         │
   ┌─────▼──────────────────────────────────────────┐
   │  Stage 7  Final optimisation (amorphous)       │
   └─────┬──────────────────────────────────────────┘
         │
   N amorphous structures  (one per input)
```

> Hybrid is cheaper than full MQ: it skips the slow heat ramp (Stage 3) by starting from a disordered structure.

</details>

---

<details>
<summary><h2>Supported backends</h2></summary>

AmorphGen supports multiple calculator backends:

| Backend | Install | Model name(s) |
|---------|---------|----------------|
| **MACE** | `pip install "amorphgen[mace]"` | `mace-mpa-0`, `mace-mpa-0-medium`, `mace-omat-0-medium`, ... (20+ variants) |
| **CHGNet** | `pip install "amorphgen[chgnet]"` | `chgnet` |
| **SevenNet** | `pip install "amorphgen[sevennet]"` | `sevennet`, `7net-mf-ompa`, `7net-l3i5`, `7net-omat`, `7net-0`, ... |
| **Classical** | built-in (no extra install) | `lennard-jones`, `buckingham` |

Only install the backend(s) you need. Classical potentials (Lennard-Jones, Buckingham+Coulomb) are built-in and require no GPU. Use `amorphgen --list-models` to see all available models.

For ensembles on a GPU there is a second execution engine, [torch-sim](https://github.com/torchsim/torch-sim), selected with `--engine torchsim`. It relaxes structures in batches and supports NVT annealing and quenching in the hybrid workflow. Batch sizes adapt to available memory. It works with MACE, SevenNet and Lennard-Jones (CHGNet and Buckingham stay on the ASE engine), needs Python 3.12+, the `[torchsim]` extra and a C/C++ compiler (it compiles kernels while it runs), and writes the same files as the ASE engine. See the [backends guide](https://smtg-bham.github.io/AmorphGen/guides/backends.html) for details.

> **ASE pass-through.** AmorphGen wraps each backend's upstream ASE calculator without modifying unit conventions, stress signs, or PBC handling; energies (eV), forces (eV/Å), stress (eV/Å³), and `atoms.pbc` are inherited directly from the upstream MLIP package. See [docs/guides/backends](https://smtg-bham.github.io/AmorphGen/guides/backends.html) for details.

### MLIP failure checks

MD and relaxation (ASE, torch-sim, and random-gen relaxation) stop on NaN/Inf,
close contacts, abrupt energy changes, or temperature/volume runaway before
saving invalid results. Configure the limits with the top-level `safety` YAML
block. Optional `repulsive_core` and `safety.reference` settings add short-range
repulsion and periodic checks against an independent model. See the
[YAML configuration guide](docs/guides/yaml-config.md#mlip-failure-checks-and-optional-stabilisation)
for defaults, units, and examples.

</details>

---

<details>
<summary><h2>Installation</h2></summary>

AmorphGen requires Python 3.10+ and supports Linux and macOS. Windows is not supported natively; use [WSL](https://learn.microsoft.com/windows/wsl/) instead.

| Task | Install from a checkout |
|---|---|
| Random generation, structure analysis and classical LJ/Buckingham pipelines | `pip install -e .` (no PyTorch) |
| MLIP relaxation and melt-quench MD | `pip install -e ".[mace]"` or `pip install -e ".[chgnet]"` |
| MACE + CHGNet | `pip install -e ".[all]"` |
| Batched relaxation and hybrid MD (`--engine torchsim`) | `pip install -e ".[mace,torchsim]"` (Python 3.12+, a C/C++ compiler, CUDA or CPU; no Apple MPS) |

Install from source:

```bash
git clone https://github.com/SMTG-Bham/AmorphGen.git
cd AmorphGen
pip install -e ".[mace,chgnet]"      # example: MACE + CHGNet
# For development instead: pip install -e ".[all,dev,docs]"
```

Conda can keep the installation isolated. The environment files in [`build_tools/`](https://github.com/SMTG-Bham/AmorphGen/tree/main/build_tools) create the environment and install the clone into it in editable mode in one step, with MACE + CHGNet (`environment.yml`) or, for development, with the torch-sim engine, pytest and the docs toolchain as well (`environment_dev.yml`):

```bash
git clone https://github.com/SMTG-Bham/AmorphGen.git
cd AmorphGen
conda env create -f build_tools/environment.yml       # or build_tools/environment_dev.yml
conda activate amorphgen                               # or amorphgen-dev
```

For PyPI installation, separate backend environments and compiler setup, see the
[installation guide](https://smtg-bham.github.io/AmorphGen/getting-started/installation.html).

> **MLIPs are optional.** The base package is deliberately torch-free.
> Install an MLIP extra only when you need MACE/CHGNet/SevenNet relaxation
> or melt-quench MD; with no torch present, `--device auto` resolves to CPU,
> and commands requesting an unavailable MLIP backend show its install command.
> `amorphgen --list-models` shows every model with installed/missing markers.

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
> MACE foundation models may fail. Prefer separate environments for these backends.

> **GPU recommended for MLIP MD.** Use `--device cuda` or `"device": "auto"`.
> Device auto-detection only runs when a job starts. On a login node with no GPU, no device message will appear until a stage is launched.

</details>

---

<details>
<summary><h2>Quick start</h2></summary>

### Command line

```bash
# -- Random generation (no crystal input needed) --
# Generate 10 random In2O3 structures (80 atoms each) and relax with MACE
amorphgen --random-gen --composition "In2O3*16" -n 10 --relax --device cpu

# Same thing with explicit atom counts
amorphgen --random-gen --composition In=32,O=48 -n 10 --relax --device cpu

# -- Melt-quench pipeline (from crystalline input) --
# Full 7-stage pipeline with MACE (default)
amorphgen POSCAR --device cuda

# Use CHGNet 
amorphgen POSCAR --model chgnet --device cpu

# List all available models
amorphgen --list-models

# -- Ensembles on a GPU with the torch-sim engine (pip install -e ".[mace,torchsim]") --
# Generate 50 structures and relax them in batches
amorphgen --random-gen --composition "GeO2*192" -n 50 --relax \
    --model mace-mpa-0 --device cuda --engine torchsim -o geo2_seeds/

# Anneal, quench and relax the whole ensemble together (stages 4-7, NVT)
amorphgen --hybrid-ensemble --input-dir geo2_seeds/random_opt/ \
    --model mace-mpa-0 --device cuda --engine torchsim \
    -o geo2_hybrid/ --resume
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
pipe.run(stages=[5, 6, 7], input_file="stage4_eq.xyz")
```

</details>

---

<details>
<summary><h2>YAML configuration</h2></summary>

Add `seed: 42` at the top level (or `--seed 42` on the command line) to make a
run reproducible: it seeds the random placement and the velocity initialisation
and thermostat noise of every MD stage. Reproducibility also depends on the
backend, hardware and package versions.
GPU calculations and interrupted/resumed MD runs need not reproduce the exact
trajectory of an uninterrupted run.

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
amorphgen --batch-opt --input-dir SiO2_sc/random_initial --work-dir SiO2_sc_opt --config random_gen_config.yaml
```

See [example_config.yaml](https://github.com/SMTG-Bham/AmorphGen/blob/main/amorphgen/configs/example_config.yaml) for annotated
configuration options.

</details>

---

<details>
<summary><h2>Random structure generation</h2></summary>

Generate random amorphous starting structures:

```bash
# Generate 20 random In₂O₃ structures (80 atoms each)
amorphgen --random-gen \
    --composition "In2O3*16" \
    --n-structures 20 \
    --work-dir In2O3_structures/

# Same with explicit atom counts and target density of 5.5 g/cm3
amorphgen --random-gen \
    --composition In=32,O=48 \
    --target-density 5.5 \
    --n-structures 20 \
    --work-dir In2O3_structures/

# Generate 10 random TiO2 and relax with MACE-MPA-0 (LBFGS; FIRE is the robust alternative) to 0.05 eV/Å on CPU; write cif files to random_structures/
amorphgen --random-gen \
    --composition "TiO2*16" \
    --n-structures 10 \
    --relax --model mace-mpa-0 \
    --optimizer LBFGS --fmax 0.05 \
    --device cpu --format cif \
    --work-dir TiO2_structures/

# Generate 10 random TiO2 at 3.5 g/cm³ and relax them with MACE-MPA-0 with --cell-filter none keeps the cubic cell fixed.
amorphgen --random-gen \
    --composition "TiO2*16" \
    --target-density 3.5 \
    --n-structures 10 \
    --relax --model mace-mpa-0 \
    --cell-filter none \
    --optimizer LBFGS --fmax 0.05 \
    --device cpu --format cif \
    --work-dir TiO2_structures/

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

**Step 1: Generate (default, no relaxation):**

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

**Step 2: Batch optimise:**

`--random-gen` writes the generated structures to `<work-dir>/random_initial/`
(and, with `--relax`, the relaxed ones to `<work-dir>/random_opt/`), so point
`--input-dir` at that subdirectory:

```bash
amorphgen --batch-opt \
    --input-dir random_Ga2O3/random_initial \
    --work-dir random_Ga2O3_opt \
    --model mace-mpa-0 --device cpu --fmax 0.01
```

```python
from amorphgen.pipeline.opt_cell import batch_optimize
from amorphgen.utils import get_calculator

calc = get_calculator(model="mace-mpa-0", device="cpu", default_dtype="float64")

batch_optimize(
    input_dir="random_Ga2O3/random_initial",
    output_dir="random_Ga2O3_opt",
    calc=calc,
)
```

The `--batch-opt` mode uses the full `opt_cell.run()` under the hood, giving
you proper logging, trajectory files, configurable optimizer/cell filter,
and configurable precision (`auto` by default).

### Coordination-aware placement

Set `--target-cn` to choose the coordination targets used during placement. New
atoms are biased toward existing under-coordinated sites, and placements that
exceed the allowed coordination (target plus tolerance) are rejected:

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

Coordination-aware placement biases the initial structure toward the requested coordination; it does not guarantee every atom reaches its target CN. Check the generated and relaxed structures with `--analyse`. Disable coordination-aware placement with `--no-sc` (legacy flag name; the placement is enabled by default whenever `--target-cn` is set or auto-detected).

</details>

---

<details>
<summary><h2>Structure analysis</h2></summary>

`--analyse` takes a directory of structures (xyz, extxyz, cif, vasp) and reports
density, bond lengths, coordination numbers, bond angles and partial RDFs. The
same run can add the structure factor, ring statistics, polyhedral connectivity,
Voronoi indices, a close-contact check and a validation against literature
ranges. Optional descriptors add crystal-like order and ordered cluster sizes,
void distributions, oxygen speciation,
stress-derived elastic moduli and harmonic vibrational DOS. Every quantity
that is plotted is also written as a CSV.

```bash
# Summary to the terminal
amorphgen --analyse --input-dir optimised_structures/

# Report + figures (RDF, coordination, angles, density) + CSVs
amorphgen --analyse --input-dir optimised_structures/ \
    --save-report report.txt --save-plot plots/

# Neutron S(q) by the direct (Debye) method, with ring statistics,
# corner/edge-sharing analysis and Voronoi indices for Ge
amorphgen --analyse --input-dir optimised_structures/ \
    --sq --sq-weighting neutron --rings --connectivity --voronoi Ge \
    --save-report report.txt --save-plot plots/

# Compare with literature ranges (a-Ga2O3, a-SiO2, a-GeO2, a-HfO2 ship in examples/)
amorphgen --analyse --input-dir optimised_structures/ \
    --reference examples/reference_a_GeO2.yaml

# Plot X-ray S(q) by Fourier transform of g(r) 
amorphgen --analyse --input-dir optimised_structures/ \
    --sq --sq-weighting xray --sq-method ft --save-plot plots/

# Plot X-ray total correlation function T(r) 
amorphgen --analyse --input-dir optimised_structures/ \
    --tr --sq-weighting xray --save-plot plots/

# Free-space sampling and oxygen connectivity, without a calculator
amorphgen --analyse --input-dir silica/ --voids --oxygen-speciation \
    --network-formers Si --save-plot descriptors/

# Crystal-like order in a phase-change ensemble
amorphgen --analyse --input-dir gete_mq/final/ --bond-order \
    --order-cutoff 3.5 \
    --qbar6-threshold 0.3 --order-min-neighbors 4 \
    --save-report gete_report.txt --save-plot gete_plots/

# Elastic response and harmonic cell modes, using the selected MLIP
amorphgen --analyse --input-dir relaxed_silica/ --elastic --vdos \
    --model mace-mpa-0 --save-plot descriptors/ --save-report descriptors.txt
```

Notes on the options:

- `--sq` computes S(q) at the reciprocal-lattice q-vectors of each cell, so the
  first sharp diffraction peak is resolved without the truncation of a Fourier
  transform of g(r). Weighting is `xray` (q-dependent Waasmaier–Kirfel form
  factors), `neutron` (Sears scattering lengths) or `unweighted`; `--sq-method ft`
  gives the g(r) transform for comparison. `--sq-partials` adds the Faber-Ziman
  partials S_ab(q) of every element pair to the CSV and a second plot. 
- `--pair-panels` draws each element pair in its own panel, for g(r) and for
  the S(q) partials, which is easier to read than one axis for a four-element
  system like IGZO.
- `--rings` counts the shortest ring per network edge, with the network former
  (Si, Ge, ...) as nodes; `--rings Ge-O` sets the pair explicitly.
- `--connectivity` reports corner-, edge- and face-sharing between cation-centred
  polyhedra and the fraction of cations in edge-sharing pairs, which separates a
  corner-sharing network glass from a random packing with the same coordination.
- `--check-dimers` flags unphysical close contacts (O–O peroxide, N–N) per structure.
- `--bond-order` reports per-atom Steinhardt q6 and Lechner–Dellago q̄6,
  ordered atom fractions and the largest connected ordered cluster, including
  periodic connections. `--order-cutoff` selects the neighbour shell, defaulting
  to `--cutoff`. The 3.5 Å example isolates the first shell of ideal rocksalt
  GeTe with lattice constant 6 Å; calibrate it for your structures. The default
  q̄6 threshold (0.3) and minimum neighbour count (4) need calibration against
  crystal and liquid references for the material; they do not identify a phase.
  `--save-plot` exports JSON, per-structure and per-atom CSVs, and a figure.
- `--voids` samples periodic point clearance using configurable atomic radii;
  `--oxygen-speciation` counts each oxygen's selected network-former neighbours.
- `--elastic` computes the stiffness tensor and Voigt/Reuss/Hill moduli from
  stresses; `--elastic-relax` adds fixed-cell atomic relaxation. `--vdos` uses
  6N force evaluations per cell for harmonic Gamma-point modes in THz.
  These four descriptors also save full per-structure JSON under `--save-plot`.
- `--smearing SIGMA` sets the Gaussian smearing of g(r) (default 0.05 Å; 0 for the
  raw histogram). `--cutoff` is `auto-rdf` (first minimum of each partial g(r),
  so every pair gets its own value), `auto` (radii table), a number in Å, or
  per-pair overrides such as `"In-O=2.6,Zn-O=2.3"` that keep `auto-rdf` for the
  other pairs. For elements bonded to several partner types (O in IGZO) the
  report adds the total coordination over all bonded partners.

`--save-plot DIR` writes RDF and coordination PNGs and CSVs, plus angle
outputs when valid triplets exist and density outputs for ensembles with
at least two structures. `--save-pdf` adds PDF copies. Optional flags add S(q),
ring and total-coordination plots and data; partial/pair-panel plots reuse
the corresponding total CSV. Connectivity and Voronoi outputs are CSV files.
See the [analysis output reference](https://smtg-bham.github.io/AmorphGen/guides/analysis.html#outputs-explained)
for filenames.

### Worked example: a multi-cation oxide (a-IGZO)

Four elements give ten element pairs, three different cation sizes and an
oxygen that is shared between them. One command covers it:

```bash
amorphgen --analyse --input-dir random_opt/ \
    --sq --sq-partials --pair-panels \
    --total-cn O --total-cn "O:In+Zn+Ga" \
    --save-report report.txt --save-plot plots/
```

What to read in the output:

- The header lists the cutoff in force for every pair. `auto-rdf` gives each
  pair its own value from the first minimum of its g(r) (Ga–O 2.03, Zn–O 2.25,
  In–O 2.47 Å here). One number for all pairs would count second-shell oxygens
  around the small Ga cation, so if you override, do it per pair:
  `--cutoff "In-O=2.6"` keeps `auto-rdf` for the rest.
- `Bonding coordination numbers` covers the cation–O pairs (Ga–O 3.9, In–O 5.1,
  Zn–O 3.9) and, because O has three partner types, a `Total coordination`
  block with `O-(Ga+In+Zn)`. Cation–cation and O–O contacts are listed apart as
  `Non-bonded contacts` and never enter the coordination or the angles.
- `--total-cn` adds any total you name: `O` counts all bonded partners,
  `O:In+Ga` only the two larger cations.
- `--sq-partials` prints the first peak of each Faber-Ziman partial S_ab(q)
  and writes them next to the total S(q). The partials do not depend on
  `--sq-weighting`; the weighting only combines them into the total.
- `--pair-panels` puts each pair in its own panel for g(r) and for S_ab(q),
  which is easier to read than ten curves on one axis.

Files this writes in `plots/`: `analysis_rdf` (all partials plus `g(r)_Total`
in the CSV), `analysis_rdf_panels.png`, `analysis_cn` (Ga–O, In–O, Zn–O and
the O total), `analysis_cn_total` (the requested totals), `analysis_sq` (total
S(q) with `s_<pair>` columns), `analysis_sq_partials.png`,
`analysis_sq_partials_panels.png`, `analysis_angles` and `analysis_density`.

The same analysis from Python:

```python
from amorphgen.analysis import StructureAnalyser

sa = StructureAnalyser("optimised_structures/")      # a directory or a list of files
print(sa.summary())
rdf  = sa.rdf(pair="Ge-O", sigma=0.05)              # r, g_r
sq   = sa.structure_factor_direct(weighting="neutron", sigma_q=0.05, partials=True)
rings = sa.ring_statistics()                         # ring_sizes, counts, fractions
conn  = sa.polyhedral_connectivity()                 # corner/edge/face sharing
sa.save_report("report.txt")
sa.plot(output_dir="plots/")
```

The analysis guide in the documentation covers the S(q) conventions and the
reference-YAML format.

</details>

---

<details>
<summary><h2>Generating multiple independent structures (batch quench)</h2></summary>

### Step 1: Run stages 1–4 and save the high-temperature trajectory

```bash
amorphgen POSCAR \
    --stages 1 2 3 4 \
    --eq-high-steps 100000 \
    --work-dir melt_run/
```

### Step 2: Batch quench N runs from snapshots

`--snapshot-dir` accepts a directory of structures or a trajectory file. The
command below extracts 20 evenly spaced frames from the saved trajectory. Use
`--burn-in-frames` to exclude an initial unequilibrated portion; choose snapshot
spacing long enough for the structural correlations relevant to your system.

```bash
amorphgen --batch-quench \
    --snapshot-dir melt_run/stage4_eq_traj.xyz \
    --n-runs 20 --select uniform \
    --batch-stages 5 6 7 \
    --work-dir batch_run/
```

```python
from amorphgen import extract_snapshots
from amorphgen.pipeline import batch_quench

snapshot_paths = extract_snapshots(
    "melt_run/stage4_eq_traj.xyz", n_snapshots=20, output_dir="snapshots",
)
results = batch_quench.run(
    snapshot_files=snapshot_paths,
    n_runs=20,
    select="uniform",
    work_dir="batch_run",
)
```

Each run gets its own subdirectory: `batch_run/run_0000/`, `batch_run/run_0001/`, …

### Resuming an interrupted batch

If a batch job times out, resubmit with `--resume`; already-completed runs are skipped:

```bash
amorphgen --batch-quench \
    --snapshot-dir melt_run/stage4_eq_traj.xyz \
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
Optimise (positions only - preserves density)
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
from amorphgen import extract_snapshots
from amorphgen.pipeline import batch_quench
from amorphgen.utils import get_calculator

# Step 1: Generate random structure (auto minsep from Shannon radii)
atoms = generate_random(
    composition={"Ti": 8, "O": 16},
    target_density=3.2,       # optional, auto-estimated if omitted
    target_cn={"Ti": 6},      # optional, enables coordination-aware placement + CN-aware radii
)

# Step 2: Optimise (positions only)
calc = get_calculator(model="chgnet", device="cpu")
optimised = opt_run(
    atoms, cfg_override={"opt": {"fmax": 0.1, "cell_filter": "none"}},
    calc=calc, work_dir="hybrid_seed",
)

# Step 3: Equilibrate at 2000 K
liquid = eq_run(optimised, cfg_override={
    "eq_high": {"ensemble": "NVT", "T": 2000, "steps": 40000, "timestep": 0.5},
}, calc=calc, stage="high", work_dir="hybrid_liquid")

# Step 4: Extract snapshots and batch quench (Stages 5 → 6 → 7)
snapshot_files = extract_snapshots(
    "hybrid_liquid/stage4_eq_traj.xyz", n_snapshots=5,
    output_dir="hybrid_snapshots", burn_in_frames=100,
)
batch_quench.run(
    snapshot_files, work_dir="hybrid_quenches", calc=calc,
    cfg_override={"quench": {"T_start": 2000}},
)
```

See [Tutorial 5](https://github.com/SMTG-Bham/AmorphGen/blob/main/Tutorials/T5_mix_random_MQ/tutorial_5_batch_quench.ipynb)
for a complete working example.

</details>

---

<details>
<summary><h2>Ensemble choice</h2></summary>

| Stage | Default | Override flag |
|-------|---------|--------------|
| Stage 2 pre-melt eq | NVT | `--eq-premelt-ensemble NPT` |
| Stage 3 melt | NPT | `--melt-ensemble NVT` |
| Stage 4 high-T eq | NPT (MTK) | `--eq-high-ensemble NVT` |
| Stage 5 quench | NVT | `--quench-ensemble NPT` |
| Stage 6 low-T eq | NVT | `--eq-low-ensemble NPT` |

</details>

---

<details>
<summary><h2>Heating / cooling rate</h2></summary>

```bash
# At a 0.5 fs timestep: 100 K/ps heating and 25 K/ps cooling
amorphgen POSCAR --timestep 0.5 \
    --melt-T-step 100 --melt-steps-per-T 2000 \
    --quench-T-step -50 --quench-steps-per-T 4000
```

For `--quench-T-step -100` and `--timestep 0.5`, common cooling rates are:

| Rate | `--quench-steps-per-T` | Time (3000 → 300 K) |
|------|------------------------|----------------------|
| 200 K/ps | 1000 | 13.5 ps |
| 100 K/ps (default) | 2000 | 27 ps |
| 10 K/ps | 20000 | 270 ps |
| 1 K/ps | 200000 | 2700 ps |

</details>

---

<details>
<summary><h2>Trajectory format</h2></summary>

| Format | Extension | Notes |
|--------|-----------|-------|
| `extxyz` | `.xyz` | **Default.** ASE extended XYZ (cell + energy + forces). Readable by OVITO, VESTA, ASE. |
| `xyz` | `.xyz` | Plain XYZ (positions only) |
| `traj` | `.traj` | ASE binary |
| `lammps-dump` | `.dump` | LAMMPS text dump |

</details>

---

<details>
<summary><h2>Available models</h2></summary>

| Name | Backend | Notes |
|------|---------|-------|
| `mace-mpa-0` | MACE | default (MPTrj + sAlex) |
| `mace-omat-0-medium` | MACE | OMAT, excellent phonons (ASL license) |
| `mace-matpes-r2scan` | MACE | MATPES, r²SCAN functional (ASL license) |
| `chgnet` | CHGNet | Charge-informed, good CPU speed |
| `7net-mf-ompa` | SevenNet | Multi-fidelity foundation, OMat+MPtrj+Alexandria |
| `lennard-jones` | Classical | Pair potential, no GPU needed |
| `buckingham` | Classical | Buckingham + Coulomb (Wolf summation), no GPU needed |

```bash
amorphgen --list-models   # full table of all models grouped by backend
```

</details>

---

<details>
<summary><h2>Full configuration reference</h2></summary>

Use [example_config.yaml](https://github.com/SMTG-Bham/AmorphGen/blob/main/amorphgen/configs/example_config.yaml) for the annotated
YAML reference and [the configuration guide](https://github.com/SMTG-Bham/AmorphGen/blob/main/docs/guides/yaml-config.md) for
precedence and stage options. Shared pipeline defaults live in
[default_config.py](https://github.com/SMTG-Bham/AmorphGen/blob/main/amorphgen/configs/default_config.py); individual stage
functions supply additional defaults. `amorphgen --help` lists CLI options.

Python callers can pass the same configuration mapping as `cfg_override`:

```python
from amorphgen import MeltQuenchPipeline

pipe = MeltQuenchPipeline(
    input_file="POSCAR",
    work_dir="my_run",
    cfg_override={
        "model": "mace-mpa-0",
        "device": "auto",
        "opt": {"fmax": 0.02},
        "quench": {"rate": 10},  # K/ps; overrides steps_per_T
    },
)
```

</details>

---

<details>
<summary><h2>Output files</h2></summary>

| Stage | Trajectory | Final structure | Log |
|-------|-----------|-----------------|-----|
| 1 | `stage1_opt.traj` | `stage1_opt.cif` + `stage1_opt.xyz` | `stage1_opt.log` |
| 2 | `stage2_eq_traj.xyz` | `stage2_eq.xyz` | `stage2_eq.log` |
| 3 | `stage3_melt_traj.xyz` | `stage3_melted.xyz` | `stage3_melt.log` |
| 4 | `stage4_eq_traj.xyz` | `stage4_eq.xyz` | `stage4_eq.log` |
| 5 | `stage5_quench_traj.xyz` | `stage5_quenched.xyz` | `stage5_quench.log` |
| 6 | `stage6_eq_traj.xyz` | `stage6_eq.xyz` | `stage6_eq.log` |
| **7** | `stage7_opt.traj` | **`stage7_opt.cif`** + `stage7_opt.xyz` | `stage7_opt.log` |

</details>

---

<details>
<summary><h2>Tutorials</h2></summary>

**Start here**:

| Tutorial | Description |
|----------|-------------|
| [Tutorial 1](https://github.com/SMTG-Bham/AmorphGen/blob/main/Tutorials/T1_5min_intro/tutorial_1_5min_intro.ipynb) | Quick-start tutorial: orientation: what it does, the three workflows, decision tree, one live demo (random + CHGNet relax on a-SiO₂) |

Workflow tutorials (each reports the wall time measured on the CPU it was validated on):

| Tutorial | Description |
|----------|-------------|
| [Tutorial 2](https://github.com/SMTG-Bham/AmorphGen/blob/main/Tutorials/T2_automated_random_gen/tutorial_2_automated_random_gen.ipynb) | Zero-config random gen: composition is the only input; auto-derive minsep, density, target CN, oxidation state across 8 material classes (Si, SiO₂, In₂O₃, CdTe, AlN, LiCl, TiO₂, Cu). Each structure is CHGNet-relaxed and saved to `output_T2/` |
| [Tutorial 3](https://github.com/SMTG-Bham/AmorphGen/blob/main/Tutorials/T3_random_gen/tutorial_3_random_generation.ipynb) | Explicit control + ensemble analysis: the opposite end of T2: hand-picked minsep (from crystalline bond lengths) and target density (from cited amorphous-thin-film references), 5-structure ensembles per system, quantitative RDF / energy / CN / bond-angle analysis vs the crystalline reference (In₂O₃, TiO₂, Al₂O₃, Ga₂O₃; MACE-MPA-0) |
| [Tutorial 4](https://github.com/SMTG-Bham/AmorphGen/blob/main/Tutorials/T4_MQ_via_7_steps/tutorial_4_melt_quench.ipynb) | Full 7-stage melt-quench from crystalline SiO₂ (CHGNet on CPU; flip the backend toggle for MACE on GPU) |
| [Tutorial 5](https://github.com/SMTG-Bham/AmorphGen/blob/main/Tutorials/T5_mix_random_MQ/tutorial_5_batch_quench.ipynb) | Hybrid workflow: random gen → high-T equilibration → batch quench (TiO₂) |
| [Tutorial 6](https://github.com/SMTG-Bham/AmorphGen/blob/main/Tutorials/T6_classical_potential/tutorial_6_classical_potential.ipynb) | Classical potential (Buckingham+Coulomb) relaxation, no GPU needed (SiO₂, Al₂O₃, TiO₂) |

Application case studies (these assume you have done the workflow tutorials):

| Tutorial | Description |
|----------|-------------|
| [Tutorial 7](https://github.com/SMTG-Bham/AmorphGen/blob/main/Tutorials/T7_application_dimer_dissociation/tutorial_7_dimer_dissociation.ipynb) | Defect chemistry: O–O peroxide dimer dissociation kinetics in amorphous In₂O₃, with Arrhenius temperature scan |

</details>

---

<details>
<summary><h2>Package layout</h2></summary>

```
AmorphGen/
├── .github/workflows/
│   └── test.yml                    ← CI (Linux/macOS, backends, min deps, wheel)
├── amorphgen/
│   ├── __init__.py                 ← public API and package version
│   ├── cli.py                      ← CLI entry point (amorphgen command)
│   ├── configs/
│   │   ├── default_config.py       ← all default parameters
│   │   ├── yaml_config.py          ← YAML config loader
│   │   └── example_config.yaml     ← example YAML with all options
│   ├── analysis/                  ← StructureAnalyser, RDF, CN, S(q), validation
│   ├── pipeline/
│   │   ├── run_pipeline.py         ← MeltQuenchPipeline orchestrator
│   │   ├── opt_cell.py             ← Stages 1 & 7 (optimisation) + batch_optimize()
│   │   ├── equilibrate.py          ← Stages 2, 4, 6 (constant-T equilibration)
│   │   ├── melt_cell.py            ← Stage 3 (heat ramp)
│   │   ├── quench.py               ← Stage 5 (cool ramp)
│   │   ├── batch_quench.py         ← batch runner: Stages 5 → 6 → 7 on N snapshots
│   │   └── random_gen.py           ← random + coordination-aware placement
│   └── utils/
│       ├── calculators.py          ← multi-backend calculator factory
│       ├── radii.py                ← Shannon/metallic radii, minsep, density estimation
│       └── common.py               ← dynamics builder, logger, trajectory writer
├── build_tools/                    ← conda environment files (user + dev)
├── test/                           ← unit and optional backend integration tests
├── pyproject.toml
├── LICENSE                         ← MIT
└── README.md
```

</details>

---

<details>
<summary><h2>HPC (SLURM) example</h2></summary>

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
    --melt-T-end 2500 --eq-high-T 2500 \
    --quench-T-start 2500
```

An ensemble of structures on one GPU can use the torch-sim
engine, which batches the structures and, together with `--resume`, can be
resubmitted into a short queue until it finishes. Outputs are written after
every chunk and MD trajectories every 100 steps, so a walltime kill costs at
most one relaxation chunk or 100 MD steps:

```bash
#!/bin/bash
#SBATCH --job-name=amorphgen_ens
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
export PYTHONUNBUFFERED=1            # progress in the log while the job runs

source /path/to/venv/bin/activate    # Python 3.12 with amorphgen[mace,torchsim]

amorphgen --hybrid-ensemble --input-dir /scratch/geo2_seeds/random_opt/ \
    --model mace-mpa-0 --device cuda \
    --engine torchsim --batch-size auto \
    --work-dir /scratch/geo2_hybrid --resume
```

Ready-made BlueBEAR scripts for generation arrays, batched relaxation, batched
MD and the GPU test suite are in `examples/`. Set `AMORPHGEN_VENV`, select your
account with `sbatch --account=your-project`, and create `logs/` before submitting.
See the [HPC guide](https://github.com/SMTG-Bham/AmorphGen/blob/main/docs/guides/hpc.md#configuring-the-bundled-examples) for
repository paths and cluster-specific setup.

</details>

---

<details>
<summary><h2>Dependencies</h2></summary>

| Package | Purpose |
|---------|---------|
| `ase` | MD engine, optimisers, I/O |
| `numpy` | Array operations |
| `scipy` | Vectorized erfc for Coulomb (classical) |
| `pyyaml` | YAML configuration |
| `matplotlib` | Analysis plots |
| `torch` | Tensor runtime installed by optional MLIP / torch-sim dependencies |
| `mace-torch` | MACE calculator (optional) |
| `chgnet` | CHGNet calculator (optional) |
| `sevenn` | SevenNet calculator (optional) |
| `torch-sim-atomistic` | Batched GPU engine for ensembles, `--engine torchsim` (optional, Python 3.12+) |

</details>

---

<details>
<summary><h2>Citation</h2></summary>

If you use AmorphGen in your research, please cite the package
and the foundation model(s) you used.

**AmorphGen:**
```bibtex
@misc{amorphgen,
  author = {Kaewmeechai, Chaiyawat and Slocombe, Louie and Scanlon, David O.},
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

</details>

---

<details>
<summary><h2>License</h2></summary>

MIT

</details>

<details>
<summary><h2>Development notes</h2></summary>

Parts of this codebase were developed with assistance from an AI tool,
Anthropic's Claude (Opus 4.8), for code drafting, refactoring, and
documentation. All AI-assisted code was reviewed, tested, and validated
by the authors, who take full responsibility for the contents of this
repository.

</details>
