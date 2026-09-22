# AmorphGen

```{image} images/logo_hero.png
:alt: AmorphGen
:width: 60%
:align: center
```

<p style="text-align: center; font-size: 1.15em; color: #555; margin-top: -8px; margin-bottom: 12px;">
Automated amorphous structure generation using machine-learning and classical interatomic potentials.
</p>

<p align="center">
  <a href="https://github.com/SMTG-Bham/AmorphGen"><img src="https://img.shields.io/badge/GitHub-source-181717?style=flat&logo=github" alt="GitHub source"></a>
  <a href="https://pypi.org/project/amorphgen/"><img src="https://img.shields.io/pypi/v/amorphgen?label=PyPI&style=flat" alt="PyPI"></a>
  <a href="https://github.com/SMTG-Bham/AmorphGen/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-yellow?style=flat" alt="License: MIT"></a>
  <a href="https://github.com/SMTG-Bham/AmorphGen/issues"><img src="https://img.shields.io/badge/issues-bug%20tracker-blue?style=flat&logo=github" alt="Issues"></a>
</p>

AmorphGen exposes three routes to amorphous structures: **random placement** from just a chemical formula, **melt-and-quench MD** from a crystal, and a **hybrid** workflow that anneals disordered inputs and quenches to low temperature. All three are powered by universal machine-learning interatomic potentials (MACE, CHGNet, SevenNet) or classical force fields (Buckingham, Lennard-Jones).

```{image} images/main_Fig.png
:alt: AmorphGen workflow: crystalline input or composition to amorphous structure
:width: 100%
:align: center
```

---

## What it does

Making an amorphous model normally means writing your own MD scripts, picking a potential for each new composition and babysitting a multi-stage run. AIMD is accurate but too slow for more than a handful of structures; classical MD is fast but needs a fitted potential for every composition.

AmorphGen wraps the whole thing in one command. You give it a formula or a crystal structure; it runs either random placement plus relaxation or a 7-stage melt-quench, using MACE, CHGNet or SevenNet as the potential (or Lennard-Jones and Buckingham+Coulomb if you have parameters). Minimum separations, density, temperatures and cooling rates have sensible automatic values and can all be overridden from the command line or a YAML file.

You get the relaxed structure, the MD trajectory and an analysis mode (RDF, coordination, angles, rings, S(q)). Structures are written as VASP, CIF or extended XYZ, so they go straight into DFT or a further relaxation at a higher level of theory.

It has been used for oxides (SiO₂, In₂O₃, TiO₂, Ga₂O₃, Al₂O₃, InGaZnO₄), halides (LiF, Li₂ZrCl₆), GaAs, Si, and nitrides (GaN, BN), mostly to prepare structures for DFT or to screen across compositions.

---

## Get started

```bash
pip install "amorphgen[mace]"

# Generate 1 random In2O3 structure (80 atoms, VASP format)
amorphgen --random-gen --composition "In2O3*16" -n 1 --format vasp

# Generate and relax with MACE
amorphgen --random-gen --composition "In2O3*16" --relax --device cpu --format vasp

# Generate 20 structures for ensemble statistics
amorphgen --random-gen --composition "In2O3*16" -n 20 --format vasp
```

See the {doc}`getting-started/installation` and {doc}`getting-started/quickstart` guides for full details.

---

## What it does

AmorphGen exposes three workflows. Pick the one that matches your starting point:

::::{grid} 3
:gutter: 3

:::{grid-item-card} 1. Random generation
:text-align: center

`--random-gen`

Place atoms into a cubic cell with automated minimum separations, density, and target CN from Shannon ionic/metallic radii. Optional relax with any backend.
:::

:::{grid-item-card} 2. Melt-and-Quench (MQ)
:text-align: center

`--mq-ensemble` (or default)

7-stage MD pipeline starting from a crystal: optimise → pre-eq → heat → high-T eq → quench → low-T eq → final optimise.
:::

:::{grid-item-card} 3. Hybrid
:text-align: center

`--hybrid-ensemble`

Anneal a directory of disordered structures (e.g. random-gen outputs) through stages 4–7. Cheaper than MQ because it skips the slow heat ramp.
:::

::::

::::{grid} 2
:gutter: 3

:::{grid-item-card} MLIP and Classical Potentials
:text-align: center

Swap calculators with a single flag. Supports MACE, CHGNet, SevenNet (MLIPs) and Lennard-Jones, Buckingham+Coulomb (classical).
:::

:::{grid-item-card} Structure Analysis
:text-align: center

RDF, coordination numbers, bond angles, ring statistics, energy ranking. CLI or Python API.
:::

::::

---

## Pipeline overview

### 1. Random generation (`--random-gen`)

```text
Composition  (e.g. "In2O3*16"  or  In=32,O=48)
         │
   ┌─────▼──────────────────────────────────────────┐
   │  Auto-derive  minsep, density, target CN       │
   │  from Shannon ionic / metallic radii           │
   └─────┬──────────────────────────────────────────┘
         │
   ┌─────▼──────────────────────────────────────────┐
   │  Random / coordination-aware placement         │
   └─────┬──────────────────────────────────────────┘
         │
   ┌─────▼──────────────────────────────────────────┐
   │  Optional relax  (--relax)                     │
   └─────┬──────────────────────────────────────────┘
         │
   N amorphous structures  (.xyz / .vasp / .cif)
```

See {doc}`guides/random-generation`.

### 2. Melt-quench (MQ)

```text
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

`--mq-ensemble` extends MQ: stages 1–4 run once, then N independent quenches (stages 5–6–7) are launched from snapshots of the stage-4 trajectory. See {doc}`guides/pipeline` and {doc}`guides/mq-ensemble`.

### 3. Hybrid: random → MQ stages 4-7 (`--hybrid-ensemble`)

```text
Directory of disordered structures  (e.g. --random-gen outputs)
         │
   ┌─────▼──────────────────────────────────────────┐
   │  Stage 4  High-T equilibration   T_melt        │
   │           NVT/NPT,  20+ ps                     │
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

See {doc}`guides/hybrid-workflow`.

---

## Features

- Random structure generation: generate amorphous structures from just a chemical formula (e.g. to model amorphous In₂O₃ for 16 formula units, `--composition "In2O3*16"`). Minimum separations, density, and coordination targets are derived automatically from Shannon ionic radii across material classes.
- 7-stage melt-and-quench pipeline: from crystalline POSCAR to relaxed amorphous structure in a single command. Configurable temperatures, cooling rates, ensembles, and timesteps.
- `--mq-ensemble`: generate N independent amorphous structures from one crystalline input in a single CLI call: shared stages 1-4, then N independent quenches via auto-extracted snapshots from the stage-4 trajectory.
- `--hybrid-ensemble`: generate an amorphous ensemble starting from disordered structures (e.g. random-gen outputs). Anneals each at high T, quenches, equilibrates, and relaxes.
- Multiple calculator backends: MACE, CHGNet, SevenNet (MLIPs) and Lennard-Jones, Buckingham+Coulomb (classical). Swap with `--model` flag.
- Structure analysis: built-in RDF, coordination numbers, bond angles, ring statistics, and energy ranking. Gaussian smearing for experimental comparison.
- Plots: vector PDF output (`--save-pdf`), 300 DPI defaults, colour-blind-safe palette.
- Energy ranking: `--rank-from-log` parses random-gen / pipeline log files and ranks structures by total energy without re-evaluating the calculator.
- HPC ready: `--resume` recovers from SLURM walltime limits. Smart checkpoint detection restarts from the last completed stage.
- CLI and Python API: every feature accessible from both the command line and Python. Full YAML configuration support.

---

## Supported backends

| Backend | Install | Model name |
|---------|---------|------------|
| **MACE** | `pip install "amorphgen[mace]"` | `mace-mpa-0` |
| **CHGNet** | `pip install "amorphgen[chgnet]"` | `chgnet` |
| **SevenNet** | `pip install "amorphgen[sevennet]"` | `sevennet`, `7net-mf-ompa` |
| **Classical** | built-in | `buckingham`, `lennard-jones` |

```bash
amorphgen --list-models   # see all 20+ model variants
```

---

## Authors & Contact

**Maintainer:** [Chaiyawat Kaewmeechai](https://cywkmc21.github.io/), University of Birmingham<br>
**Email:** `c[dot]kaewmeechai[at]bham[dot]ac[dot]uk`

**Bug reports / feature requests:** [Open an issue on GitHub](https://github.com/SMTG-Bham/AmorphGen/issues).<br>
For research collaborations or scientific questions, please email the maintainer above.

---

## Citing AmorphGen

If you use AmorphGen in your research, please cite the JOSS paper (in
preparation) and the GitHub repository:

```bibtex
@misc{amorphgen,
  author = {Kaewmeechai, Chaiyawat and Scanlon, David O.},
  title  = {AmorphGen: A Python package for amorphous structure generation
            with machine-learning and classical interatomic potentials},
  year   = {2026},
  url    = {https://github.com/SMTG-Bham/AmorphGen}
}
```

A Zenodo DOI for each tagged release will be added on first stable release.
Please also cite the underlying machine-learning interatomic potential you
use (MACE, CHGNet, SevenNet) and any reference structures or experimental
data you compare against.

---

```{toctree}
:maxdepth: 2
:caption: Getting Started

getting-started/installation
getting-started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: User Guide

guides/random-generation
guides/pipeline
guides/mq-ensemble
guides/batch-quench
guides/hybrid-workflow
guides/best-practices
guides/analysis
guides/backends
guides/yaml-config
guides/benchmarks
guides/api-reference
guides/hpc
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/index
```

```{toctree}
:maxdepth: 2
:caption: Validation

validation/index
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/calculators
api/random-gen
api/pipeline
api/analysis
api/cli
api/config
api/utils
```

```{toctree}
:maxdepth: 1
:caption: Development

contributing
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
