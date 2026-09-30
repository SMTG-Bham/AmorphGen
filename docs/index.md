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

AmorphGen exposes three routes to amorphous structures: **random placement** from just a chemical formula, **melt-and-quench MD** from a crystal, and a **hybrid** workflow that anneals disordered inputs and quenches to low temperature. Random placement runs without a potential. Relaxation and MD use machine-learning interatomic potentials (MACE, CHGNet, SevenNet) or classical force fields (Buckingham, Lennard-Jones).

```{image} images/main_Fig.png
:alt: AmorphGen workflow: crystalline input or composition to amorphous structure
:width: 100%
:align: center
```

---

## Get started

Generate one structure from a composition with the base package:

```bash
pip install amorphgen

# 16 formula units of In2O3 = 80 atoms
amorphgen --random-gen --composition "In2O3*16" --seed 42 \
    --format vasp --work-dir in2o3_random
```

The structure is written to `in2o3_random/random_initial/`. To relax an ensemble,
install a calculator backend and add `--relax`:

```bash
pip install "amorphgen[mace]"
amorphgen --random-gen --composition "In2O3*16" -n 5 --seed 42 \
    --relax --model mace-mpa-0 --device cpu --format vasp \
    --work-dir in2o3_relaxed
```

Initial structures go in `in2o3_relaxed/random_initial/` and relaxed structures
in `in2o3_relaxed/random_opt/`. Placement provides starting configurations;
check the relaxed density, bonding and convergence before using them in a study.

See {doc}`getting-started/installation` for platform and backend requirements,
and {doc}`getting-started/quickstart` for the other workflows.

```{note}
These docs follow the repository's `main` branch, which can contain features
added after the latest PyPI release. See {doc}`changelog` for release boundaries
and the installation guide for installing from source.
```

## Choose a workflow

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Random generation
:link: guides/random-generation
:link-type: doc

Start with a **composition**. Place atoms using estimated density, minimum
separations and coordination targets, with optional relaxation using a potential.

`--random-gen`
:::

:::{grid-item-card} Melt-and-quench
:link: guides/pipeline
:link-type: doc

Start with a **crystal**. Run the seven-stage pipeline, or share stages 1–4
before running separate quenches from high-temperature snapshots.

Default pipeline or `--mq-ensemble`
:::

:::{grid-item-card} Hybrid ensemble
:link: guides/hybrid-workflow
:link-type: doc

Start with **disordered structures**. Run high-temperature equilibration,
quenching, low-temperature equilibration and final relaxation (stages 4–7)
for each input.

`--hybrid-ensemble`
:::

::::

The {doc}`guides/best-practices` guide covers choosing and checking a protocol.
For multiple quenches, see {doc}`guides/mq-ensemble` or {doc}`guides/batch-quench`.

## Analyse and configure

- {doc}`guides/analysis`: RDFs, coordination, bond angles, rings, structure factors
  and plots, using the CLI or Python API.
- {doc}`guides/yaml-config`: save a protocol, set temperatures and cooling rates,
  and control reproducibility.
- {doc}`guides/hpc`: run and resume jobs on a cluster.
- {doc}`tutorials/index`: notebooks demonstrating the workflows.
- {doc}`validation/index`: a worked comparison with reference data and its limits.

## Supported backends

| Backend | Install | Example `--model` value |
|---------|---------|-------------------------|
| MACE | `pip install "amorphgen[mace]"` | `mace-mpa-0` |
| CHGNet | `pip install "amorphgen[chgnet]"` | `chgnet` |
| SevenNet | `pip install "amorphgen[sevennet]"` | `sevennet`, `7net-mf-ompa` |
| Classical | Included in `pip install amorphgen` | `buckingham`, `lennard-jones` |

Classical potentials require parameters appropriate to the system. See
{doc}`guides/backends` for model selection and backend compatibility, or run
`amorphgen --list-models` for the registered model names.

The optional **torch-sim engine** batches relaxation and hybrid ensembles with
MACE, SevenNet or Lennard-Jones. For MACE, install
`pip install "amorphgen[mace,torchsim]"` and select `--engine torchsim`.
It requires Python 3.12 or newer; hybrid MD supports NVT only. See the
[installation instructions](getting-started/installation.md#the-torch-sim-engine).

---

## Authors & Contact

**Authors:** [Chaiyawat Kaewmeechai](https://orcid.org/0000-0003-1603-2659), [Louie Slocombe](https://orcid.org/0000-0002-6986-5526) and [David O. Scanlon](https://orcid.org/0000-0001-9174-8601)<br>
**Maintainer:** [Chaiyawat Kaewmeechai](https://cywkmc21.github.io/), University of Birmingham<br>
**Email:** `c[dot]kaewmeechai[at]bham[dot]ac[dot]uk`

**Bug reports / feature requests:** [Open an issue on GitHub](https://github.com/SMTG-Bham/AmorphGen/issues).<br>
For research collaborations or scientific questions, please email the maintainer above.

---

## Citing AmorphGen

If you use AmorphGen in your research, cite the software version you used.
The repository includes a [CITATION.cff](https://github.com/SMTG-Bham/AmorphGen/blob/main/CITATION.cff)
file; a BibTeX citation is:

```bibtex
@misc{amorphgen,
  author = {Kaewmeechai, Chaiyawat and Slocombe, Louie and Scanlon, David O.},
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
:hidden:
:maxdepth: 2
:caption: Getting Started

getting-started/installation
getting-started/quickstart
```

```{toctree}
:hidden:
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
:hidden:
:maxdepth: 2
:caption: Tutorials

tutorials/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Validation

validation/index
```

```{toctree}
:hidden:
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
:hidden:
:maxdepth: 1
:caption: Methodology

notes/sq_xrd_methodology
notes/sq_xrd_credits
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Development

contributing
changelog
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
