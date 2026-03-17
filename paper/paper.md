---
title: 'AmorphGen: A Python package for amorphous structure generation by melt-quench simulation and random placement using universal machine-learning force fields'
tags:
  - Python
  - molecular dynamics
  - amorphous materials
  - machine learning interatomic potentials
  - materials science
  - melt-quench
  - random structure generation
authors:
  - name: Chaiyawat Kaewmeechai
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: David O. Scanlon
    orcid: 
    affiliation: 1
affiliations:
  - name: School of Chemistry, University of Birmingham, Edgbaston, Birmingham B15 2TT, United Kingdom
    index: 1
date: 2026-03-16
bibliography: paper.bib
---

# Summary

Amorphous metal oxides are crucial to a wide range of technological applications, including thin-film transistors, transparent conducting electrodes, memristive devices, and photovoltaic contacts [@nomura2004; @buchholz2014]. Understanding and engineering their properties requires accurate atomic-scale structural models. Two complementary computational approaches are widely used: (1) **melt-and-quench molecular dynamics**, in which a crystalline structure is heated above the melting point, equilibrated as a disordered liquid, and cooled to produce an amorphous glass; and (2) **random placement**, in which atoms are placed stochastically into a simulation cell subject to minimum separation and density constraints, then relaxed to a local energy minimum.

`AmorphGen` is an open-source Python package that integrates both methods within a single, model-agnostic framework. It supports multiple universal machine-learning interatomic potentials (MLIPs) as the energy and force engine including MACE-MP [@batatia2023foundation], CHGNet [@deng2023chgnet], and M3GNet [@chen2022m3gnet]. This enables quality MD simulations across the periodic table without system-specific parameterisation. `AmorphGen` wraps the Atomic Simulation Environment (ASE) [@larsen2017] and provides a reproducible, configurable, seven-stage melt-and-quench pipeline, a random structure generator, and a batch workflow for producing statistically independent structure ensembles. The package is designed for minimal user intervention and full HPC compatibility.

# Statement of Need

Despite the widespread use of melt-and-quench MD for generating amorphous oxide structures, no standardised open-source pipeline exists that (1) automates all stages of the protocol, (2) works across diverse compositions without force-field parameterisation, (3) supports multiple MLIP backends interchangeably, and (4) is accessible to researchers without deep MD expertise. Most published melt-and-quench workflows are tailored to a single material system, making them difficult to adapt to new compositions or interatomic potential models.

The emergence of universal MLIPs such as MACE-MP [@batatia2023foundation], CHGNet [@deng2023chgnet], and M3GNet [@chen2022m3gnet] has transformed the accessibility of high-quality atomistic simulation. However, applying these models to amorphous structure generation still requires non-trivial workflow construction: managing sequential simulation stages, selecting appropriate ensembles and thermostats, handling cell control, managing trajectory output, deploying on GPU-based HPC facilities, and generating statistically independent structure ensembles. `AmorphGen` simplifies this process by providing a ready-to-use toolkit with protocols developed and validated across multiple metal oxide systems including In$_2$O$_3$, TiO$_2$, Ga$_2$O$_3$, and SiO$_2$.

The primary target users are computational materials scientists working on amorphous oxide semiconductors, dielectrics, and glasses who need to generate structural models for property prediction, DFT benchmarking, or machine-learning training datasets. The package is also suitable as a teaching tool for MD methodology, as the modular stage architecture allows individual stages to be run and inspected independently.

# Software Design

## Multi-backend calculator factory

`AmorphGen` provides a unified `get_calculator()` function that routes to the appropriate MLIP backend based on a model name string. Supported backends are:

- **MACE** (20+ model variants via `mace-torch`): `mace-mpa-0`, `mace-omat-0-medium`, `mace-matpes-r2scan`, etc.
- **CHGNet** (via `chgnet`): `chgnet`
- **M3GNet** (via `matgl`): `m3gnet`

Each backend is lazy-imported so that users only need to install the backend(s) they use. Custom fine-tuned models are also supported via a file path.

## Seven-stage melt-and-quench pipeline

The pipeline implements a validated melt-and-quench protocol:

1. **Structure optimisation** (LBFGS/FIRE + cell filter): relaxes the crystalline input to the MLIP potential energy surface.
2. **Pre-melt equilibration at 300 K** (NVT, 50 ps): thermalises the crystal before heating.
3. **NPT heat ramp** (300 K → T$_\text{melt}$): expands the cell to the melt density at a configurable heating rate (default 100 K/ps).
4. **High-temperature equilibration at T$_\text{melt}$** (NVT): fully disorders the liquid structure. Supports optional snapshot sampling for batch structure generation.
5. **NVT cooling ramp** (T$_\text{melt}$ → 300 K): quenches the liquid to an amorphous solid at a configurable cooling rate (default 100 K/ps).
6. **Low-temperature equilibration at 300 K** (NVT): relaxes the quenched structure.
7. **Final optimisation** (LBFGS/FIRE + cell filter): minimises residual forces to produce the final amorphous structure.

The pipeline is orchestrated by `MeltQuenchPipeline`, a single-class interface that manages the calculator lifecycle, working directory management, and inter-stage data passing. All simulation parameters are controlled through a hierarchical configuration dictionary with sensible defaults. Both a Python API and a command-line interface (`amorphgen`) are provided.

## Random structure generation

`AmorphGen` also provides a random placement module that generates amorphous starting structures by stochastically placing atoms into a simulation cell subject to user-defined minimum separation distances and target density constraints. Generated structures can optionally be relaxed using any supported MLIP backend. This approach is useful for rapidly exploring configuration space, generating training data for machine-learning models, or providing initial structures for subsequent melt-and-quench refinement.

## Batch quench workflow

The Stage 4 trajectory can be sampled at regular intervals to extract decorrelated liquid snapshots, which are then independently quenched through Stages 5–7 to produce an ensemble of statistically independent amorphous structures. This is essential for sampling the amorphous configuration space and for computing ensemble-averaged structural properties such as radial distribution functions and bond angle distributions. The batch workflow supports interruption and resumption via a `--resume` flag, making it suitable for HPC environments with job time limits.

# State of the Field

The most common existing approaches for amorphous structure generation are:

- **AIMD with VASP or CP2K**: high accuracy but limited to ~100–200 atoms and ~100 ps total simulation time due to DFT cost [@buchholz2014]. Not accessible to researchers without DFT expertise and HPC allocations.
- **Classical MD with LAMMPS or GROMACS**: fast and scalable, but requires system-specific force-field parameterisation (e.g., BKS potential for SiO$_2$, Rivera potential for In$_2$O$_3$) that is unavailable for many compositions and may not transfer to doped or mixed systems.
- **pyiron**: a general-purpose atomistic simulation framework [@janssen2019] that provides workflow infrastructure but does not implement a melt-and-quench protocol or integrate universal MLIP models.
- **ASE scripting**: the ASE library [@larsen2017] provides all the low-level components (MD integrators, optimisers, calculators), but assembling a complete melt-and-quench pipeline requires significant user effort and domain knowledge.

`AmorphGen` occupies a distinct niche: it provides a complete, validated, multi-stage protocol specifically designed for amorphous structure generation, built on universal MLIPs that require no system-specific parameterisation, supports multiple backends interchangeably, and is packaged for immediate use by non-specialists.

# Research Impact

`AmorphGen` was developed and validated as part of ongoing research at the University of Birmingham on amorphous metal oxide semiconductors for thin-film transistor applications. The pipeline has been applied to In$_2$O$_3$, TiO$_2$, Ga$_2$O$_3$, and SiO$_2$, with structural validation against experimental radial distribution functions, bond angle distributions, and density data. Results obtained using `AmorphGen` are the subject of a manuscript currently in preparation.

The package is deployed on the Baskerville GPU HPC facility (University of Birmingham).

# Acknowledgements

The authors thank the Baskerville HPC facility (EPSRC EP/T022221/1) for computational resources.

# References
