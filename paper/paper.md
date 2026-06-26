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
date: 2026-04-30
bibliography: paper.bib
---

# Summary

Amorphous metal oxides are crucial to a wide range of technological applications, including thin-film transistors, transparent conducting electrodes, memristive devices, and photovoltaic contacts [@nomura2004; @buchholz2014]. Understanding and engineering their properties requires accurate atomic-scale structural models. Two complementary computational approaches are widely used: (1) **melt-and-quench molecular dynamics**, in which a crystalline structure is heated above the melting point, equilibrated as a disordered liquid, and cooled to produce an amorphous glass; and (2) **random placement**, in which atoms are placed stochastically into a simulation cell subject to minimum separation and density constraints, then relaxed to a local energy minimum.

`AmorphGen` is an open-source Python package that integrates both methods within a single, model-agnostic framework. It supports multiple universal machine-learning interatomic potentials (MLIPs) as the energy and force engine including MACE-MP [@batatia2023foundation], CHGNet [@deng2023chgnet], and SevenNet [@park2024sevennet]. This enables quality MD simulations across the periodic table without system-specific parameterisation. `AmorphGen` wraps the Atomic Simulation Environment (ASE) [@larsen2017] and provides a reproducible, configurable, seven-stage melt-and-quench pipeline, a random structure generator, and a batch workflow for producing statistically independent structure ensembles. The package is designed for minimal user intervention and full HPC compatibility.

# Statement of Need

Despite the widespread use of melt-and-quench MD for generating amorphous oxide structures, no standardised open-source pipeline exists that (1) automates all stages of the protocol, (2) works across diverse compositions without force-field parameterisation, (3) supports multiple MLIP backends interchangeably, and (4) is accessible to researchers without deep MD expertise. Most published melt-and-quench workflows are tailored to a single material system, making them difficult to adapt to new compositions or interatomic potential models.

The emergence of universal MLIPs such as MACE-MP [@batatia2023foundation], CHGNet [@deng2023chgnet], and SevenNet [@park2024sevennet] has transformed the accessibility of high-quality atomistic simulation. However, applying these models to amorphous structure generation still requires non-trivial workflow construction: managing sequential simulation stages, selecting appropriate ensembles and thermostats, handling cell control, managing trajectory output, deploying on GPU-based HPC facilities, and generating statistically independent structure ensembles. `AmorphGen` simplifies this process by providing a ready-to-use toolkit with protocols developed and validated across multiple metal oxide systems including In$_2$O$_3$, TiO$_2$, Ga$_2$O$_3$, and SiO$_2$.

The primary target users are computational materials scientists working on amorphous oxide semiconductors, dielectrics, and glasses who need to generate structural models for property prediction, DFT benchmarking, or machine-learning training datasets. The package is also suitable as a teaching tool for MD methodology, as the modular stage architecture allows individual stages to be run and inspected independently.

# Software Design

## Multi-backend calculator factory

`AmorphGen` provides a unified `get_calculator()` function that routes to the appropriate MLIP backend based on a model name string. Supported backends are:

- **MACE** (20+ model variants via `mace-torch`): `mace-mpa-0`, `mace-omat-0-medium`, `mace-matpes-r2scan`, etc.
- **CHGNet** (via `chgnet`): `chgnet`
- **SevenNet** (via `sevenn`): `sevennet`, `7net-mf-ompa`, `7net-l3i5`, `7net-omat`, etc., including multi-fidelity foundation models with automatic `modal` selection.

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

`AmorphGen` also provides a random placement module that generates amorphous starting structures by stochastically placing atoms into a simulation cell subject to pairwise minimum separation constraints. The algorithm uses constrained random sequential placement, optionally enhanced by coordination-aware placement that targets specific coordination numbers by biasing each new atom toward existing under-coordinated sites. Generated structures can be relaxed using any supported MLIP backend.

### Automated minimum separation distances

A key challenge in random structure generation is determining appropriate minimum interatomic distances for each element pair without manual input. `AmorphGen` automates this using a bonding-type-aware scheme based on Shannon effective ionic radii [@shannon1976]:

1. **Bond classification**: each element pair is classified as ionic (metal--nonmetal), covalent (metalloid--metalloid), or metallic (metal--metal) based on the van Arkel--Ketelaar triangle.
2. **Radius selection**: for ionic pairs, Shannon ionic radii are used, with coordination-number-specific values (CN=4 or CN=6) selected automatically from the target coordination number when provided. Metallic pairs use Goldschmidt metallic radii (CN=12), and metalloid pairs use covalent radii.
3. **Scale factors**: the sum of radii is scaled by a bonding-type-dependent factor to give the minimum separation: 0.85 for ionic, covalent, and metallic bonds; 0.80 for small-anion packing contacts (O--O, F--F); and 0.70 for large-anion packing (Cl--Cl, Br--Br). These factors produce distances at 80--90% of equilibrium crystalline bond lengths, consistent with AIRSS practice [@pickard2011].
4. **Metal--metal in ionic context**: for M--M pairs in oxide or halide compounds, the minimum separation is taken as the maximum of the metallic distance and a geometric estimate from edge-sharing polyhedra: $d_\text{M-M} = \max(d_\text{metallic},\, \sqrt{2}\, d_\text{M-X})$, where $d_\text{M-X}$ is the Shannon-derived metal--anion distance.

This scheme is fully composition-agnostic and has been validated across SiO$_2$, GeO$_2$, TiO$_2$, Al$_2$O$_3$, Ga$_2$O$_3$, In$_2$O$_3$, InGaZnO$_4$, and Li$_2$ZrCl$_6$ without any system-specific parameters.

### Automated density estimation

The cell volume (and hence the initial density) is estimated using a three-path approach:

1. **User-specified density** (`--target-density`): used directly when available.
2. **Elemental solid densities**: for pure elements and all-metal alloys, the crystalline density (from CRC Handbook data) is scaled by 0.80 to provide headroom for the random sequential placement algorithm.
3. **Shannon sphere packing**: for compounds containing nonmetal elements (oxides, halides, etc.), the cell volume is estimated from the total ionic sphere volume $V_\text{sphere} = \sum_i \frac{4}{3}\pi r_i^3$ divided by a material-class-dependent packing factor. Shannon CN=6 ionic radii are used for all elements. The packing factor is determined by automatic classification of the compound type:

| Material class | Packing factor | Examples |
|:---|:---:|:---|
| Covalent oxide | 0.50 | SiO$_2$, GeO$_2$ |
| Metal oxide | 0.52 | In$_2$O$_3$, TiO$_2$, Al$_2$O$_3$ |
| Halide | 0.58 | Li$_2$ZrCl$_6$, NaCl |
| Sulfide | 0.55 | Li$_2$S, ZnS |
| Nitride | 0.52 | Si$_3$N$_4$, AlN |

The classification is based on the presence of metal versus metalloid cations and the anion species. This approach produces estimated densities within 62--101% of experimental values across the tested systems, with covalent oxides (SiO$_2$: 98%, GeO$_2$: 101%) and heavy metal oxides (In$_2$O$_3$: 86%) performing best. An auto-retry mechanism expands the cell by 5% if placement fails, ensuring robustness for all compositions.

### Coordination-aware placement

When target coordination numbers are specified (e.g. Si=4, O=2 for SiO$_2$), each new atom is placed within the bonding shell ($d_\text{minsep} \le d \le d_\text{max}$, where $d_\text{max} = 1.5 \times d_\text{minsep}$) of an existing under-coordinated atom rather than at a purely random position. An over-coordination check rejects placements that would push any neighbour beyond its target coordination number. This produces structures with short-range order closer to the target amorphous topology, reducing the relaxation effort required by the MLIP. Targets are auto-detected for each material class (e.g. Si=4, In=5, Zr=6) but can be overridden per element.

## Batch quench workflow

The Stage 4 trajectory can be sampled at regular intervals to extract decorrelated liquid snapshots, which are then independently quenched through Stages 5–7 to produce an ensemble of statistically independent amorphous structures. This is essential for sampling the amorphous configuration space and for computing ensemble-averaged structural properties such as radial distribution functions and bond angle distributions. The batch workflow supports interruption and resumption via a `--resume` flag, making it suitable for HPC environments with job time limits.

# State of the Field

The most common existing approaches for amorphous structure generation are:

- **AIMD with VASP or CP2K**: high accuracy but limited to ~100–200 atoms and ~100 ps total simulation time due to DFT cost [@buchholz2014]. Not accessible to researchers without DFT expertise and HPC allocations.
- **Classical MD with LAMMPS or GROMACS**: fast and scalable, but requires system-specific force-field parameterisation (e.g., BKS potential for SiO$_2$, Rivera potential for In$_2$O$_3$) that is unavailable for many compositions and may not transfer to doped or mixed systems.
- **pyiron**: a general-purpose atomistic simulation framework [@janssen2019] that provides workflow infrastructure but does not implement a melt-and-quench protocol or integrate universal MLIP models.
- **ASE scripting**: the ASE library [@larsen2017] provides all the low-level components (MD integrators, optimisers, calculators), but assembling a complete melt-and-quench pipeline requires significant user effort and domain knowledge.

`AmorphGen` occupies a distinct niche: it provides a complete, validated, multi-stage protocol specifically designed for amorphous structure generation, built on universal MLIPs that require no system-specific parameterisation, supports multiple backends interchangeably, and is packaged for immediate use by non-specialists.

# Validation

To verify that `AmorphGen` reproduces structures consistent with rigorous DFT melt-quench MD, we benchmarked the package against published a-Ga$_2$O$_3$ structures generated by the authors using PBE0 hybrid-DFT melt-quench [@kaewmeechai2025]. Four ensembles were compared at matched composition (Ga$_{160}$O$_{240}$, 400 atoms) and matched cell shape (cubic):

1. **Reference DFT melt-quench** (n=20) from [@kaewmeechai2025].
2. **AmorphGen random + relax** (n=20) using CHGNet.
3. **AmorphGen melt-quench from crystalline supercell** (n=3) using CHGNet, following the protocol of [@kaewmeechai2025] with the heating-rate adjustment necessitated by MLIP cost.
4. **AmorphGen hybrid (random + quench)** (n=20) using CHGNet, in which random structures are annealed at 3000 K (within CHGNet's training window) and quenched at 100 K/ps.

Across all four ensembles, the mean Ga--O bond distance agrees within 1--2%, mean coordination numbers within 1%, and bond-angle distributions within 2% of the reference DFT structures. Density agreement is ~4%, with the small underestimate consistent with the well-known cell-expansion behaviour of CHGNet relative to PBE0. `AmorphGen` therefore reproduces rigorous DFT melt-quench structures at a fraction of the compute cost, and the agreement between random-placement, melt-quench, and hybrid workflows confirms internal consistency of the package across its three structure-generation routes. A built-in `--reference` flag automates this comparison against literature ranges defined by the user in YAML format.

# Research Impact

`AmorphGen` was developed and validated as part of ongoing research at the University of Birmingham on amorphous metal oxide semiconductors for thin-film transistor applications. The pipeline has been applied to In$_2$O$_3$, TiO$_2$, Ga$_2$O$_3$, and SiO$_2$, with structural validation against experimental radial distribution functions, bond angle distributions, and density data. Results obtained using `AmorphGen` are the subject of a manuscript currently in preparation.

The package is deployed on both the Baskerville GPU HPC facility and BlueBEAR (University of Birmingham), and supports common Slurm-based HPC environments via per-stage checkpointing and a `--resume` flag.

# Acknowledgements

The authors thank the Baskerville HPC facility (EPSRC EP/T022221/1) for computational resources.

# References
