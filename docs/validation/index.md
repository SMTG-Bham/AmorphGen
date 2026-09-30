# Validation

This page records a comparison against a DFT-PBE0 melt-quench reference for
amorphous Ga₂O₃. It compares random placement followed by CHGNet relaxation,
the **hybrid workflow** (`--hybrid-ensemble`, stages 4–7), and the full
melt-quench workflow with CHGNet on GPU.

The reported ensembles contain **N = 20 structures** each. The repository
includes the comparison figure and one representative structure, but not
the full ensembles or the complete run configurations and seeds. The
commands below show how to run a similar comparison; they do not reproduce
these numerical results exactly.

```{note}
The reported coordination and bond-angle analysis uses `--cutoff auto-rdf`,
the analysis default since v1.0.0rc2. It estimates each bond cutoff from
the first minimum of the partial RDF and falls back to a radii-based value
when no suitable minimum is found. The alternative `--cutoff auto` uses
minimum-separation estimates and can truncate broad first shells. Inspect
the RDF and selected cutoff when comparing coordination numbers between
studies. Changes to cutoff detection can also change reanalysed results;
see {doc}`/changelog`.
```

## a-Ga₂O₃

::::{grid} 2
:gutter: 3
:margin: 0

:::{grid-item}
:columns: 7

**System:** Ga₁₆₀O₂₄₀ supercell (400 atoms), N=20 structures per ensemble

**Reference:** DFT-PBE0 melt-quench ensemble from
[Kaewmeechai, Strand & Shluger, *Phys. Rev. B* **111**, 035203 (2025)](https://doi.org/10.1103/PhysRevB.111.035203)

**Workflow:** Four ensembles compared, DFT-PBE0 (PRB 2025), AmorphGen
Random + CHGNet relax, AmorphGen Hybrid (Stages 4-7), AmorphGen Full MQ
(Stages 1-7, NPT throughout).
:::

:::{grid-item}
:columns: 5

```{figure} /images/validation/ga2o3/structure.png
:alt: Representative AmorphGen a-Ga2O3 structure
:width: 70%

*a-Ga₂O₃ structure (Ga: dark, O: red)*
```
:::

::::

### Results vs DFT-PBE0 + experiment

| Metric | DFT-PBE0 | Random | Hybrid | Full MQ | Experiment |
|---|---|---|---|---|---|
| Density (g/cm³) | **4.83** | 4.63 | 4.70 | 4.37 | 4.78–4.84 |
| Ga–O bond (Å) | 1.895 | 1.925 | 1.922 | **1.913** | — |
| Ga–O coordination | **4.42** | 4.38 | 4.42 | 4.33 | ~4.5 (EXAFS) |
| O–Ga–O angle (°) | 107.5 | 107.7 | 107.6 | **107.9** | — |

```{note}
In this comparison, the local structural averages are close to the PBE0
reference across all three AmorphGen workflows. Density agreement varies:
the Full MQ ensemble is about 9% less dense than the PBE0 ensemble.
The guide examples use NVT for hybrid MD and NPT for full MQ, so their
volume constraints differ. Density differences depend on both the
potential and the simulation protocol.
```

### Validation figure

```{image} /images/validation/ga2o3/fig_validation.png
:alt: a-Ga2O3 validation - four-way comparison (DFT-PBE0, Random, Hybrid, Full MQ)
:width: 100%
:align: center
```

*Panels: (a) partial RDFs (Ga–O solid, Ga–Ga dashed, O–O dotted);
(b) coordination distributions (Ga-centred top half, O-centred bottom);
(c) bond-angle distributions (O–Ga–O solid, Ga–O–Ga dashed);
(d) per-structure density compared with PBE0 reference and experimental
range. Blue = DFT-PBE0 reference, orange = AmorphGen Random,
green = AmorphGen Hybrid, pink = AmorphGen Full MQ.*

### Run a similar comparison

Run from a repository checkout after installing CHGNet. Save the example
configurations from {doc}`/guides/hybrid-workflow` and
{doc}`/guides/mq-ensemble` as `hybrid.yaml` and `mq.yaml`, and provide your own
`Ga2O3_supercell.xyz` for the full MQ option. Review the temperatures,
durations, and cell constraints for your system before running.

```bash
# 1. Generate and relax 20 random structures (Ga160O240, 400 atoms)
amorphgen --random-gen \
    --composition "Ga2O3*80" \
    --n-structures 20 \
    --relax --model chgnet --device cuda --cell-filter none \
    --work-dir random_ga2o3/

# 2. Run either Hybrid or Full MQ workflow

#   Option A - Hybrid (from the random placements, Stages 4-7)
amorphgen --hybrid-ensemble \
    --input-dir random_ga2o3/random_initial/ \
    --config hybrid.yaml \
    --work-dir hybrid_ga2o3/

#   Option B - Full melt-quench from a crystal supercell
amorphgen Ga2O3_supercell.xyz \
    --mq-ensemble --n-structures 20 \
    --config mq.yaml \
    --work-dir mq_ga2o3/

# 3. Analyse against the bundled reference YAML (mq_ga2o3/final/ for Option B)
amorphgen --analyse \
    --input-dir hybrid_ga2o3/final/ \
    --reference examples/reference_a_Ga2O3.yaml \
    --save-report report.txt --save-plot plots/ --save-pdf
```

Both ensemble modes collect their final structures in `<work-dir>/final/`.
Analyse `random_ga2o3/random_opt/` for the random-plus-relaxation comparison.
Use `amorphgen.analysis.compare_ensembles` to plot the ensembles together;
see {doc}`/api/analysis`.

Reference files for a-SiO₂, a-GeO₂, a-HfO₂ and a-IrO₂ sit next to it in
`examples/` in the repository. The reference file
`examples/reference_a_Ga2O3.yaml` contains literature ranges for automatic
match/concern/fail scoring. These example files are not installed by the
Python wheel.

## Data availability

A representative a-Ga₂O₃ structure is included in the repository at
`examples/validation/ga2o3/example_final.xyz`. The full 20-structure
ensembles are not included. To inspect the bundled structure:

```bash
amorphgen --analyse examples/validation/ga2o3/example_final.xyz \
    --reference examples/reference_a_Ga2O3.yaml
```

Its individual metrics need not equal the ensemble averages in the table.

## Reference data sources

- a-Ga₂O₃: Kaewmeechai, Strand & Shluger, *Phys. Rev. B* **111** (2025) 035203 (DFT-PBE0 ensemble); Stehlik et al., *J. Non-Cryst. Solids* **458** (2017) 14 (neutron + EXAFS); Yoshioka et al., *J. Phys. Condens. Matter* **19** (2007) 346211 (DFT-MD).
