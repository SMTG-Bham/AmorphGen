# Validation

AmorphGen has been benchmarked against a DFT-PBE0 melt-quench reference for
amorphous Ga₂O₃. The results below come from running the **hybrid workflow**
(`--hybrid-ensemble`) — random placement followed by stages 4-7 (high-T
anneal → quench → low-T eq → final relax) with CHGNet on GPU, NPT throughout —
alongside Random and Full melt-quench ensembles for comparison.

All ensembles are **N = 20 structures**. Plots and reports below come
from the built-in analysis (`amorphgen --analyse`) so they exactly match
what you would see if you re-ran this workflow yourself.

```{note}
**Analysis cutoff convention.** All coordination and bond-angle numbers
on this page are produced with the default `--cutoff auto-rdf`, which
finds the first minimum of each partial RDF and is the standard
convention in neutron-diffraction analysis of glasses and liquids. The
legacy `--cutoff auto` (minsep-based) is kept for the
placement / repair stages of `--random-gen` but **should not be used
for analysis** of materials with broad first-shell distributions — it
systematically truncates the first peak and under-counts coordination.
This was the default in AmorphGen v1.0.0rc1 and earlier; from v1.0.0 the
analysis default is `auto-rdf`.
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

**Workflow:** Four ensembles compared — DFT-PBE0 (PRB 2025), AmorphGen
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
Local structure (bond, coordination, angles) matches the PBE0 reference
across all three AmorphGen workflows. Density agreement varies: Random
and Hybrid stay closer to PBE0 because their cells are constrained near
the auto-estimated density; the Full MQ workflow (NPT throughout)
exposes CHGNet's preferred equilibrium density, which is ~9% lower than
PBE0 — a known limitation of MPtrj-trained MLIPs for oxide glasses.
```

### Validation figure

```{image} /images/validation/ga2o3/fig_validation.png
:alt: a-Ga2O3 validation — four-way comparison (DFT-PBE0, Random, Hybrid, Full MQ)
:width: 100%
:align: center
```

*Panels: (a) partial RDFs (Ga–O solid, Ga–Ga dashed, O–O dotted);
(b) coordination distributions (Ga-centred top half, O-centred bottom);
(c) bond-angle distributions (O–Ga–O solid, Ga–O–Ga dashed);
(d) per-structure density compared with PBE0 reference and experimental
range. Blue = DFT-PBE0 reference, orange = AmorphGen Random,
green = AmorphGen Hybrid, pink = AmorphGen Full MQ.*

### Reproduce

```bash
# 1. Generate 20 random a-Ga2O3 structures
amorphgen --random-gen \
    --composition "Ga2O3*32" \
    --n-structures 20 \
    --work-dir random_ga2o3/

# 2. Run either Hybrid or Full MQ workflow

#   Option A — Hybrid (from random inputs, Stages 4-7)
amorphgen --hybrid-ensemble \
    --input-dir random_ga2o3/ \
    --config examples/hybrid_stages_4567_cuda.yaml \
    --work-dir hybrid_ga2o3/

#   Option B — Full melt-quench from a crystal supercell
amorphgen Ga2O3_supercell.xyz \
    --mq-ensemble --n-structures 20 \
    --config examples/mq_stages_1234_cuda.yaml \
    --work-dir mq_ga2o3/

# 3. Analyse against the bundled reference YAML
mkdir final && cp <workflow_dir>/run_*/run_0000/final_amorphous.xyz final/
amorphgen --analyse \
    --input-dir final/ \
    --reference examples/reference_a_Ga2O3.yaml \
    --save-report report.txt --save-plot plots/ --save-pdf
```

The reference file `examples/reference_a_Ga2O3.yaml` is shipped with
AmorphGen and contains the PBE0 ensemble metrics plus experimental
neutron/EXAFS data for automatic match/concern/fail scoring.

## Data availability

A representative a-Ga₂O₃ structure ships with AmorphGen at
`examples/validation/ga2o3/example_final.xyz`. The full 20-structure
ensemble will be archived on Zenodo at the first stable release (v1.0.0);
a DOI link will be added here.

## Reference data sources

- **a-Ga₂O₃**: Kaewmeechai, Strand & Shluger, *Phys. Rev. B* **111** (2025) 035203 (DFT-PBE0 ensemble); Stehlik et al., *J. Non-Cryst. Solids* **458** (2017) 14 (neutron + EXAFS); Yoshioka et al., *J. Phys. Condens. Matter* **19** (2007) 346211 (DFT-MD).
