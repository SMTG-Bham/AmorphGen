# Analysis

AmorphGen's `--analyse` mode computes structural descriptors for an
ensemble of amorphous structures and produces publication-quality
figures plus CSV data for every plot.

```bash
amorphgen --analyse --input-dir my_structures/ --save-plot plots/
```

That single command computes density, partial radial distribution
functions, coordination distributions, bond-angle distributions, and a
per-structure density violin — saved as four PNG figures plus CSV
companions.

## Recipes

Pick the scenario that matches what you want:

::::::{tab-set}

:::::{tab-item} Quick start

The minimum command, four standalone figures (no comparison, no
validation against literature):

```bash
amorphgen --analyse \
    --input-dir hybrid_ga2o3/random_opt/ \
    --save-plot plots/ \
    --save-pdf
```

Output:

```
plots/
├── analysis_rdf.{png,pdf,csv}        # partial RDFs (all pairs)
├── analysis_cn.{png,pdf,csv}         # coordination distribution
├── analysis_angles.{png,pdf,csv}     # bond-angle distributions
└── analysis_density.{png,pdf,csv}    # per-structure density violin
```

The CSV files contain the raw numbers (r, g(r), CN counts, angle
histograms, per-structure densities) so you can re-plot in any tool.

:::::

:::::{tab-item} Single ensemble + text report

Add a full text report alongside the plots:

```bash
amorphgen --analyse \
    --input-dir hybrid_ga2o3/random_opt/ \
    --save-report report.txt \
    --save-plot plots/ \
    --save-pdf
```

The report shows: density mean ± std, bond distances (mean / std / count
per pair), coordination numbers (with distribution histograms), bond
angles (mean / std / count per triplet).

To also see the breakdown **per structure** (one row per file with
density, energy, CN), add ``--per-structure``:

```bash
amorphgen --analyse \
    --input-dir hybrid_ga2o3/random_opt/ \
    --per-structure
```

If the structure files don't carry per-atom energies in their headers
(VASP, CIF), AmorphGen automatically reads ``random_gen.log`` in the
parent directory (when present) to fill in the E/atom column.

:::::

:::::{tab-item} Validate against a reference YAML

Compare your ensemble against literature ranges with automatic
match/concern/fail scoring:

```bash
amorphgen --analyse \
    --input-dir hybrid_ga2o3/random_opt/ \
    --reference examples/reference_a_Ga2O3.yaml \
    --save-report report.txt \
    --save-plot plots/ \
    --save-pdf
```

The report adds a section like:

```
Validation: a-Ga2O3
  Descriptor              Computed        Expected  Units    Verdict
  ----------------------------------------------------------------------
  Density                    4.37    [4.70, 5.10]  g/cm^3   fail
  Bond Ga-O                  1.91    [1.85, 1.95]  A        match
  CN Ga-O                    4.42    [4.00, 4.80]           match
  Angle Ga-O-Ga             116.8  [110.00, 130.0]  deg     match
  Angle O-Ga-O              108.2  [100.00, 115.0]  deg     match
  ----------------------------------------------------------------------
  Summary: 4 match, 0 concern, 1 fail (out of 5 metrics)
```

AmorphGen ships one reference YAML at
``examples/reference_a_Ga2O3.yaml``. Write your own for other systems
by following the same schema.

:::::

:::::{tab-item} Compare multiple ensembles

For overlaying Random vs Hybrid vs DFT-reference (or any combination),
use the Python API. There's no single CLI flag for this yet —
``compare_ensembles()`` is the entry point:

```python
from amorphgen.analysis import EnsembleSpec, compare_ensembles

compare_ensembles(
    ensembles=[
        EnsembleSpec("DFT-PBE0", "prb_ensemble/*.cif"),
        EnsembleSpec("Random",   "random_inputs/*.vasp"),
        EnsembleSpec("Hybrid",   "hybrid_run/*.xyz"),
    ],
    rdf_pairs=[("Ga-O", "-"), ("Ga-Ga", "--"), ("O-O", ":")],
    cn_top_key="Ga-O",
    cn_bot_key="O-Ga",
    angle_keys=[("O-Ga-O", "-"), ("Ga-O-Ga", "--")],
    exp_density=(4.78, 4.84),
    output_dir="comparison/",
    prefix="ga2o3",
)
```

Output: `comparison/ga2o3_rdf.{png,pdf,csv}`,
`comparison/ga2o3_coordination.{...}`,
`comparison/ga2o3_angles.{...}`,
`comparison/ga2o3_density.{...}` — same layout as `--analyse
--save-plot`, but each figure overlays all listed ensembles with
distinct colours from the Okabe-Ito palette.

See {doc}`/validation/index` for fully-worked examples of this on four
material systems.

:::::

::::::

## Cutoff

The cutoff defines what counts as a "first-shell" bond and affects
coordination, bond-length statistics, and bond-angle triplets. Default
in v1.0.0+ is `auto-rdf`, which finds the first minimum of each partial
RDF — the standard convention in neutron-diffraction analysis of
glasses.

| Cutoff mode | When to use |
|---|---|
| **`auto-rdf`** (default) | All structural analysis. Robust across systems with broad bond-length distributions (a-Si, a-HfO₂, chalcogenides). |
| `auto` | Legacy. Uses minsep from Shannon/Cordero/Goldschmidt radii. Fast but can under-count coordination for systems with long first-shell bonds. |
| Numeric, e.g. `--cutoff 2.5` | Single cutoff (in Å) for all pairs. Useful for tight-bonded covalent networks. |
| Per-pair dict via YAML | Custom per-pair cutoffs for unusual chemistries. |

Setting an explicit cutoff is rarely needed; `auto-rdf` handles
practically every amorphous system AmorphGen targets.

## Structure factor S(q) and simulated XRD



## Full CLI flag reference

```bash
amorphgen --analyse \
    --input-dir DIR_OF_STRUCTURES \
    [--cutoff MODE_OR_NUMBER] \
    [--per-structure] \
    [--save-report FILE] \
    [--save-plot DIR] \
    [--save-pdf] \
    [--reference YAML] \
    [--smearing SIGMA] \
    [--total-rdf] \
    [--dpi N] \
    [--show-title]
```

| Flag | What it does |
|---|---|
| `--input-dir DIR` | Directory of structure files (``.xyz``, ``.extxyz``, ``.cif``, ``.vasp``). Globs everything matching these extensions. |
| `--cutoff MODE` | `auto-rdf` (default), `auto`, or a number in Å. |
| `--per-structure` | Print a per-structure table (one row per file: density, E/atom, CN). |
| `--save-report FILE` | Write the full text report (densities, bond distances, coordination, angles) to a file. |
| `--save-plot DIR` | Save the four standard figures (RDF, CN, angles, density) plus CSV data into ``DIR``. |
| `--save-pdf` | Also save vector PDF copies alongside the PNGs. |
| `--reference YAML` | Validate against the literature ranges in YAML, print a match/concern/fail table. |
| `--smearing SIGMA` | Gaussian smearing of the RDF in Å (typical: ``0.02–0.05`` to compare against experimental neutron data). |
| `--total-rdf` | Overlay the total g(r) on the partial-RDF plot. |
| `--dpi N` | PNG DPI (default 300). |
| `--show-title` | Add titles to each plot (default off — captions usually clearer in figures). |

## Outputs explained

For each ensemble, ``--analyse --save-plot DIR`` writes:

| File | What's in it |
|---|---|
| `analysis_rdf.png` / `.pdf` | Partial RDFs for all unique pairs in the system. One line per pair using the Okabe-Ito palette. |
| `analysis_rdf.csv` | Columns: ``r(A), g(r)_<pair1>, g(r)_<pair2>, …``. Re-plot in any tool. |
| `analysis_cn.png` / `.pdf` | Coordination distribution. For binary AB systems (e.g. SiO₂), shown as **mirrored bars**: A-B on top, B-A reflected below the zero line. For mono-element systems (a-Si), shown side-by-side. |
| `analysis_cn.csv` | Per-pair CN counts as percentages of the centred atom population. |
| `analysis_angles.png` / `.pdf` | Bond-angle histograms (normalised). One line per triplet. |
| `analysis_angles.csv` | Raw angle values, one row per triplet observation. |
| `analysis_density.png` / `.pdf` | Per-structure density violin with jittered scatter and mean ± std label. |
| `analysis_density.csv` | One row per structure: ``structure_index, density_g_per_cm3``. |

## Python API

For programmatic access — useful in scripts, notebooks, and the
comparison workflow:

```python
from amorphgen.analysis import StructureAnalyser

sa = StructureAnalyser("hybrid_ga2o3/random_opt/")   # accepts a dir OR list of files
sa.summary()                                         # print structural summary

# Individual descriptors
rho = sa.density()
print(rho["mean"], rho["std"], rho["values"])

rdf = sa.rdf(pair="Ga-O", sigma=0.05)
print(rdf["r"], rdf["g_r"])

cn = sa.coordination()
print(cn["Ga-O"]["mean"], cn["Ga-O"]["distribution"])

ang = sa.bond_angles()
print(ang["O-Ga-O"]["mean"])

# Multi-ensemble comparison
from amorphgen.analysis import EnsembleSpec, compare_ensembles
compare_ensembles(...)        # see "Compare multiple ensembles" tab above

# Validate against reference
from amorphgen.analysis import validate_against_reference, format_validation_report
import yaml
with open("examples/reference_a_Ga2O3.yaml") as f:
    ref = yaml.safe_load(f)
print(format_validation_report(validate_against_reference(sa, ref)))
```

See {doc}`/api/analysis` for the full Python API.

## Troubleshooting


