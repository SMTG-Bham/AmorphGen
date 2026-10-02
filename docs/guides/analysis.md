# Analysis

AmorphGen's `--analyse` mode computes structural descriptors for an
ensemble of amorphous structures and writes a figure and a CSV
file for each quantity.

```bash
amorphgen --analyse --input-dir my_structures/ --save-plot plots/
```

That single command computes density, partial radial distribution
functions, coordination distributions, bond-angle distributions, and a
per-structure density violin, saved as up to four PNG figures plus CSV
companions. The density plot and CSV require at least two structures;
angle output requires valid bond-angle triplets.

Directory input reads `.xyz`, `.extxyz`, `.vasp` and `.cif` files in that
directory. Files with the same stem count once, in that format priority
order. Each file contributes its last frame; to analyse a trajectory as an
ensemble, first extract snapshots or pass a list of frames to the Python API.

## Spread and uncertainty of the mean

Each input structure is one independent sampling unit. Pooled site, bond and
angle spreads remain available as the legacy `mean`/`std` fields and the
explicit `pooled_mean`/`pooled_std` fields. They describe structural disorder;
they are not errors on an ensemble mean. Density retains its population
spread over structures in `std`.

Scalar results include an `uncertainty` summary calculated from **one mean per
structure**, giving every contributing structure equal weight. This matters
when cell sizes, numbers of bonds or numbers of angles differ. Its fields are:

| Field | Meaning |
|---|---|
| `per_structure` | Values in input order; `None` for missing descriptors |
| `mean`, `std` | Equal-weight mean and sample SD between structures (`ddof=1`) |
| `sem` | Standard error, sample SD divided by the square root of the number of contributing structures |
| `ci_low`, `ci_high` | Student-t confidence interval with `n - 1` degrees of freedom; 95% by default |
| `bootstrap_low`, `bootstrap_high` | Percentile interval of resampled structure means; 1,000 draws and seed 0 by default |
| `n_structures`, `n_total_structures` | Contributing structures and all supplied structures |
| `n_per_point` | Number contributing to each curve bin (scalar for a scalar descriptor) |

With fewer than two contributing structures, SEM and interval bounds are
`None` (blank in CSV). Missing descriptors are excluded, not replaced with
zero. A present species with no neighbors has zero coordination; a missing
central species has no coordination observation. Repeating sites within one
structure does not increase the independent sample count.

`rdf()`, `averaged_rdf()`, `structure_factor()`, `structure_factor_direct()`,
`total_correlation()` and `angle_distribution()` return per-structure curves
and pointwise uncertainty on common grids. Angle histograms are normalized
within each structure before averaging. S(q) and T(r) transformations use each
structure's own composition and density before averaging, preserving their
covariance. Direct S(q) bins with no reciprocal vectors are missing; their
`n_per_point` can be smaller than the ensemble size. The `n_per_bin` field
counts reciprocal vectors, not independent samples.
The weighted Fourier-transform S(q) needs at least two atoms of every species
for its same-species RDF normalization; use `structure_factor_direct()` for
singleton dopants. An unestimable partial is never silently treated as an
observed zero curve.

```python
sa = StructureAnalyser("ensemble/")
cn = sa.coordination("Si-O")["Si-O"]
print(cn["pooled_std"], cn["uncertainty"]["sem"])
rdf = sa.rdf(confidence=0.95, n_bootstrap=2000, seed=42)
angles = sa.angle_distribution("O-Si-O", bins=90, seed=42)
tr = sa.total_correlation(weighting="xray", seed=42)
# T(r)'s primary uncertainty is for T_r; other curves have separate summaries.
print(tr["curve_uncertainty"]["G_r"]["ci_low"])
```

Bootstrap resampling selects **whole structures**, retaining correlations
between bins. The shaded bands are **pointwise**, not simultaneous confidence
bands for the entire curve. Set `n_bootstrap=0` to skip bootstrap draws in the
curve APIs. These intervals quantify sampling of independent configurations;
they do not include force-field bias, finite-cell error, cutoff selection, or
transform-parameter uncertainty. Correlated trajectory frames require
independent sampling or a separate correlation/block analysis before using
these intervals.

RDF, angle, S(q), and T(r) exports include companion
`*_uncertainty.csv`, `*_per_structure.csv`, and `*_uncertainty.json` files.
The JSON records confidence level, seed, resampling count, and sampling unit;
the CSV includes contributing counts, SEM, t bounds and bootstrap bounds.
Raw angle CSV rows also carry the structure index. The density plot shows a
t interval for the mean, alongside individual structures.
`analysis_statistics.json` retains core scalar descriptors and their aligned
per-structure observations; `analysis_statistics.csv` separates pooled spread
from structure means, sample SD, SEM and t intervals.

Coordination and oxygen-speciation outputs distinguish `fraction_of_sites`
(pooled sites, between 0 and 1) from `fraction_of_structures` (the fraction of
all input structures containing **at least one** site in that category).
Structure prevalence is not a distribution: categories can overlap and need
not sum to one. Crystal-like order and dimer/connectivity reports make the
same distinction. Per-structure site-fraction summaries estimate the
**equal-weight mean site fraction**, which can differ from the pooled fraction.
Optional descriptors (rings, Voronoi, oxygen speciation, bond order, voids,
elastic moduli, vibrational DOS and energy) also retain per-structure values
and named uncertainty summaries. Existing void Monte Carlo errors remain
separate from uncertainty across structures.

Reference validation uses the structure-weighted mean and its t interval.
A confidence interval admitting both in-range and out-of-range values is
`inconclusive`, even when its mean is inside the reference range. A finite
mean with no estimable interval is also `inconclusive`; an absent descriptor
remains `n/a`. Reports include intervals and count inconclusive verdicts.

(ensemble-convergence)=
## Declared tolerances and ensemble convergence

Declare an absolute tolerance for the uncertainty of each descriptor's
**ensemble mean**, in its native units. A tolerance of `0.02` for density
means a 95% Student-t interval half-width no greater than 0.02 g/cm³; it is
neither a relative percentage nor a bound on the spread of individual structures.

```bash
amorphgen --analyse --input-dir ensemble/ \
    --tolerance density=0.02 \
    --tolerance coordination.Si-O=0.05 \
    --tolerance bond_angle.O-Si-O=1.0 \
    --convergence-confidence 0.95 --save-plot analysis/ \
    --save-report analysis.txt
```

Each `--tolerance NAME=VALUE` enables convergence reporting. Use `--convergence`
alone to inspect available descriptor names and their uncertainty before
declaring tolerances. Default names include `density`, `coordination.PAIR`,
`total_coordination.ELEMENT`, `bond_distance.PAIR` and `bond_angle.TRIPLET`.
Pair ordering is significant: silica bond distance is `bond_distance.O-Si`,
whereas coordination has separate `coordination.Si-O` and `coordination.O-Si`.
Unknown names and nonpositive or nonfinite tolerances are rejected.

Selected optional analyses add named descriptors, such as
`bond_order.ordered_fraction`, `voids.accessible_fraction`, `sq.total`,
`sq.Si-O` with `--sq-partials`, and `tr.T_r`. A tolerance on `rdf.total` or
`rdf.PAIR` also computes that RDF for convergence. For curve descriptors the
tolerance must hold at **every point**: the plotted quantity is the largest
pointwise half-width, not a simultaneous confidence band for the entire curve.
Optional descriptors require their corresponding analysis flags.

The equivalent YAML entries live under `analysis`:

```yaml
analysis:
  convergence: true
  tolerances:
    density: 0.02
    coordination.Si-O: 0.05
  convergence_confidence: 0.95
  convergence_max_structures: 1000000
```

CLI tolerance declarations replace YAML tolerances of the same name and
preserve the others. `--convergence-max-structures` sets the upper search bound
for forecasts; it never truncates the input ensemble.

```python
from amorphgen.analysis import (
    StructureAnalyser, format_convergence_report, save_convergence_report,
)

sa = StructureAnalyser("ensemble/")
report = sa.convergence_report({"density": 0.02, "coordination.Si-O": 0.05})
print(format_convergence_report(report))
save_convergence_report(report, "analysis/", save_pdf=True)

# Include any aligned per-structure descriptor, including a whole curve.
rdf = sa.rdf(n_bootstrap=0)
curve_report = sa.convergence_report(
    {"rdf.total": 0.05}, descriptors={"rdf.total": rdf["uncertainty"]},
    sizes=[2, 5, 10, 20, 50, 100],
)
```

The curves are independent of generation order. For complete scalar data,
AmorphGen plots

$$h(n) = t_{(1+c)/2,\,n-1}\,s_N / \sqrt{n},$$

where $c$ is the chosen confidence and $s_N$ is the sample standard deviation
of all $N$ structures. For $2 \leq n \leq N$, this is the exact
root-mean-square Student-t half-width over **all** subsets of size $n$,
because their mean sample variance equals
$s_N^2$. No randomized shuffling, seed, or generation-order prefixes enter
the calculation. A vector descriptor takes the maximum of these componentwise
RMS values, not the RMS of the subset-wise maxima. At $n=N$ the curve equals
the observed full-ensemble half-width; beyond $N$ it is an extrapolation.
The interval formula follows the
[NIST Student-t confidence interval](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm).

Missing observations are excluded rather than treated as zero. If a component
appears in $k$ of $N$ structures, planning at size $n$ uses
$\lfloor nk/N\rfloor$ contributing observations with the observed sample
variance. This is an availability-adjusted approximation, not an exact
all-subset result. Any component with fewer than two observations makes that
descriptor's uncertainty and forecast unavailable; empty bins are retained.

The report returns `met`, `not_met`, `insufficient_data`, or `undeclared`
per descriptor and overall. Undeclared descriptors do not decide the overall
status. It estimates the smallest total ensemble size satisfying all declared
tolerances under unchanged variance and availability, and subtracts the
current size to give the additional structures needed. A met tolerance needs
zero additional structures; an estimate beyond the search bound is reported
as `exceeds_max_structures`, with no invented finite forecast.

Forecasts assume independent structures and representative, stable variance
and missingness. They do not account for correlated trajectory frames,
force-field bias, finite-size error or undiscovered rare configurations.
Zero observed variance yields a zero estimated half-width once two values
exist, but does not establish zero population variance. Reassess the report
as new independent structures arrive.

With `--save-plot`, exports include `analysis_convergence.json` (full report,
strict JSON with missing values as `null`), `analysis_convergence.txt`,
`analysis_convergence_summary.csv`, and `analysis_convergence_curves.csv`
(one row per descriptor, planned size and point). Each descriptor has its
own figure, for example `analysis_convergence_density.png`, with an optional
PDF. Figures distinguish the observed endpoint, declared tolerance, solid
planning curve and dashed projection through the estimated target. CSV
columns retain counts, confidence, units in the summary, forecast status,
and pointwise uncertainty so the decisions can be reproduced.

## Recipes

Common cases:

::::::{tab-set}

:::::{tab-item} Quick start

For an ensemble with at least two structures and valid angle triplets,
this writes four standalone figures:

```bash
amorphgen --analyse \
    --input-dir hybrid_ga2o3/final/ \
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

The CSV files contain RDF values, coordination percentages, raw angle
observations and per-structure densities so you can re-plot in any tool.

:::::

:::::{tab-item} Multi-cation oxide (a-IGZO)

Four elements, ten element pairs, three cation sizes and a shared oxygen.
One command reports and plots all of it:

```bash
amorphgen --analyse \
    --input-dir igzo_final/ \
    --sq --sq-partials --pair-panels --save-pdf \
    --total-cn O --total-cn "O:In+Ga" \
    --save-report report.txt \
    --save-plot plots/
```

The report header shows the cutoff in force for every pair. With the
default ``auto-rdf`` each pair gets its own value from the first minimum
of its g(r):

```
  Cutoff mode: auto (RDF)
    Ga-O: 2.03 A
    In-O: 2.47 A
    O-Zn: 2.25 A
    ...
```

One cutoff for all pairs is the thing to avoid here: 2.47 Å (the In–O
value) would count second-shell oxygens around Ga and lift the Ga–O
coordination from 3.9 to 4.2. To override one pair and keep ``auto-rdf``
for the rest, use ``--cutoff "In-O=2.6"``.

The coordination part of the report separates bonds from contacts:

```
  Bonding coordination numbers:
  Ga-O: mean=3.9 +/- 0.3 [3,4]
  In-O: mean=5.1 +/- 0.6 [4,6]
  Zn-O: mean=3.9 +/- 0.5 [3,5]
  O-Ga: ...  O-In: ...  O-Zn: ...

  Total coordination (all bonded partners):
  O-(Ga+In+Zn): mean=3.2 +/- 0.5 [2,4]

  Total coordination (requested):
  O-(all bonded): mean=3.2 +/- 0.5 [2,4]
  O-(In+Ga): mean=2.3 +/- 0.8 [0,4]

  Non-bonded contacts:
  Ga-In: ...  In-In: ...  O-O: ...
```

Cation–O pairs are bonds; cation–cation and O–O pairs are second-shell
contacts listed apart and excluded from the coordination and the angles.
The ``O-(Ga+In+Zn)`` line appears on its own whenever an element has more
than one bonded partner type; ``--total-cn`` adds any centre and partner
set you name (``O`` for all bonded partners, ``O:In+Ga`` for a subset).

``--sq-partials`` prints the first peak of each Faber-Ziman partial
S_ab(q) and writes the partials next to the total; ``--pair-panels`` puts
every pair in its own panel for g(r) and for S_ab(q).

Output:

```
plots/
├── analysis_rdf.{png,pdf,csv}              # partials; CSV also has g(r)_Total
├── analysis_rdf_panels.{png,pdf}           # one panel per pair
├── analysis_cn.{png,pdf,csv}               # Ga-O, In-O, Zn-O and O-(Ga+In+Zn)
├── analysis_cn_total.{png,pdf,csv}         # the --total-cn requests
├── analysis_sq.{png,pdf,csv}               # total S(q); CSV has s_<pair> columns
├── analysis_sq_partials.{png,pdf}          # partials on one axis
├── analysis_sq_partials_panels.{png,pdf}   # one panel per pair
├── analysis_angles.{png,pdf,csv}
└── analysis_density.{png,pdf,csv}
```

Python equivalent:

```python
from amorphgen.analysis import StructureAnalyser

sa = StructureAnalyser("igzo_final/", cutoff="In-O=2.6")       # rest auto-rdf
sa.summary()
sa.total_coordination(centre="O", partners=["In", "Ga"])
sq = sa.structure_factor_direct(weighting="xray", sigma_q=0.05, partials=True)
sq["partials"]["In-O"]
sa.plot(output_dir="plots/", save_pdf=True,
        pair_panels=True, total_cn=["O", "O:In+Ga"])
from amorphgen.analysis.plotting import plot_sq
plot_sq(sq, output_dir="plots/", save_pdf=True, pair_panels=True)
```

:::::

:::::{tab-item} Single ensemble + text report

Add a full text report alongside the plots:

```bash
amorphgen --analyse \
    --input-dir hybrid_ga2o3/final/ \
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
    --input-dir hybrid_ga2o3/final/ \
    --per-structure
```

If the structure files don't carry energies (VASP, CIF), AmorphGen looks
for ``random_gen.log`` alongside the files and in their parent directory
to fill in the E/atom column. Keep the original ``random_NNNN`` filenames
so the log entries can be matched to the correct structures.

:::::

:::::{tab-item} Validate against a reference YAML

Compare your ensemble against literature ranges with automatic
match/concern/fail scoring:

```bash
amorphgen --analyse \
    --input-dir hybrid_ga2o3/final/ \
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

AmorphGen ships reference YAMLs for a-Ga₂O₃, a-SiO₂, a-GeO₂, a-HfO₂ and
a-IrO₂ in ``examples/`` (``reference_a_<system>.yaml``). The GeO₂ file
notes that the neutron partial structure factors of Salmon et al. (2005)
allow a partial-by-partial comparison with ``--sq --sq-weighting neutron``.
Write your own for other systems by following the same schema.

:::::

:::::{tab-item} Compare multiple ensembles

For overlaying Random vs Hybrid vs DFT-reference (or any combination),
use the Python API. There's no single CLI flag for this yet;
``compare_ensembles()`` is the entry point:

```python
from amorphgen.analysis import EnsembleSpec, compare_ensembles

compare_ensembles(
    ensembles=[
        EnsembleSpec("DFT-PBE0", "prb_ensemble/*.cif"),
        EnsembleSpec("Random",   "random_inputs/*.vasp"),
        EnsembleSpec("Hybrid",   "hybrid_run/final/*.xyz"),
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
`comparison/ga2o3_density.{...}`, same layout as `--analyse
--save-plot`, but each figure overlays all listed ensembles with
distinct colours from the Okabe-Ito palette.

See {doc}`/validation/index` for a Ga₂O₃ comparison and the available
reference data.

:::::

::::::

## Which contacts count as bonds?

The bonding coordination report, total coordination, bond angles and CN
plot share the same bonding rule. In compounds with anions, only
cation–anion pairs count as bonds; charge balance determines the anions.
Without anions, the radii classification decides: same-element pairs
count in a single-element system or a metal-rich alloy (at least 70 %
metal atoms), and unlike-element pairs count when classified as ionic,
covalent or metallic.

Hydrogenated group-IV networks contain only C, Si and/or Ge plus H, with
at most one H per host atom. Their host keeps the bonding rules of the
H-free composition: Si–Si in a-Si:H, Ge–Ge in a-Ge:H and C–C in a-C:H
count as bonds. In a-SiC:H only Si–C host pairs count, and in a-SiGe:H
only Si–Ge host pairs count. Every host–H pair counts as a bond; H–H
never does. These pairs still have to lie within their distance cutoffs.
Metal hydrides, hydroxides, compositions with other elements and those
with more H than host atoms retain the usual rules.

## Cutoff

The cutoff defines what counts as a "first-shell" bond and affects
coordination, bond-length statistics, and bond-angle triplets. The default
is `auto-rdf`, which finds the first minimum of each partial
RDF, the standard convention in neutron-diffraction analysis of
glasses. The minimum is read at a resolution of 0.25 Å, so a flat step
or a noise dip on the falling side of the first peak, common in small
cells, does not end the shell. Where g(r) is zero over a range, the
cutoff goes in the middle of it.

| Cutoff mode | When to use |
|---|---|
| **`auto-rdf`** (default) | All structural analysis. Robust across systems with broad bond-length distributions (a-Si, a-HfO₂, chalcogenides). |
| `auto` | Legacy. Uses minsep from Shannon/Cordero/Goldschmidt radii. Fast but can under-count coordination for systems with long first-shell bonds. |
| Numeric, e.g. `--cutoff 2.5` | Single cutoff (in Å) for all pairs. Useful for tight-bonded covalent networks. |
| Per-pair overrides, e.g. `--cutoff "In-O=2.6,Zn-O=2.3"` | Fix the pairs you name and keep `auto-rdf` for the rest. Prefix a base rule to change the rest: `"auto,In-O=2.6"` or `"2.4,In-O=2.6"`. |
| Dict via YAML or API | `cutoff: {In-O: 2.6}` completes the unlisted pairs from `auto-rdf`; add `default: 2.4` (or a mode name) to change that. |

Inspect the RDF and the reported cutoffs before interpreting coordination.
An explicit cutoff can help when a first minimum is ambiguous. One number for every
pair is the option to avoid in a multi-cation oxide: for a-IGZO the In–O
first minimum sits at 2.47 Å and Ga–O at 2.03 Å, and a single 2.47 Å
cutoff raises the Ga–O coordination from 3.9 to 4.2 by admitting
second-shell oxygens. The report header lists the cutoff in force for
every pair, so a per-pair override is easy to check. These numerical values are an
example, not fixed cutoffs for every IGZO ensemble.

For elements bonded to more than one partner type (O in IGZO, bonded to
Ga, In and Zn) the report adds a `Total coordination` block with the
first-shell count over all bonded partners, next to the per-pair O–Ga,
O–In and O–Zn entries. The label reads centre first, then the partners in
brackets, the same order as the per-pair entries (`O-Ga` is O with Ga
around it). To choose the centre and the partners yourself, use
`--total-cn` (repeatable) or the YAML list `total_cn:`:

```bash
amorphgen --analyse --input-dir DIR --total-cn O --total-cn "O:In+Ga"
```

`O` counts every bonded partner; `O:In+Ga` counts only the named ones.
From Python, `sa.total_coordination(centre="O", partners=["In", "Ga"])`
returns the same statistics.

## Structure factor S(q) and simulated XRD

AmorphGen provides two structure-factor methods: a direct sum over the
reciprocal-lattice vectors of a periodic cell, and a Fourier transform of
the radial distribution function. The CLI uses the direct method by
default. Both support X-ray, neutron and unweighted totals.

The direct method avoids truncating g(r), but still has finite-cell and
sampling limits. The FT method can show termination ripples and altered
peak heights; there is no universal correction factor between the two.
See {doc}`/notes/sq_xrd_methodology` for the conventions and limitations.

::::{tab-set}

:::{tab-item} Structure factor S(q)
:sync: directmethod

From the CLI:

```bash
amorphgen --analyse --input-dir DIR --sq --sq-weighting neutron --save-plot plots/
```

This writes `analysis_sq.png` and `analysis_sq.csv`. With the default
smoothing, the CSV contains `q_invA`, `s_q`, `s_q_raw` and `n_per_bin`.
Add `--sq-partials` to print the first peak of each Faber–Ziman partial,
append `s_<A-B>` columns to the CSV and write `analysis_sq_partials.png`.
The partials describe the geometry and do not depend on the weighting.

From Python, for an ensemble of GeO₂ structures:

```python
from amorphgen.analysis import StructureAnalyser

sa = StructureAnalyser("geo2_ensemble/")
sq = sa.structure_factor_direct(weighting="neutron", sigma_q=0.05,
                                partials=True)
sq["partials"]["Ge-O"]
```

For each nonzero reciprocal-lattice vector, the implementation returns
Faber–Ziman-normalized scattering:

$$S(\vec q) = 1 + \frac{I(\vec q)/N - \langle f^2(q)\rangle}
{\langle f(q)\rangle^2}, \qquad
I(\vec q) = \left|\sum_i f_i(q)e^{i\vec q\cdot\vec r_i}\right|^2.$$

Here $c_\alpha=N_\alpha/N$,
$\langle f\rangle=\sum_\alpha c_\alpha f_\alpha$ and
$\langle f^2\rangle=\sum_\alpha c_\alpha f_\alpha^2$.
Subtracting the self-scattering term gives the high-q limit $S(q)\to1$
for uncorrelated positions. The raw intensity divided by
$N\langle f\rangle^2$ has a different limit for mixtures.

With cell vectors as rows of $\mathbf A$, the sampled vectors are

$$\vec G = 2\pi\,\vec n\mathbf A^{-\top}, \qquad
\vec n\in\mathbb Z^3, \quad 0<|\vec G|\le q_{\max}.$$

Values are averaged into shells by $|\vec G|$, pooling vectors across
structures with the same composition. `n_per_bin` records their count;
empty shells contain `NaN`. For a cubic cell of side $L$, the smallest
nonzero q is $2\pi/L$. Increasing `nq` adds bins but cannot add independent
low-q information.

`sigma_q` (`--sq-smooth`) applies Gaussian smoothing weighted by each
shell's vector count. Smoothing can broaden features, so compare with
`s_q_raw`. The CLI defaults to 0.05 Å⁻¹; the Python method
defaults to 0 (no smoothing). Empty shells remain `NaN`.

With `partials=True`, partials are averaged over the same shells. Their
weighted sum reproduces the total exactly for q-independent weights
(such as neutron scattering lengths). For X-rays the weights vary
within a shell, so recombining the binned partials using weights at bin
centres is approximate.

Use `--sq-method ft` or `sa.structure_factor(weighting="xray")` for the
FT method. Its Python default weighting is `"unweighted"`; the direct
method and CLI default to `"xray"`.

:::

:::{tab-item} Simulated XRD I(2θ)
:sync: xrdrecipe

The normalized structure factor is not a raw diffractometer intensity.
For an X-ray coherent-scattering profile, first restore the self-scattering
term and form-factor scale, then map momentum transfer to angle:

$$I_{\mathrm{coh}}(q)/N = \langle f(q)\rangle^2[S(q)-1]
+\langle f^2(q)\rangle, \qquad q=\frac{4\pi\sin\theta}{\lambda}.$$

The following post-processing example uses Cu-Kα wavelength and the
composition of the first structure. It approximates the form factors
at each bin centre; use narrow q bins when comparing intensities.

```python
import numpy as np
from amorphgen.analysis import StructureAnalyser
from amorphgen.analysis.rdf import xray_form_factor

sa = StructureAnalyser("ga2o3_ensemble/")
sq = sa.structure_factor_direct(weighting="xray", qmax=8.0, nq=400)
q, s, counts = (np.asarray(sq[k]) for k in ("q", "s_q", "n_per_bin"))
symbols = sa.atoms_list[0].get_chemical_symbols()
fractions = {el: symbols.count(el) / len(symbols) for el in set(symbols)}
f = {el: xray_form_factor(el, q) for el in fractions}
f_mean = sum(fractions[el] * f[el] for el in fractions)
f2_mean = sum(fractions[el] * f[el]**2 for el in fractions)
coherent = f_mean**2 * (s - 1.0) + f2_mean

wavelength = 1.5406  # Å, Cu-Kα
keep = (counts >= 5) & np.isfinite(coherent)
if np.count_nonzero(keep) < 2:
    raise ValueError("Too few populated q bins; use more structures or wider bins")
two_theta = np.linspace(20.0, 90.0, 1000)
q_at_angle = 4.0 * np.pi * np.sin(np.deg2rad(two_theta / 2)) / wavelength
intensity = np.interp(q_at_angle, q[keep], coherent[keep],
                      left=np.nan, right=np.nan)
```

This gives a coherent-scattering profile before instrument and sample
corrections. Polarization, geometry, absorption, background and resolution
must match the measurement. A powder-diffraction Lorentz–polarization
factor applied to `S(q)` alone is not a general prediction of an amorphous
sample's measured trace. Any additional broadening should come from the
instrument's resolution; the structural halo width is already present
in the calculated profile.

AmorphGen currently has no `xrd_pattern()` convenience method; this is
Python post-processing. For a comparison to an experimentally reduced
Faber–Ziman `S(Q)`, compare directly to the calculated `S(q)` using matching
weights and normalization.

:::

:::{tab-item} Weighting
:sync: weighting

Both methods combine partial structure factors using

$$S(q)=\frac{\sum_{\alpha,\beta}c_\alpha c_\beta f_\alpha(q)
f_\beta(q)S_{\alpha\beta}(q)}
{\left[\sum_\alpha c_\alpha f_\alpha(q)\right]^2}.$$

The sum counts ordered pairs, so unlike-element pairs occur twice.
Different weights change how the partial peaks and troughs contribute;
they do not guarantee a particular peak height.

| `weighting` | Per-element factor | Use |
|---|---|---|
| `"xray"` | q-dependent neutral-atom form factor | X-ray total scattering |
| `"neutron"` | Tabulated coherent scattering length | Neutron total scattering |
| `"unweighted"` | 1 for every species | Geometric structure comparisons |

X-ray form factors use the five-Gaussian parametrization of
[Waasmaier & Kirfel (1995)](https://doi.org/10.1107/S0108767394013292):

$$f_\alpha(q)=\sum_{i=1}^5 a_i\exp[-b_i(q/4\pi)^2]+c.$$

The table covers neutral atoms H–Cf. Its fitted values approach the
atomic number at q = 0; the code uses the q-dependent values throughout.
Anomalous scattering corrections are not included.

Neutron weights use the built-in common-element table from
[Sears (1992)](https://doi.org/10.1080/10448639208218770).
Unsupported species raise an error. Scattering lengths can be negative
and depend on isotope; the API chooses weights from element symbols,
so changing an atom's mass does not select an isotope-specific length.

:::

:::{tab-item} Physical validity
:sync: physics

Use the same normalization, scattering weights, q range and smoothing
when comparing calculations and experiments. A summary of the relevant
conventions is [Keen (2001)](https://doi.org/10.1107/S0021889800019993).
The direct reciprocal-space calculation and the FT of g(r) are described
in {doc}`/notes/sq_xrd_methodology`.

The repository's `TestSqNormalisation` tests in `test/test_analysis.py`
check the high-q limit, the FT integrand, scattering-table entries,
q-dependent X-ray weighting, smoothing and neutron partial recombination.
These are implementation checks; they do not establish universal accuracy
bounds or validate every material against experiment.

| Limitation | How to assess it |
|---|---|
| Finite cell and sparse low-q shells | Inspect `n_per_bin`; compare larger cells and more independent configurations. |
| Finite-r truncation in the FT method | Vary `rmax` within the cell's useful range and compare with the direct method. |
| Smoothing and finite q-bin width | Compare raw data and narrower bins; report `sigma_q` and `nq`. |
| Neutral-atom X-ray weights; element-based neutron weights | Check whether anomalous or isotope-specific scattering matters for the experiment. |
| Thermal motion | Use representative configurations at the comparison temperature; an extra Debye–Waller correction may double-count sampled motion. |
| Instrument and sample effects | Apply corrections appropriate to the measured quantity and geometry. |

Cite AmorphGen for the software and the primary literature for the
scattering conventions and tables; see {doc}`/notes/sq_xrd_credits`.

:::

::::

### Example: a-Ga₂O₃

For a directory of Ga₂O₃ configurations with the same composition:

```python
from amorphgen.analysis import StructureAnalyser

sa = StructureAnalyser("ga2o3_ensemble/", cutoff="auto-rdf")
sq = sa.structure_factor_direct(weighting="xray", qmax=12.0, nq=300,
                                sigma_q=0.05, partials=True)
# sq["s_q_raw"] retains the unsmoothed total for comparison.
```

Peak positions and intensities depend on the supplied structures, cell
size and settings. The same API applies to other compositions; select
weights and a q range appropriate to the reference data.

### Open issues / future work

- An `xrd_pattern()` convenience method with explicit intensity conventions
  and instrument settings.

## Crystal-like order and the largest ordered cluster

Use `--bond-order` to look for residual or newly formed crystal-like regions
in a quenched ensemble, including phase-change systems such as GeTe:

```bash
amorphgen --analyse --input-dir gete_mq/final/ --bond-order \
    --order-cutoff 3.5 \
    --qbar6-threshold 0.3 --order-min-neighbors 4 \
    --save-report gete_report.txt --save-plot gete_plots/
```

This geometry-only descriptor reports each atom's Steinhardt $q_6$ and
Lechner–Dellago $\bar q_6$, the fraction of atoms classified as ordered, and
the largest connected ordered cluster in each structure. For atom $i$ with
neighbour shell $N(i)$, the complex spherical-harmonic coefficients are

$$q_{6m}(i)=\frac{1}{|N(i)|}\sum_{j\in N(i)}Y_{6m}(\hat{\mathbf r}_{ij}),
\qquad q_6(i)=\sqrt{\frac{4\pi}{13}\sum_{m=-6}^{6}|q_{6m}(i)|^2}.$$

Lechner–Dellago averaging includes the central atom and its neighbours,
**before** taking the rotationally invariant norm:

$$\bar q_{6m}(i)=\frac{q_{6m}(i)+\sum_{j\in N(i)}q_{6m}(j)}{|N(i)|+1},
\qquad \bar q_6(i)=\sqrt{\frac{4\pi}{13}\sum_{m=-6}^{6}|\bar q_{6m}(i)|^2}.$$

These follow [Steinhardt, Nelson and Ronchetti (1983)](https://doi.org/10.1103/PhysRevB.28.784)
and [Lechner and Dellago (2008)](https://doi.org/10.1063/1.2977970).
Averaging the scalar $q_6$ values would give a different descriptor.

The neighbour shell uses all element pairs within `--order-cutoff`, falling
back to the analyser's `--cutoff` when no order-specific cutoff is supplied.
It does not apply the chemical bonding filter used for coordination and angles.
Periodic images contribute their actual bond directions; cluster sizes count
unique atoms in the supplied cell. Two ordered atoms belong to the same
cluster when connected by a path of cutoff neighbours that are all ordered,
including connections across periodic boundaries. This is a connectivity
measure; it does not additionally require aligned $q_{6m}$ vectors.

An atom is ordered when `qbar6 >= qbar6_threshold` and it has at least
`order_min_neighbors` neighbours. The defaults, 0.3 and 4, are **heuristic**.
They do not identify a crystal phase or give a universal crystalline volume
fraction. Calibrate the cutoff and threshold using crystalline and liquid
references at relevant temperatures, especially for GeTe. Separate partial-RDF
cutoffs can include different geometric shells for different element pairs;
inspect the resolved cutoffs and use an explicit scalar or pair cutoff when
needed. Compare distributions as well as ordered fractions and cluster sizes.

The example's 3.5 Å order cutoff illustrates the first shell of an **ideal
rocksalt GeTe cell with lattice constant 6 Å**. Its six neighbours lie at
3 Å and give $q_6=\bar q_6=\sqrt{1/8}\approx0.35355$; all atoms therefore
pass the 0.3 threshold. With pairwise `auto-rdf` cutoffs, the same ideal cell
can include twelve same-element neighbours as well, giving
$\bar q_6\approx0.26517$ and no ordered atoms at that threshold. The explicit
`--order-cutoff` prevents changing the shell used for the other chemical
analyses. This ideal-cell example is not a calibration for thermally distorted
or rhombohedral GeTe; inspect short and long bonds and reference distributions
before choosing the shell for a phase-change workflow.

To reproduce the ideal-cell check without a calculator:

```python
from ase.build import bulk
from amorphgen.analysis import compute_bond_order

ideal = bulk("GeTe", "rocksalt", a=6.0, cubic=True).repeat((2, 2, 2))
frame = compute_bond_order([ideal], cutoff=3.5)["per_structure"][0]
print(frame["qbar6_mean"])           # approximately 0.353553
print(frame["largest_cluster_size"]) # 64: the whole cell
```

The equivalent YAML settings are:

```yaml
analysis:
  bond_order: true
  qbar6_threshold: 0.3
  order_min_neighbors: 4
  order_cutoff: 3.5       # illustrative ideal-rocksalt shell; calibrate for your system
  cutoff: auto-rdf
```

`--save-report` includes the order summary; `--save-plot` adds
`analysis_bond_order.json`, `analysis_bond_order.csv`,
`analysis_bond_order_atoms.csv` and
`analysis_bond_order.png` (`.pdf` with `--save-pdf`). JSON retains the
per-atom values, labels and resolved cutoffs for reproducible comparisons.

```python
from amorphgen.analysis import StructureAnalyser
from amorphgen.analysis.descriptors import save_descriptor

sa = StructureAnalyser("gete_mq/final/", cutoff="auto-rdf")
order = sa.bond_order(qbar6_threshold=0.3, min_neighbors=4, cutoff=3.5)
frame = order["per_structure"][0]
print(frame["ordered_fraction"], frame["largest_cluster_size"])
save_descriptor("bond_order", order, "gete_plots/")
```

`ordered_fraction` and `largest_cluster_fraction` use all atoms in each
structure as the denominator. The top-level result summarizes structures
with equal weight; inspect `per_structure` for individual clusters and
per-atom arrays. An isolated atom has $q_6=\bar q_6=0$ and is disordered.

For a crystal-started `--mq-ensemble` run, a separate automatic
`melt_memory` report measures how much of the initially ordered atom
population is also ordered after heating and in the high-temperature
snapshots. See {doc}`mq-ensemble` for the definition and its limitations.

## Void, oxygen, elastic and vibrational descriptors

These four descriptors are opt-in. Void sampling and oxygen speciation use
geometry alone and do not load an ML model. Elastic and vibrational analysis
load the selected calculator (`--model`, `--model-path`, `--device`) and
evaluate new configurations; saved single-point stresses or forces are
insufficient.

### Void distribution

```bash
amorphgen --analyse --input-dir silica/ --voids \
    --void-samples 20000 --void-probe-radius 0.5 --void-seed 42 \
    --save-plot descriptors/
```

Uniform random points sample **point clearance**: the distance to the nearest
atomic-sphere surface, in Å. A point is accessible when its clearance is at
least the probe radius. This measures local free space; it does not find
connected pores, pore throats or maximal cavities. The reported radius is
clearance, not diameter, and sampled maxima underestimate the true maximum.

Atomic spheres use ASE covalent radii by default. Choose a consistent radius
convention for comparisons; override individual elements with YAML
`analysis.void_radii` or the Python `radii` argument. The histogram density
integrates to one over accessible points, while `bin_volume_fraction` sums
to the accessible fraction. Ensemble fractions are weighted by cell volume;
`accessible_volume` is the mean accessible volume per structure. Standard
errors describe Monte Carlo sampling only. Per-structure 95% Wilson intervals
also cover cases where no accessible points were found; neither measure
captures variation between structures. All cells must be fully periodic in 3D.

### Bridging and non-bridging oxygen

```bash
amorphgen --analyse --input-dir aluminosilicate/ --oxygen-speciation \
    --network-formers Si,Al --cutoff "Si-O=2.0,Al-O=2.3" \
    --save-plot descriptors/
```

Each oxygen is classified by its number of neighbouring selected network
formers: zero = `free`, one = `non_bridging`, two = `bridging`, three =
`tricluster`, and four or more = `higher_coordinated`. Fractions pool oxygen
counts across structures. The selection defaults to the Al, B, Ge, P and Si
present. Select the appropriate formers explicitly for other oxides and
exclude modifiers such as Na or Ca. All ensemble structures must have the
same element set; analyse different chemistries separately.

This uses the analyser's pair cutoffs and periodic neighbours. Check those
cutoffs before interpreting the counts. `free` means no selected former
neighbour; the descriptor does not infer charge, bond order or hydroxyl
speciation. The former/modifier distinction follows the connectivity
convention described by [Stebbins and Xu](https://www.nature.com/articles/36312).

### Elastic moduli from calculator stresses

```bash
amorphgen --analyse --input-dir relaxed_silica/ --elastic \
    --model mace-mpa-0 --device cpu --elastic-strain 0.005 \
    --save-plot descriptors/ --save-report descriptors.txt
```

Central differences require 13 stress evaluations per structure. The default
keeps fractional atomic coordinates fixed under strain (clamped ions).
`--elastic-relax` instead optimises internal positions at each fixed cell,
including the reference; `--fmax` and `--opt-steps` control convergence.
Optimise the starting cell separately for equilibrium moduli. Input
structures remain unchanged and failed internal relaxation raises an error.

The symmetrized stiffness tensor uses engineering strain in ASE Voigt order
`xx, yy, zz, yz, xz, xy`; stiffness and moduli are in GPa. The report includes
Voigt, Reuss and Hill bulk, shear and Young's moduli, dimensionless Poisson
ratios, residual stress and stability diagnostics. Reuss/Hill estimates are
unavailable for unstable or ill-conditioned tensors. Residual stress above
0.1 GPa is flagged: these are static tangent stress-strain coefficients with
no finite-pressure correction. Check convergence with strain amplitude and
calculator precision. The averaging equations follow the
[NIST atomman reference](https://www.ctcms.nist.gov/potentials/atomman/tutorial/3.1._ElasticConstants_class.html).

### Harmonic vibrational density of states

```bash
amorphgen --analyse --input-dir relaxed_silica/ --vdos \
    --model mace-mpa-0 --device cpu --vdos-displacement 0.01 \
    --vdos-sigma 0.1 --vdos-npoints 800 --save-plot descriptors/
```

The mass-weighted force-constant matrix is built using central finite
differences, as in [ASE's harmonic vibration formulation](https://docs.ase-lib.org/_modules/ase/vibrations/data.html).
This requires **6N force evaluations** and dense diagonalisation of a
`3N × 3N` matrix per structure. Begin with small cells to assess cost.
The spectrum contains all 3N modes of each supplied cell; for periodic
structures these are Gamma-point modes, without Brillouin-zone sampling.

Optimise the reference positions beforehand. No automatic optimisation or
acoustic sum rule is applied, and atomic constraints are rejected. Negative
plotted frequencies denote imaginary modes. The Gaussian width is in THz;
the displacement is in Å. The total DOS integrates to one on its returned
grid, with equal weight per mode across structures. Element projections use
squared mass-weighted eigenvector components and sum to the total DOS;
they are not scattering-weighted experimental intensities. Check displacement,
broadening and grid convergence before interpreting fine features.

With `--save-plot`, each requested descriptor writes full JSON (including
per-structure results), a CSV and a PNG; add `--save-pdf` for PDF figures.
`--save-report` appends the compact text summaries. Python methods return
results without automatically writing files:

```python
from amorphgen.analysis import StructureAnalyser
from amorphgen.analysis.descriptors import save_descriptor

sa = StructureAnalyser("silica/", cutoff={"Si-O": 2.0})
voids = sa.void_distribution(n_samples=20000, probe_radius=0.5, seed=42)
oxygen = sa.oxygen_speciation(network_formers=["Si"])
save_descriptor("voids", voids, "descriptors/", save_pdf=True)

# Optional model-backed descriptors; install the matching backend extra.
from amorphgen.utils import get_calculator

calc = get_calculator(model="mace-mpa-0", device="cpu")
elastic = sa.elastic_moduli(calculator=calc, strain=0.005, relax=False)
vdos = sa.vibrational_dos(calculator=calc, displacement=0.01,
                         sigma=0.1, npoints=800)
print(elastic["ensemble"]["moduli"]["hill"])
print(vdos["imaginary_modes"])
```

Both calculator-backed methods also accept live calculators attached to
the input ASE objects when `calculator` is omitted.

## Full CLI flag reference

The brackets below denote optional arguments; omit the brackets when running
the command.

```text
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
    [--sq] [--sq-weighting {xray,neutron,unweighted}] \
    [--sq-method {direct,ft}] [--sq-smooth SIGMA_Q] [--sq-partials] \
    [--pair-panels] [--total-cn SPEC] \
    [--tr] [--tr-qrange QMIN QMAX] [--tr-window {lorch,none}] [--tr-scan] \
    [--check-dimers] [--rings [PAIR]] [--voronoi [ELEMENT]] [--connectivity] \
    [--bond-order] [--order-cutoff MODE_OR_NUMBER] \
    [--qbar6-threshold FLOAT] [--order-min-neighbors INT] \
    [--dpi N] \
    [--show-title]
```

| Flag | What it does |
|---|---|
| `--input-dir DIR` | Read structure files in this directory. Same-stem duplicates count once, preferring ``.xyz``, then ``.extxyz``, ``.vasp`` and ``.cif``. |
| `--cutoff MODE` | `auto-rdf` (default: first minimum of each partial g(r)), `auto` (radii table), a number in Å, or per-pair overrides such as `"In-O=2.6,Zn-O=2.3"` that keep `auto-rdf` for the other pairs (`"auto,In-O=2.6"` or `"2.4,In-O=2.6"` change the base). |
| `--per-structure` | Print a per-structure table (one row per file: density, E/atom, CN). |
| `--save-report FILE` | Write the full text report (densities, bond distances, coordination, angles) to a file. |
| `--save-plot DIR` | Save available standard figures (RDF, CN, angles, density) plus CSV data into ``DIR``. |
| `--save-pdf` | Also save vector PDF copies alongside the PNGs. |
| `--convergence` | Report available descriptor names, uncertainty versus ensemble size, and declared tolerance status. |
| `--tolerance NAME=VALUE` | Declare an absolute Student-t mean interval half-width in descriptor units; repeat for each descriptor. Enables convergence reporting. |
| `--convergence-confidence FLOAT` | Confidence for convergence intervals and forecasts (default 0.95). |
| `--convergence-max-structures INT` | Largest total ensemble size searched for the forecast (default 1000000). |
| `--reference YAML` | Validate against the literature ranges in YAML, print a match/concern/fail table. |
| `--smearing SIGMA` | Gaussian smearing of the RDF in Å (default 0.05, roughly thermal broadening; 0 for the raw histogram). |
| `--total-rdf` | Overlay the total g(r) on the partial-RDF plot. |
| `--sq` | Compute the total structure factor S(q) and save it as PNG + CSV under ``--save-plot``. |
| `--sq-weighting` | `xray` (default, Waasmaier–Kirfel form factors), `neutron` (Sears scattering lengths) or `unweighted`. |
| `--sq-method` | `direct` (default): reciprocal-lattice sum. `ft`: Fourier transform of g(r), with finite-r truncation effects. |
| `--sq-smooth SIGMA_Q` | Gaussian re-binning width in Å⁻¹ for the direct S(q) (default 0.05; 0 = raw). Raw values are kept in the CSV. |
| `--sq-partials` | With `--sq` (direct method): the Faber-Ziman partial structure factors S_ab(q) of every element pair. First peaks printed, `s_<pair>` columns in `analysis_sq.csv`, `analysis_sq_partials.png`. |
| `--tr` | Total correlation function T(r) = 4πrρg(r), obtained by transforming the direct S(q). Match the reference's scattering weights, normalization, q range and window before comparing. Writes `analysis_tr.png` and a CSV with r, the weighted g(r), T(r) and the reduced PDF G(r). |
| `--tr-qrange QMIN QMAX` | Integration limits for `--tr` (default 0.3 20). Set them to the experiment's own range: qmax fixes the real-space resolution and the truncation ripple. |
| `--tr-window {lorch,none}` | Window for the `--tr` transform. `lorch` damps the qmax truncation ripple at the cost of broader peaks; match whichever the paper used. |
| `--tr-scan` | With `--tr`, sweep qmax and the window and report changes in the first T(r) peak and its integrated scattering-weighted count. This measures sensitivity to transform settings, not a statistical error bar or a species-resolved coordination number. |
| `--pair-panels` | One small panel per element pair for the partial g(r) (`analysis_rdf_panels.png`) and, with `--sq-partials`, for S_ab(q) (`analysis_sq_partials_panels.png`). |
| `--total-cn SPEC` | Total first-shell coordination of one element over several partner types, repeatable: `O` counts every bonded partner, `O:In+Ga` only the named ones. Printed, and plotted as `analysis_cn_total.png` + CSV. |
| `--check-dimers` | Report unphysical close contacts (O–O peroxide, N–N) per structure. |
| `--rings [PAIR]` | Ring statistics (shortest ring per network edge). Nodes default to the least electronegative element; `--rings Ge-O` sets nodes–bridge explicitly. Added to the report; `analysis_rings.{csv,png}` under ``--save-plot``. |
| `--voronoi [ELEMENT]` | Voronoi indices <n3 n4 n5 n6> for all atoms or one element. Added to the report; `analysis_voronoi.csv` under ``--save-plot``. |
| `--connectivity` | Corner/edge/face sharing between cation-centred polyhedra (two cations sharing one anion = corner, two = edge, three or more = face) and the percentage of cations in at least one edge- or face-sharing pair, which is near zero in a corner-sharing network glass and tens of percent in a random packing. Added to the report; `analysis_connectivity.csv` under ``--save-plot``. |
| `--bond-order` | Steinhardt $q_6$, Lechner–Dellago $\bar q_6$, ordered atom fraction and largest connected ordered cluster. |
| `--order-cutoff MODE` | Neighbour cutoff for bond order and MQ melt-memory reports; accepts the same scalar, pair and automatic forms as `--cutoff`. Defaults to `--cutoff` and otherwise overrides it only for bond order. |
| `--qbar6-threshold FLOAT` | Minimum $\bar q_6$ for classifying an atom as ordered (default 0.3; calibrate for the material and shell). Also applies to MQ melt-memory reports. |
| `--order-min-neighbors INT` | Minimum neighbour count for an ordered atom (default 4). Also applies to MQ melt-memory reports. |
| `--voids` | Periodic point-clearance distribution and accessible volume. |
| `--void-samples INT`, `--void-bins INT` | Monte Carlo points per cell (default 10000) and histogram bins (50). |
| `--void-probe-radius FLOAT`, `--void-seed INT` | Probe radius in Å (default 0) and sampling seed (0). |
| `--oxygen-speciation`, `--network-formers Si,Al` | Oxygen connectivity classes; formers default to the Al/B/Ge/P/Si present. |
| `--elastic`, `--elastic-strain FLOAT` | Stress-derived tensor and isotropic moduli; strain amplitude defaults to 0.005. |
| `--elastic-relax` | Relax internal positions at each fixed cell, using `--fmax` and `--opt-steps`. |
| `--vdos`, `--vdos-displacement FLOAT` | Harmonic cell modes; displacement defaults to 0.01 Å. |
| `--vdos-sigma FLOAT`, `--vdos-npoints INT` | Gaussian width in THz (default 0.1) and frequency-grid points (400). |
| `--dpi N` | PNG DPI (default 300). |
| `--show-title` | Add titles to each plot (default off, captions usually clearer in figures). |

## Outputs explained

For each ensemble, ``--analyse --save-plot DIR`` writes the applicable files
below. PDF copies require ``--save-pdf``; optional descriptors require the
listed flags.

| File | What's in it |
|---|---|
| `analysis_rdf.png` / `.pdf` | Partial RDFs for all unique pairs in the system. One line per pair using the Okabe-Ito palette. |
| `analysis_rdf.csv` | Columns: ``r(A), g(r)_Total, g(r)_<pair1>, g(r)_<pair2>, …``. The total g(r) is always in the CSV; it is drawn only with ``--total-rdf``. Re-plot in any tool. |
| `analysis_rdf_panels.png` / `.pdf` | With ``--pair-panels``: the same partials, one small panel per pair (shared axes). |
| `analysis_sq_partials.png`, `analysis_sq_partials_panels.png` | With ``--sq --sq-partials``: the Faber-Ziman partials S_ab(q) on one axis and, with ``--pair-panels``, one panel per pair. |
| `analysis_cn.png` / `.pdf` | Coordination distribution of the bonded pairs. Binary AB systems (SiO₂) as **mirrored bars**: A-B on top, B-A reflected below the zero line. Multi-cation compounds (IGZO) as one panel per cation-centred pair (Ga-O, In-O, Zn-O) plus the anion total over all its cations (O-(Ga+In+Zn)). Mono-element systems (a-Si) and alloys side-by-side. |
| `analysis_cn.csv` | Per-pair CN counts as percentages of the centred atom population, plus the anion-total rows. |
| `analysis_cn_total.png` / `.csv` | With ``--total-cn``: one panel per requested total (``O``, ``O:In+Ga``). |
| `analysis_angles.png` / `.pdf` | Bond-angle histograms (normalised). One line per triplet. |
| `analysis_angles.csv` | Raw angle values, one row per triplet observation. |
| `analysis_density.png` / `.pdf` | Per-structure density violin with jittered scatter and mean ± std label (at least two structures). |
| `analysis_density.csv` | One row per structure: ``structure_index, density_g_per_cm3`` (at least two structures). |
| `analysis_sq.png` / `.pdf`, `analysis_sq.csv` | With ``--sq``: S(q) and, for the direct method, the number of q-vectors per bin; raw values are also saved when smoothing is enabled. |
| `analysis_tr.png` / `.pdf`, `analysis_tr.csv` | With ``--tr``: the total correlation function and its scattering-weighted g(r) and reduced PDF G(r). |
| `analysis_rings.png` / `.csv` | With ``--rings``: ring-size distribution (size, count, percent of edges). |
| `analysis_voronoi.csv` | With ``--voronoi``: the ten most common Voronoi indices with counts and percentages. |
| `analysis_connectivity.csv` | With ``--connectivity``: corner/edge/face link percentages and the edge-sharing cation fraction, overall and per structure. |
| `analysis_bond_order.{json,csv,png,pdf}`, `analysis_bond_order_atoms.csv` | With `--bond-order`: per-structure ordered fraction and largest ordered cluster, a $\bar q_6$ histogram, and per-atom $q_6$, $\bar q_6$, neighbour counts and cluster labels in JSON and the atom CSV. PDF requires `--save-pdf`. |
| `analysis_voids.{json,csv,png,pdf}` | With `--voids`: clearance density and volume fractions; JSON includes sampling uncertainties and per-structure statistics. PDF requires `--save-pdf`. |
| `analysis_oxygen_speciation.{json,csv,png,pdf}` | Oxygen counts/fractions by class; JSON also contains each oxygen's former coordination. |
| `analysis_elastic.{json,csv,png,pdf}`, `analysis_elastic_tensor.csv` | Modulus means/std/counts and mean stiffness heatmap; JSON includes raw/symmetrized tensors and diagnostics per structure. |
| `analysis_vdos.{json,csv,png,pdf}` | Total and element-projected DOS; JSON includes individual mode frequencies and per-structure diagnostics. |
| `analysis_convergence.{json,txt}` | With `--convergence` or `--tolerance`: declared bounds, observed uncertainty, planning curves and required-size forecasts with assumptions. |
| `analysis_convergence_summary.csv`, `analysis_convergence_curves.csv` | One row per descriptor for decisions/counts/units; one row per descriptor, planned size and point for uncertainty curves. |
| `analysis_convergence_DESCRIPTOR.png` / `.pdf` | One independent figure per descriptor with tolerance, observed endpoint and estimated required size. PDF requires `--save-pdf`. |

## Python API

For programmatic access, useful in scripts, notebooks, and the
comparison workflow:

```python
from amorphgen.analysis import StructureAnalyser

sa = StructureAnalyser("hybrid_ga2o3/final/")   # accepts a dir OR list of files
sa.summary()                                # prints and returns the structural summary

# Individual descriptors
rho = sa.density()
print(rho["mean"], rho["std"], rho["values"])

rdf = sa.rdf(pair="Ga-O", sigma=0.05)
print(rdf["r"], rdf["g_r"])

cn = sa.coordination()
print(cn["Ga-O"]["mean"], cn["Ga-O"]["distribution"])

ang = sa.bond_angles()
print(ang["O-Ga-O"]["mean"])

# Structure factor: direct method, neutron weighting, Faber-Ziman partials
sq = sa.structure_factor_direct(weighting="neutron", sigma_q=0.05, partials=True)
print(sq["q"], sq["s_q"], sq["partials"]["Ga-Ga"])

# Multi-ensemble comparison
from amorphgen.analysis import EnsembleSpec, compare_ensembles
# Call compare_ensembles with the arguments in "Compare multiple ensembles" above.

# Validate against reference
from amorphgen.analysis import validate_against_reference, format_validation_report
import yaml
with open("examples/reference_a_Ga2O3.yaml") as f:
    ref = yaml.safe_load(f)
print(format_validation_report(validate_against_reference(sa, ref)))
```

See {doc}`/api/analysis` for the full Python API.

## Troubleshooting

### "Density is high but CN looks too low"

The default `auto-rdf` cutoff handles most systems, but if you're using
the legacy `--cutoff auto` (minsep-based), it may truncate the first
RDF peak for materials with broad bond distributions. Symptom: many
atoms appear under-coordinated. Fix: drop the `--cutoff auto` and let
the default `auto-rdf` resolve it.

### "Per-structure E/atom shows N/A"

If your structures are in VASP or CIF format, they don't carry energy
in their headers. AmorphGen falls back to reading ``random_gen.log``
in the parent directory (the file written by ``amorphgen --random-gen
--relax``). If you've moved the structures away from their original
``--random-gen`` output, place the log in their parent directory, or use
``--format xyz`` / ``--format extxyz`` (which embed energy in the
comment line) when generating.

### "Si–Si appears as a bond in a-SiO₂"

That's an analysis artifact. Same-element pairs in multi-element ionic
compounds (Si–Si in SiO₂, Hf–Hf in HfO₂, Ga–Ga in Ga₂O₃) are
**second-shell contacts mediated through the anion**, not first-shell
bonds. They are listed under `Non-bonded contacts` and excluded from
bonding coordination, total coordination, bond-angle triplets and the
CN plot. For SiO₂, the relevant bonding CNs are Si–O and O–Si. In
a-Si and a-Si:H, however, Si–Si is a host-network bond and is included.

### "RDF goes to zero suddenly at large r"

Check the plotted r range and the cell size. The default `rmax` is half
the smallest cell-vector length, rounded down to 0.1 Å. Extending the
range can introduce correlations from periodic replicas; use a larger
cell to investigate longer-range structure.
