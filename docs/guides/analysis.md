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

- Per-bin uncertainty estimates from independent configurations.
- An `xrd_pattern()` convenience method with explicit intensity conventions
  and instrument settings.

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
