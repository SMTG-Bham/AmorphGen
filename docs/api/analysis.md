# Structure analysis

The ``amorphgen.analysis`` module provides ensemble structural analysis for
amorphous structure files: pair distribution functions, structure factors,
total correlation functions, coordination
numbers, bond angles, bond-orientational order, ring statistics, Voronoi metrics, void distributions,
oxygen speciation, elastic moduli, harmonic vibrational DOS, energy ranking,
and validation against literature reference ranges.

The CLI entry point is ``amorphgen --analyse <FILE_OR_DIR>``; the Python
API is the ``StructureAnalyser`` class.

## `StructureAnalyser`

```{eval-rst}
.. autoclass:: amorphgen.analysis.StructureAnalyser
   :members:
   :undoc-members:
   :show-inheritance:
```

## Ring sizes and void clearance

```python
from amorphgen.analysis import StructureAnalyser
from amorphgen.analysis.descriptors import save_descriptor

sa = StructureAnalyser("structures/", cutoff={"Si-O": 2.0})
rings = sa.ring_statistics(bond_pair=("Si", "O"), max_ring=16)
voids = sa.void_distribution(n_samples=20000, probe_radius=0.5, seed=42,
                             probe_radii=[0, 0.25, 0.5, 0.75, 1.0])
save_descriptor("rings", rings, "analysis/")
save_descriptor("voids", voids, "analysis/")
```

Ring `counts` and legacy `total_rings` count shortest-cycle observations per
network edge, not unique cycles. `mean_ring_size`, `std_ring_size` (population
spread), `min_ring_size` and `max_ring_size` summarize the resolved edges.
`n_network_edges`, `n_ring_edges`, `n_unresolved_edges` and `ring_edge_fraction`
report search coverage. An unresolved edge may close beyond `max_ring`;
undefined size statistics and coverage are `None`. The resolved cutoff,
counting convention and per-structure observations accompany the result.

Void `probe_curve` contains sorted unique `radii` and the aligned
`accessible_fraction`, `accessible_volume` and corresponding `*_stderr`
arrays. All thresholds use the same samples; these errors quantify Monte
Carlo noise. Omitting `probe_radii` uses the histogram bin edges. The curve
can include radii below the base `probe_radius`, independently of the
histogram. `clearance_quantiles` gives empirical p10/p50/p90 clearances
conditional on the base probe, using cell-volume weights across structures.
No accessible samples gives `None` quantiles. Clearance describes local free
space, not connected pores or maximal cavities.

Both results include `per_structure` observations and separate `uncertainty`
summaries of equal-weight structure means. See {doc}`/guides/analysis` for
normalization, interpretation and exported files.

## Cutoff robustness

`StructureAnalyser.cutoff_robustness(window=0.1, points=5)` measures contact
and coordination sensitivity around the analyser's resolved pair cutoffs.
`window` is a finite positive half-width in Å; `points` is an odd integer
of at least three. Automatic cutoffs are resolved once and frozen while
the same offset is applied to each positive pair cutoff. Values are clipped
at zero and zero cutoffs stay zero throughout the sweep.

The report includes pooled undirected periodic contact counts and the share
whose inclusion changes from the lower to the upper endpoint, using the
upper endpoint contact count as denominator. No contacts gives an undefined
share. Coordination is directional, with pooled central-site means,
per-structure means and equal-weight structure means retained. The ordinary
`distance <= pair cutoff` and `distance < largest cutoff` boundary rules
apply at every point. See {doc}`/guides/analysis` for interpretation.

`summary(show_angles=True, cutoff_window=0.1)` and
`per_structure_summary(cutoff_window=0.1)` include the five-point report by
default. `plot(..., cutoff_window=0.1)` exports it when `save_csv=True`.

```python
from amorphgen.analysis import (
    StructureAnalyser, format_cutoff_robustness, save_cutoff_robustness,
)

sa = StructureAnalyser("structures/", cutoff="auto-rdf")
report = sa.cutoff_robustness(window=0.15, points=7)
print(format_cutoff_robustness(report))
paths = save_cutoff_robustness(report, output_dir="analysis/", prefix="analysis")
```

`save_cutoff_robustness(report, output_dir=".", prefix="analysis")` writes
`analysis_cutoff_robustness.json`, `analysis_cutoff_robustness_pairs.csv`
and `analysis_cutoff_robustness_coordination.csv` with the default prefix.

```{eval-rst}
.. autofunction:: amorphgen.analysis.format_cutoff_robustness

.. autofunction:: amorphgen.analysis.save_cutoff_robustness
```

## Measured scattering and XRD

`StructureAnalyser.compare_experiment()` loads measured S(q) or T(r),
calculates the matching ensemble curve and reports residuals and metrics with
pointwise ensemble bands. The separate loader and comparison functions accept
already calculated curves. `StructureAnalyser.xrd_pattern()` returns coherent
X-ray intensity per atom on a physical 2θ axis. See {doc}`/guides/analysis`
for file formats, metric definitions, CLI examples and intensity conventions.

```{eval-rst}
.. autofunction:: amorphgen.analysis.load_experiment

.. autofunction:: amorphgen.analysis.compare_experiment

.. autofunction:: amorphgen.analysis.format_experiment_report

.. autofunction:: amorphgen.analysis.save_experiment_comparison

.. autofunction:: amorphgen.analysis.compute_xrd_pattern

.. autofunction:: amorphgen.analysis.save_xrd_pattern
```

## Optional material descriptors

The following functions are also exported from `amorphgen.analysis`.
Their `StructureAnalyser` counterparts are `bond_order()`, `void_distribution()`,
`oxygen_speciation()`, `elastic_moduli()` and `vibrational_dos()`.
All return ensemble and per-structure results; the geometry functions need
no calculator. Elastic and vibrational calculations require a live ASE
calculator that can evaluate strained or displaced configurations.

```python
from ase.io import read
from amorphgen.analysis import compute_void_distribution, compute_oxygen_speciation

frames = read("silica.extxyz", index=":")
voids = compute_void_distribution(frames, n_samples=20000, seed=42)
oxygen = compute_oxygen_speciation(frames, network_formers=["Si"],
                                  cutoff={"Si-O": 2.0})
```

See {doc}`/guides/analysis` for physical conventions, calculator cost,
CLI examples and exports. `amorphgen.analysis.descriptors.save_descriptor`
accepts a result with the name `bond_order`, `voids`, `oxygen_speciation`, `elastic` or
`vdos` and writes JSON, CSV and PNG, plus PDF with `save_pdf=True`.

```{eval-rst}
.. autofunction:: amorphgen.analysis.compute_bond_order

.. autofunction:: amorphgen.analysis.compute_void_distribution

.. autofunction:: amorphgen.analysis.compute_oxygen_speciation

.. autofunction:: amorphgen.analysis.compute_elastic_moduli

.. autofunction:: amorphgen.analysis.compute_vibrational_dos
```

### Bond order result

```python
from amorphgen.analysis import StructureAnalyser

sa = StructureAnalyser("structures/", cutoff="auto-rdf")
order = sa.bond_order(qbar6_threshold=0.3, min_neighbors=4,
                      cutoff=3.5)  # example order shell; choose for the material
frame = order["per_structure"][0]
print(frame["ordered_fraction"], frame["largest_cluster_size"])
print(frame["q6"], frame["qbar6"], frame["cluster_ids"])
```

`bond_order(..., cutoff=None)` uses the analyser's resolved cutoff unless
overridden. The standalone `compute_bond_order(atoms_list, cutoff="auto-rdf",
qbar6_threshold=0.3, min_neighbors=4)` accepts ASE frames directly. Neither
method loads a calculator.

`parameters` records the resolved `cutoff`, `qbar6_threshold` and
`min_neighbors`. Each `per_structure` item contains `index`, `n_atoms`,
per-atom arrays `q6`, `qbar6`, `neighbor_counts`, `ordered`, `cluster_ids`,
and scalar `ordered_count`, `ordered_fraction`, `largest_cluster_size`,
`largest_cluster_fraction`, `q6_mean` and `qbar6_mean`. Cluster IDs are -1
for disordered atoms. Fractions divide by all atoms in the structure;
cluster sizes count unique cell atoms. Top-level scalar summaries are
arithmetic means across structures, including `largest_cluster_size`.

See {doc}`/guides/analysis` for the order equations and threshold calibration,
and {doc}`/guides/mq-ensemble` for the automatic initial-crystal retention
report.

### Initial-crystal retention

`compute_melt_memory(initial_atoms, frames, cutoff="auto-rdf",
qbar6_threshold=0.3, min_neighbors=4)` takes an original ASE structure and
a mapping from checkpoint labels to ASE frames. Its `initial` and
`comparisons` entries contain order summaries; `survival_fraction` divides
the retained initially ordered atom count by the original ordered count.
`parameters` stores the fixed cutoff resolved from the input.

No initially ordered atoms gives an unavailable (`None`) survival fraction.
Atom-count or element-sequence mismatches raise `ValueError` in this Python
helper; the automatic MQ file report instead records unavailable checkpoint
rows with reasons. Both require preserved atom indices and measure endpoint
retention, so neither establishes uninterrupted crystal survival.

```{eval-rst}
.. autofunction:: amorphgen.analysis.compute_melt_memory

.. autofunction:: amorphgen.analysis.format_melt_memory
```

## Ensemble convergence

`StructureAnalyser.convergence_report(tolerances=None, *, descriptors=None,
confidence=0.95, sizes=None, max_structures=1000000)` gathers density,
coordination, total coordination, bond-distance and bond-angle summaries.
`descriptors` adds named aligned per-structure arrays or uncertainty summaries
containing `per_structure`. Tolerances are positive absolute Student-t mean
interval half-widths, with curve descriptors assessed at every point.

The standalone `convergence_report` accepts a mapping of descriptor names to
scalar or structure-by-component observations, with missing entries as `None`
or nonfinite numbers. The report includes `descriptors`, observed counts and
half-widths, `curve` data, declared tolerance statuses and estimated total and
additional structure counts. Arrays and missing values are JSON-compatible.
Canonical reductions make the curves and forecasts independent of input order.

```python
from amorphgen.analysis import (
    convergence_report, format_convergence_report, save_convergence_report,
)

report = convergence_report(
    {"density": [2.20, 2.25, 2.18, 2.23]}, {"density": 0.02},
    confidence=0.95, max_structures=10000,
)
print(format_convergence_report(report))
paths = save_convergence_report(report, "analysis/", save_pdf=True)
```

Complete-data planning curves are exact componentwise all-subset RMS
Student-t half-widths for sizes 2 through the observed ensemble size; larger
sizes extrapolate the variance model. Vector summaries take the maximum of
those componentwise values.
Missing data use an observed-availability approximation. These are conditional
precision forecasts under independent sampling, not empirical histories or
simultaneous confidence bands. See {ref}`ensemble-convergence` for the formulas,
CLI/YAML examples, status meanings and exported files.

```{eval-rst}
.. autofunction:: amorphgen.analysis.convergence_report

.. autofunction:: amorphgen.analysis.format_convergence_report

.. autofunction:: amorphgen.analysis.save_convergence_report
```

## Reference-validation helpers

For comparing computed metrics against literature ranges (used by
``amorphgen --analyse --reference REF.yaml``):

```{eval-rst}
.. automodule:: amorphgen.analysis.validate
   :members: validate_against_reference, format_validation_report
   :show-inheritance:
```

## Energy ranking helpers

For parsing ``random_gen.log`` and ranking generated structures by total
energy (used by ``amorphgen --rank-from-log LOG``):

```{eval-rst}
.. automodule:: amorphgen.analysis.energy
   :members: rank_from_log, format_log_ranking
   :show-inheritance:
```

## Comparing ensembles

Use `EnsembleSpec` to describe each structure set and `compare_ensembles`
to save RDF, coordination, bond-angle, and density comparisons.

```{eval-rst}
.. autoclass:: amorphgen.analysis.EnsembleSpec
   :members:

.. autofunction:: amorphgen.analysis.compare_ensembles
```

## Submodule reference

``StructureAnalyser`` delegates to focused submodules; advanced users can
import these directly:

| Submodule | Provides |
|---|---|
| ``analysis.rdf`` | Pair distribution function g(r), partial RDFs, S(q), T(r) |
| ``analysis.structure`` | Coordination numbers, bond distances, bond angles |
| ``analysis.bond_order`` | Steinhardt $q_6$, Lechner–Dellago $\bar q_6$ and periodic ordered clusters |
| ``analysis.melt_memory`` | Initial ordered-atom retention at MQ melt endpoints and snapshots |
| ``analysis.rings`` | Shortest-path ring statistics with periodic-image closure |
| ``analysis.voronoi`` | Voronoi cell volumes and connectivity |
| ``analysis.voids`` | Periodic Monte Carlo point clearance and accessible volume |
| ``analysis.oxygen`` | Oxygen classes by network-former coordination |
| ``analysis.elasticity`` | Stress-derived stiffness and Voigt/Reuss/Hill moduli |
| ``analysis.vibrations`` | Harmonic cell-mode DOS and element projections |
| ``analysis.descriptors`` | Optional descriptor summaries and JSON/CSV/figure export |
| ``analysis.convergence`` | Order-independent uncertainty curves, tolerances and sample-size planning |
| ``analysis.convergence_output`` | Convergence text, JSON, CSV and per-descriptor figures |
| ``analysis.energy`` | Total-energy parsing and ranking |
| ``analysis.cutoff`` | Bond-cutoff selection from g(r) first minimum |
| ``analysis.plotting`` | Publication-quality matplotlib helpers |
| ``analysis.validate`` | Reference-YAML validation |
| ``analysis.comparison_plots`` | Multi-ensemble comparison plots and CSV output |
