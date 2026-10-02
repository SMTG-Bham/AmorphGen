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
| ``analysis.energy`` | Total-energy parsing and ranking |
| ``analysis.cutoff`` | Bond-cutoff selection from g(r) first minimum |
| ``analysis.plotting`` | Publication-quality matplotlib helpers |
| ``analysis.validate`` | Reference-YAML validation |
| ``analysis.comparison_plots`` | Multi-ensemble comparison plots and CSV output |
