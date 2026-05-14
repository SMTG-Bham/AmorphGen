# Utilities

User-facing helper functions used across the pipeline.

## Common helpers

MD-dynamics builder, trajectory I/O, config merging, density helper
(``amorphgen.utils.common``).

```{eval-rst}
.. automodule:: amorphgen.utils.common
   :members:
   :show-inheritance:
```

## Radii, classification and auto-derivation

Shannon ionic / Cordero covalent / Goldschmidt metallic radii, bonding-type
classification, automatic minsep + density + target-CN derivation, and
charge-balance oxidation-state inference (``amorphgen.utils.radii``).
These helpers are exercised end-to-end in **Tutorial 2**
("Zero-config random structure generation").

```{eval-rst}
.. automodule:: amorphgen.utils.radii
   :members: default_minsep, estimate_cell_length, auto_target_cn,
             infer_oxidation_state, get_ionic_radius
   :show-inheritance:
```

## Format conversion

Format-conversion helper used by ``amorphgen --convert`` and the convert-mode
YAML config (``amorphgen.utils.convert``).

```{eval-rst}
.. automodule:: amorphgen.utils.convert
   :members:
   :show-inheritance:
```

## MD equilibration convergence analysis

Functions for assessing whether an MD equilibration stage has converged
(``amorphgen.utils.equilibration``).  Useful as a post-pipeline quality
check: did the high-T equilibration in your melt-quench actually reach
steady-state, or did you quench from a still-thermalising liquid?

The high-level entry point is ``convergence_report``, which generates
energy-vs-time, block-averaging, MSD, temperature, and time-window-RDF
plots in one call.  Lower-level functions are exposed for users who
want to assemble custom diagnostics.

```{eval-rst}
.. automodule:: amorphgen.utils.equilibration
   :members: parse_md_log, extract_energies, running_average,
             plot_energy_convergence, block_average_test,
             plot_block_averages, compute_msd, plot_msd,
             plot_temperature, plot_rdf_time_windows,
             compute_cn_vs_time, plot_cn_vs_time,
             convergence_report
   :show-inheritance:
```

## Other utility modules

- {doc}`calculators` — multi-backend calculator factory (``get_calculator``).
- ``amorphgen.utils.classical`` — Lennard-Jones and Buckingham+Coulomb
  calculators; loaded via ``get_calculator("lennard-jones" | "buckingham", ...)``.
