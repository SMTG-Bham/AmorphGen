# Structure analysis

The ``amorphgen.analysis`` module provides ensemble structural analysis for
amorphous structure files: pair distribution functions, coordination
numbers, bond angles, ring statistics, Voronoi metrics, energy ranking,
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

## Submodule reference

``StructureAnalyser`` delegates to focused submodules; advanced users can
import these directly:

| Submodule | Provides |
|---|---|
| ``analysis.rdf`` | Pair distribution function g(r), partial RDFs |
| ``analysis.structure`` | Coordination numbers, bond distances, bond angles |
| ``analysis.rings`` | Ring statistics (King's shortest-path) |
| ``analysis.voronoi`` | Voronoi cell volumes and connectivity |
| ``analysis.energy`` | Total-energy parsing and ranking |
| ``analysis.cutoff`` | Bond-cutoff selection from g(r) first minimum |
| ``analysis.plotting`` | Publication-quality matplotlib helpers |
| ``analysis.validate`` | Reference-YAML validation |
