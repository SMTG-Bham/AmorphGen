---
orphan: true
---

# Scattering methods and citations

AmorphGen implements standard scattering conventions in
`amorphgen/analysis/rdf.py`, exposed through `StructureAnalyser` and the CLI.
Cite the software version used, together with the sources relevant to the
calculation. The repository's
[CITATION.cff](https://github.com/SMTG-Bham/AmorphGen/blob/main/CITATION.cff)
contains the software citation.

## Sources used by the implementation

| Topic | Reference |
|---|---|
| Total-scattering normalization and correlation-function conventions | [Keen, *J. Appl. Cryst.* **34** (2001), 172–177](https://doi.org/10.1107/S0021889800019993) |
| Faber–Ziman partial weighting | [Faber & Ziman, *Philos. Mag.* **11** (1965), 153–173](https://doi.org/10.1080/14786436508211931) |
| q-dependent X-ray form factors | [Waasmaier & Kirfel, *Acta Cryst. A* **51** (1995), 416–431](https://doi.org/10.1107/S0108767394013292) |
| Coherent neutron scattering lengths | [Sears, *Neutron News* **3** (1992), 26–37](https://doi.org/10.1080/10448639208218770) |

The direct method evaluates atomic scattering amplitudes at nonzero
reciprocal-lattice vectors and averages over spherical shells. It subtracts
the self-scattering contribution to produce Faber–Ziman `S(q)` with a
high-q limit of 1. X-ray weights use q-dependent neutral-atom form factors,
not constant atomic numbers. See {doc}`sq_xrd_methodology` for the equations.

## Reporting a calculation

A methods description should state:

- The AmorphGen version, input configurations, cell sizes and temperature.
- Whether `structure_factor_direct()` or the FT-based `structure_factor()`
  was used, and the scattering weights.
- The q range, number of bins and smoothing width; for the FT method, `rmax`.
- The normalization of the experimental reference and any additional
  instrument or sample corrections.

For example, after filling in the settings actually used:

> Structure factors were computed with AmorphGen by summing atomic
> scattering amplitudes over the periodic cells' reciprocal-lattice vectors
> and averaging into spherical q shells. X-ray weights used the
> Waasmaier–Kirfel neutral-atom form factors. The self-scattering term was
> removed using the Faber–Ziman normalization.

Add the software and method citations to this description. If reporting an
XRD intensity profile, also describe the conversion from normalized `S(q)`
to coherent intensity and the measurement-specific corrections. Agreement
with another package or an experiment should be supported by a reproducible
comparison of the same inputs and conventions.
