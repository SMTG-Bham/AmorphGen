# Best practices & limitations

Practical guidance for getting physically reliable amorphous structures out of
AmorphGen — and an honest account of where the underlying methods break down.

## Prefer NVT annealing over NPT melt-quench with foundation MLIPs

The single most important recommendation: **with universal/foundation MLIPs,
generate amorphous structures by random placement + low-temperature annealing
(or the hybrid route), rather than a full crystal melt-quench under NPT.**

Universal MLIPs (MACE-MP, CHGNet, SevenNet, …) are trained almost entirely on
*near-equilibrium crystalline* data. A high-temperature liquid is out of that
distribution on every axis at once — large forces, broken/over-coordinated
bonds, high kinetic energy — so the model extrapolates and the melt is
unreliable. The failure is worst under **NPT**, because the barostat acts on the
stress tensor, which is even more sensitive to out-of-distribution force error
than the energy: wrong stresses drive the cell to expand, collapsing the
predicted density.

AmorphGen's {doc}`random-generation` and {doc}`hybrid-workflow` routes are built
around the reliable regime: they start from in-distribution disordered
configurations and anneal at low temperature, in a **fixed cell (NVT-like)** when
you keep the auto-estimated density. This avoids the unstable liquid entirely.

```{tip}
If you must run a melt-quench, prefer **NVT** (fixed cell) over NPT, keep the
melt temperature as low as the chemistry allows, and treat any large
mid-quench volume change as a red flag rather than a result.
```

### Decision guide

| Situation | Recommended route |
|-----------|-------------------|
| New composition, no crystal needed | `--random-gen` + `--relax` (NVT-like, fixed cell) |
| Want a diverse amorphous ensemble cheaply | `--hybrid-ensemble` (random → anneal → quench) |
| You have a crystal and want classic MQ | Full pipeline, but prefer **NVT** stages; watch the density |
| MLIP keeps over-expanding under NPT | Switch to random-gen / hybrid, or fix the cell |

## Trust the relaxed density, set it when the estimate is shaky

The auto-estimated density is a *starting cell* heuristic (class-aware sphere
packing on Shannon/Cordero/Goldschmidt radii). It is good for common oxides but
approximate for unusual chemistries.

```{warning}
For elements missing from the radii tables, or compositions far from the tuned
material classes, the auto density can be off. AmorphGen prints
`NOTE: Auto density is approximate for this composition` when confidence is low —
in that case pass `--target-density` (or a `cell_length`) explicitly, or relax
with a cell filter and trust the **relaxed** density, not the estimate.
```

Dense rutile-type dioxides are the usual culprits: the generic `metal_oxide`
packing factor under-predicts them, which is why rutile-type MO₂ oxides
(TiO₂, SnO₂, RuO₂, IrO₂, OsO₂, …) are routed to a denser `rutile_dioxide`
class. They are identified geometrically — an MO₂ whose 4+ cation radius is
below the rutile/fluorite cutoff (~0.70 Å) — so fluorite dioxides (ZrO₂, HfO₂,
CeO₂) correctly stay `metal_oxide`. See {doc}`../validation/index` for the
validated density ranges.

## Known limitations

- **Foundation-MLIP reliability** — accuracy on amorphous/liquid configurations,
  and on elements sparsely represented in training data, is not guaranteed.
  High-temperature liquid sampling under NPT is the main failure mode (see
  above); validate against AIMD or experiment for any new chemistry (see
  {doc}`../validation/index`).
- **Density estimation is approximate** — the auto density is a class-aware
  sphere-packing *estimate* for the starting cell, tuned on common material
  classes. It can be off for unusual chemistries; override with
  `--target-density` and trust the relaxed density.
- **Element coverage of the radii tables** — minimum-separation and density
  estimation use Shannon/Cordero/Goldschmidt radii for a curated element set.
  Elements outside it fall back to approximate values, degrading the auto
  density (e.g. set an explicit `--target-density`, or add the element to
  `amorphgen/utils/radii.py`).
- **Single-point relaxation leaves voids** — a 0 K relax of one random structure
  can stay porous; use the hybrid (anneal + quench) route for densification.
- **Ensembles, not single structures** — one structure is not statistically
  representative of an amorphous phase. Generate an ensemble (e.g. `-n 20`) and
  average for any reported property.
- **Structure generation only** — AmorphGen produces relaxed atomic structures,
  not electronic-structure or transport properties; those need a separate
  DFT/post-processing step on the generated models.
- **Classical potentials need parameters** — the Lennard-Jones and
  Buckingham+Coulomb backends require user-supplied parameters; they are a fast
  starting point, not a substitute for a fitted potential.
- **Resume granularity** — stage-level only. A run killed mid-stage re-runs that
  whole stage on `--resume`; the partial trajectory is preserved but not
  continued from.

## Further reading

For the reliability literature behind the NVT-over-NPT recommendation, see the
broader work on finite-temperature reliability of foundation atomistic models.
