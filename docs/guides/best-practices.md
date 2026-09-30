# Best practices & limitations

Practical guidance for getting physically reliable amorphous structures out of
AmorphGen, and an honest account of where the underlying methods break down.

(prefer-nvt-annealing-over-npt-melt-quench-with-foundation-mlips)=
## Choose the ensemble and calculator together

Validate the chosen calculator for the composition, density and temperature
range of the intended workflow. A model's accuracy on a crystalline starting
structure does not establish its accuracy on random, amorphous or liquid
configurations. Check its training coverage and compare representative
configurations with reference calculations or measurements.

NPT uses the calculator's stress tensor to change the cell. Inspect both the
volume trajectory and the resulting density. NVT keeps the cell fixed during
MD; it is useful when a justified target density is available, but fixing the
cell does not validate the predicted forces or structure.

AmorphGen's {doc}`random-generation` and {doc}`hybrid-workflow` routes let you
start from disordered configurations and choose an anneal temperature. Set
each MD stage to **NVT** and use `--cell-filter none` for relaxation when
holding the starting density fixed; the random-gen and hybrid CLI modes
otherwise default to isotropic cell relaxation (`cubic`). The full pipeline
defaults to NPT for stages 3 and 4, so override both for fixed-volume MD.

### Decision guide

| Situation | Recommended route |
|-----------|-------------------|
| New composition, no crystal needed | `--random-gen --relax --cell-filter none` when keeping a validated density fixed |
| Want a diverse amorphous ensemble cheaply | `--hybrid-ensemble` (random → anneal → quench) |
| You have a crystal and want classic MQ | Full pipeline; choose NVT or NPT according to the density constraint and calculator validation |
| Cell expands unexpectedly under NPT | Check model stresses, timestep, coupling parameters and reference density before continuing |

## Choosing the anneal temperature

A static relaxation can leave a random-gen seed in a local minimum with
connectivity defects. Annealing can allow further rearrangements; the useful
temperature and duration depend on the system and calculator. Track both
energy and structural metrics during the anneal (for example, bridging-anion
fraction and network-former coordination). A flat coordination metric alone
does not distinguish a converged network from a trapped one.

An exploratory a-SiO₂ run (72 atoms, CHGNet, NVT) previously recorded in
this guide gave the following coordination fractions at 600 K. The complete
input, model version and trajectory are not supplied with this table, so it
is an illustration of what to monitor, not a reproducible temperature recipe:

| Time at 600 K | Si CN=4 | bridging O |
|---|---|---|
| 0 ps (relaxed seed) | 79% | 79% |
| 3 ps | 92% | 100% |
| 5 ps | 100% | 100% |

A separate 1500 K, 8 ps anneal of the same starting seed was reported to
end at 96% Si CN=4 and 98% bridging O after re-relaxation, with an energy
about 38 meV/atom lower (−590.99 vs −588.23 eV for 72 atoms). These
exploratory observations do not establish a generally sufficient temperature
or duration for silica or other network glasses.

For your own protocol:

- Compare several temperatures and durations using the same calculator and
  analysis settings.
- Inspect density, coordination, RDF and any relevant connectivity or ring
  metrics; extend sampling when the properties of interest are still changing.
- Re-relax sampled structures before comparing minimum energies. Use the same
  relaxation settings and calculator; annealing does not guarantee a lower
  final energy for every seed.

## Validate the density and override an uncertain estimate

The auto-estimated density is a *starting cell* heuristic (class-aware sphere
packing on Shannon/Cordero/Goldschmidt radii). It is good for common oxides but
approximate for unusual chemistries.

```{warning}
For elements missing from the radii tables, or compositions far from the tuned
material classes, the auto density can be off. AmorphGen prints
`NOTE: Auto density is approximate for this composition` when confidence is low,
in that case pass `--target-density` explicitly (or `cell_length_ang` in the Python
API). A cell relaxation provides a model-dependent density that also needs
validation.
```

Dense rutile-type dioxides are the usual culprits: the generic `metal_oxide`
packing factor under-predicts them, which is why rutile-type MO₂ oxides
(TiO₂, SnO₂, RuO₂, IrO₂, OsO₂, …) are routed to a denser `rutile_dioxide`
class. They are identified geometrically: an MO₂ whose 4+ cation radius is
below the rutile/fluorite cutoff (~0.70 Å), so fluorite dioxides (ZrO₂, HfO₂,
CeO₂) use the separate `fluorite_dioxide` class. See {doc}`random-generation`
for density-estimation classes; validate the resulting density for your material.

## Known limitations

- Foundation-MLIP reliability: accuracy on amorphous/liquid configurations,
  and on elements sparsely represented in training data, is not guaranteed.
  Validate the intended temperature and ensemble against reference
  calculations or experiment for any new chemistry (see
  {doc}`../validation/index`).
- Density estimation is approximate: the auto density is a class-aware
  sphere-packing *estimate* for the starting cell, tuned on common material
  classes. It can be off for unusual chemistries; override with
  `--target-density` and validate the relaxed density.
- Element coverage of the radii tables: minimum-separation and density
  estimation use Shannon/Cordero/Goldschmidt radii for a curated element set.
  Elements outside it fall back to approximate values, degrading the auto
  density (e.g. set an explicit `--target-density`, or add the element to
  `amorphgen/utils/radii.py`).
- Local relaxation can leave voids: a 0 K relaxation of one random structure
  can stay porous. Annealing can rearrange the network, but NVT and fixed-cell
  relaxation preserve the total density; densification requires a volume change.
- Ensembles, not single structures: one structure is not statistically
  representative of an amorphous phase. Generate an ensemble (e.g. `-n 20`) and
  average for any reported property.
- Structure generation only: AmorphGen produces relaxed atomic structures,
  not electronic-structure or transport properties; those need a separate
  DFT/post-processing step on the generated models.
- Classical potentials need suitable parameters: the Lennard-Jones and
  Buckingham+Coulomb backends accept explicit parameter sets. Defaults or
  illustrative parameters do not validate a potential for your material.
- Resume granularity: `--resume` skips completed stages and continues an
  interrupted MD stage from its last saved trajectory frame; only the
  optimisation stages (1 and 7) restart from the beginning.

## Further reading

See {doc}`benchmarks` for exploratory comparisons and
{doc}`../validation/index` for the validation examples supplied with this project.
