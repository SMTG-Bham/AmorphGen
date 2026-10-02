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

## Quantitative protocol: starting values and convergence

Use the following as a **study design for bulk inorganic glasses**, especially
oxides. These are proposed starting budgets to test, not validated settings for
every material or calculator. The package defaults and the suggested trials
are listed separately. Choose the observables and acceptable errors before
running the convergence study.

| Parameter | Current package behaviour | Suggested first study |
|-----------|---------------------------|-----------------------|
| MD timestep | **0.5 fs** in every MD stage | Compare **0.5 and 0.25 fs**. For H-containing systems or unusually fast vibrations, start at **0.25 fs** and compare **0.125 fs**. Try **1 fs** only after checking against smaller timesteps. |
| Cell size | Supplied by the input structure or random-gen composition; no universal atom-count default | Pilot with **200–300 atoms**, then roughly double or triple the atom count at the same composition and density. Include **600–1500 atoms** in the size study when affordable; continue larger if the target property still changes. |
| High-temperature hold | **10 ps** (`20000` steps at 0.5 fs) | Compare **20, 50 and 100 ps**; extend further until structural relaxation and snapshot decorrelation are resolved. |
| Cooling rate | **100 K/ps** from 100 K segments, 2000 steps per segment and 0.5 fs | Compare **100, 10 and 1 K/ps** with the same endpoints and preparation protocol. |
| Low-temperature hold | **10 ps** (`20000` steps at 0.5 fs) | Start with **10 ps**, compare **20 ps**, and use identical final relaxation settings. |
| Ensemble size | `-n` / `--n-structures` defaults to **1**; mode-specific behaviour is described below | Pilot with **10 independent preparations**, then compare **20 and 40** using uncertainty on the property of interest. |

For context, a published silica study used a 1 fs timestep, 3000 atoms and
six independent starting configurations at each of 0.1, 1, 10 and 100 K/ps
with the Jakse potential. Those choices demonstrate a rate-sensitivity study;
they do not validate a foundation MLIP at the same settings.
See [Yang and Schwalbe-Koda, *npj Computational Materials* (2026)](https://www.nature.com/articles/s41524-025-01901-1).

### Timestep and physical duration

Convert steps to time explicitly:

$$t\;[\mathrm{ps}] = N_{\mathrm{steps}}\,\Delta t\;[\mathrm{fs}]/1000.$$

Thus **100000 steps at 0.5 fs = 50 ps**. When halving the timestep, double
the number of steps in each fixed-temperature stage to preserve its duration.
For ramps, set `rate` explicitly so that their physical duration also stays
fixed; otherwise changing the timestep while keeping `steps_per_T` changes
the heating/cooling rate. `--timestep` applies to all MD stages, while YAML
allows stage-specific values.

Compare ensemble structural metrics and relaxed energies, rather than expecting
individual stochastic trajectories to match. Check the hottest stage as well
as the final glass. Finite energies and a stable thermostatted temperature
alone do not establish timestep convergence. If testing NVE energy drift, use
a separate ASE run: AmorphGen's pipeline exposes only NVT and NPT.
See [ASE's timestep guidance](https://docs.ase-lib.org/ase/md.html#choosing-the-time-step).

### Cooling rate and simulation cost

With a cooling-rate magnitude $R$ in K/ps, the nominal quench duration is
$t_q = |T_{\mathrm{start}}-T_{\mathrm{end}}|/R$.
**1 K/ps = $10^{12}$ K/s**. For **3000 → 300 K**, 100 K temperature
segments and a 0.5 fs timestep:

| `quench.rate` (K/ps) | Rate (K/s) | Steps per segment | Total MD steps | Quench duration |
|----------------------|------------|-------------------|----------------|-----------------|
| 100 | $10^{14}$ | 2000 | 54000 | 27 ps |
| 10 | $10^{13}$ | 20000 | 540000 | 270 ps |
| 1 | $10^{12}$ | 200000 | 5400000 | 2700 ps (2.7 ns) |

Set `quench.rate` in YAML; there is no `--quench-rate` flag. `rate` takes
precedence over `steps_per_T`. AmorphGen uses stepped temperature holds,
not a continuously changing target: there are 27 holds in this example,
excluding the starting temperature and including 300 K. Steps per hold are
rounded to an integer; a final partial temperature interval still receives a
full hold. The table is exact for these divisible endpoints. For other
schedules, report the actual duration as well as the nominal rate.

At the selected rate, also compare **100 K and 50 K** segment sizes, holding
`rate` fixed. Keep `melt.T_end`, `eq_high.T` and `quench.T_start` equal when
changing the maximum temperature; they are independent configuration keys.
Compare coordination, relaxed energy and medium-range structure, not only the
first RDF peak. If changes remain significant at the slowest affordable rate,
report the rate dependence rather than claiming cooling-rate convergence.

### Cell size and accessible length scales

Keep composition, density and preparation schedule fixed while increasing
size. For example, at an **illustrative fixed density of 2.20 g/cm³**, cubic
SiO₂ cells have the following dimensions (calculated from formula masses):

| Random-gen composition | Atoms | Cubic edge $L$ (Å) | Half-box distance $L/2$ (Å) |
|------------------------|-------|--------------------|-----------------------------|
| `SiO2*24` | 72 | 10.29 | 5.14 |
| `SiO2*72` | 216 | 14.84 | 7.42 |
| `SiO2*216` | 648 | 21.40 | 10.70 |
| `SiO2*432` | 1296 | 26.96 | 13.48 |

The 72-atom cell is useful for a smoke test, but cannot independently resolve
an RDF out to 10 Å. For cubic cells, keep the RDF range below **$L/2$**;
10 Å correlations require **$L>20$ Å**. Use a common RDF range and binning
when comparing sizes. For skewed cells, check periodic-image geometry rather
than assuming half the shortest cell-vector length is sufficient. Enlarging
a cell by tiling one finished glass does not create new independent disorder;
prepare the larger systems separately.

Use `--composition` to choose random-gen atom counts and `--target-density`
to choose the starting volume. Use `--retry-mode reduce-minsep` or `none`
when placement retries must not expand the cell, and inspect the resulting
minimum separations. For fixed-density comparisons, use NVT in every MD stage
and `cell_filter: none` in both initial and final relaxation. Larger cells
may be needed for rings, voids, rare defects or mechanical properties even
when nearest-neighbour coordination has converged.

### Independent structures and a quantitative stopping rule

An ensemble of 20 files is not necessarily 20 independent samples.
`--mq-ensemble --n-structures 20 --select decorrelated` requests **up to 20**
snapshots from a shared melt. Inspect `snapshot_sampling.json` for the actual
count, unresolved diagnostics and estimated effective count, and inspect
`melt_memory` for retained crystalline order. A 10 ps plateau is not extended
automatically to fill the request. Increase the plateau and repeat preparation
with independent seeds to check sensitivity to the shared melt.

For `--hybrid-ensemble`, generate the desired number of random seeds first;
hybrid processes all inputs in the first matching format and does not use
`-n` to limit them.
Different noise seeds alone do not prove independence if structures remain
trapped in the same initial network. See {doc}`mq-ensemble` and
{doc}`hybrid-workflow` for sampling details, and the
[PyMBAR timeseries documentation](https://pymbar.readthedocs.io/en/stable/timeseries.html)
for statistical inefficiency and correlated-sample subsampling.

For a scalar property measured once per independent structure, report the
mean, sample standard deviation $s$, and the 95% confidence interval:

$$\bar{x}\;\pm\;t_{0.975,n-1}\,s/\sqrt{n}.$$

AmorphGen's {doc}`analysis` already exports these ensemble uncertainties from
per-structure values. The t interval assumes independent observations and is
exact for normally distributed structure-level values. Inspect the distribution
for small or skewed samples; bootstrap intervals can help, but a small ensemble
cannot establish rare-defect statistics. The exported intervals do not
automatically correct for correlated MQ snapshots. For example,
$s=0.10$ g/cm³ across 20 independent glasses gives
a 95% interval half-width of about **0.047 g/cm³**, falling to **0.032 g/cm³**
at 40 if the spread remains unchanged. Counting more atoms in one structure
does not increase the number of independent preparations.

As an example error budget, require changes below **0.02 Å** in an RDF peak
position, **2 percentage points** in a coordination fraction, **5 meV/atom**
in mean relaxed energy, or **1%** in mean density when volume can vary.
Choose thresholds relevant to the intended property; these are proposed
tolerances, not accuracy guarantees. For each timestep, size, duration and
rate comparison, require the **entire 95% confidence interval for the mean
difference** to lie inside the chosen ± tolerance. Calculate that difference
interval separately, using independent-ensemble statistics or paired differences
when reusing the same independent starting structures. If intervals are too
wide to distinguish a change, increase the independent sample count.
For ensemble-size convergence, compare
10 → 20 → 40 preparations and require the mean to stabilise and its interval
half-width to fall below the tolerance. Density fixed by NVT is an input,
not a convergence result. Sampling precision does not remove calculator bias.

### Worked starting configuration

This explicit **NVT pilot** uses a 50 ps high-temperature hold and a 100 K/ps
quench. Supply a roughly 200–300-atom input cell at a justified density and a
calculator validated for the chosen temperature range. **3000 K is an example**;
verify liquid formation or anneal relaxation for your system before sampling.

```yaml
# Save as protocol.yaml; select the calculator in the command below.
seed: 42
opt:
  cell_filter: none
  fmax: 0.01
  max_steps: 1000
eq_premelt:
  ensemble: NVT
  T: 300
  timestep: 0.5
  steps: 20000           # 10 ps
melt:
  ensemble: NVT
  T_start: 300
  T_end: 3000
  T_step: 100
  timestep: 0.5
  rate: 100              # K/ps; 27 ps heating
eq_high:
  ensemble: NVT
  T: 3000
  timestep: 0.5
  steps: 100000          # 50 ps
quench:
  ensemble: NVT
  T_start: 3000
  T_end: 300
  T_step: -100
  timestep: 0.5
  rate: 100              # K/ps; 27 ps cooling per snapshot
eq_low:
  ensemble: NVT
  T: 300
  timestep: 0.5
  steps: 20000           # 10 ps per snapshot
final_opt:
  cell_filter: none
  fmax: 0.01
  max_steps: 1000
```

```bash
# Replace the example model with the calculator validated for your system.
amorphgen input.xyz --mq-ensemble --n-structures 10 --select decorrelated \
    --config protocol.yaml --model chgnet -o pilot_100Kps/
```

The example inherits the NVT Langevin friction **0.01 fs⁻¹** (100 fs damping
time). With 10 selected snapshots, it requests **457 ps** of aggregate MD:
87 ps shared preparation plus 10 × 37 ps per-snapshot cooling/equilibration;
optimisations add cost. Fewer selected snapshots reduce that total. Use the
same configuration with `--hybrid-ensemble --input-dir ...` for stages 4–7
on random-gen inputs; stages 1–3 are skipped. For every convergence variant,
save a separate YAML and output directory: `--resume` is for continuing the
same protocol, not changing it.

Record the composition, atom count and cell, density constraint, calculator
version/checkpoint, timestep, ensembles and coupling parameters, all stage
temperatures/durations, cooling rate and segment size, seeds, selected and
effective sample counts, relaxation force threshold and achieved convergence,
and analysis settings with uncertainty. Keep the YAML, `run_manifest.json`,
sampling reports and trajectories with the results.

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
