# YAML configuration

AmorphGen accepts a YAML config file via `--config <file>` (CLI) or `cfg_override=load_yaml_config(...)` (Python). YAML is recommended for any non-trivial workflow because it:

- Keeps simulation parameters in version control alongside the code that produced them
- Records the protocol; combine it with saved inputs, a seed and software/model versions for reproducibility
- Reads cleanly compared to a long CLI flag chain
- Supports comments to document why each parameter is set

## Configuration precedence

```
CLI flags  >  YAML config  >  DEFAULT_CONFIG
```

Anything passed on the CLI overrides the YAML; YAML overrides defaults. So you can keep a baseline YAML and tweak one parameter at the command line:

```bash
amorphgen POSCAR --config full_pipeline.yaml --device cuda \
    --melt-T-end 4000 --eq-high-T 4000 --quench-T-start 4000
```

Stage temperatures are independent: changing the melt endpoint does not automatically update the high-temperature plateau or quench start. The example above changes all three.

## Structure

A YAML config is a nested dictionary. Top-level keys map to calculator settings and stage names; stage values are themselves dictionaries:

```yaml
model: mace-mpa-0
device: cuda
default_dtype: float64

opt:
  fmax: 0.05
  optimizer: LBFGS
  cell_filter: FrechetCellFilter

eq_premelt:
  ensemble: NVT
  T: 300
  steps: 5000
  timestep: 0.5

melt:
  ensemble: NPT
  T_start: 300
  T_end: 3000
  rate: 100        # K/ps
```

Only the keys you want to override need to be present; anything you omit falls back to the default.

## Example: full melt-quench pipeline

```yaml
# Save as full_pipeline.yaml
model: mace-mpa-0
device: cuda
default_dtype: float64

opt:
  fmax: 0.05
  max_steps: 500
  optimizer: LBFGS
  cell_filter: none

eq_premelt:
  ensemble: NVT
  T: 300
  steps: 10000        # 5 ps at 0.5 fs
  timestep: 0.5

melt:
  ensemble: NPT
  npt_method: berendsen  # robust during the 300→3000 K ramp
  T_start: 300
  T_end: 3000
  T_step: 100
  rate: 100           # K/ps
  timestep: 0.5

eq_high:
  ensemble: NPT       # default high-temperature ensemble
  npt_method: mtk     # Nose-Hoover-chain NPT sampling
  T: 3000
  steps: 50000        # 25 ps at 0.5 fs
  timestep: 0.5

quench:
  ensemble: NVT
  T_start: 3000
  T_end: 300
  rate: 100
  timestep: 0.5

eq_low:
  ensemble: NVT
  T: 300
  steps: 10000        # 5 ps at 0.5 fs
  timestep: 0.5
```

Run with:

```bash
amorphgen POSCAR --config full_pipeline.yaml -o my_run/
```

## Example: hybrid (random + quench)

Skip stages 1–3 (already disordered starting structure), anneal at high T, quench:

```yaml
# hybrid.yaml
model: chgnet         # default_dtype left at auto (float32, CHGNet's only precision)
device: cuda

eq_high:
  ensemble: NVT
  T: 3000
  steps: 20000        # 10 ps anneal at 0.5 fs
  timestep: 0.5
  friction: 0.01

quench:
  ensemble: NVT
  T_start: 3000
  T_end: 300
  rate: 100
  timestep: 0.5
  friction: 0.01

eq_low:
  ensemble: NVT
  T: 300
  steps: 5000
  timestep: 0.5
  friction: 0.01

opt:
  fmax: 0.05
  optimizer: LBFGS
  cell_filter: cubic
```

Run via batch-quench:

```bash
amorphgen --batch-quench --snapshot-dir random_inputs/ \
    --config hybrid.yaml --batch-stages 4 5 6 7 \
    -o hybrid_runs/
```

A MACE version with a 100 ps anneal ships as `examples/hybrid_airss_mq.yaml`.

## Example: validation reference YAML

For `--analyse --reference`, write a structured reference of expected literature ranges. This adds a match/concern/fail validation table to the analysis output.

```yaml
# Save as reference_a_Ga2O3.yaml; verify ranges against the cited sources
system: a-Ga2O3

references:
  - "Kaewmeechai, Strand & Shluger, Phys. Rev. B 111, 035203 (2025)"
  - "Stehlik et al., J. Non-Cryst. Solids 458 (2017) 14"

density:
  expected: [4.70, 5.10]      # g/cm^3
  units: "g/cm^3"

bond_distances:
  Ga-O:
    expected: [1.85, 1.95]    # Angstrom
    units: "A"

coordination:
  Ga-O:
    mean_expected: [4.0, 4.8]
  O-Ga:
    mean_expected: [2.7, 3.0]

bond_angles:
  Ga-O-Ga:
    expected: [110.0, 130.0]
    units: "deg"
  O-Ga-O:
    expected: [100.0, 115.0]
    units: "deg"
```

Run with:

```bash
amorphgen --analyse --input-dir my_structures/ \
    --cutoff auto-rdf \
    --reference reference_a_Ga2O3.yaml
```

Each metric is reported as **match** (within range), **concern** (within ~5% of either bound), **fail** (outside range), or **n/a** when the structures do not have it (an element that is absent, or no contact within the cutoff). A bond, and the two end atoms of an angle, can be written in either order (`Si-O` or `O-Si`). A coordination entry is directional: `Si-O` counts the O around Si. The table checks the supplied ranges; it does not independently verify their provenance or establish model accuracy.

## Example: classical potential

Buckingham + Coulomb for SiO₂:

```yaml
model: buckingham
device: cpu

classical_params:
  params:
    Si-O: {A: 18003.76, rho: 0.2052, C: 133.54}
    O-O:  {A: 1388.77,  rho: 0.3623, C: 175.0}
  charges: {Si: 2.4, O: -1.2}
  cutoff: 10.0
  coulomb: true
  coulomb_method: ewald  # default; use wolf for the approximate alternative

opt:
  fmax: 0.05
  optimizer: FIRE
  cell_filter: none

# The built-in Buckingham calculator provides no stress tensor.
# Override the two default NPT stages for a full pipeline.
melt:
  ensemble: NVT
eq_high:
  ensemble: NVT
```

The other MD stages default to NVT, and Stage 7 inherits the fixed-cell
`opt:` settings. See {doc}`backends` for parameter and electrostatics options.

## Loading YAML in Python

```python
from amorphgen.configs import load_yaml_config
from amorphgen import MeltQuenchPipeline

cfg = load_yaml_config("full_pipeline.yaml")
pipe = MeltQuenchPipeline("POSCAR", cfg_override=cfg)
pipe.run()
```

`load_yaml_config()` checks known keys and types, prints warnings for unknown keys, and raises `ValueError` for detected validation errors. Unknown keys are retained, so a warning does not establish that a key is used by the selected mode.

## Stage-1 vs stage-7 optimisation: the `final_opt` fallback

Both Stage 1 (initial crystal opt) and Stage 7 (final amorphous opt) use the structure-optimiser code. By default, both read from the same `opt:` block. If you want them to differ (e.g. a tighter `fmax` for the final amorphous structure, or `FrechetCellFilter` for full cell relaxation while Stage 1 keeps the cell fixed), add a separate `final_opt:` block:

```yaml
# Stage 1 - initial crystal opt
opt:
  fmax: 0.05
  max_steps: 200
  optimizer: LBFGS
  cell_filter: none           # fixed cell for the crystal

# Stage 7 - final amorphous opt (overrides only the keys you specify)
final_opt:
  fmax: 0.01                  # tighter convergence
  max_steps: 500
  optimizer: LBFGS
  cell_filter: FrechetCellFilter   # relax cell and positions; validate density
```

Stage 7 inherits `opt:` and applies the individual keys in `final_opt:` on top.
Partial overrides, including a CLI flag such as `--format`, preserve all other
optimisation settings from `opt:`. In 1.0.0rc4, the YAML validator warns that
`final_opt` is an unknown top-level key; the pipeline still applies this block.

## Selecting an NPT integrator

Any stage with `ensemble: NPT` accepts a `npt_method:` key that picks among three ASE integrators:

| `npt_method` | ASE class | When to use it |
|---|---|---|
| `berendsen` *(default)* | `NPTBerendsen` | Robust during the 300 → 3000 K heating ramp. Does not sample the NPT ensemble correctly; avoid deriving thermodynamic response functions from its fluctuations. |
| `mtk` | `IsotropicMTKNPT` | Martyna-Tobias-Klein Nose-Hoover-chain NPT. **NPT ensemble sampling.** Default for `eq_high`. May become unstable during rapid temperature ramps. |
| `parrinello-rahman` | `MelchionnaNPT` | Nose-Hoover + Parrinello-Rahman **flexible cell** (volume *and* shape evolve). Useful for anisotropic glasses. Requires upper-triangular cell, ASE will raise otherwise. |

The defaults are NPT/Berendsen for stage 3 and NPT/MTK for stage 4; stages 2, 5 and 6 use NVT. Set `ensemble: NPT` and `npt_method` explicitly in any other stage where pressure control is intended.

### Tuning the Berendsen barostat

If the melt ramp produces volume excursions that are too large, two knobs tighten the response without changing the integrator:

| Key | Default | Effect |
|---|---|---|
| `taup_factor` | `10.0` | Ratio `taup / ttime`. Larger → slower, more stable barostat. Also applied to MTK `pdamp`. |
| `compressibility_GPa` | `100.0` | Reference bulk modulus in GPa; the Berendsen compressibility is its inverse, `1/(compressibility_GPa × GPa)`. A larger value reduces the volume response. |

**Example: slower volume response during a melt ramp**

```yaml
melt:
  ensemble: NPT
  npt_method: berendsen
  taup_factor: 30.0              # slower barostat (3× default)
  compressibility_GPa: 200.0     # larger reference bulk modulus
```

Inspect the volume trajectory after changing the coupling parameters; a slower response does not establish that the predicted density is correct.

**Example: restore the legacy NVT plateau**

The default for `eq_high` is NPT/MTK. To revert to constant-volume behaviour:

```yaml
eq_high:
  ensemble: NVT
  T: 3000
```

## The `analysis` block

`--analyse` reads its defaults from an `analysis:` block; any CLI flag
overrides the YAML value.

```yaml
analysis:
  cutoff: auto-rdf        # or auto, a number in A, or a dict of per-pair
                          # overrides such as {In-O: 2.6}; unlisted pairs
                          # keep auto-rdf, or the dict's "default" entry
  smearing: 0.05          # RDF Gaussian smearing (A); 0 = raw histogram
  per_structure: true
  check_dimers: true
  save_report: report.txt
  save_plot: plots/
  pair_panels: true       # one panel per pair for g(r) and S_ab(q)
  total_cn: [O, "O:In+Ga"]  # total coordination of a centre over chosen partners
  sq: true                # structure factor
  sq_weighting: neutron   # xray | neutron | unweighted
  sq_method: direct       # direct | ft
  sq_smooth: 0.05         # re-binning width for the direct method (1/A); 0 = raw
  sq_partials: true       # also the Faber-Ziman partials S_ab(q) (direct method)
  tr: true                # total correlation function T(r) = 4 pi r rho g(r)
  tr_qrange: [0.3, 20.0]  # integration limits (1/A); match the experiment
  tr_window: lorch        # lorch | none
  tr_scan: true           # sweep qmax / window and report the spread
  rings: true             # or a nodes-bridge pair such as Ge-O
  voronoi: Ge             # or true for all atoms
  connectivity: true      # corner/edge/face sharing of cation polyhedra
```

## Reproducibility: the `seed` key

A top-level `seed:` (or `--seed INT` on the command line) makes a run
use controlled random-number streams. The same integer seeds the random placement in
`--random-gen` and, in every MD stage, the Maxwell–Boltzmann velocity
initialisation and the Langevin thermostat noise. Each stage and each run
directory (`run_0007/`) gets its own stream derived from the seed, so stage 4 of
run 7 draws the same noise whatever ran before it.

```yaml
seed: 42
```

What this does and does not guarantee:

- The seed controls random-number streams. Identical structures and trajectories
  also require the same inputs, configuration, numerical backend and deterministic
  operations; a seed alone does not guarantee bit-identical MLIP results.
- CPU or GPU numerical differences can grow during MD. Compare ensemble statistics
  across environments rather than expecting individual frames to match.
- A frame-level `--resume` restores saved atomic state and momenta, but not
  the thermostat random-number state or full barostat state. It is not an exact
  continuation of an uninterrupted trajectory.
- NPT stages (Berendsen, MTK) contain no randomness beyond the initial
  velocities.

Without a seed (the default) placement is still reproducible if `random_gen:
seed:` is set, but the MD stages are not.

## Tips

- Keep YAMLs in version control. They're tiny and document your protocol.
- Mix YAML + CLI for parameter sweeps: a baseline YAML, with the swept variable on the CLI. A quench-rate sweep, with the default 100 K per segment at 0.5 fs:
  ```bash
  # 4000 / 2000 / 1000 steps per segment = 50 / 100 / 200 K/ps
  for steps in 4000 2000 1000; do
      amorphgen POSCAR --config baseline.yaml --quench-steps-per-T $steps -o run_${steps}steps/
  done
  ```
  Leave `rate:` out of the `quench:` block of `baseline.yaml` for this: when set, it takes precedence over `steps_per_T`, including a value given on the CLI.
- Comment liberally: `# ...` after any value explains *why* you chose it. Reviewers and future you will thank you.
- Pre-built examples ship in `examples/`: `full_pipeline.yaml`, `hybrid_airss_mq.yaml`, `fast_test.yaml`, `reference_a_Ga2O3.yaml`.
