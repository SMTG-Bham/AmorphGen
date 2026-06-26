# YAML configuration

AmorphGen accepts a YAML config file via `--config <file>` (CLI) or `cfg_override=load_yaml_config(...)` (Python). YAML is recommended for any non-trivial workflow because it:

- Keeps simulation parameters in version control alongside the code that produced them
- Makes runs reproducible — one file describes the entire protocol
- Reads cleanly compared to a long CLI flag chain
- Supports comments to document why each parameter is set

## Configuration precedence

```
CLI flags  >  YAML config  >  DEFAULT_CONFIG
```

Anything passed on the CLI overrides the YAML; YAML overrides defaults. So you can keep a baseline YAML and tweak one parameter at the command line:

```bash
amorphgen POSCAR --config full_pipeline.yaml --device cuda --eq-high-T 4000
```

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

Only the keys you want to override need to be present — anything you omit falls back to the default.

## Example: full melt-quench pipeline

```yaml
# examples/full_pipeline.yaml
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
  steps: 10000        # 10 ps at 1 fs
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
  ensemble: NPT       # NEW default — was NVT
  npt_method: mtk     # Nose-Hoover-chain canonical fluctuations
  T: 3000
  steps: 50000        # 50 ps
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
  steps: 10000        # 10 ps
  timestep: 0.5
```

Run with:

```bash
amorphgen POSCAR --config full_pipeline.yaml -o my_run/
```

## Example: hybrid (random + quench)

Skip stages 1–3 (already disordered starting structure), anneal at high T, quench:

```yaml
# examples/hybrid_airss_mq.yaml
model: chgnet
device: cuda
default_dtype: float64

eq_high:
  ensemble: NVT
  T: 3000
  steps: 20000        # 20 ps anneal
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
    --config hybrid_airss_mq.yaml --batch-stages 4 5 6 7 \
    -o hybrid_runs/
```

## Example: validation reference YAML

For `--analyse --reference`, write a structured reference of expected literature ranges. This adds a match/concern/fail validation table to the analysis output.

```yaml
# examples/reference_a_Ga2O3.yaml
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

Each metric is reported as **match** (within range), **concern** (within ~5% of either bound), or **fail** (outside range). Fast, defensible answer to "do my structures agree with the literature?"

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
  alpha: 0.2          # Wolf-summation damping (1/A)
  coulomb: true

opt:
  fmax: 0.05
  optimizer: FIRE
  cell_filter: none
```

## Loading YAML in Python

```python
from amorphgen.configs import load_yaml_config
from amorphgen import MeltQuenchPipeline

cfg = load_yaml_config("full_pipeline.yaml")
pipe = MeltQuenchPipeline("POSCAR", cfg_override=cfg)
pipe.run()
```

`load_yaml_config()` validates the YAML and emits warnings for unknown keys.

## Stage-1 vs stage-7 optimisation: the `final_opt` fallback

Both Stage 1 (initial crystal opt) and Stage 7 (final amorphous opt) use the structure-optimiser code. By default, both read from the same `opt:` block. If you want them to differ (e.g. a tighter `fmax` for the final amorphous structure, or `FrechetCellFilter` for full cell relaxation while Stage 1 keeps the cell fixed), add a separate `final_opt:` block:

```yaml
# Stage 1 — initial crystal opt
opt:
  fmax: 0.05
  max_steps: 200
  optimizer: LBFGS
  cell_filter: none           # fixed cell for the crystal

# Stage 7 — final amorphous opt (overrides only the keys you specify)
final_opt:
  fmax: 0.01                  # tighter convergence
  max_steps: 500
  optimizer: LBFGS
  cell_filter: FrechetCellFilter   # full cell relax for accurate density
```

When `final_opt:` is absent, Stage 7 silently falls back to `opt:`. This is fine for many workflows but worth knowing if you're producing publication-quality structures.

## Selecting an NPT integrator

Any stage with `ensemble: NPT` accepts a `npt_method:` key that picks among three ASE integrators:

| `npt_method` | ASE class | When to use it |
|---|---|---|
| `berendsen` *(default)* | `NPTBerendsen` | Robust during the 300 → 3000 K heating ramp. Averages are correct; fluctuation-derived quantities (heat capacity, isothermal compressibility) are not. |
| `mtk` | `IsotropicMTKNPT` | Martyna-Tobias-Klein Nose-Hoover-chain NPT. **True canonical fluctuations.** Default for `eq_high`. May become unstable during rapid temperature ramps. |
| `parrinello-rahman` | `MelchionnaNPT` | Nose-Hoover + Parrinello-Rahman **flexible cell** (volume *and* shape evolve). Useful for anisotropic glasses. Requires upper-triangular cell — ASE will raise otherwise. |

Recommended pattern: Berendsen on the heating and cooling ramps (stages 3, 5), MTK on the equilibration plateaux (stages 2, 4, 6). The current default config follows this for stages 3 and 4.

### Tuning the Berendsen barostat

If the melt ramp produces volume excursions that are too large, two knobs tighten the response without changing the integrator:

| Key | Default | Effect |
|---|---|---|
| `taup_factor` | `10.0` | Ratio `taup / ttime`. Larger → slower, more stable barostat. Also applied to MTK `pdamp`. |
| `compressibility_GPa` | `100.0` | Reference compressibility for Berendsen (`1/(compressibility_GPa × GPa)`). Default is liquid-like; oxides with bulk modulus 150–300 GPa benefit from 200 GPa. |

**Example: oxide-tuned melt ramp**

```yaml
melt:
  ensemble: NPT
  npt_method: berendsen
  taup_factor: 30.0              # slower barostat (3× default)
  compressibility_GPa: 200.0     # stiffer, oxide-realistic
```

On a Cu/EMT benchmark at 1500 K over 300 fs, these settings reduce the maximum volume excursion by ~74 % compared with the defaults.

**Example: restore the legacy NVT plateau**

The new default for `eq_high` is NPT/MTK. To revert to constant-volume behaviour:

```yaml
eq_high:
  ensemble: NVT
  T: 3000
```

## Tips

- **Keep YAMLs in version control.** They're tiny and document your protocol.
- **Mix YAML + CLI** for parameter sweeps: a baseline YAML, with the swept variable on the CLI:
  ```bash
  for rate in 50 100 200; do
      amorphgen POSCAR --config baseline.yaml --quench-rate $rate -o run_${rate}Kps/
  done
  ```
- **Comment liberally** — `# ...` after any value explains *why* you chose it. Reviewers and future you will thank you.
- **Pre-built examples** ship in `examples/`: `full_pipeline.yaml`, `hybrid_airss_mq.yaml`, `fast_test.yaml`, `reference_a_Ga2O3.yaml`.
