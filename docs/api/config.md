# Configuration

AmorphGen uses a nested dictionary for pipeline configuration. Most defaults are
defined in `default_config.py`; the calculator and stage functions supply the
remaining defaults. Override them with CLI flags, YAML config files, or the
Python API.

## Default configuration

```{eval-rst}
.. automodule:: amorphgen.configs.default_config
   :members:
```

## Configuration keys

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| top-level | `model` | `"mace-mpa-0"` | MLIP model name |
| top-level | `device` | `"auto"` | Compute device (`cpu`, `cuda`, `mps`) |
| top-level | `default_dtype` | `"auto"` | Calculator precision: `float32` for CHGNet; `float64` for MACE and SevenNet |
| top-level | `engine` | `"ase"` | Ensemble engine (`ase` or optional `torchsim`; see {doc}`/guides/backends`) |
| top-level | `seed` | `None` | Seed for random placement and MD random streams |
| `opt` | `fmax` | `0.01` | Force convergence (eV/A) |
| `opt` | `max_steps` | `1000` | Max optimisation steps |
| `opt` | `optimizer` | `"LBFGS"` | ASE optimiser |
| `opt` | `cell_filter` | `"FrechetCellFilter"` | Cell filter: FrechetCellFilter, UnitCellFilter, ExpCellFilter, StrainFilter, cubic, none |
| `melt` | `T_start` | `300` | Start temperature (K) |
| `melt` | `T_end` | `3000` | Melt temperature (K) |
| `melt` | `T_step` | `100` | Temperature step per segment (K) |
| `melt` | `rate` | `None` | Heating rate (K/ps), auto-calculates steps_per_T |
| `melt` | `steps_per_T` | `2000` | MD steps per temperature step |
| `melt` | `timestep` | `0.5` | MD timestep (fs) |
| `melt` | `ensemble` | `"NPT"` | Heating ensemble (Berendsen barostat by default) |
| `melt` | `make_cubic` | `False` | Explicitly reshape the input cell before heating |
| `eq_premelt` | `ensemble` | `"NVT"` | Pre-melt ensemble |
| `eq_premelt` | `T` | `300` | Pre-melt equilibration T (K) |
| `eq_premelt` | `steps` | `100000` | Pre-melt equilibration steps (50 ps) |
| `eq_high` | `T` | `3000` | High-T equilibration T (K) |
| `eq_high` | `steps` | `20000` | High-T equilibration steps (10 ps) |
| `eq_high` | `ensemble` | `"NPT"` | High-T ensemble (MTK barostat by default) |
| `quench` | `T_start` | `3000` | Quench start T (K) |
| `quench` | `T_end` | `300` | Quench end T (K) |
| `quench` | `T_step` | `-100` | Temperature step (K, negative for cooling) |
| `quench` | `steps_per_T` | `2000` | MD steps per temperature step |
| `quench` | `rate` | `None` | Cooling rate (K/ps), auto-calculates steps_per_T |
| `quench` | `ensemble` | `"NVT"` | Cooling ensemble |
| `eq_low` | `T` | `300` | Low-T equilibration T (K) |
| `eq_low` | `steps` | `20000` | Low-T equilibration steps (10 ps) |
| `eq_low` | `ensemble` | `"NVT"` | Low-T ensemble |

All MD stages default to a `0.5` fs timestep. The `opt` section supplies
settings for stages 1 and 7; a partial `final_opt` section overrides individual
settings for stage 7. The CLI defaults to `cell_filter: cubic` for
`--random-gen --relax`, `--batch-opt`, and `--hybrid-ensemble` unless a filter
is selected explicitly. Classical calculators ignore `default_dtype`.

See {doc}`/guides/yaml-config` for the `safety` and `repulsive_core` blocks,
including per-step failure limits and optional reference-model spot checks.

## Configuration precedence

```
CLI arguments > YAML config (--config) > DEFAULT_CONFIG
```

## Override examples

### Python API

```python
from amorphgen import MeltQuenchPipeline

pipe = MeltQuenchPipeline(
    input_file="POSCAR",
    work_dir="my_run",
    cfg_override={
        "model": "mace-mpa-0",
        "device": "cuda",
        "opt": {"fmax": 0.05, "cell_filter": "none"},
        "eq_high": {"T": 3000, "steps": 200000, "timestep": 0.5},
        "quench": {"rate": 100, "timestep": 0.5},
    },
)
atoms = pipe.run(stages=[1, 4, 5, 6, 7], resume=True)
```

### YAML config

```yaml
model: mace-mpa-0
device: cuda

opt:
  fmax: 0.05
  cell_filter: none

eq_high:
  ensemble: NVT
  T: 3000
  steps: 200000
  timestep: 0.5

quench:
  ensemble: NVT
  T_start: 3000
  T_end: 300
  T_step: -100
  rate: 100
  timestep: 0.5
```
