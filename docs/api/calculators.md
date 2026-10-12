# Calculators

The calculator module provides a unified interface to MLIP, classical, ACE and
LAMMPS backends.

## Calculator factory

```{eval-rst}
.. autofunction:: amorphgen.utils.calculators.get_calculator
```

```{eval-rst}
.. autofunction:: amorphgen.utils.calculators.list_models
```

## Backend detection

```{eval-rst}
.. autofunction:: amorphgen.utils.calculators.backend_for
.. autofunction:: amorphgen.utils.calculators.backend_available
.. autofunction:: amorphgen.utils.calculators.available_backends
.. autofunction:: amorphgen.utils.calculators.require_backend
.. autofunction:: amorphgen.utils.calculators.require_potential
```

## MACE models

The following MACE foundation model names are registered by AmorphGen.
Install the `mace` extra to use them; model weights may download on first use.
Slash-separated sizes below represent separate keys (for example,
`mace-mp-0b3-small`, `mace-mp-0b3-medium`, and `mace-mp-0b3-large`).

| Key | Variant |
|-----|---------|
| `mace-mp-0a-small/medium/large` | MP-0a (initial release) |
| `mace-mp-0b-small/medium/large` | MP-0b (improved pair repulsion) |
| `mace-mp-0b2-small/medium/large` | MP-0b2 (high-pressure stability) |
| `mace-mp-0b3-small/medium/large` | MP-0b3 (fixed phonons) |
| `mace-mpa-0`, `mace-mpa-0-medium` | MPA-0; AmorphGen default |
| `mace-omat-0-small/medium`, `mace-omat-0` | OMAT-0; unsuffixed alias selects medium |
| `mace-matpes-pbe`, `mace-matpes-r2scan` | MATPES models |
| `mace-mh-0`, `mace-mh-1` | Multi-domain models |
| `mace-omol` | Molecular model |

## CHGNet

`get_calculator("chgnet", device="auto")` loads CHGNet. Install the `chgnet`
extra; CHGNet supports `float32` only, which `default_dtype="auto"` selects.

## SevenNet models

| Key | Variant |
|-----|---------|
| `sevennet`, `sevennet-mf`, `7net-mf-ompa` | Multi-fidelity foundation (OMat+MPtrj+Alexandria); SevenNet default |
| `7net-mf-0` | Multi-fidelity baseline |
| `7net-omat` | OMat-only |
| `7net-l3i5` | Improved equivariant features |
| `7net-0` | Original release (Jul 2024) |
| `7net-omni` | Multi-task |

Multi-fidelity (`mf`) variants accept a `modal` kwarg (`'mpa'` default, or `'omat24'`):

```python
from amorphgen.utils.calculators import get_calculator

calc = get_calculator("7net-mf-ompa", device="auto")  # MPtrj+Alexandria modality
calc = get_calculator("7net-mf-ompa", device="auto", modal="omat24")  # OMat modality
```

Install the `sevennet` extra in its own environment because its e3nn
requirement conflicts with the MACE extra. Use `amorphgen --list-models` or
`list_models()` for the registered models and backend installation status.

## Classical potentials

Built-in pair potentials for initial structure preparation. No extra install needed.

These calculators provide energy and forces, but no stress tensor. Use
`cell_filter: none` for optimisation and `ensemble: NVT` for MD; cell
relaxation and NPT require a calculator that provides stress. CPU execution
uses NumPy; GPU execution requires PyTorch.

| Model name | Potential | Parameters required |
|------------|-----------|-------------------|
| `lennard-jones` / `lj` | 4*eps*[(sig/r)^12 - (sig/r)^6] | `epsilon`, `sigma` per pair |
| `buckingham` / `buck` | A*exp(-r/rho) - C/r^6 + Coulomb | `A`, `rho`, `C` per pair + charges |

Parameters are passed via `classical_params` in YAML config or Python API:

```python
from amorphgen.utils.calculators import get_calculator

calc = get_calculator("buckingham", classical_params={
    "params": {("Si", "O"): {"A": 18003.76, "rho": 0.2052, "C": 133.54}},
    "charges": {"Si": 2.4, "O": -1.2},
    "cutoff": 10.0,
})
```

```{eval-rst}
.. autoclass:: amorphgen.utils.classical.LennardJonesCalculator
.. autoclass:: amorphgen.utils.classical.BuckinghamCalculator
```

## ACE and LAMMPS potential files

| Model name | Potential | Settings |
|------------|-----------|----------|
| `ace` | ACE potential file via pyace's `PyACECalculator` | `model_path` (`.yaml` / `.yace` / `.ace`, which alone selects ACE); optional `ace_params` |
| `lammps` | any LAMMPS pair style via ASE's `LAMMPSlib` | `lammps_params` (`pair_style`, `pair_coeff`, optional `elements`, `commands`, `masses`, `lammps_header`, `amendments`, `log_file`) |

Both run on the CPU in float64 and supply stress; neither runs on the torch-sim
engine. Install the `ace` or `lammps` extra. See {doc}`../guides/backends` for
the LAMMPS type-order rules.

```python
from amorphgen.utils.calculators import get_calculator

calc = get_calculator(model_path="output_potential.yaml")
calc = get_calculator("lammps", lammps_params={
    "pair_style": "tersoff", "pair_coeff": "* * SiC.tersoff Si C"})
```

```{eval-rst}
.. autofunction:: amorphgen.utils.lammps_potential.lammps_setup
.. autoclass:: amorphgen.utils.lammps_potential.LAMMPSCalculator
```

## Deprecated aliases

```{eval-rst}
.. autofunction:: amorphgen.utils.calculators.get_mace_calculator
```
