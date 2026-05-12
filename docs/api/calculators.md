# Calculators

The calculator module provides a unified interface to MLIP and classical backends.

## Calculator factory

```{eval-rst}
.. autofunction:: amorphgen.utils.calculators.get_calculator
```

```{eval-rst}
.. autofunction:: amorphgen.utils.calculators.list_models
```

## Backend detection

```{eval-rst}
.. autofunction:: amorphgen.utils.calculators._detect_backend
```

## MACE models

The following MACE foundation models are available via `get_calculator()`:

| Key | Variant |
|-----|---------|
| `mace-mp-0a-small/medium/large` | MP-0a (initial release) |
| `mace-mp-0b-small/medium/large` | MP-0b (improved pair repulsion) |
| `mace-mp-0b2-small/medium/large` | MP-0b2 (high-pressure stability) |
| `mace-mp-0b3-small/medium/large` | MP-0b3 (fixed phonons) |
| `mace-mpa-0` | Latest MPA model |

## SevenNet models

| Key | Variant |
|-----|---------|
| `sevennet`, `7net-mf-ompa` | Multi-fidelity foundation (OMat+MPtrj+Alexandria) — recommended |
| `7net-mf-0` | Multi-fidelity baseline |
| `7net-omat` | OMat-only |
| `7net-l3i5` | Improved equivariant features |
| `7net-0` | Original release (Jul 2024) |
| `7net-omni` | Multi-task |

Multi-fidelity (`mf`) variants accept a `modal` kwarg (`'mpa'` default, or `'omat24'`):

```python
calc = get_calculator("7net-mf-ompa", device="auto")              # PBE
calc = get_calculator("7net-mf-ompa", device="auto", modal="omat24")  # PBE+U
```

Use `amorphgen --list-models` or `list_models()` for the complete list.

## Classical potentials

Built-in pair potentials for initial structure preparation. No extra install needed.

| Model name | Potential | Parameters required |
|------------|-----------|-------------------|
| `lennard-jones` / `lj` | 4*eps*[(sig/r)^12 - (sig/r)^6] | `epsilon`, `sigma` per pair |
| `buckingham` / `buck` | A*exp(-r/rho) - C/r^6 + Coulomb | `A`, `rho`, `C` per pair + charges |

Parameters are passed via `classical_params` in YAML config or Python API:

```python
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

## Deprecated aliases

```{eval-rst}
.. autofunction:: amorphgen.utils.calculators.get_mace_calculator
```
