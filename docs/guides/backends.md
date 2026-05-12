# Calculator backends

AmorphGen supports multiple calculator backends through a unified factory: three machine-learning interatomic potentials (MLIPs) and two classical pair potentials.

```{note}
**ASE convention pass-through.**  `amorphgen.utils.calculators.get_calculator()` is a thin wrapper around each backend's upstream ASE calculator (`MACECalculator`, `CHGNetCalculator`, `SevenNetCalculator`, plus the built-in classical calculators).  AmorphGen does **not** apply any custom unit conversion, stress-sign flip, or PBC override — energies are returned in eV, forces in eV/Å, stress in eV/Å³, and `atoms.pbc` is passed through unchanged.  This means cross-backend numerical consistency is inherited directly from the upstream MLIP package.  If you upgrade `mace-torch`, `chgnet`, or `sevenn` and observe a sudden density / energy shift, check the upstream calculator's release notes for unit-convention changes before assuming an AmorphGen regression.
```

## MLIP backends

### MACE

The default backend. Provides 20+ pre-trained foundation models including `mace-mpa-0`, `mace-mp-0`, and size variants (small, medium, large).

```python
calc = get_calculator(model="mace-mpa-0", device="auto")
```

`device="auto"` picks CUDA → MPS → CPU automatically; pass `"cpu"` / `"cuda"` / `"mps"` explicitly to override.

**Install:** `pip install amorphgen[mace]`

### CHGNet

Crystal Hamiltonian Graph Neural Network. Good balance of speed and accuracy, especially on CPU.

```python
calc = get_calculator(model="chgnet")
```

**Install:** `pip install amorphgen[chgnet]`

**Precision (dtype):** CHGNet is trained and benchmarked at `float32`. AmorphGen's CHGNet loader enforces this — passing `default_dtype="float64"` raises `NotImplementedError` with a clear message pointing the user to MACE, because CHGNet's `composition_model` submodule builds its input feature vectors via a path that bypasses `torch.get_default_dtype()` and crashes at forward time when the rest of the model is upcast. Keeping `float32` (the default) is the recommended path for MD; switch to MACE if you genuinely need `float64` for static-energy precision.

**MD performance heads-up:** CHGNet's `CHGNetCalculator.calculate()` rebuilds the atomic graph (neighbour list + edges + line graph) from scratch on every MD step. For systems above ~200 atoms or with high density (e.g. a-Ga₂O₃ at 400 atoms), the per-step cost on an A100 is around 500 ms — substantially slower than the AdvanceSoft H100 benchmark (≈ 84 ms/step at 400 atoms for Li₁₀GeP₂S₁₂) would predict, mostly because (a) denser oxides have more graph edges per atom and (b) the ASE → pymatgen → graph round-trip carries Python overhead. For large-system MD where speed matters, MACE (which caches neighbour lists internally) is 3–5× faster at the same system size.

### SevenNet

Equivariant graph neural network from KAIST (MDIL-SNU). Foundation models pre-trained on Materials Project / OMat / Alexandria. Multi-fidelity (`mf`) variants combine multiple DFT datasets.

```python
calc = get_calculator(model="7net-mf-ompa", device="auto")  # ★ recommended default
calc = get_calculator(model="sevennet")                     # alias for 7net-mf-ompa
calc = get_calculator(model="7net-l3i5")                    # lighter, single-fidelity
calc = get_calculator(model="7net-omat")                    # OMat-trained
```

For multi-fidelity models (`7net-mf-*`), AmorphGen defaults `modal='mpa'` (MPtrj+Alexandria, PBE). Override with the `modal` kwarg if you want `'omat24'` (PBE+U).

**Install:** `pip install amorphgen[sevennet]` (no DGL dep, works on Mac/Linux)

:::{warning}
**Use a separate environment for SevenNet.** SevenNet depends on `e3nn>=0.5`,
while pre-trained MACE foundation models (`mace-mpa-0`, ...) were pickled
with `e3nn==0.4.x`. e3nn 0.5+ changed the `_codegen` storage format from
2-tuples to 3-tuples; loading a MACE foundation model with the newer e3nn
raises:

    ValueError: too many values to unpack (expected 2)
        in e3nn/util/codegen/_mixin.py:115 (`__setstate__`)

The default `[all]` extra therefore installs MACE + CHGNet (no SevenNet).
To use SevenNet, create a dedicated env:

```bash
conda create -n amorphgen-sevennet python=3.11
conda activate amorphgen-sevennet
pip install -e ".[sevennet,chgnet]"
```

The `[full]` extra installs everything in one env, but loading MACE
foundation models will then fail unless you use a `mace-torch` release
that supports e3nn 0.5+. CHGNet is unaffected (no e3nn dependency).
:::

### Custom / fine-tuned models

```python
calc = get_calculator(model="mace", model_path="/path/to/finetuned.model")
```

## Classical potentials

Built-in pair potentials for initial structure preparation. No extra install or GPU required. Parameters are provided via YAML config or the `classical_params` keyword.

### Lennard-Jones

Standard 12-6 pair potential. V(r) = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6].

```python
calc = get_calculator("lennard-jones", classical_params={
    "params": {("Ar", "Ar"): {"epsilon": 0.0104, "sigma": 3.40}},
    "cutoff": 10.0,
})
```

### Buckingham + Coulomb

Buckingham short-range potential with Wolf summation for long-range Coulomb. V(r) = A*exp(-r/rho) - C/r^6 + q_i*q_j/(4*pi*eps0*r). Rigid-ion model (no core-shell).

```python
calc = get_calculator("buckingham", classical_params={
    "params": {
        ("Si", "O"): {"A": 18003.76, "rho": 0.2052, "C": 133.54},
        ("O", "O"):  {"A": 1388.77,  "rho": 0.3623, "C": 175.0},
    },
    "charges": {"Si": 2.4, "O": -1.2},
    "cutoff": 10.0,
    "alpha": 0.2,      # Wolf damping (1/A)
    "coulomb": True,    # set False for Buckingham-only
})
```

Classical potentials can also be specified via YAML config:

```yaml
model: buckingham
device: cpu
classical_params:
  params:
    Si-O: {A: 18003.76, rho: 0.2052, C: 133.54}
    O-O:  {A: 1388.77,  rho: 0.3623, C: 175.0}
  charges: {Si: 2.4, O: -1.2}
  cutoff: 10.0
opt:
  fmax: 0.05
  optimizer: FIRE
  cell_filter: none
```

See `amorphgen/configs/example_classical.yaml` for a complete example.

## Listing available models

```python
from amorphgen.utils.calculators import list_models
list_models()
```

Or from the CLI:

```bash
amorphgen --list-models
```
