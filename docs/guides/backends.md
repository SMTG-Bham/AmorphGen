# Calculator backends

AmorphGen supports multiple calculator backends through a unified factory: three machine-learning interatomic potentials (MLIPs) and two classical pair potentials.

```{note}
`amorphgen.utils.calculators.get_calculator()` is a thin wrapper around each backend's upstream ASE calculator (`MACECalculator`, `CHGNetCalculator`, `SevenNetCalculator`, plus the built-in classical calculators).  AmorphGen does **not** apply any custom unit conversion, stress-sign flip, or PBC override - energies are returned in eV, forces in eV/Å, stress in eV/Å³, and `atoms.pbc` is passed through unchanged.  This means cross-backend numerical consistency is inherited directly from the upstream MLIP package.  If you upgrade `mace-torch`, `chgnet`, or `sevenn` and observe a sudden density / energy shift, check the upstream calculator's release notes for unit-convention changes before assuming an AmorphGen regression.
```

## MLIP backends

### MACE

The default backend. Provides 20+ pre-trained foundation models including `mace-mpa-0`, `mace-mp-0`, and size variants (small, medium, large).

```python
calc = get_calculator(model="mace-mpa-0", device="auto")
```

`device="auto"` picks CUDA → MPS → CPU automatically; pass `"cpu"` / `"cuda"` / `"mps"` explicitly to override.

Install: `pip install amorphgen[mace]`

### CHGNet

Crystal Hamiltonian Graph Neural Network. Good balance of speed and accuracy, especially on CPU.

```python
calc = get_calculator(model="chgnet")
```

Install: `pip install amorphgen[chgnet]`

Precision: CHGNet is trained and benchmarked at `float32`. AmorphGen's CHGNet loader enforces this, passing `default_dtype="float64"` raises `NotImplementedError` with a clear message pointing the user to MACE, because CHGNet's `composition_model` submodule builds its input feature vectors via a path that bypasses `torch.get_default_dtype()` and crashes at forward time when the rest of the model is upcast. Keeping `float32` (the default) is the recommended path for MD; switch to MACE if you genuinely need `float64` for static-energy precision.

A note on MD speed: CHGNet's `CHGNetCalculator.calculate()` rebuilds the atomic graph (neighbour list + edges + line graph) from scratch on every MD step. For systems above ~200 atoms or with high density (e.g. a-Ga₂O₃ at 400 atoms), the per-step cost on an A100 is around 500 ms, substantially slower than the AdvanceSoft H100 benchmark (≈ 84 ms/step at 400 atoms for Li₁₀GeP₂S₁₂) would predict, mostly because (a) denser oxides have more graph edges per atom and (b) the ASE → pymatgen → graph round-trip carries Python overhead. For large-system MD where speed matters, MACE (which caches neighbour lists internally) is 3–5× faster at the same system size.

### SevenNet

Equivariant graph neural network from KAIST (MDIL-SNU). Foundation models pre-trained on Materials Project / OMat / Alexandria. Multi-fidelity (`mf`) variants combine multiple DFT datasets.

```python
calc = get_calculator(model="7net-mf-ompa", device="auto")  # ★ recommended default
calc = get_calculator(model="sevennet")                     # alias for 7net-mf-ompa
calc = get_calculator(model="7net-l3i5")                    # lighter, single-fidelity
calc = get_calculator(model="7net-omat")                    # OMat-trained
```

For multi-fidelity models (`7net-mf-*`), AmorphGen defaults `modal='mpa'` (MPtrj+Alexandria, PBE). Override with the `modal` kwarg if you want `'omat24'` (PBE+U).

Install: `pip install amorphgen[sevennet]` (no DGL dep, works on Mac/Linux)

:::{warning}
Use a separate environment for SevenNet: it depends on `e3nn>=0.5`,
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

Buckingham short-range potential with Ewald summation for the long-range Coulomb term. V(r) = A*exp(-r/rho) - C/r^6 + q_i*q_j/(4*pi*eps0*r). Rigid-ion model (no core-shell). The Ewald real-space part runs over the pair cutoff with alpha = 3.5/cutoff, the reciprocal-space part is summed in NumPy; energies match a reference Ewald to 0.02 meV/atom. The damped-shifted Wolf sum is still available as `coulomb_method: wolf`, but note it is only approximate (about 10 % force error for an ionic melt at a 10 A cutoff), so use it for speed comparisons rather than production runs.

```python
calc = get_calculator("buckingham", classical_params={
    "params": {
        ("Si", "O"): {"A": 18003.76, "rho": 0.2052, "C": 133.54},
        ("O", "O"):  {"A": 1388.77,  "rho": 0.3623, "C": 175.0},
    },
    "charges": {"Si": 2.4, "O": -1.2},
    "cutoff": 10.0,
    "coulomb": True,           # set False for Buckingham-only
    "coulomb_method": "ewald", # or "wolf" (approximate, faster)
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


## Batched relaxation with torch-sim (optional)

[torch-sim](https://github.com/torchsim/torch-sim) relaxes many structures in one
batched MLIP call with automatic GPU memory management. AmorphGen can hand the
ensemble modes to it:

```bash
pip install "amorphgen[torchsim]"          # Python >= 3.12; CUDA GPU or CPU (no Apple MPS)

amorphgen --batch-opt --input-dir random_structures/random_initial/ \
    -m mace-mpa-0 --engine torchsim -o relaxed/
amorphgen --random-gen --composition "GeO2*192" -n 20 --relax \
    -m mace-mpa-0 --engine torchsim -o geo2_ensemble/
```

or `engine: torchsim` in the YAML. Only `--batch-opt` and `--random-gen --relax`
use it: all structures are relaxed together with torch-sim's FIRE optimiser
instead of one after another through ASE. Output files, names and logs are the
same as with the ASE engine, so `--analyse` and everything downstream is
unchanged.

What carries over: `-f/--fmax`, `--opt-steps`, `-O` (LBFGS by default, or FIRE, BFGS,
gradient descent) and the cell filter (`cubic` maps
to torch-sim's unit-cell filter with hydrostatic strain, `FrechetCellFilter` to
its Frechet filter, `none` to fixed cell). Supported models: MACE foundation
models and `.model` files, SevenNet checkpoints, Lennard-Jones (single
sigma/epsilon; used by the tests). CHGNet and Buckingham+Coulomb have no
torch-sim implementation and raise a clear error; use the ASE engine for those.

With a cell filter the convergence test also requires the pressure to be below
`pressure_tol_gpa` (0.02 GPa by default, settable under `opt:`), so the returned
cells are at zero pressure like the ASE path. torch-sim's own cell-force criterion
is loose for cells of hundreds of atoms and left residuals of 0.1 to 0.3 GPa. The
`.cif` convenience copy is written next to the `.xyz`, as in the ASE path. Results
are not bit-identical to the ASE path (different optimiser implementations) and
may land in different local minima, as any two optimisers do on a random start. The single-structure melt-quench
pipeline always uses ASE. The MD stages of the hybrid mode are not batched yet.


### Batched MD for the hybrid workflow

`--hybrid-ensemble --engine torchsim` runs stages 4 to 7 for all input
structures together: one batched NVT-Langevin integration for the
high-temperature stage, the quench ramp (segment temperatures as a per-step
schedule) and the low-temperature stage, then the batched relaxation. Every run
still gets its own `run_NNNN/` directory with `stage4_eq.log`,
`stage4_eq_traj.xyz` (a frame every 100 steps, with momenta), and so on, and
the results are collected into `final/` as usual, so `--analyse` and the
downstream tools see no difference.

What differs from the ASE path:

- NVT only. torch-sim's NPT is Langevin-based and is not mapped; a YAML with an
  NPT stage raises a clear error.
- Resume works at run level and at frame level. Runs whose
  `final_amorphous.xyz` exists are skipped. For a chunk that was killed inside
  an MD stage, `--resume` finds the last frame that every run of the chunk has
  reached, cuts any run that got further back to that frame, and continues the
  stage from there with the momenta stored in the trajectory, so a walltime
  kill costs at most 100 steps per stage. Stages already finished for the whole
  chunk are skipped. The chunking must be the same on resume (same inputs and
  chunk size). An `auto` chunk size is therefore written to `batch_size.json`
  in the work directory and read back on `--resume` instead of probing again,
  so a resubmitted job script re-chunks identically.
- Runs are processed in chunks of `--batch-size` structures. The default,
  `auto`, integrates a short probe of the first structure on the GPU, reads
  the peak memory it needed, and sizes the chunk to use about half of the
  card, so the chunk follows the cell size and the model. Give an integer to
  fix it (for 600-atom cells in float64 on a 40 GB card, 4 to 5 is the
  practical limit); on CPU `auto` means 16.
- The `seed` seeds torch's generator per stage, so a batch is reproducible for
  the same set of inputs and chunking; the thermostat noise stream is shared
  across the runs of a chunk.
- The thermostat is torch-sim's Langevin (same friction, `0.01/fs` by default,
  as ASE's) but not the same integrator step, so trajectories are statistically
  equivalent to the ASE path, not identical.

GPU tests for both engines live in `test/test_torchsim_gpu.py` (skipped without
CUDA); `examples/run_gpu_tests_bluebear.slurm` runs them, plus the Tier 3 MACE
integration tests, on one BlueBEAR GPU in about ten minutes.
