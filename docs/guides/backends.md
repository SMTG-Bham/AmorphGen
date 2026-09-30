# Calculator backends

AmorphGen supports three machine-learning interatomic potential (MLIP)
backends and two classical pair potentials through one calculator factory.

The Python examples on this page use:

```python
from amorphgen.utils import get_calculator
```

```{note}
The factory delegates MLIP calculations to the upstream ASE calculator. It
does not convert energies, forces or stresses, change the stress sign, or
change `atoms.pbc`. ASE units are eV for energy, eV/Å for forces and eV/Å³ for
stress when the calculator supplies it. Results depend on the selected model
and backend version.
```

## MLIP backends

### MACE

The default backend. Supported model names include `mace-mpa-0`,
`mace-mp-0b2-medium`, and `mace-omat-0-medium`. Run `amorphgen --list-models`
for the full registry; available sizes depend on the model family.

```python
calc = get_calculator(model="mace-mpa-0", device="auto")
```

`device="auto"` picks CUDA → MPS → CPU automatically. On Apple Silicon,
MACE and SevenNet default to float64, which MPS cannot represent; use
`device="cpu"` (CLI: `--device cpu`) with those defaults. CHGNet uses float32
and has an MPS loading path. See the
[installation guide](../getting-started/installation.md#backend-compatibility).

Install: `pip install "amorphgen[mace]"`

### CHGNet

Crystal Hamiltonian Graph Neural Network. Select `chgnet` to load its
pre-trained model.

```python
calc = get_calculator(model="chgnet")
```

Install: `pip install "amorphgen[chgnet]"`

CHGNet uses `float32`. AmorphGen rejects `default_dtype="float64"` because
the model's composition features remain in float32 and would cause a dtype
mismatch. Leave the dtype at `auto` or set it to `float32`.

Backend speed depends on the model, system size, composition and hardware.
Measure a short run of your own system before choosing a production protocol.

### SevenNet

Equivariant graph neural network. Foundation models pre-trained on Materials Project / OMat / Alexandria. Multi-fidelity (`mf`) variants combine multiple DFT datasets.

```python
calc = get_calculator(model="7net-mf-ompa", device="auto")  # ★ recommended default
calc = get_calculator(model="sevennet")                     # alias for 7net-mf-ompa
calc = get_calculator(model="7net-l3i5")                    # lighter, single-fidelity
calc = get_calculator(model="7net-omat")                    # OMat-trained
```

For multi-fidelity models (`7net-mf-*`), AmorphGen defaults to `modal="mpa"`.
For the ASE engine, pass another modality supported by your checkpoint through
`get_calculator(..., modal="omat24")`.

Install: `pip install "amorphgen[sevennet]"`

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
pip install "amorphgen[sevennet,chgnet]"
```

The `[full]` extra installs the three MLIP backends in one environment;
it does not include the optional torch-sim engine. Loading MACE foundation
models can then fail unless the installed `mace-torch` release supports
e3nn 0.5+. CHGNet is unaffected (no e3nn dependency).
:::

### Custom / fine-tuned models

```python
calc = get_calculator(model="mace", model_path="/path/to/finetuned.model")
```

## Classical potentials

Built-in pair potentials for initial structure preparation. CPU calculations
need no extra dependencies; optional CUDA acceleration requires PyTorch.
Parameters are provided via YAML config or the `classical_params` keyword.
Set `device="cpu"` explicitly to use the NumPy implementation.

The built-in ASE classical calculators supply energies and forces, but no
stress tensor. Use `cell_filter: none` for relaxation and NVT for every MD
stage; NPT and cell relaxation require stress. The torch-sim Lennard-Jones
model supports stress and cell relaxation.

### Lennard-Jones

Standard 12-6 pair potential. V(r) = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6].

```python
calc = get_calculator("lennard-jones", device="cpu", classical_params={
    "params": {("Ar", "Ar"): {"epsilon": 0.0104, "sigma": 3.40}},
    "cutoff": 10.0,
})
```

### Buckingham + Coulomb

Buckingham short-range potential with an optional Coulomb term and a rigid-ion
model (no core-shell). The short-range term is
`A * exp(-r / rho) - C / r^6`. Ewald summation is the default for electrostatics;
`coulomb_method: wolf` selects an approximate damped-shifted Wolf sum.
The real-space cutoff and charge parameters affect the result, so validate
them for the system and conditions you plan to simulate.

```python
calc = get_calculator("buckingham", device="cpu", classical_params={
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

See the bundled
[`example_classical.yaml`](https://github.com/SMTG-Bham/AmorphGen/blob/main/amorphgen/configs/example_classical.yaml)
for a complete example.

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
pip install "amorphgen[mace,torchsim]"   # Python 3.12+; C/C++ compiler required

amorphgen --batch-opt --input-dir random_structures/random_initial/ \
    -m mace-mpa-0 --engine torchsim -o relaxed/
amorphgen --random-gen --composition "GeO2*192" -n 20 --relax \
    -m mace-mpa-0 --engine torchsim -o geo2_ensemble/
```

or `engine: torchsim` in the YAML. In `--batch-opt` and `--random-gen --relax`,
structures are relaxed in batches. Select the optimiser with `-O` (LBFGS by
default). The output layout remains compatible with the ASE engine, so
`--analyse` can read the results. Hybrid MD also supports torch-sim, as
described below.

The engine needs a C/C++ compiler wherever it runs, and pip does not install one:
torch-sim's neighbour list goes through `torch.compile`, and without a compiler
the first relaxation stops with `InvalidCxxCompiler`. See
[the installation page](../getting-started/installation.md#the-torch-sim-engine)
for how to get one.

What carries over: `-f/--fmax`, `--opt-steps`, `-O` (LBFGS by default, or FIRE
and BFGS) and the cell filter (`cubic` maps
to torch-sim's unit-cell filter with hydrostatic strain, `FrechetCellFilter` to
its Frechet filter, `none` to fixed cell). Gradient descent is available through
YAML/Python as `optimizer: gradient_descent`; ASE-only choices such as `MDMin`
and `BFGSLineSearch` are rejected by this engine. Supported models: MACE foundation
models and `.model` files, SevenNet checkpoints, and Lennard-Jones with a
single sigma/epsilon pair. For mixtures with different pair parameters, use
ASE. CHGNet and Buckingham+Coulomb have no torch-sim implementation and
raise a clear error; use the ASE engine for those.

With a cell filter, convergence also requires the absolute mean pressure to
be below `pressure_tol_gpa` (0.02 GPa by default, settable under `opt:`).
Reaching the step limit can still leave a structure unconverged. A `.cif`
convenience copy is written next to each relaxed `.xyz` in `--batch-opt` and
`--random-gen --relax`, as in the ASE path. Results
are not bit-identical to the ASE path (different optimiser implementations) and
may land in different local minima, as any two optimisers do on a random start. The single-structure melt-quench
pipeline uses ASE; batched hybrid MD is described below.


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
  chunk size). An `auto` chunk size is written to
  `quench_runs/batch_size.json` under the hybrid work directory and read back
  on `--resume` instead of probing again.
- Runs are processed in chunks of `--batch-size` structures. The default,
  `auto`, probes the largest of the first four unfinished structures on the
  GPU and estimates a chunk size using half of the card's memory as its
  budget. Memory use depends on atom count, density, model and precision;
  reduce the chunk size if the estimate is too large. On CPU `auto` means 16.
- The `seed` seeds torch-sim's state generator per stage, chunk, resume block
  and job index. Reproducibility requires the same inputs, chunking and
  resume pattern. Each chunk draws its own velocities and thermostat noise;
  different SLURM array task IDs get separate streams even with the same
  `--seed`. Without a seed, each batch gets fresh entropy.
- SevenNet runs in float32 on this engine (its torch-sim wrapper accepts no
  other precision); multi-fidelity checkpoints get `modal="mpa"` as on the ASE
  path.
- The hybrid mode switches the MD stages to NVT when no ensemble was chosen
  (the pipeline default for stage 4 is NPT) and refuses an explicit NPT before
  starting.
- The thermostat is torch-sim's Langevin (same friction, `0.01/fs` by default,
  as ASE's), with a different integration scheme. Do not expect identical
  trajectories; compare equilibrated observables when validating a protocol.

GPU tests for both engines live in `test/test_torchsim_gpu.py` (skipped without
CUDA); its MACE checks also require `--run-mace`, which permits model downloads.
`examples/run_gpu_tests_bluebear.slurm` runs them, plus the Tier 3 MACE
integration tests, on one BlueBEAR GPU. Adapt its environment settings using
{doc}`hpc`.
