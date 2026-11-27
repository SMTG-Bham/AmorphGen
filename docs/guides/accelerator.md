# Accelerating MACE with Symmetrix-XL

`--accelerator` evaluates a MACE model with
[Symmetrix-XL](https://pypi.org/project/symmetrix-xl/) instead of PyTorch. The
model is the same, so the structures are the same; only the evaluation is
faster. On an NVIDIA A100 a force call is about 10-25 times faster than
`mace-torch`, and the forces agree with MACE to a few thousandths of a meV/Å.

It is optional. Without the package, and with the default `--accelerator none`,
nothing changes.

## Install

```bash
pip install "amorphgen[symmetrix-xl]"            # Linux x86-64
pip install "amorphgen[symmetrix-xl,torchsim]"   # with batched ensembles (Python 3.12+)
```

Then install the Symmetrix-XL GPU backend that matches your card. It is a
separate package because it depends on the GPU generation, for example:

```bash
pip install "symmetrix-xl-cuda13-sm80>=0.2,<0.3"   # NVIDIA A100 generation
symmetrix backend show                             # the backend that will be used
```

:::{warning}
Without a matching GPU backend Symmetrix-XL runs on the CPU and reports no
error. Check `symmetrix backend show` on the machine that runs the job.
:::

Symmetrix-XL publishes wheels for Linux x86-64 only. On macOS the extra tries
to build it from source, which needs a Fortran compiler; the rest of AmorphGen
is unaffected.

## Use

Add the flag to any command that uses a MACE model:

```bash
amorphgen --hybrid-ensemble --input-dir random_structures/random_opt/ \
    --config hybrid.yaml -m mace-mpa-0 -d cuda \
    --accelerator symmetrix-xl -o hybrid_run/
```

| Value | Behaviour |
|---|---|
| `none` | Plain MACE (the default). |
| `auto` | Use Symmetrix-XL when it is installed and the model is MACE; otherwise run unchanged. One script then works with and without the package. |
| `symmetrix-xl` | Require it. The run is refused before any work if it cannot be used: package not installed, or a model that is not MACE. |

In YAML:

```yaml
model: mace-mpa-0
device: cuda
accelerator:
  engine: symmetrix-xl
```

In Python:

```python
from amorphgen.utils.calculators import get_calculator

calc = get_calculator("mace-mpa-0", device="cuda",
                      accelerator={"engine": "symmetrix-xl", "elements": ["Ga", "O"]})
```

Foundation-model names and your own `.model` files (`--model-path`) both work.
Other backends (CHGNet, SevenNet, classical potentials, ACE, LAMMPS) are not
accelerated: `auto` runs them unchanged and `symmetrix-xl` refuses them.

## What happens on the first run

1. **Conversion.** The model is converted once into a *model store*
   (`~/.amorphgen/models`, or `$AMORPHGEN_MODEL_STORE`). For `mace-mpa-0` this
   takes about a minute and 274 MB. Parallel jobs convert a model only once.
2. **Check against MACE.** Energy, forces and stress of a small test cell with
   the elements of the run are compared with `mace-torch`. If the largest
   force difference exceeds `accelerator.tolerance` (default 1 meV/Å) the run
   is refused. The result is recorded, so later runs with the same model,
   elements and options skip the check.
3. **Kernel compilation.** Symmetrix-XL compiles GPU kernels for the model the
   first time it is used (about half a minute per precision) and caches them.

Later runs reuse all three.

## Batched ensembles with torch-sim

With `--engine torchsim` (see {doc}`backends`) the structures of a chunk are
evaluated in one Symmetrix-XL call:

```bash
amorphgen --hybrid-ensemble --input-dir random_structures/random_opt/ \
    --config hybrid.yaml -m mace-mpa-0 -d cuda --default-dtype float32 \
    --engine torchsim --accelerator symmetrix-xl -o hybrid_run/ --resume
```

- The chunk size is 32 structures with `--batch-size auto`; set `--batch-size`
  or `accelerator.batch_max_atoms` to change it.
- The neighbour graph is kept on the GPU between MD steps and rebuilt only when
  an atom has moved further than a skin distance allows
  (`accelerator.batch_skin`, default 0.5 Å; 0 rebuilds every step).
- The first joined call is compared with a single-structure call, and the run
  stops if they differ.
- `--resume` works as for the plain torch-sim engine.

This route was developed against Symmetrix-XL 0.2 and uses functions the
package does not document. If a later version changes them, AmorphGen prints a
message and continues on a slower route that builds the graph on the CPU at
every call; results are unaffected. Versions outside the 0.2 series print a
warning at start.

Long-range, field-coupled and dispersion-corrected models are evaluated one
structure per call.

## Settings

All keys of the YAML `accelerator` block:

| Key | Default | Meaning |
|---|---|---|
| `engine` | `none` | `none`, `auto` or `symmetrix-xl`. |
| `elements` | from the run | Elements for the check against MACE. |
| `tolerance` | `1.0` | Largest accepted force difference from MACE, in meV/Å. |
| `validate` | `true` | Run the check against MACE. |
| `spline_points` | `1024` | Radial spline resolution of the converted model. |
| `convert` | `true` | Convert a model that is not in the store; `false` refuses instead. |
| `store` | see above | Folder of the model store. |
| `options` | `{}` | Options passed to Symmetrix-XL (`--accelerator-option key=value`). |
| `batch_max_atoms` | none | Upper bound on the atoms in one joined call (torch-sim). |
| `batch_skin` | `0.5` | Skin of the kept neighbour lists in Å (torch-sim). |

Environment variables:

| Variable | Meaning |
|---|---|
| `AMORPHGEN_MODEL_STORE` | Folder of the model store. Point it at shared storage so a group converts each model once. |
| `SYMMETRIX_JIT_CACHE` | Where Symmetrix-XL keeps its compiled kernels. |
| `SYMMETRIX_BACKEND` | Selects the Symmetrix-XL backend when several are installed. |

The accelerator used, its version and the converted model file are written to
the run's provenance record.

## Measured performance

One NVIDIA A100, `mace-mpa-0`, float32. Times depend on the system and the
hardware; treat them as a guide.

Force difference from MACE (float64) and speed per call:

| System | Atoms | Largest force difference | Speed-up per call |
|---|---|---|---|
| Ga2O3 | 400 | 0.003 meV/Å | 25x |
| SiO2 | 81 | 0.003 meV/Å | 13x |
| InGaZnO4 | 84 | 0.001 meV/Å | 12x |

Hybrid anneal of 16 Ga2O3 cells of 80 atoms (6,800 MD steps and a relaxation
each), wall time for the whole set:

| Route | Time |
|---|---|
| ASE engine, MACE | about 3,200 s (extrapolated from 4 structures) |
| torch-sim engine, MACE | 1,152 s |
| ASE engine, accelerator | 639 s |
| torch-sim engine, accelerator | 344 s |

## Limits

- MACE models only. Tested with `mace-mpa-0`; other MACE models are converted
  and checked the same way.
- Tested on NVIDIA A100 cards. Other cards need their own Symmetrix-XL GPU
  backend package.
- NPT under the torch-sim engine has not been tested with the accelerator.
- The original Symmetrix package (`wcwitt/symmetrix`) is not supported. It
  uses the same import name, so it cannot be installed next to Symmetrix-XL.

See {doc}`hpc` for cluster settings.
