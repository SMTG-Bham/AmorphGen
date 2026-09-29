# Installation

## Requirements

- Python 3.10 to 3.14. CHGNet publishes wheels up to 3.12 only; on 3.13 and
  3.14 pip compiles it from source, which needs a C compiler.
- ASE (Atomic Simulation Environment)
- An MLIP backend **only** for MLIP relaxation / melt-quench MD; the base
  install is deliberately torch-free
- A C/C++ compiler **only** for the torch-sim engine
  ([see below](#the-torch-sim-engine))

## Pick the install for your task

| I want to… | Install |
|---|---|
| generate random structures, analyse trajectories (RDF, CN, S(q), plots), classical LJ/Buckingham pipelines | `pip install amorphgen` (~80 MB, no PyTorch) |
| MLIP relaxation & melt-quench MD | `pip install "amorphgen[mace]"` or `"amorphgen[chgnet]"` |
| everything (MACE + CHGNet) | `pip install "amorphgen[all]"` |
| batched GPU relaxation and MD of ensembles (`--engine torchsim`) | `pip install "amorphgen[mace,torchsim]"` (Python 3.12+, and a C/C++ compiler) |

On a torch-free install, `--device auto` resolves to CPU and any
calculator-requiring command fails fast with the exact install line to copy.
`amorphgen --list-models` shows every model with installed/missing markers.

## Install from PyPI

Install AmorphGen with your preferred backend:

```bash
# Lightweight (no PyTorch): generation + analysis + classical potentials
pip install amorphgen

# MACE only
pip install "amorphgen[mace]"

# CHGNet only
pip install "amorphgen[chgnet]"

# SevenNet only
pip install "amorphgen[sevennet]"

# MACE + CHGNet (recommended default)
pip install "amorphgen[mace,chgnet]"

# SevenNet + CHGNet (alternative env for SevenNet users)
pip install "amorphgen[sevennet,chgnet]"

# Conflict-free convenience bundle (MACE + CHGNet + analysis)
pip install "amorphgen[all]"

# Development install
pip install "amorphgen[all,dev]"

# torch-sim engine for batched ensembles on a GPU (Python 3.12+, add to any MLIP extra)
pip install "amorphgen[mace,torchsim]"
```

:::{warning}
Do not install MACE and SevenNet in the same environment: SevenNet
depends on `e3nn>=0.5`, while pre-trained MACE foundation models
(`mace-mpa-0`, …) were pickled with `e3nn==0.4.x` and fail to load
against the newer `e3nn`. The `[all]` extra therefore includes only
MACE + CHGNet (not SevenNet).

`[mace]`, `[chgnet]`, and `[sevennet]` are each fine on their own;
just don't combine `[mace]` and `[sevennet]`. The recommended pattern
is two conda environments:

```bash
# env A - MACE + CHGNet (default)
conda create -n amorphgen python=3.11
conda activate amorphgen
pip install "amorphgen[mace,chgnet]"

# env B - SevenNet + CHGNet (when SevenNet is required)
conda create -n amorphgen-sevennet python=3.11
conda activate amorphgen-sevennet
pip install "amorphgen[sevennet,chgnet]"
```

CHGNet has no `e3nn` dependency and is safe alongside either backend.
See {doc}`../guides/backends` for the full explanation of the conflict.
:::

### The torch-sim engine

The `[torchsim]` extra adds a second execution engine (`--engine torchsim`)
that relaxes, anneals and quenches all structures of an ensemble in one
batched call. It needs Python 3.12 and a CUDA GPU or CPU (Apple MPS is not
supported) and works with MACE, SevenNet and Lennard-Jones. Everything else
runs unchanged on the default ASE engine. See {doc}`../guides/backends`.

It also needs a C/C++ compiler wherever it runs, and pip cannot install one:
torch-sim's neighbour list goes through `torch.compile`, which builds its
kernels with the system compiler the first time it runs. Without one, the first
relaxation stops with `InvalidCxxCompiler: No working C++ compiler found`. Most
Linux machines and clusters have gcc already (`g++ --version` prints its
version); otherwise install one:

```bash
sudo apt install build-essential                       # Ubuntu, Debian, WSL
conda install -c conda-forge c-compiler cxx-compiler   # Linux, into the active conda environment, no root
xcode-select --install && brew install libomp          # macOS: clang, and the OpenMP Apple's clang lacks
```

On a cluster whose compute nodes have no compiler, load one in the job script
(`module load GCC`; the name varies by site).

## Install from source

```bash
git clone https://github.com/SMTG-Bham/AmorphGen.git
cd AmorphGen
pip install -e ".[mace,chgnet,dev]"
```

## Install with conda

AmorphGen is not on conda-forge, but a conda environment is the cleanest way to
isolate it; on HPC, conda manages the CUDA toolchain. From a clone, the
environment files in
[`build_tools/`](https://github.com/SMTG-Bham/AmorphGen/tree/main/build_tools)
create it in one step and install the checkout in editable mode:

```bash
git clone https://github.com/SMTG-Bham/AmorphGen.git
cd AmorphGen

# MACE + CHGNet
conda env create -f build_tools/environment.yml
conda activate amorphgen

# or, for development, with the torch-sim engine, pytest and the docs toolchain too
conda env create -f build_tools/environment_dev.yml
conda activate amorphgen-dev
```

Or create the environment with conda, then install AmorphGen into it with pip:

```bash
conda create -n amorphgen python=3.11
conda activate amorphgen

# from PyPI (once released):
pip install "amorphgen[mace,chgnet]"

# or from source:
git clone https://github.com/SMTG-Bham/AmorphGen.git
cd AmorphGen
pip install -e ".[mace,chgnet]"
```

CHGNet is safe alongside MACE or SevenNet; just don't put MACE and SevenNet in
the same environment (see the warning above, or the {doc}`../guides/backends`
page for the full explanation).

## Backend compatibility

| Backend | PyPI package | GPU support | Mac (Apple Silicon) |
|---------|-------------|-------------|---------------------|
| MACE    | `mace-torch` | CUDA yes | CPU + MPS yes |
| CHGNet  | `chgnet`    | CUDA yes | CPU + MPS yes |
| SevenNet | `sevenn`   | CUDA yes | CPU + MPS yes |
| Classical (LJ, Buckingham) | built-in | N/A | CPU yes |
| torch-sim engine (`--engine torchsim`) | `torch-sim-atomistic` | CUDA yes | CPU only, no MPS |

## HPC setup (SLURM)

```bash
# Load your cluster's CUDA module (name varies by site)
module load CUDA/11.8.0

# MACE + CHGNet env (recommended default)
conda create -n amorphgen python=3.11
conda activate amorphgen
pip install "amorphgen[mace,chgnet]"

# For batched ensembles on the GPU, use Python 3.12 and add the torch-sim extra
# (the jobs then need a C/C++ compiler too, see "The torch-sim engine" above)
conda create -n amorphgen-ts python=3.12
conda activate amorphgen-ts
pip install "amorphgen[mace,torchsim]"
```

If you also want SevenNet, create a second environment as described
in the warning above and switch between them in your SLURM scripts via
`source activate amorphgen` or `source activate amorphgen-sevennet`.

## Verify installation

```python
import amorphgen
print(amorphgen.__version__)  # 1.0.0

from amorphgen.utils.calculators import list_models
list_models()  # prints all available models grouped by backend
```
