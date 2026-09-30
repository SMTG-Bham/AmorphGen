# Installation

## Requirements

- Linux or macOS. Windows is not supported natively; use
  [WSL](https://learn.microsoft.com/windows/wsl/) instead.
- Python 3.10+ for the base package. Optional backends may impose additional
  Python-version or build requirements. The supplied standard conda environment
  allows Python 3.10–3.12; the development environment requires 3.12 for torch-sim.
- Core dependencies, including ASE, are installed automatically by pip
- An MLIP backend **only** for MLIP relaxation / melt-quench MD; the base
  install is deliberately torch-free
- A C/C++ compiler for the torch-sim engine (and any dependency built from source)
  ([see below](#the-torch-sim-engine))

## Pick the install for your task

| I want to… | Install |
|---|---|
| generate random structures, analyse trajectories (RDF, CN, S(q), plots), classical LJ/Buckingham pipelines | `pip install amorphgen` (no PyTorch) |
| MLIP relaxation & melt-quench MD | `pip install "amorphgen[mace]"` or `pip install "amorphgen[chgnet]"` |
| MACE + CHGNet | `pip install "amorphgen[all]"` |
| batched GPU relaxation and MD of ensembles (`--engine torchsim`) | `pip install "amorphgen[mace,torchsim]"` (Python 3.12+, and a C/C++ compiler) |

On a torch-free install, `--device auto` resolves to CPU. Classical potentials
work on CPU without PyTorch; commands requesting an unavailable MLIP backend report
the required install command.
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

# Convenience bundle: MACE + CHGNet (analysis is included in the base package)
pip install "amorphgen[all]"

# torch-sim with MACE (Python 3.12+; also supports SevenNet or Lennard-Jones)
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
for `--random-gen --relax`, `--batch-opt` and `--hybrid-ensemble`. It batches
relaxations and supports NVT annealing and quenching in the hybrid workflow.
It needs Python 3.12+ and a CUDA GPU or CPU (Apple MPS is not supported), and
works with MACE, SevenNet and Lennard-Jones. CHGNet and Buckingham use the
default ASE engine. On Python below 3.12 the extra does not install torch-sim.
See {doc}`../guides/backends`.

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
pip install -e ".[mace,chgnet]"

# For development, include tests and the documentation toolchain:
# pip install -e ".[all,dev,docs]"
```

## Install with conda

The supplied conda environments install Python from conda-forge and AmorphGen
with pip. From a clone, the environment files in
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

# from PyPI:
pip install "amorphgen[mace,chgnet]"

# or from source:
git clone https://github.com/SMTG-Bham/AmorphGen.git
cd AmorphGen
pip install -e ".[mace,chgnet]"
```

## Backend compatibility

| Backend | PyPI package | GPU support | Mac (Apple Silicon) |
|---------|-------------|-------------|---------------------|
| MACE    | `mace-torch` | CUDA yes | CPU; see MPS note below |
| CHGNet  | `chgnet`    | CUDA yes | CPU + MPS yes |
| SevenNet | `sevenn`   | CUDA yes | CPU; see MPS note below |
| Classical (LJ, Buckingham) | built-in | Optional CUDA path with PyTorch | CPU yes |
| torch-sim engine (`--engine torchsim`) | `torch-sim-atomistic` | CUDA yes | CPU only, no MPS |

On Apple Silicon, `--device auto` can select MPS. MACE and SevenNet default
to float64, which MPS cannot represent; use `--device cpu` for these defaults.
CHGNet uses float32 and includes an MPS loading path.

## HPC setup (SLURM)

```bash
# Follow your cluster's instructions to initialise conda before these commands.

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

Initialise conda in each SLURM job and activate the matching environment with
`conda activate amorphgen` or `conda activate amorphgen-ts`. For SevenNet, use
its separate environment as described above. The compute node needs an NVIDIA
driver compatible with the installed PyTorch CUDA build; these environment
files do not install a driver or a standalone CUDA toolkit. See
{doc}`../guides/hpc` for job examples.

## Verify installation

```python
import amorphgen
print(amorphgen.__version__)  # installed package version

from amorphgen.utils.calculators import list_models
list_models()  # prints registered models and backend installation status
```

Or verify the CLI without loading or downloading a model:

```bash
amorphgen --version
amorphgen --list-models
```
