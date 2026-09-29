# Conda environments

| File | Environment | Installs |
|---|---|---|
| `environment.yml` | `amorphgen` | AmorphGen with MACE and CHGNet, the recommended default |
| `environment_dev.yml` | `amorphgen-dev` | the same plus the torch-sim engine, C/C++ compilers, pytest and the Sphinx toolchain |

Both take Python from conda-forge and install this checkout with pip in editable
mode, so a `git pull` or a local edit takes effect without reinstalling. The
Python dependencies come from the extras in the root `pyproject.toml`, which
stays the one place they are declared. conda resolves both files to Python 3.12,
the newest version CHGNet publishes wheels for and the oldest the torch-sim
engine runs on.

On Linux, pip's PyTorch wheel bundles the CUDA libraries it was built with, so a
GPU node needs an NVIDIA driver recent enough for that CUDA release: compare
`python -c "import torch; print(torch.version.cuda)"` with the CUDA version
`nvidia-smi` reports, and on an older driver install the matching PyTorch build
from pytorch.org into the environment. Each environment takes about 6 GB, mostly
PyTorch and its CUDA libraries.

Run the commands below from the root of this checkout.

## Using AmorphGen

```bash
conda env create -f build_tools/environment.yml
conda activate amorphgen
amorphgen --list-models
```

For batched ensembles on a GPU, add the torch-sim engine (`--engine torchsim`):

```bash
pip install -e ".[torchsim]"
```

## Development

```bash
conda env create -f build_tools/environment_dev.yml
conda activate amorphgen-dev
pytest test/
```

When `pyproject.toml` gains or changes a dependency, rerun the pip step with
`conda env update -f build_tools/environment_dev.yml`. See
[CONTRIBUTING.md](../CONTRIBUTING.md) for the test and documentation checks.

## SevenNet

Neither file installs SevenNet: it needs `e3nn>=0.5`, which breaks loading the
MACE foundation models. Give it an environment of its own, as described in
`docs/guides/backends.md`.
