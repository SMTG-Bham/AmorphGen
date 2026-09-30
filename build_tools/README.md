# Conda environments

| File | Environment | Installs |
|---|---|---|
| `environment.yml` | `amorphgen` | AmorphGen with MACE and CHGNet, the recommended default |
| `environment_dev.yml` | `amorphgen-dev` | the same plus the torch-sim engine, C/C++ compilers, pytest and the Sphinx toolchain |

Both take Python from conda-forge and install this checkout with pip in editable
mode, so a `git pull` or a local edit takes effect without reinstalling. The
Python dependencies come from the extras in the root
[pyproject.toml](../pyproject.toml). Both files allow Python 3.10–3.12, and conda
normally selects 3.12. The torch-sim engine requires Python 3.12+: verify
`python --version` before using it, since its pip dependency is skipped on
older Python versions.

On Linux, pip's PyTorch wheel bundles the CUDA libraries it was built with, so a
GPU node needs an NVIDIA driver recent enough for that CUDA release: compare
`python -c "import torch; print(torch.version.cuda)"` with the CUDA version
`nvidia-smi` reports, and on an older driver install the matching PyTorch build
from PyTorch into the environment. These files do not install an NVIDIA driver
or a standalone CUDA toolkit; disk use varies with the selected PyTorch build.

Run the commands below from the root of this checkout.

## Using AmorphGen

```bash
conda env create -f build_tools/environment.yml
conda activate amorphgen
amorphgen --list-models
```

For batched ensembles on a GPU, add the torch-sim engine (`--engine torchsim`).
It compiles kernels while it runs, so it also needs a C/C++ compiler, which
`environment.yml` does not install (`environment_dev.yml` does). Skip the first
line if `g++ --version` already works:

```bash
conda install -c conda-forge c-compiler cxx-compiler
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
[the backends guide](../docs/guides/backends.md).
