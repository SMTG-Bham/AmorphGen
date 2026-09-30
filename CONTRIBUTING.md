# Contributing to AmorphGen

Thank you for your interest in contributing to AmorphGen! This document provides
guidelines for contributing to this project.

## Getting started

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/AmorphGen.git
   cd AmorphGen
   ```
3. Install in development mode:
   ```bash
   pip install -e ".[dev,docs]"
   ```
4. Create a branch for your changes:
   ```bash
   git switch -c my-feature
   ```

## Development setup

AmorphGen requires Python ≥ 3.10. The conda development environment installs
your clone in editable mode with MACE, CHGNet, the torch-sim engine, pytest and
the Sphinx toolchain:

```bash
conda env create -f build_tools/environment_dev.yml
conda activate amorphgen-dev
```

For the core package, tests, and documentation without an MLIP backend:

```bash
pip install -e ".[dev,docs]"
```

For MACE, CHGNet, and torch-sim development, use Python 3.12 and install
`pip install -e ".[all,torchsim,dev,docs]"`. The torch-sim engine also needs
a C/C++ compiler on `PATH`. SevenNet's e3nn dependency conflicts with the
MACE extra, so use a separate environment for SevenNet.

## Running tests

```bash
# Run the suite available in the current environment
pytest test/ -v --tb=short

# Run with MACE integration tests (requires mace-torch + GPU recommended)
pytest test/ -v --tb=short --run-mace
```

Tests requiring an uninstalled optional backend are skipped. GitHub Actions
tests the core package on Python 3.10 to 3.14 and runs backend tests in
separate jobs on every push and pull request to `main` and `dev`:

| Job | Checks |
|---|---|
| `lint` | syntax errors and undefined names (ruff) in the package, tests, docs config and tutorial notebooks |
| `test` | the torch-free suite on Python 3.10 to 3.14 on Linux, and on 3.14 on macOS |
| `backends` | the full suite with CPU-only PyTorch, the torch-sim engine and CHGNet, with a coverage report |
| `min-deps` | the core suite on Python 3.10 with direct dependencies at their lowest allowed versions |
| `package` | the sdist and wheel build, the README links as PyPI renders them, and the sdist's tests run against the installed wheel |
| `light-install` | a bare `pip install` (no extras) stays torch-free |

The documentation builds with Sphinx warnings as errors (`docs.yml`), the
conda environments in `build_tools/` are built and tested whenever they or
`pyproject.toml` change (`conda.yml`), and a weekly canary relaxes a structure
with CHGNet and SevenNet (`canary.yml`). The MACE tests (`--run-mace`) and the
CUDA tests (`test/test_torchsim_gpu.py`) need a model download or a GPU and are
run outside CI, before releases.

To run the `backends` or `min-deps` job locally:

```bash
# backends, in a Python 3.12 environment; torch-sim tests need a C/C++
# compiler on PATH (the conda development environment installs one)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[torchsim,chgnet,dev]"
pytest test/

# min-deps, in a fresh Python 3.10 environment (needs uv)
uv pip install --resolution lowest-direct -e ".[dev]"
pytest test/
```

If `min-deps` fails because the code requires a newer dependency, either
restore compatibility or raise the relevant lower bound in `pyproject.toml`.

## Code style

- Follow PEP 8 conventions
- Use type hints where practical
- Add docstrings to all public functions and classes
- Keep imports organised: standard library, third-party, then local

## How to contribute

### Reporting bugs

Open an issue on GitHub with:

- A clear description of the problem
- Steps to reproduce the issue
- The full error traceback
- Your Python version and OS
- Which MLIP backend you are using (MACE, CHGNet, SevenNet)

### Suggesting features

Open an issue on GitHub describing:

- What the feature would do
- Why it would be useful
- Any relevant references or examples

### Submitting changes

1. Make your changes on a feature branch
2. Add or update tests for any new functionality
3. Ensure all tests pass locally: `pytest test/ -v`
4. Commit with a clear message describing the change
5. Push to your fork and open a pull request against `main`

### Adding a new MLIP backend

AmorphGen is designed to be model-agnostic. To add a new backend:

1. In `amorphgen/utils/calculators.py`:
   - Add a `_load_<backend>(model, device, **kwargs)` function (see
     the existing `_load_mace`, `_load_chgnet`, `_load_sevennet` for
     reference).
   - Wire it into the dispatch inside `get_calculator()`.
   - Add model names to the appropriate registry constant (e.g.
     `MACE_FOUNDATION_MODELS`, `SEVENNET_MODELS`).
2. Add an optional dependency group in `pyproject.toml`.
3. Add dispatch / smoke tests in `test/test_calculators.py` (see
   `TestBackendRouting` for the pattern).
4. Update the README with the new backend.

### Tutorials

Notebook tutorials live under `Tutorials/`. New tutorials should:

- Use a clear `TN_<topic>/tutorial_N_<topic>.ipynb` directory layout.
- Be self-contained (input files alongside the notebook).
- Run end-to-end on CPU within ~15 minutes where possible, or document
  the GPU / wall-time requirement at the top of the notebook.

### Documentation

Sphinx docs live under `docs/`. From the repository root, install the docs
extra and build with the same warning checks as CI:

```bash
pip install -e ".[docs]"
make -C docs html SPHINXOPTS="-W --keep-going"
# output in docs/_build/html/
```

User-facing changes (new flags, new modes) should be reflected in the
relevant guide (`docs/guides/*.md`).

## Project structure

```
amorphgen/
├── cli.py                  ← command-line interface
├── configs/
│   ├── default_config.py   ← all default parameters
│   └── yaml_config.py      ← YAML loader + schema validation
├── pipeline/
│   ├── run_pipeline.py     ← MeltQuenchPipeline orchestrator
│   ├── opt_cell.py         ← Stages 1 & 7 (optimisation)
│   ├── equilibrate.py      ← Stages 2, 4, 6 (equilibration)
│   ├── melt_cell.py        ← Stage 3 (heat ramp)
│   ├── quench.py           ← Stage 5 (cool ramp)
│   ├── batch_quench.py     ← batch / hybrid / MQ-ensemble runner
│   └── random_gen.py       ← random structure placement
├── analysis/
│   ├── analyser.py         ← StructureAnalyser entry point
│   ├── rdf.py              ← radial distribution functions
│   ├── structure.py        ← coordination, bond angles
│   ├── rings.py            ← ring statistics
│   ├── voronoi.py          ← Voronoi tessellation
│   └── energy.py           ← per-structure energy ranking
└── utils/
    ├── calculators.py      ← multi-backend calculator factory
    ├── classical.py        ← Lennard-Jones, Buckingham+Coulomb
    ├── radii.py            ← Shannon / Cordero / Goldschmidt radii,
    │                          bond + material-class classifiers,
    │                          minsep + density estimators
    ├── common.py           ← MD-dynamics builder, trajectory I/O
    └── equilibration.py    ← block-average convergence diagnostics
```

## Code of conduct

Everyone taking part in AmorphGen is expected to follow the
[code of conduct](https://github.com/SMTG-Bham/AmorphGen/blob/main/CODE_OF_CONDUCT.md),
which also says how to report a problem.

## Questions?

Open an issue on GitHub or contact the maintainers directly.
