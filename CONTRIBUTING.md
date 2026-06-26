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
   pip install -e ".[all,dev]"
   ```
4. Create a branch for your changes:
   ```bash
   git checkout -b my-feature
   ```

## Development setup

AmorphGen requires Python ≥ 3.10. Install all dependencies including test tools:

```bash
pip install -e ".[all,dev]"
```

This installs all MLFF backends (MACE, CHGNet, M3GNet) plus pytest.

## Running tests

```bash
# Run the full test suite
pytest test/ -v --tb=short

# Run with MACE integration tests (requires mace-torch + GPU recommended)
pytest test/ -v --tb=short --run-mace
```

All tests must pass on Python 3.10, 3.11, and 3.12 before a pull request
will be merged. GitHub Actions CI runs automatically on all pull requests.

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
- Which MLFF backend you are using (MACE, CHGNet, M3GNet)

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

### Adding a new MLFF backend

AmorphGen is designed to be model-agnostic. To add a new backend:

1. Add a new section in `amorphgen/utils/calculators.py`:
   - Add the backend to `_detect_backend()`
   - Create a `_get_<backend>_calculator()` function
   - Add model names to the appropriate registry
2. Add an optional dependency group in `pyproject.toml`
3. Add tests in `test/test_utils.py` for backend detection
4. Update the README and `paper.md` with the new backend

## Project structure

```
amorphgen/
├── cli.py                  ← command-line interface
├── configs/
│   └── default_config.py   ← all default parameters
├── pipeline/
│   ├── run_pipeline.py     ← MeltQuenchPipeline orchestrator
│   ├── opt_cell.py         ← Stages 1 & 7 (optimisation)
│   ├── equilibrate.py      ← Stages 2, 4, 6 (equilibration)
│   ├── melt_cell.py        ← Stage 3 (heat ramp)
│   ├── quench.py           ← Stage 5 (cool ramp)
│   ├── batch_quench.py     ← batch quench workflow
│   └── random_gen.py       ← random structure placement
└── utils/
    ├── calculators.py      ← multi-backend calculator factory
    └── common.py           ← shared utilities
```

## Code of conduct

Please be respectful and constructive in all interactions. We are committed
to providing a welcoming and inclusive experience for everyone.

## Questions?

Open an issue on GitHub or contact the maintainers directly.
