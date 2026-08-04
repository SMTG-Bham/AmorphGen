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

This installs all MLIP backends (MACE, CHGNet, SevenNet) plus pytest.

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

Sphinx docs live under `docs/`. To build locally:

```bash
cd docs
make html
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

Please be respectful and constructive in all interactions. We are committed
to providing a welcoming and inclusive experience for everyone.

## Questions?

Open an issue on GitHub or contact the maintainers directly.
