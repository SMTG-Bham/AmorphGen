"""Shared fixtures and opt-in model tests.

The default suite runs core tests and any installed optional CPU backends.
CUDA tests require suitable hardware; real MACE tests additionally require
``--run-mace`` because they can download a foundation model.
"""

import matplotlib
import pytest

from ase.build import bulk
from ase.calculators.emt import EMT

# The tests save figures and never show them. The package leaves the backend
# to the user, so the suite picks Agg itself and runs the same with or without
# a display.
matplotlib.use("Agg")


# ── CLI options ───────────────────────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption("--run-mace", action="store_true", default=False,
                     help="Run real MACE tests (may download models; GPU recommended)")


def pytest_configure(config):
    config.addinivalue_line("markers", "mace: requires real MACE calculator")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-mace"):
        skip = pytest.mark.skip(reason="Pass --run-mace to run")
        for item in items:
            if item.get_closest_marker("mace") is not None:
                item.add_marker(skip)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_work_dir(tmp_path, monkeypatch):
    """Provide a clean temporary working directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def cu_bulk():
    """4-atom FCC copper cell — works with EMT."""
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    return atoms


@pytest.fixture
def cu_supercell():
    """32-atom copper supercell for MD tests."""
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    return atoms


@pytest.fixture
def emt_calc():
    """ASE's built-in EMT calculator (no GPU needed)."""
    return EMT()
