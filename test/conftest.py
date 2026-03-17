"""
tests/conftest.py
-----------------
Shared pytest fixtures for amorphgen tests.

Tiers:
  Tier 1 – Pure unit tests (no calculator needed)
  Tier 2 – Integration tests with ASE's EMT calculator (no GPU)
  Tier 3 – Full MACE/CHGNet/M3GNet tests (@pytest.mark.mace, etc.)

Usage:
    pytest                        # Tiers 1 + 2
    pytest -m mace --run-mace     # Tier 3 MACE tests
"""

import os
import shutil
import tempfile
import pytest
import numpy as np

from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT


# ── CLI options ───────────────────────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption("--run-mace", action="store_true", default=False,
                     help="Run tests requiring real MACE model (GPU + internet)")


def pytest_configure(config):
    config.addinivalue_line("markers", "mace: requires real MACE calculator")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-mace"):
        skip = pytest.mark.skip(reason="Pass --run-mace to run")
        for item in items:
            if "mace" in item.keywords:
                item.add_marker(skip)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_work_dir(tmp_path):
    """Provide a clean temporary working directory."""
    old = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(old)


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
