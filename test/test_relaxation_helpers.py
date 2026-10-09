"""The two ASE relaxation entry points share target construction semantics."""

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from amorphgen.pipeline import opt_cell, random_gen
from amorphgen.utils.relaxation import build_cell_filter


@pytest.mark.parametrize("name", ["LBFGS", "FIRE", "BFGSLineSearch", "BFGS", "MDMin"])
def test_relaxation_entry_points_resolve_the_same_optimizer(name):
    import ase.optimize

    assert opt_cell._get_optimizer(name) is getattr(ase.optimize, name)
    assert random_gen._get_optimizer_class(name) is getattr(ase.optimize, name)


def test_fixed_target_and_cubic_pressure_force_scale():
    atoms = bulk("Cu", a=3.8, cubic=True) * (2, 2, 2)
    atoms.calc = EMT()
    assert build_cell_filter(atoms, "none") is atoms
    assert build_cell_filter(atoms, None) is atoms
    cubic = build_cell_filter(atoms, "cubic")
    # Cell force rows retain the virial scale, rather than being divided by N.
    expected = -np.trace(atoms.get_stress(voigt=False)) / 3 * atoms.get_volume()
    np.testing.assert_allclose(cubic.get_forces()[-3:], np.eye(3) * expected, atol=1e-10)


def test_unknown_filter_retains_frechet_fallback():
    from ase.filters import FrechetCellFilter

    atoms = bulk("Cu", cubic=True)
    assert isinstance(build_cell_filter(atoms, "historical-unknown-name"), FrechetCellFilter)
