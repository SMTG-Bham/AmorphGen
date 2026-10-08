"""Cation-cation minimum separations in cation-rich compounds.

The M-M floor of a compound with anions was max(metallic contact,
sqrt(2) x d(M-X)), capped at 2.80 A: two octahedra sharing an edge, a 90
degree M-X-M angle. That holds while an anion has at most 6 cations around it
(MO, M2O3, MO2). In antifluorite Li2O / Li2S, anti-perovskite Li3OCl and Li3N
the cations crowd an anion more closely, at 70.5 or 60 degrees, and Li+ is far
smaller than the Li atom the metallic radius measures: the old floors sat above
the crystals' own Li-Li distances (Li3N 2.58 against 2.11 A, Li2O 2.60 against
2.31 A), and random placement could not reach the crystal density at all.
"""

import logging

import numpy as np
import pytest
from ase.build import bulk
from ase.neighborlist import neighbor_list
from ase.spacegroup import crystal

from amorphgen.pipeline.random_gen import generate_random
from amorphgen.utils.radii import (
    _cation_contact_factor,
    auto_target_cn,
    default_minsep,
)


def _symbols(composition):
    return [s for s, n in composition.items() for _ in range(n)]


def _floors(composition):
    """The minsep table generate_random builds for this composition."""
    target_cn, _ = auto_target_cn(composition)
    return default_minsep(_symbols(composition), target_cn=target_cn)


def _shortest(atoms, cutoff=6.0):
    """Shortest distance of each element pair in a crystal."""
    symbols = np.array(atoms.get_chemical_symbols())
    i, j, d = neighbor_list("ijd", atoms, cutoff)
    out = {}
    for a, b, dist in zip(symbols[i], symbols[j], d):
        key = "-".join(sorted((a, b)))
        out[key] = min(out.get(key, np.inf), dist)
    return out


def _density(atoms):
    return atoms.get_masses().sum() * 1.66054 / atoms.get_volume()


# Experimental room-temperature structures.
CRYSTALS = {
    "Li3N": (crystal(["N", "Li", "Li"], [(0, 0, 0), (0, 0, 0.5), (1 / 3, 2 / 3, 0)],
                     spacegroup=191, cellpar=[3.648, 3.648, 3.875, 90, 90, 120]),
             {"Li": 48, "N": 16}, "Li-Li"),
    "Li2O": (bulk("OLi2", "fluorite", a=4.611), {"Li": 64, "O": 32}, "Li-Li"),
    "Li2S": (bulk("SLi2", "fluorite", a=5.708), {"Li": 64, "S": 32}, "Li-Li"),
    "Na2S": (bulk("SNa2", "fluorite", a=6.54), {"Na": 64, "S": 32}, "Na-Na"),
    "Li3OCl": (crystal(["Cl", "O", "Li"], [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
                       spacegroup=221, cellpar=[3.91] * 3 + [90] * 3),
               {"Li": 48, "O": 16, "Cl": 16}, "Li-Li"),
}


def test_contact_factor_follows_the_narrowest_angle():
    assert _cation_contact_factor(4) == 2**0.5
    assert _cation_contact_factor(6) == 2**0.5          # octahedral edge, 90 deg
    assert _cation_contact_factor(8) == pytest.approx(2 / 3**0.5, abs=1e-5)  # cube, 70.5 deg
    assert _cation_contact_factor(12) == pytest.approx(1.0)  # 60 deg


@pytest.mark.parametrize("name", sorted(CRYSTALS))
def test_cation_floor_below_the_crystal(name):
    atoms, composition, pair = CRYSTALS[name]
    floors, shortest = _floors(composition), _shortest(atoms)
    # the cation pair at 0.7-0.92 of the crystal's, as a bond floor is at 0.8
    assert 0.70 * shortest[pair] <= floors[pair] <= 0.92 * shortest[pair]
    # and no floor of the compound excludes the crystal's own contacts
    for key, floor in floors.items():
        assert floor < 0.95 * shortest[key], (key, floor, shortest[key])


@pytest.mark.parametrize("name, fraction", [
    ("Li3N", 1.0), ("Li2S", 1.0), ("Li2O", 0.85),
])
def test_cation_rich_places_near_the_crystal_density(name, fraction):
    """The old floors stalled placement down to 0.60-0.70 of these densities."""
    atoms, composition, _ = CRYSTALS[name]
    small = {s: n * 100 // sum(composition.values()) for s, n in composition.items()}
    rho = fraction * _density(atoms)
    for seed in (1, 2):
        placed = generate_random(small, target_density=rho, seed=seed,
                                 retry_mode="none", max_attempts_per_atom=50000)
        assert _density(placed) == pytest.approx(rho, rel=1e-3)


@pytest.mark.parametrize("composition, pair, expected", [
    # metal-rich: more cations than any anion can hold, so the metals touch
    ({"Ni": 80, "P": 20}, "Ni-Ni", 0.85 * 2 * 1.24),
    ({"Fe": 48, "C": 16}, "Fe-Fe", 0.85 * 2 * 1.26),
])
def test_metal_rich_takes_the_metallic_contact(composition, pair, expected):
    assert _floors(composition)[pair] == pytest.approx(expected)


@pytest.mark.parametrize("composition, pair, expected", [
    # anions with at most 6 cations: the octahedral-edge rule, unchanged
    ({"In": 88, "O": 132}, "In-In", 2.8),
    ({"Na": 108, "Cl": 108}, "Na-Na", 2.8),
    ({"Mg": 108, "O": 108}, "Mg-Mg", 2.72),
    ({"Ga": 108, "N": 108}, "Ga-Ga", 2.320017),
    ({"Li": 108, "Cl": 108}, "Li-Li", 2.8),
    ({"Ca": 72, "F": 144}, "Ca-Ca", 2.8),
    ({"Li": 32, "Si": 8, "O": 32}, "Li-Li", 2.596496),   # anion CN exactly 6
    ({"Li": 24, "P": 8, "O": 32}, "Li-Li", 2.596496),
    ({"Na": 16, "O": 16, "H": 16}, "Na-Na", 2.8),        # the H+ counts 1, not 6
    ({"Mg": 48, "N": 32}, "Mg-Mg", 2.72),
])
def test_ordinary_compounds_unchanged(composition, pair, expected):
    assert _floors(composition)[pair] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("composition, pair, expected", [
    # without target CNs the gate takes the automatic ones, so the H+ of a
    # hydroxide counts 1 and not the default 6
    ({"Na": 16, "O": 16, "H": 16}, "Na-Na", 2.8),
    ({"Mg": 12, "O": 24, "H": 24}, "Mg-Mg", 2.72),
    ({"Li": 29, "P": 10, "O": 33, "N": 5}, "Li-Li", 2.668621),
    ({"Li": 32, "Si": 8, "O": 32}, "Li-Li", 2.596496),
])
def test_table_without_target_cn_unchanged(composition, pair, expected):
    assert default_minsep(_symbols(composition))[pair] == pytest.approx(expected, abs=1e-6)


def test_cation_rich_floor_is_logged(caplog):
    with caplog.at_level(logging.INFO, logger="amorphgen.utils.radii"):
        _floors({"Li": 48, "N": 16})
    assert "M-M cation-rich, anion CN=12.0" in caplog.text
