"""Minimum separations of the families a Materials Project audit found blocked.

Against the shortest contacts of 30,616 experimentally observed structures
(ICSD-matched, within 25 meV/atom of the hull), 11.9 % of the inorganic ones had
a floor above one of their own contacts. Most came from rules written for
oxides: anions kept apart even where they must bond (S2 2- in FeS2, Se chains in
Ge20Se80, C2 2- in CaC2, N3 - in NaN3, O2 2- in Li2O2), the H-H of a hydride or a
water molecule held at anion or end-to-end distance, P sized as P3- in a
covalent metal phosphide, the O-O edge of a nitrate or carbonate triangle under
the packing floor, and the centre of an oxoanion left an anion when its metal
(Ag, Tl, Mn, Cr) has a top state it does not take.
"""

import numpy as np
import pytest
from ase.build import bulk
from ase.spacegroup import crystal

from amorphgen_test_helpers import (
    minsep_floors as _floors,
    shortest_pair_distances as _shortest,
)

from amorphgen.analysis.structure import compute_dimers
from amorphgen.pipeline.random_gen import generate_random
from amorphgen.utils.radii import (
    _homopolar_anion,
    cation_nonmetals,
    infer_oxidation_state,
)


def _anions(composition):
    from amorphgen.utils.radii import NONMETALS
    centres = cation_nonmetals(composition)
    return [s for s in composition if s in NONMETALS and s not in centres]


# ── Oxoanion centres beside a metal below its top state ──────────

@pytest.mark.parametrize("composition, expected", [
    ({"Ag": 2, "S": 1, "O": 4}, {"S"}),       # Ag+3 at the top state tied -4 / +4
    ({"Ag": 3, "P": 1, "O": 4}, {"P"}),
    ({"Tl": 3, "P": 1, "O": 4}, {"P"}),       # needs Tl+ in the table
    ({"Ag": 1, "I": 1, "O": 3}, {"I"}),
    ({"Mn": 1, "O": 2, "H": 1}, {"H"}),       # MnOOH
    ({"Cr": 16, "O": 32, "H": 16}, {"H"}),    # CrOOH: Cr+6 covers O and H-
    # ... and the cells whose roles must not move
    ({"Si": 4, "O": 6, "C": 1}, set()),
    ({"Si": 2, "C": 1, "N": 2}, set()),
    ({"Ti": 2, "C": 1, "N": 1}, set()),
    ({"Sb": 16, "O": 45, "Cl": 5}, set()),
    ({"Ge": 20, "S": 10, "Se": 70}, set()),
    ({"Li": 29, "P": 10, "O": 33, "N": 5}, {"P"}),
    ({"V": 1, "O": 1, "Cl": 1}, set()),
    ({"Na": 32, "Cl": 32, "O": 1}, set()),
    ({"Ti": 24, "O": 48, "F": 1}, set()),
])
def test_oxoanion_centres(composition, expected):
    assert set(cation_nonmetals(composition)) == expected


def test_silver_sulfate_is_a_sulfate():
    comp = {"Ag": 2, "S": 1, "O": 4}
    assert infer_oxidation_state("Ag", comp) == 1
    assert infer_oxidation_state("S", comp) == 6
    assert _floors(comp)["O-S"] < 1.3          # the S-O bond, not anion packing


# ── Anions that must bond to each other ──────────────────────────

@pytest.mark.parametrize("composition, anion, floor", [
    ({"Ge": 20, "Se": 80}, "Se", 1.92),
    ({"As": 30, "S": 70}, "S", 1.68),
    ({"Fe": 32, "S": 64}, "S", 1.68),
    ({"Li": 32, "S": 32}, "S", 1.68),
    ({"Ca": 32, "C": 64}, "C", 1.06),
    ({"Y": 32, "C": 32, "I": 32}, "C", 1.06),
    ({"Na": 16, "N": 48}, "N", 0.99),
    ({"Li": 32, "O": 32}, "O", 0.92),
    ({"K": 32, "O": 64}, "O", 0.92),
    ({"Cs": 16, "I": 48}, "I", 2.22),
])
def test_anion_excess_bonds(composition, anion, floor):
    assert _homopolar_anion(composition, _anions(composition)) == anion
    assert _floors(composition)[f"{anion}-{anion}"] == pytest.approx(floor, abs=0.01)


@pytest.mark.parametrize("composition", [
    {"Ge": 32, "Se": 64}, {"As": 40, "S": 60}, {"Zn": 32, "S": 32},
    {"Li": 16, "Ni": 16, "O": 32},     # Ni3+ above the table must not read as a peroxide
    {"Ba": 16, "Ni": 16, "O": 48},
    {"Sb": 16, "O": 45, "Cl": 5}, {"V": 16, "O": 40, "F": 1},
    {"Si": 32, "O": 64, "F": 2}, {"Li": 29, "P": 10, "O": 33, "N": 5},
    {"Si": 4, "O": 6, "C": 1}, {"C": 64}, {"Se": 64},
])
def test_no_anion_excess(composition):
    assert _homopolar_anion(composition, _anions(composition)) is None


# ── H-H, M-P, and the nitrate / carbonate edge ───────────────────

@pytest.mark.parametrize("composition", [
    {"Na": 16, "O": 16, "H": 16}, {"Ca": 8, "S": 8, "O": 48, "H": 32},  # gypsum
    {"H": 24, "P": 8, "O": 32},
])
def test_two_h_can_share_an_oxygen(composition):
    assert _floors(composition)["H-H"] < 0.85 * 1.52   # the H-H of a water molecule


@pytest.mark.parametrize("composition", [
    {"Ti": 32, "H": 64}, {"Mg": 32, "H": 64}, {"Li": 16, "B": 16, "H": 64},
])
def test_hydride_h_h(composition):
    assert _floors(composition)["H-H"] == pytest.approx(0.8 * 2.1)


@pytest.mark.parametrize("composition, pair", [
    ({"Ni": 64, "P": 32}, "Ni-P"), ({"Co": 48, "P": 48}, "Co-P"),
    ({"Ni": 80, "P": 20}, "Ni-P"),
])
def test_covalent_metal_phosphide(composition, pair):
    assert _floors(composition)[pair] < 0.85 * 2.2     # bonds of 2.2-2.3 A


@pytest.mark.parametrize("composition, edge", [
    ({"Na": 20, "N": 20, "O": 60}, 2.17), ({"Ca": 12, "C": 12, "O": 36}, 2.22),
])
def test_trigonal_oxoanion_edge(composition, edge):
    assert _floors(composition)["O-O"] < 0.9 * edge


@pytest.mark.parametrize("composition", [
    {"Li": 24, "P": 8, "O": 32}, {"Li": 16, "S": 8, "O": 32}, {"Si": 32, "O": 64},
])
def test_tetrahedral_oxoanions_keep_packing(composition):
    assert _floors(composition)["O-O"] == pytest.approx(2.24)


# ── Real crystals and placement ──────────────────────────────────

CRYSTALS = {
    "FeS2": (crystal(["Fe", "S"], [(0, 0, 0), (0.385, 0.385, 0.385)],
                     spacegroup=205, cellpar=[5.417] * 3 + [90] * 3),
             {"Fe": 32, "S": 64}),
    "CaC2": (crystal(["Ca", "C"], [(0, 0, 0), (0, 0, 0.406)], spacegroup=139,
                     cellpar=[3.89, 3.89, 6.38, 90, 90, 90]), {"Ca": 32, "C": 64}),
    "TiH2": (bulk("TiH2", "fluorite", a=4.454), {"Ti": 32, "H": 64}),
}


@pytest.mark.parametrize("name", sorted(CRYSTALS))
def test_no_floor_excludes_the_crystal(name):
    atoms, composition = CRYSTALS[name]
    floors, shortest = _floors(composition), _shortest(atoms, cutoff=4.5)
    for key, floor in floors.items():
        if key in shortest:
            assert floor < 0.95 * shortest[key], (key, floor, shortest[key])


@pytest.mark.parametrize("composition, rho", [
    ({"Fe": 32, "S": 64}, 5.01), ({"Na": 24, "N": 72}, 1.85),
])
def test_anion_excess_places_near_the_crystal_density(composition, rho):
    """The S-S and N-N packing floors stalled these below 0.85 of the crystal."""
    for seed in (1, 2):
        generate_random(composition, target_density=0.85 * rho, seed=seed,
                        retry_mode="none", max_attempts_per_atom=50000)


def test_ge20se80_places_se_se_bonds():
    atoms = generate_random({"Ge": 20, "Se": 80}, target_density=4.32 * 0.85, seed=1,
                            retry_mode="none", max_attempts_per_atom=50000)
    s = np.array(atoms.get_chemical_symbols())
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, 9.0)
    se = s == "Se"
    assert (d[np.ix_(se, se)] < 2.6).any()


# ── compute_dimers reads the stoichiometry ───────────────────────

def test_dimers_keep_flagging_a_peroxide():
    """One of each would read Li2O as LiO, a peroxide, and drop the O-O flag."""
    atoms = bulk("OLi2", "fluorite", a=4.611).repeat(2)
    o = next(i for i, a in enumerate(atoms) if a.symbol == "O")
    atoms.append("O")
    atoms.positions[-1] = atoms.positions[o] + [1.45, 0, 0]
    assert compute_dimers([atoms])["pairs"]["O-O"]["count"] == 1


def test_dimers_accept_a_persulfide():
    atoms, _ = CRYSTALS["FeS2"]
    assert compute_dimers([atoms])["pairs"] == {}
