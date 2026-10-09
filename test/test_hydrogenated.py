"""Hydrogenated group-IV networks: a-Si:H, a-Ge:H, a-C:H, a-SiC:H.

H is on the anion table as the H- of LiH, so a-Si:H was classed "hydride"
and sized with the Si4+ ionic radius (15 g/cm3 against a measured ~2.2),
and a-C:H was a covalent carbide with H as its anion (3.5 g/cm3 against
1.2-2.0) and a target CN of 6 for H. As anions, C and H were also kept
2.24 A from everything, so no C-C or C-H bond could be placed and neither
system could be generated at all.
"""

from itertools import pairwise

import numpy as np
import pytest
from ase import Atoms
from ase.neighborlist import neighbor_list

from amorphgen_test_helpers import (
    composition_symbols as _symbols,
    estimated_density as _density,
)

from amorphgen.utils.radii import (
    _classify_compound, auto_target_cn, default_minsep, format_auto_derive_summary, infer_oxidation_state,
)
from amorphgen.pipeline.random_gen import _auto_dmax, generate_random


# ── Which compositions are networks ──────────────────────────────

@pytest.mark.parametrize("composition", [
    {"Si": 64, "H": 8},                 # a-Si:H
    {"Ge": 64, "H": 8},                 # a-Ge:H
    {"C": 70, "H": 30},                 # a-C:H
    {"C": 50, "H": 50},                 # polymer-like a-C:H, one H per C
    {"Si": 32, "C": 32, "H": 8},        # a-SiC:H
    {"Si": 32, "Ge": 32, "H": 8},       # a-SiGe:H
    {"C": 1, "H": 1},                   # a bare element set (compute_dimers)
])
def test_hydrogenated_networks(composition):
    assert _classify_compound(composition) == "hydrogenated_network"


@pytest.mark.parametrize("composition, expected", [
    ({"Li": 16, "H": 16}, "hydride"),                 # the real hydrides
    ({"Mg": 8, "H": 16}, "hydride"),
    ({"Na": 36, "Al": 36, "H": 144}, "hydride"),
    ({"Ti": 72, "H": 144}, "hydride"),
    ({"Si": 64}, "group_iv"),                         # the H-free hosts
    ({"C": 64}, "elemental_semiconductor"),
    ({"Si": 32, "C": 32}, "covalent_carbide"),
])
def test_unchanged_classes(composition, expected):
    assert _classify_compound(composition) == expected


@pytest.mark.parametrize("composition", [
    {"Si": 32, "O": 16, "H": 8},        # a second anion
    {"Si": 30, "N": 40, "H": 10},
    {"Si": 64, "H": 8, "F": 2},
    {"Si": 60, "Al": 4, "H": 8},        # a metal
    {"B": 40, "C": 10, "H": 10},        # a host outside C, Si, Ge
    {"C": 40, "H": 60},                 # more H than host: chains, molecules
    {"Si": 8, "H": 32},
])
def test_outside_the_network_gate(composition):
    """Left to the class rules for the other element, as before."""
    assert _classify_compound(composition) != "hydrogenated_network"


# ── Density ──────────────────────────────────────────────────────

def test_a_si_h_density():
    """Was 15.1 g/cm3 (Si4+ ionic radius); glow-discharge a-Si:H is ~2.2."""
    assert 2.1 <= _density({"Si": 64, "H": 8}) <= 2.35


def test_a_c_h_density():
    """Was 3.5 g/cm3 whatever the H content; measured a-C:H is 1.2-2.0 and
    falls as H rises (hard a-C:H 1.6-2.2 at 30-40 % H, polymer-like 1.2-1.6
    at 40-50 %)."""
    rho = [_density({"C": 100 - h, "H": h}) for h in (10, 20, 30, 40, 50)]
    assert 1.6 <= rho[2] <= 2.0
    assert 1.1 <= rho[4] <= 1.4
    assert all(a > b for a, b in pairwise(rho))


@pytest.mark.parametrize("host", [
    {"Si": 64}, {"C": 64}, {"Ge": 64}, {"Si": 32, "C": 32}, {"Si": 32, "Ge": 32},
])
def test_density_continuous_with_the_host(host):
    """One H changes the estimate by its own share of the volume (2.4 % in
    C64H1, whose C atoms are small), not by a class jump: the host keeps its
    Cordero radii and packing factor."""
    with_h = dict(host, H=1)
    assert _density(with_h) == pytest.approx(_density(host), rel=0.03)


def test_hydride_density_unchanged():
    assert _density({"Li": 16, "H": 16}) == pytest.approx(0.544, abs=0.005)
    assert _density({"Na": 36, "Al": 36, "H": 144}) == pytest.approx(0.966, abs=0.005)


# ── Target CN and oxidation states ───────────────────────────────

@pytest.mark.parametrize("composition, expected", [
    ({"Si": 64, "H": 8}, {"Si": 4, "H": 1}),       # was Si 6 (hydride)
    ({"C": 70, "H": 30}, {"C": 4, "H": 1}),        # was H 6 (carbide)
    ({"Si": 32, "C": 32, "H": 8}, {"Si": 4, "C": 4, "H": 1}),
    ({"Li": 16, "H": 16}, {"Li": 6}),              # hydride unchanged
])
def test_target_cn(composition, expected):
    target_cn, tol = auto_target_cn(composition)
    assert target_cn == expected and tol == 0


def test_no_oxidation_states():
    """Charge balance against H- made Si50H50 Si+1."""
    assert infer_oxidation_state("Si", {"Si": 50, "H": 50}) is None
    assert infer_oxidation_state("Li", {"Li": 16, "H": 16}) == 1


# ── Minimum separations and bonds ────────────────────────────────

@pytest.mark.parametrize("composition, pair, bond", [
    ({"C": 70, "H": 30}, "C-C", 1.54),     # was 2.24 (C4- anion packing)
    ({"C": 70, "H": 30}, "C-H", 1.09),     # was 2.24
    ({"Si": 64, "H": 8}, "H-Si", 1.48),
    ({"Si": 64, "H": 8}, "Si-Si", 2.35),
    ({"Ge": 64, "H": 8}, "Ge-H", 1.53),
])
def test_bond_minsep(composition, pair, bond):
    target_cn, _ = auto_target_cn(composition)
    ms = default_minsep(_symbols(composition), target_cn=target_cn)
    assert 0.75 * bond < ms[pair] < 0.9 * bond


@pytest.mark.parametrize("composition, geminal", [
    ({"C": 70, "H": 30}, 1.78),            # H-C-H of a CH2
    ({"Si": 64, "H": 8}, 2.42),            # H-Si-H of a SiH2
    ({"Si": 32, "C": 32, "H": 8}, 1.78),
])
def test_h_h_floor(composition, geminal):
    """Two H on one host atom can be placed; an H2 molecule (0.74 A) cannot
    (was 2.24 A, which forbade every CH2)."""
    ms = default_minsep(_symbols(composition))
    assert 0.74 < ms["H-H"] < geminal


@pytest.mark.parametrize("composition, host", [
    ({"Si": 64, "H": 8}, {"Si": 64}),
    ({"C": 70, "H": 30}, {"C": 70}),
    ({"Si": 32, "C": 32, "H": 8}, {"Si": 32, "C": 32}),   # SiC keeps C-C apart
    ({"Si": 32, "Ge": 32, "H": 8}, {"Si": 32, "Ge": 32}),
])
def test_host_keeps_its_own_floors(composition, host):
    target_cn, _ = auto_target_cn(composition)
    ms = default_minsep(_symbols(composition), target_cn=target_cn)
    host_cn, _ = auto_target_cn(host)
    ms_host = default_minsep(_symbols(host), target_cn=host_cn)
    assert {k: v for k, v in ms.items() if "H" not in k.split("-")} == ms_host


@pytest.mark.parametrize("composition, bonds", [
    # the host bonds as it does without H (Si-Si as in a-Si, Si-Ge as in
    # SiGe, only Si-C in SiC); H counted as an anion took Si-Si and Si-Ge away
    ({"Si": 64, "H": 8}, {"Si-Si", "H-Si"}),
    ({"C": 70, "H": 30}, {"C-C", "C-H"}),
    ({"Si": 32, "C": 32, "H": 8}, {"C-Si", "C-H", "H-Si"}),
    ({"Si": 32, "Ge": 32, "H": 8}, {"Ge-Si", "Ge-H", "H-Si"}),
])
def test_bonds_of_the_placement(composition, bonds):
    target_cn, _ = auto_target_cn(composition)
    ms = default_minsep(_symbols(composition), target_cn=target_cn)
    dmax = _auto_dmax(ms, target_cn, composition=composition)
    assert set(dmax) == bonds
    for pair in bonds:
        a, b = pair.split("-")
        assert dmax[pair] == pytest.approx((1.2 if a == b else 1.5) * ms[pair])


def test_auto_derive_summary():
    comp = {"C": 70, "H": 30}
    target_cn, _ = auto_target_cn(comp)
    ms = default_minsep(_symbols(comp), target_cn=target_cn)
    line = format_auto_derive_summary(comp, target_cn, ms, 1.84, 9.23)
    assert "→ hydrogenated_network" in line and "OS{" not in line
    assert "C-C:1.22 covalent" in line and "C-H:0.86 covalent" in line
    assert "H-H:1.21 geminal" in line


def test_compute_dimers_a_c_h():
    """compute_dimers reads the floors from the element set; a-C:H's C-C
    bonds and CH2 pairs are not dimers (were, at 0.85 x 2.24 A), H2 is."""
    from amorphgen.analysis.structure import compute_dimers

    def cell(pair_syms, d):
        syms = list(pair_syms) + ["C", "H"]
        pos = [[1.0, 1.0, 1.0], [1.0 + d, 1.0, 1.0], [8.0, 8.0, 8.0],
               [12.0, 12.0, 12.0]]
        return Atoms(syms, positions=pos, cell=[20, 20, 20], pbc=True)

    assert compute_dimers([cell("CC", 1.54)])["total"] == 0
    assert compute_dimers([cell("HH", 1.78)])["total"] == 0
    assert "H-H" in compute_dimers([cell("HH", 0.74)])["pairs"]


# ── Generated structures ─────────────────────────────────────────

@pytest.mark.parametrize("composition, host_bond", [
    ({"Si": 64, "H": 8}, {"Si": 1.48}),
    ({"C": 70, "H": 30}, {"C": 1.09}),
    ({"Si": 32, "C": 32, "H": 8}, {"Si": 1.48, "C": 1.09}),
])
def test_generated_cells(composition, host_bond):
    """Placement used to fail after four cell expansions. Now the cell is
    placed at the estimated density and every H is bonded to a host."""
    for seed in (1, 2):
        atoms = generate_random(composition, seed=seed)
        rho = atoms.get_masses().sum() * 1.66054 / atoms.get_volume()
        assert rho == pytest.approx(_density(composition), rel=1e-6)
        s = np.array(atoms.get_chemical_symbols())
        i, j, d = neighbor_list("ijd", atoms, 2.0)
        for k in np.where(s == "H")[0]:
            near = [(s[jj], dd) for ii, jj, dd in zip(i, j, d) if ii == k]
            assert any(sym in host_bond and dd <= 1.25 * host_bond[sym]
                       for sym, dd in near), (seed, k, near)
