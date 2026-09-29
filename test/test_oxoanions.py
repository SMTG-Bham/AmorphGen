"""Random generation of oxoanion compounds (issue #17).

The centre of an oxoanion (P in a phosphate, S in a sulfate, C in a carbonate,
N in a nitrate, Cl in a perchlorate) and the H of a hydroxide are nonmetals, so
the radii rules used to treat them as anions: P-O was kept 2.46 A apart against
a 1.53 A bond, the S of Li2SO4 was counted as S2- (Li+5), and no generated P or
C had an O within bonding distance. Borates and K/Ba silicates had the related
fault of treating a cation-cation pair (Na-B, K-Si) as an ionic bond.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.neighborlist import neighbor_list

from amorphgen.utils.radii import (
    _classify_compound, auto_target_cn, cation_nonmetals, classify_bond,
    default_minsep, estimate_cell_length, format_auto_derive_summary,
    infer_oxidation_state,
)
from amorphgen.pipeline.random_gen import _auto_dmax, generate_random


def _symbols(composition):
    return [s for s, n in composition.items() for _ in range(n)]


def _density(composition):
    L = estimate_cell_length(composition)
    m = sum(atomic_masses[atomic_numbers[e]] * n for e, n in composition.items())
    return m * 1.66054 / L ** 3


# ── Which nonmetals are cations ──────────────────────────────────

@pytest.mark.parametrize("composition, expected", [
    ({"Li": 3, "P": 1, "O": 4}, {"P"}),            # phosphate
    ({"Li": 2, "S": 1, "O": 4}, {"S"}),            # sulfate
    ({"Ca": 1, "C": 1, "O": 3}, {"C"}),            # carbonate
    ({"Na": 1, "N": 1, "O": 3}, {"N"}),            # nitrate
    ({"Na": 1, "Cl": 1, "O": 4}, {"Cl"}),          # perchlorate
    ({"K": 1, "I": 1, "O": 3}, {"I"}),             # iodate
    ({"Na": 1, "O": 1, "H": 1}, {"H"}),            # hydroxide
    ({"Li": 1, "H": 2, "P": 1, "O": 4}, {"H", "P"}),
    ({"P": 2, "O": 5}, {"P"}),                     # no metal at all
    ({"Li": 3, "P": 1, "S": 4}, {"P"}),            # thiophosphate
    ({"Li": 29, "P": 10, "O": 33, "N": 5}, {"P"}),  # LiPON: N stays an anion
    # ... and the nonmetals that stay anions
    ({"Si": 1, "C": 1}, set()),                    # carbide
    ({"Si": 4, "O": 6, "C": 1}, set()),            # oxycarbide: the carbide C
    ({"Si": 2, "C": 1, "N": 2}, set()),            # SiCN
    ({"C": 70, "H": 30}, set()),                   # a-C:H: H is less electronegative
    ({"In": 1, "P": 1}, set()),
    ({"Ni": 80, "P": 20}, set()),
    ({"Li": 1, "H": 1}, set()),                    # hydride
    ({"La": 2, "O": 2, "S": 1}, set()),            # oxysulfide
    ({"Na": 1, "Ta": 1, "O": 1, "Cl": 4}, set()),  # oxychloride
    ({"Ta": 1, "O": 1, "N": 1}, set()),            # oxynitride
])
def test_cation_nonmetals(composition, expected):
    assert cation_nonmetals(composition) == expected


# ── Oxidation states, classes, target CN ─────────────────────────

@pytest.mark.parametrize("composition, sym, expected", [
    ({"Li": 2, "S": 1, "O": 4}, "Li", 1),       # was 5 (S counted as S2-)
    ({"Li": 2, "S": 1, "O": 4}, "S", 6),
    ({"Na": 2, "S": 1, "O": 3}, "S", 4),        # sulfite
    ({"Ca": 1, "S": 1, "O": 4}, "Ca", 2),       # was 10
    ({"Na": 1, "N": 1, "O": 3}, "Na", 1),       # was 9
    ({"Na": 1, "N": 1, "O": 3}, "N", 5),
    ({"Na": 1, "N": 1, "O": 2}, "N", 3),        # nitrite
    ({"Na": 1, "Cl": 1, "O": 4}, "Cl", 7),
    ({"K": 1, "I": 1, "O": 3}, "I", 5),
    ({"Li": 3, "P": 1, "O": 4}, "P", 5),
    ({"Ca": 1, "C": 1, "O": 3}, "C", 4),
    ({"Na": 1, "O": 1, "H": 1}, "Na", 1),       # was 3 (H counted as H-)
    ({"Na": 2, "Te": 1, "O": 3}, "Na", 1),      # tellurite, was 4
    ({"Si": 4, "O": 6, "C": 1}, "C", None),     # the carbide C is an anion
])
def test_oxidation_states(composition, sym, expected):
    assert infer_oxidation_state(sym, composition) == expected


@pytest.mark.parametrize("composition, expected", [
    ({"Li": 16, "S": 8, "O": 32}, "metal_oxide"),    # was high_valent_oxide (Li+5)
    ({"Ca": 8, "S": 8, "O": 32}, "metal_oxide"),     # was high_valent_oxide
    ({"Na": 12, "N": 12, "O": 36}, "metal_oxide"),   # was high_valent_oxide
    ({"Na": 6, "Cl": 6, "O": 24}, "metal_oxide"),    # was oxyhalide
    ({"K": 6, "I": 6, "O": 18}, "metal_oxide"),      # was oxyhalide
    ({"Mg": 12, "O": 24, "H": 24}, "metal_oxide"),   # was high_valent_oxide
    ({"Be": 8, "S": 8, "O": 32}, "metal_oxide"),     # was the BeO covalent network
    ({"P": 8, "O": 20}, "covalent_oxide"),           # unchanged
    ({"S": 8, "O": 24}, "covalent_oxide"),
    ({"V": 8, "P": 8, "O": 40}, "high_valent_oxide"),  # high-valent through its V
    ({"Bi": 16, "O": 16, "Cl": 16}, "oxyhalide"),    # real oxyhalides unchanged
    ({"Sb": 16, "O": 45, "Cl": 5}, "oxyhalide"),     # needs Sb(V) as Sb's top state
])
def test_classes(composition, expected):
    assert _classify_compound(composition) == expected


@pytest.mark.parametrize("composition, sym, cn", [
    ({"Li": 3, "P": 1, "O": 4}, "P", 4),
    ({"Li": 2, "S": 1, "O": 4}, "S", 4),
    ({"Na": 2, "S": 1, "O": 3}, "S", 3),        # pyramidal sulfite
    ({"Ca": 1, "C": 1, "O": 3}, "C", 3),
    ({"Na": 1, "N": 1, "O": 3}, "N", 3),
    ({"Na": 1, "N": 1, "O": 2}, "N", 2),        # bent nitrite
    ({"Na": 1, "Cl": 1, "O": 4}, "Cl", 4),
    ({"K": 1, "I": 1, "O": 3}, "I", 3),
    ({"Na": 1, "O": 1, "H": 1}, "H", 1),
    ({"P": 2, "O": 5}, "P", 4),
])
def test_target_cn(composition, sym, cn):
    target_cn, _ = auto_target_cn(composition)
    assert target_cn[sym] == cn


def test_target_cn_keeps_metal_rules():
    """The metal cations keep their class rule; only the centre is added.
    P2O5 has no metal, so its P is strict (tolerance 0)."""
    cn, tol = auto_target_cn({"Li": 24, "P": 8, "O": 32})
    assert cn == {"Li": 5, "P": 4} and tol == 1
    cn, tol = auto_target_cn({"Li": 16, "S": 8, "O": 32})
    assert cn == {"Li": 5, "S": 4}               # was Li 6 (high_valent_oxide)
    cn, tol = auto_target_cn({"P": 16, "O": 40})
    assert cn == {"P": 4} and tol == 0           # was {} (no SC placement)


# ── Minimum separations ──────────────────────────────────────────

@pytest.mark.parametrize("composition, pair, bond", [
    ({"Li": 3, "P": 1, "O": 4}, "O-P", 1.53),
    ({"Li": 2, "S": 1, "O": 4}, "O-S", 1.47),
    ({"Ca": 1, "C": 1, "O": 3}, "C-O", 1.285),
    ({"Na": 1, "N": 1, "O": 3}, "N-O", 1.25),
    ({"Na": 1, "Cl": 1, "O": 4}, "Cl-O", 1.44),
    ({"K": 1, "I": 1, "O": 3}, "I-O", 1.81),
    ({"Na": 1, "O": 1, "H": 1}, "H-O", 0.97),
    ({"P": 2, "O": 5}, "O-P", 1.53),
    ({"Li": 3, "P": 1, "S": 4}, "P-S", 2.04),
    ({"Li": 29, "P": 10, "O": 33, "N": 5}, "N-P", 1.65),
])
def test_centre_ligand_minsep_below_the_bond(composition, pair, bond):
    """Was P-O 2.46, S-O 2.27, C-O 2.24, H-O 2.24 A (both treated as anions).
    Now the Shannon cation radius gives ~0.8 of the bond, as for Si-O."""
    ms = default_minsep(_symbols(composition))
    assert 0.75 * bond < ms[pair] < 0.9 * bond


@pytest.mark.parametrize("composition, pair", [
    ({"Li": 3, "P": 1, "O": 4}, "Li-P"),
    ({"Ca": 1, "C": 1, "O": 3}, "C-Ca"),
    ({"Li": 2, "S": 1, "O": 4}, "Li-S"),
    ({"Na": 1, "N": 1, "O": 3}, "N-Na"),
    ({"Na": 2, "B": 4, "O": 7}, "B-Na"),         # was 0.90 A
    ({"Li": 2, "B": 4, "O": 7}, "B-Li"),         # was 0.70 A
    ({"K": 2, "Si": 1, "O": 3}, "K-Si"),         # was 1.31 A
    ({"Ba": 1, "Si": 1, "O": 3}, "Ba-Si"),       # was 1.29 A
])
def test_cation_pairs_are_second_shell(composition, pair):
    """Two cations only meet across an anion: never closer than the cation's
    own bond to it, and not a bond for the coordination-aware placement."""
    ms = default_minsep(_symbols(composition))
    a, b = pair.split("-")
    assert classify_bond(a, b, composition) in ("cation-cation", "covalent",
                                                "metallic")
    bonds = [v for k, v in ms.items() if (a in k.split("-")) != (b in k.split("-"))
             and "O" in k.split("-")]
    assert ms[pair] > max(bonds)
    target_cn, _ = auto_target_cn(composition)
    dmax = _auto_dmax(ms, target_cn, composition=composition)
    assert pair not in dmax


def test_same_centre_floor_above_its_homonuclear_bond():
    """Two centres of one element keep a floor above their X-X bond length /
    0.85, so neither the placement nor compute_dimers lets a P-P, S-S or C-C
    bond through; the H-H floor still allows the 1.51 A of a water molecule."""
    for comp, pair, bond in [({"Li": 3, "P": 1, "O": 4}, "P-P", 2.21),
                             ({"P": 2, "O": 5}, "P-P", 2.21),
                             ({"Li": 2, "S": 1, "O": 4}, "S-S", 2.05),
                             ({"Ca": 1, "C": 1, "O": 3}, "C-C", 1.54),
                             ({"Na": 1, "N": 1, "O": 3}, "N-N", 1.45)]:
        assert default_minsep(_symbols(comp))[pair] * 0.85 > bond, pair
    h = default_minsep(_symbols({"H": 2, "O": 1}))["H-H"]
    assert 0.74 / 0.85 < h < 1.51 / 0.85


def test_compute_dimers_still_flags_homonuclear_bonds():
    """compute_dimers reads the floors from the element set alone; a P-P or
    S-S bond is still a dimer, a water molecule's H-H is not."""
    from amorphgen.analysis.structure import compute_dimers

    def cell(pair_syms, d, others):
        syms = list(pair_syms) + others
        pos = [[1.0, 1.0, 1.0], [1.0 + d, 1.0, 1.0]]
        pos += [[6.0 + 3.5 * (k % 3), 6.0 + 3.5 * (k // 3), 12.0]
                for k in range(len(others))]
        return Atoms(syms, positions=pos, cell=[20, 20, 20], pbc=True)

    assert "P-P" in compute_dimers([cell("PP", 2.2, ["O"] * 5)])["pairs"]
    assert "P-P" in compute_dimers([cell("PP", 2.2, ["Li", "O", "O"])])["pairs"]
    assert "S-S" in compute_dimers([cell("SS", 2.05, ["O"] * 6)])["pairs"]
    assert "H-H" not in compute_dimers([cell("HH", 1.51, ["O"])])["pairs"]
    assert "H-H" in compute_dimers([cell("HH", 0.74, ["O"])])["pairs"]


def test_unchanged_where_nonmetals_stay_anions():
    """The carbide C of an oxycarbide keeps its anion treatment (bonded to Si,
    kept off O), as does every nonmetal of a compound with no oxoanion."""
    ms = default_minsep(_symbols({"Si": 16, "O": 24, "C": 4}))
    assert ms["C-O"] > 2.0 and ms["C-Si"] < 1.6
    assert "C" not in (auto_target_cn({"Si": 16, "O": 24, "C": 4})[0] or {})
    assert classify_bond("Si", "C", {"Si": 32, "C": 32}) == "covalent"


# ── Density ──────────────────────────────────────────────────────

@pytest.mark.parametrize("composition, crystal_rho", [
    ({"Li": 16, "S": 8, "O": 32}, 2.22),     # was 0.65 of crystal
    ({"Ca": 8, "S": 8, "O": 32}, 2.96),      # was 0.60
    ({"Ca": 12, "C": 12, "O": 36}, 2.71),    # was 0.63
    ({"Li": 16, "C": 8, "O": 24}, 2.11),     # was 0.61
    ({"Na": 12, "N": 12, "O": 36}, 2.26),    # was 0.72
    ({"Na": 6, "Cl": 6, "O": 24}, 2.50),     # was 0.64
    ({"Mg": 12, "O": 24, "H": 24}, 2.34),    # was 0.52
    ({"Be": 8, "S": 8, "O": 32}, 2.44),      # was 1.87 (covalent-network radii)
    ({"Li": 24, "P": 8, "O": 32}, 2.48),     # unchanged, P already P5+
])
def test_salt_density(composition, crystal_rho):
    """The centre is sized as the small cation it is (S6+, C4+, N5+, H+), not
    as S2- / C4- / N3- / H-, so the estimate lands where the other ionic
    classes do, 75-100 % of the crystal density."""
    assert 0.75 <= _density(composition) / crystal_rho <= 1.0


def test_auto_derive_summary_reports_the_roles():
    comp = {"Li": 16, "S": 8, "O": 32}
    target_cn, _ = auto_target_cn(comp)
    ms = default_minsep(_symbols(comp), target_cn=target_cn)
    line = format_auto_derive_summary(comp, target_cn, ms, 1.9, 9.2)
    assert "→ metal_oxide" in line and "OS{Li:+1, S:+6}" in line
    assert "O-S:1.22 ionic" in line and "Li-S:2.21 cation-cation" in line
    assert "S-S:2.58 cation-cation" in line


# ── Generated structures ─────────────────────────────────────────

@pytest.mark.parametrize("composition, centre, bond, min_mean", [
    ({"Li": 24, "P": 8, "O": 32}, "P", 1.53, 2.5),
    ({"Li": 16, "S": 8, "O": 32}, "S", 1.47, 2.5),
    ({"Ca": 12, "C": 12, "O": 36}, "C", 1.285, 1.5),
    ({"Na": 12, "N": 12, "O": 36}, "N", 1.25, 1.5),
    ({"P": 16, "O": 40}, "P", 1.53, 3.0),
])
def test_generated_centres_are_bonded(composition, centre, bond, min_mean):
    """Every generated P had its nearest O at >= 2.43 A, every C at >= 2.25 A.
    Now nearly all centres have O at bonding distance: the placement is
    coordination-aware for them (target CN), as it is for Si in a silicate."""
    for seed in (1, 2):
        atoms = generate_random(composition, seed=seed)
        s = np.array(atoms.get_chemical_symbols())
        i, j, d = neighbor_list("ijd", atoms, 1.25 * bond)
        cns = np.array([np.sum((i == k) & (s[j] == "O"))
                        for k in np.where(s == centre)[0]])
        assert np.mean(cns >= 1) >= 0.85, (seed, cns)
        assert cns.mean() >= min_mean, (seed, cns)
