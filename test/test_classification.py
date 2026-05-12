"""Tests for material classification and classical calculators."""

import pytest
import numpy as np
from ase import Atoms

from amorphgen.utils.radii import _classify_compound, auto_target_cn, default_minsep


# ── Material classification ──────────────────────────────────────

@pytest.mark.parametrize("composition, expected", [
    # Group IV
    ({"Si": 64}, "group_iv"),
    ({"Ge": 64}, "group_iv"),
    ({"Si": 32, "Ge": 32}, "group_iv"),
    # Pnictides (III-V)
    ({"Ga": 16, "As": 16}, "pnictide"),
    ({"In": 16, "P": 16}, "pnictide"),
    ({"In": 16, "As": 16}, "pnictide"),
    ({"Ga": 16, "Sb": 16}, "pnictide"),
    # Chalcogenides
    ({"Zn": 16, "S": 16}, "chalcogenide"),
    ({"Cd": 16, "Te": 16}, "chalcogenide"),
    ({"Ge": 16, "Te": 16}, "chalcogenide"),
    ({"Sb": 8, "Te": 12}, "chalcogenide"),
    ({"Bi": 8, "Te": 12}, "chalcogenide"),
    # Covalent oxides
    ({"Si": 16, "O": 32}, "covalent_oxide"),
    ({"Ge": 16, "O": 32}, "covalent_oxide"),
    ({"B": 8, "O": 12}, "covalent_oxide"),
    # Metal oxides
    ({"Al": 16, "O": 24}, "metal_oxide"),
    ({"In": 16, "O": 24}, "metal_oxide"),
    ({"Ti": 16, "O": 32}, "metal_oxide"),
    ({"Mg": 16, "O": 16}, "metal_oxide"),
    ({"Ca": 16, "O": 16}, "metal_oxide"),
    ({"Zn": 16, "O": 16}, "metal_oxide"),
    # Halides
    ({"Na": 16, "Cl": 16}, "halide"),
    ({"Li": 8, "Zr": 4, "Cl": 24}, "halide"),
    ({"Li": 16, "F": 16}, "halide"),
    # Nitrides
    ({"Al": 16, "N": 16}, "nitride"),
    ({"Ga": 16, "N": 16}, "nitride"),
    ({"Si": 12, "N": 16}, "nitride"),
    # Carbides — split by cation chemistry (2026-05-11):
    # main-group / metalloid + C  → covalent_carbide (SiC, B4C)
    # d-block transition metal + C → transition_metal_carbide (TiC, WC, ZrC)
    ({"Si": 16, "C": 16}, "covalent_carbide"),
    ({"B": 16, "C": 4},   "covalent_carbide"),
    ({"Ti": 16, "C": 16}, "transition_metal_carbide"),
    ({"W": 16, "C": 16},  "transition_metal_carbide"),
    ({"Zr": 16, "C": 16}, "transition_metal_carbide"),
    # Hydrides
    ({"Li": 16, "H": 16}, "hydride"),
    ({"Mg": 8, "H": 16}, "hydride"),
    # Borides
    ({"Ti": 8, "B": 16}, "boride"),
    ({"La": 4, "B": 24}, "boride"),
    # Alloys
    ({"Ni": 16, "Ti": 16}, "alloy"),
    ({"Cu": 16, "Zn": 16}, "alloy"),
    ({"Fe": 12, "Si": 4}, "alloy"),
])
def test_classify_compound(composition, expected):
    assert _classify_compound(composition) == expected


def test_classify_pure_sb_not_pnictide():
    """Pure Sb should not be classified as pnictide."""
    cls = _classify_compound({"Sb": 64})
    assert cls != "pnictide"


def test_classify_bn_is_nitride():
    """BN should classify as nitride (N is nonmetal anion)."""
    assert _classify_compound({"B": 16, "N": 16}) == "nitride"


def test_classify_default_fallback():
    """Unknown composition should return 'default'."""
    # Pure oxygen — O is nonmetal, no cations, no metal, no metalloid
    cls = _classify_compound({"O": 32})
    assert cls == "default"


def test_classify_oxynitride():
    """Oxynitride with metal falls to metal_oxide (O checked first)."""
    cls = _classify_compound({"Si": 12, "Al": 6, "O": 6, "N": 10})
    assert cls == "metal_oxide"


# ── auto_target_cn ───────────────────────────────────────────────

def test_auto_cn_sio2():
    cn, tol = auto_target_cn({"Si": 16, "O": 32})
    assert cn is not None
    assert cn["Si"] == 4


def test_auto_cn_al2o3():
    cn, tol = auto_target_cn({"Al": 16, "O": 24})
    assert cn is not None
    assert cn["Al"] == 5


def test_auto_cn_pure_si():
    cn, tol = auto_target_cn({"Si": 64})
    assert cn is not None
    assert cn["Si"] == 4
    assert tol == 0


def test_auto_cn_gaas():
    cn, tol = auto_target_cn({"Ga": 16, "As": 16})
    assert cn is not None
    assert cn["Ga"] == 4
    assert cn["As"] == 4


# ── default_minsep ───────────────────────────────────────────────

def test_minsep_symmetric():
    """A-B and B-A should give the same key."""
    ms = default_minsep({"Si": 16, "O": 32})
    assert "O-Si" in ms or "Si-O" in ms


def test_minsep_mm_capped():
    """M-M minsep should not exceed _MAX_SAME_ELEMENT_MINSEP."""
    from amorphgen.utils.radii import _MAX_SAME_ELEMENT_MINSEP
    ms = default_minsep({"Ca": 16, "O": 16})
    assert ms["Ca-Ca"] <= _MAX_SAME_ELEMENT_MINSEP


def test_minsep_positive():
    """All minsep values should be positive."""
    ms = default_minsep({"In": 16, "O": 24})
    for v in ms.values():
        assert v > 0


# ── Classical calculators ────────────────────────────────────────

def test_lj_energy_forces():
    """LJ calculator should give negative energy for Ar dimer at equilibrium."""
    from amorphgen.utils.classical import LennardJonesCalculator

    # Two Ar atoms in a box, near LJ minimum (~3.82 A for sigma=3.40)
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [3.82, 0, 0]],
                  cell=[10, 10, 10], pbc=True)
    calc = LennardJonesCalculator(
        params={("Ar", "Ar"): {"epsilon": 0.0104, "sigma": 3.40}},
        cutoff=8.0,
    )
    atoms.calc = calc
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    assert e < 0
    assert f.shape == (2, 3)
    # Forces should be small near equilibrium
    assert np.max(np.abs(f)) < 0.05


def test_buckingham_energy_forces():
    """Buckingham calculator should give finite energy and forces."""
    from amorphgen.utils.classical import BuckinghamCalculator

    # Simple SiO2 cell
    atoms = Atoms("SiO2",
                  positions=[[0, 0, 0], [1.6, 0, 0], [0, 1.6, 0]],
                  cell=[5, 5, 5], pbc=True)
    calc = BuckinghamCalculator(
        params={("Si", "O"): {"A": 18003.7572, "rho": 0.205205, "C": 133.5381},
                ("O", "O"): {"A": 1388.7730, "rho": 0.362319, "C": 175.0}},
        charges={"Si": 2.4, "O": -1.2},
        cutoff=5.0,
    )
    atoms.calc = calc
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    assert np.isfinite(e)
    assert np.all(np.isfinite(f))
    assert f.shape == (3, 3)


def test_buckingham_no_coulomb():
    """Buckingham with coulomb=False should still work."""
    from amorphgen.utils.classical import BuckinghamCalculator

    atoms = Atoms("SiO2",
                  positions=[[0, 0, 0], [1.8, 0, 0], [0, 1.8, 0]],
                  cell=[5, 5, 5], pbc=True)
    calc = BuckinghamCalculator(
        params={("Si", "O"): {"A": 18003.7572, "rho": 0.205205, "C": 133.5381}},
        coulomb=False,
        cutoff=5.0,
    )
    atoms.calc = calc
    e = atoms.get_potential_energy()
    assert np.isfinite(e)


def test_lj_no_stress():
    """LJ calculator should not claim to implement stress."""
    from amorphgen.utils.classical import LennardJonesCalculator
    assert "stress" not in LennardJonesCalculator.implemented_properties


def test_buckingham_no_stress():
    """Buckingham calculator should not claim to implement stress."""
    from amorphgen.utils.classical import BuckinghamCalculator
    assert "stress" not in BuckinghamCalculator.implemented_properties
