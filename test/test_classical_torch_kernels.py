"""Torch kernels of the classical calculators agree with the NumPy kernels, plus their guards."""
from __future__ import annotations

import itertools
import sys
from math import erfc, exp, pi, sqrt

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.neighborlist import neighbor_list

import amorphgen.utils.classical as classical
from amorphgen.utils.classical import BuckinghamCalculator, LennardJonesCalculator

KE = 14.399645                                     # e^2 / (4 pi eps0), eV A
NACL_MADELUNG = -1.747565 * KE / 2.82              # eV per ion pair, a = 5.64 A

LJ_AR = {("Ar", "Ar"): {"epsilon": 0.0104, "sigma": 3.40}}
LJ_AR_KR = {**LJ_AR, ("Kr", "Kr"): {"epsilon": 0.0140, "sigma": 3.65}}   # no Ar-Kr
BUCK_NACL = {("Na", "Cl"): {"A": 2000.0, "rho": 0.30, "C": 0.0},
             ("Cl", "Cl"): {"A": 1000.0, "rho": 0.35, "C": 20.0}}       # no Na-Na
BUCK_SIO2_AR = {("Si", "O"): {"A": 18003.76, "rho": 0.2052, "C": 133.54},
                ("O", "O"): {"A": 1388.77, "rho": 0.3624, "C": 175.0},
                ("Ar", "Ar"): {"A": 1000.0, "rho": 0.30, "C": 50.0}}    # no Si-Si, X-Ar
NACL_Q = {"Na": 1.0, "Cl": -1.0}


def _numpy_kernel_called(*args, **kwargs):
    raise AssertionError("the NumPy kernel ran on the torch path")


@pytest.fixture
def evaluate(monkeypatch):
    """evaluate(atoms, calc, kernel) -> (energy, forces) from the 'numpy' or 'torch' kernels.

    The torch kernels only use ``device`` as a torch device string, so
    routing device='cpu' through them runs the CUDA/MPS code on the CPU.
    """
    pytest.importorskip("torch")
    monkeypatch.setattr(classical, "_torch", None)

    def run(atoms, calc, kernel):
        with monkeypatch.context() as m:
            if kernel == "torch":
                m.setattr(classical, "_use_gpu", lambda device: True)
                for cls in (LennardJonesCalculator, BuckinghamCalculator):
                    m.setattr(cls, "_calc_cpu", staticmethod(_numpy_kernel_called))
            atoms = atoms.copy()
            atoms.calc = calc
            return atoms.get_potential_energy(), atoms.get_forces()
    return run


def _argon(repeat, *, cubic=True, kr_every=None):
    atoms = bulk("Ar", "fcc", a=5.26, cubic=cubic).repeat(repeat)
    if kr_every:
        atoms.symbols[::kr_every] = "Kr"
    atoms.rattle(0.15, seed=1)
    return atoms


def _nacl(cubic=False, repeat=2):
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=cubic).repeat(repeat)
    atoms.rattle(0.15, seed=2)
    return atoms


def _silica_with_argon():
    """3 Si, 6 O (charged, neutral overall) and 3 uncharged Ar on a jittered grid."""
    rng = np.random.default_rng(7)
    grid = np.array(list(itertools.product(range(3), repeat=3))) * 2.5
    sites = rng.choice(len(grid), 12, replace=False)
    return Atoms("Si3O6Ar3", positions=grid[sites] + rng.normal(0.0, 0.1, (12, 3)),
                 cell=[7.5, 7.5, 7.5], pbc=True)


def _has_own_image_pairs(atoms, cutoff):
    i, j = neighbor_list("ij", atoms, cutoff)
    return bool(np.any(i == j))


def _assert_same(numpy_result, torch_result):
    (e_np, f_np), (e_t, f_t) = numpy_result, torch_result
    assert np.abs(f_np).max() > 1e-3               # a non-trivial comparison
    assert e_t == pytest.approx(e_np, rel=1e-12, abs=1e-10)
    assert f_t.dtype == np.float64
    np.testing.assert_allclose(f_t, f_np, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("atoms, params, cutoff, own_images", [
    (_argon(2), LJ_AR, 5.0, False),
    (_argon(1), LJ_AR, 8.0, True),
    (_argon((2, 2, 2), cubic=False), LJ_AR, 8.0, True),
    (_argon(2, kr_every=3), LJ_AR_KR, 5.0, False),
], ids=["cutoff<L/2", "own-images", "triclinic", "pair-without-params"])
def test_lennard_jones_torch_kernel_matches_numpy(evaluate, atoms, params, cutoff, own_images):
    assert _has_own_image_pairs(atoms, cutoff) is own_images
    def make():
        return LennardJonesCalculator(params, cutoff=cutoff, device="cpu")
    _assert_same(evaluate(atoms, make(), "numpy"), evaluate(atoms, make(), "torch"))


@pytest.mark.parametrize("method", ["ewald", "wolf"])
@pytest.mark.parametrize("atoms, params, charges, cutoff, own_images", [
    (_nacl(), BUCK_NACL, NACL_Q, 6.0, False),
    (_nacl(cubic=True, repeat=1), BUCK_NACL, NACL_Q, 7.0, True),
    (_silica_with_argon(), BUCK_SIO2_AR, {"Si": 2.4, "O": -1.2}, 6.0, False),
    (_nacl(), BUCK_NACL, None, 6.0, False),
    (_nacl(), {}, NACL_Q, 6.0, False),
], ids=["nacl", "nacl-own-images", "charged-and-neutral-species", "no-charges",
        "coulomb-only"])
def test_buckingham_torch_kernel_matches_numpy(evaluate, method, atoms, params, charges,
                                               cutoff, own_images):
    assert _has_own_image_pairs(atoms, cutoff) is own_images
    def make():
        return BuckinghamCalculator(params, charges=charges, cutoff=cutoff,
                                    coulomb_method=method, device="cpu")
    _assert_same(evaluate(atoms, make(), "numpy"), evaluate(atoms, make(), "torch"))


def test_torch_kernel_lennard_jones_lattice_sum_with_own_images(evaluate):
    """A one-atom fcc cell interacts only with its own images: brute-force lattice sum."""
    eps, sig, cutoff = 0.0104, 3.40, 8.0
    atoms = bulk("Ar", "fcc", a=5.26)
    n = np.array(list(itertools.product(range(-4, 5), repeat=3)))
    r = np.linalg.norm(n @ atoms.cell[:], axis=1)
    r = r[(r > 0) & (r < cutoff)]
    expected = 0.5 * np.sum(4 * eps * ((sig / r) ** 12 - (sig / r) ** 6))
    energy, forces = evaluate(atoms, LennardJonesCalculator(LJ_AR, cutoff=cutoff), "torch")
    assert energy == pytest.approx(expected, rel=1e-12)
    np.testing.assert_allclose(forces, 0.0, atol=1e-12)


@pytest.mark.parametrize("cubic", [False, True])
def test_torch_kernel_ewald_gives_nacl_madelung_energy(evaluate, cubic):
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=cubic)      # 2 or 8 ions, cutoff > L
    calc = BuckinghamCalculator({}, charges=NACL_Q, cutoff=7.0)
    energy, forces = evaluate(atoms, calc, "torch")
    assert energy / (len(atoms) // 2) == pytest.approx(NACL_MADELUNG, abs=2e-3)
    np.testing.assert_allclose(forces, 0.0, atol=1e-6)


@pytest.mark.parametrize("formula, make_calc", [
    ("Ar2", lambda: LennardJonesCalculator(LJ_AR, cutoff=6.0, device="cuda")),
    ("NaCl", lambda: BuckinghamCalculator(BUCK_NACL, charges=NACL_Q, cutoff=6.0,
                                          device="cuda")),
], ids=["lennard-jones", "buckingham"])
def test_gpu_device_without_torch_names_both_remedies(monkeypatch, formula, make_calc):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(classical, "_torch", None)
    atoms = Atoms(formula, positions=[(0, 0, 0), (3.5, 0, 0)], cell=[12, 12, 12], pbc=True)
    atoms.calc = make_calc()
    with pytest.raises(ImportError, match="requires PyTorch") as exc:
        atoms.get_potential_energy()
    assert "pip install torch" in str(exc.value)
    assert "device='cpu'" in str(exc.value)
    assert exc.value.__cause__ is None


def test_lennard_jones_with_no_parameterised_pair_in_range_is_zero():
    atoms = Atoms("ArKr", positions=[(0, 0, 0), (3.8, 0, 0)], cell=[12, 12, 12], pbc=True)
    atoms.calc = LennardJonesCalculator(LJ_AR_KR, cutoff=6.0)
    assert atoms.get_potential_energy() == 0.0
    np.testing.assert_array_equal(atoms.get_forces(), np.zeros((2, 3)))


def test_ewald_refuses_a_cluster_and_wolf_gives_the_damped_shifted_pair_energy():
    r, rc, alpha = 2.8, 12.0, 0.2
    dimer = Atoms("NaCl", positions=[(0, 0, 0), (r, 0, 0)], cell=[20, 20, 20], pbc=False)
    dimer.calc = BuckinghamCalculator({}, charges=NACL_Q, cutoff=rc)
    with pytest.raises(ValueError, match="periodic cell"):
        dimer.get_potential_energy()

    dimer.calc = BuckinghamCalculator({}, charges=NACL_Q, cutoff=rc, coulomb_method="wolf")
    pair = -KE * (erfc(alpha * r) / r - erfc(alpha * rc) / rc)
    self_term = -KE * 2 * (erfc(alpha * rc) / (2 * rc) + alpha / sqrt(pi))
    dpair_dr = -KE * (-erfc(alpha * r) / r ** 2
                      - 2 * alpha / sqrt(pi) * exp(-(alpha * r) ** 2) / r)
    assert dimer.get_potential_energy() == pytest.approx(pair + self_term, rel=1e-5)
    np.testing.assert_allclose(dimer.get_forces(),
                               [[dpair_dr, 0, 0], [-dpair_dr, 0, 0]], rtol=1e-5)
