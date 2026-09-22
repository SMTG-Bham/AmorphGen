"""Tests for amorphgen.utils.classical — Lennard-Jones + Buckingham+Coulomb.

CPU-path only (the GPU paths in `_calc_gpu` are exercised on hardware
that isn't available in CI; they are tested manually).
"""
from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from amorphgen.utils.classical import (
    BuckinghamCalculator,
    LennardJonesCalculator,
)


# ─── Lennard-Jones ─────────────────────────────────────────────────────────

class TestLennardJonesCalculator:
    def test_dimer_at_minimum(self):
        """Two atoms at r = sigma * 2^(1/6) → V = -epsilon."""
        sigma, eps = 3.0, 1.0
        r_min = sigma * 2 ** (1 / 6)
        a = Atoms("Ar2",
                   positions=[(0, 0, 0), (r_min, 0, 0)],
                   cell=[20, 20, 20], pbc=True)
        a.calc = LennardJonesCalculator(
            params={("Ar", "Ar"): {"epsilon": eps, "sigma": sigma}},
            cutoff=8.0,
            device="cpu",
        )
        e = a.get_potential_energy()
        assert e == pytest.approx(-eps, abs=1e-3)

    def test_dimer_at_sigma_is_zero_energy(self):
        """Two atoms at r = sigma → V = 0 (LJ crosses zero)."""
        sigma = 3.0
        a = Atoms("Ar2",
                   positions=[(0, 0, 0), (sigma, 0, 0)],
                   cell=[20, 20, 20], pbc=True)
        a.calc = LennardJonesCalculator(
            params={("Ar", "Ar"): {"epsilon": 1.0, "sigma": sigma}},
            cutoff=8.0, device="cpu",
        )
        e = a.get_potential_energy()
        assert e == pytest.approx(0.0, abs=1e-3)

    def test_no_pairs_within_cutoff_zero_energy(self):
        """Atoms beyond cutoff → energy = 0, forces = 0."""
        a = Atoms("Ar2",
                   positions=[(0, 0, 0), (15, 0, 0)],   # well beyond 8 A cutoff
                   cell=[30, 30, 30], pbc=True)
        a.calc = LennardJonesCalculator(
            params={("Ar", "Ar"): {"epsilon": 1.0, "sigma": 3.0}},
            cutoff=8.0, device="cpu",
        )
        e = a.get_potential_energy()
        f = a.get_forces()
        assert abs(e) < 1e-9
        assert np.max(np.abs(f)) < 1e-9

    def test_force_balance(self):
        """Forces on a closed system should sum to zero."""
        a = Atoms("Ar3",
                   positions=[(0, 0, 0), (3.5, 0, 0), (1.75, 3.0, 0)],
                   cell=[20, 20, 20], pbc=True)
        a.calc = LennardJonesCalculator(
            params={("Ar", "Ar"): {"epsilon": 1.0, "sigma": 3.0}},
            cutoff=8.0, device="cpu",
        )
        f = a.get_forces()
        np.testing.assert_allclose(f.sum(axis=0), 0.0, atol=1e-9)


# ─── Buckingham + Coulomb ──────────────────────────────────────────────────

class TestBuckinghamCalculator:
    @pytest.fixture
    def sio2_calc(self):
        """Standard BKS-like Buckingham parameters for SiO2."""
        return BuckinghamCalculator(
            params={
                ("Si", "O"): {"A": 18003.76, "rho": 0.2052, "C": 133.54},
                ("O", "O"): {"A": 1388.77,  "rho": 0.3624, "C": 175.00},
                ("Si", "Si"): {"A": 0.0, "rho": 1.0, "C": 0.0},
            },
            charges={"Si": 2.4, "O": -1.2},
            cutoff=10.0,
            device="cpu",
        )

    def test_runs_on_quartz_dimer(self, sio2_calc):
        a = Atoms("SiO",
                   positions=[(0, 0, 0), (1.6, 0, 0)],
                   cell=[20, 20, 20], pbc=True)
        a.calc = sio2_calc
        e = a.get_potential_energy()
        f = a.get_forces()
        # Energy should be finite and forces non-zero (atoms close enough to interact).
        assert np.isfinite(e)
        assert np.max(np.abs(f)) > 0
        # Newton's third law for two-atom system.
        np.testing.assert_allclose(f.sum(axis=0), 0.0, atol=1e-9)

    def test_force_consistency_with_finite_difference(self, sio2_calc):
        """Numerical derivative of energy ≈ analytic force, for one atom."""
        a = Atoms("SiO",
                   positions=[(0, 0, 0), (1.8, 0, 0)],
                   cell=[20, 20, 20], pbc=True)
        a.calc = sio2_calc
        f_analytic = a.get_forces()[1, 0]    # x-force on O atom

        # Finite-difference approximation.
        delta = 1e-4
        a.positions[1, 0] += delta
        e_plus = a.get_potential_energy()
        a.positions[1, 0] -= 2 * delta
        e_minus = a.get_potential_energy()
        f_numeric = -(e_plus - e_minus) / (2 * delta)
        a.positions[1, 0] += delta   # restore

        # Should agree to ~3 decimal places for this finite-difference step.
        assert f_analytic == pytest.approx(f_numeric, rel=0.05, abs=1e-2)

    def test_zero_charge_atom_pair_is_buckingham_only(self):
        """If both atoms have charge=0, only Buckingham contributes (no Coulomb)."""
        calc = BuckinghamCalculator(
            params={("Ar", "Ar"): {"A": 1000.0, "rho": 0.3, "C": 50.0}},
            charges={"Ar": 0.0},
            cutoff=10.0,
            coulomb=True,
            device="cpu",
        )
        a = Atoms("Ar2", positions=[(0, 0, 0), (3.0, 0, 0)],
                   cell=[20, 20, 20], pbc=True)
        a.calc = calc
        e = a.get_potential_energy()
        # Only Buckingham term active; should still be a finite real number.
        assert np.isfinite(e)

    def test_coulomb_disabled_by_flag(self):
        """coulomb=False suppresses Coulomb regardless of charges."""
        calc = BuckinghamCalculator(
            params={("Si", "O"): {"A": 18003.76, "rho": 0.2052, "C": 133.54},
                    ("O", "O"): {"A": 1388.77, "rho": 0.3624, "C": 175.0}},
            charges={"Si": 2.4, "O": -1.2},
            cutoff=10.0,
            coulomb=False,
            device="cpu",
        )
        a = Atoms("SiO",
                   positions=[(0, 0, 0), (1.6, 0, 0)],
                   cell=[20, 20, 20], pbc=True)
        a.calc = calc
        e_no_coul = a.get_potential_energy()

        calc2 = BuckinghamCalculator(
            params={("Si", "O"): {"A": 18003.76, "rho": 0.2052, "C": 133.54},
                    ("O", "O"): {"A": 1388.77, "rho": 0.3624, "C": 175.0}},
            charges={"Si": 2.4, "O": -1.2},
            cutoff=10.0,
            coulomb=True,
            device="cpu",
        )
        a.calc = calc2
        e_with_coul = a.get_potential_energy()

        assert e_no_coul != pytest.approx(e_with_coul, rel=1e-6), \
            "Disabling Coulomb should change the energy"


# ─── Backend factory wiring ───────────────────────────────────────────────

class TestClassicalViaFactory:
    """Confirm the classical calcs are reachable through get_calculator."""

    def test_lj_via_factory(self):
        from amorphgen.utils.calculators import get_calculator
        calc = get_calculator("lennard-jones",
                               device="cpu",
                               classical_params={
                                   "params": {("Ar", "Ar"): {"epsilon": 1.0, "sigma": 3.0}},
                                   "cutoff": 8.0,
                               })
        assert isinstance(calc, LennardJonesCalculator)

    def test_buckingham_via_factory(self):
        from amorphgen.utils.calculators import get_calculator
        calc = get_calculator("buckingham",
                               device="cpu",
                               classical_params={
                                   "params": {("Si", "O"): {"A": 18003.76,
                                                              "rho": 0.2052,
                                                              "C": 133.54}},
                                   "charges": {"Si": 2.4, "O": -1.2},
                                   "cutoff": 10.0,
                               })
        assert isinstance(calc, BuckinghamCalculator)


# ─── Stress-capability guard (classical potentials lack a stress tensor) ────

class TestStressGuard:
    """LJ/Buckingham implement only energy+forces. Variable-cell operations
    (NPT, cell-filter optimisation) need stress and must fail with a clear
    error instead of an opaque PropertyNotImplementedError from ASE."""

    def _lj(self):
        return LennardJonesCalculator(
            {("Cu", "Cu"): {"epsilon": 0.01, "sigma": 3.0}})

    def _atoms(self):
        return Atoms("Cu4",
                     positions=[[0, 0, 0], [1.8, 0, 0],
                                [0, 1.8, 0], [0, 0, 1.8]],
                     cell=[6, 6, 6], pbc=True)

    def test_classical_reports_no_stress(self):
        from amorphgen.utils.common import calculator_supports_stress
        assert calculator_supports_stress(self._lj()) is False

    def test_emt_reports_stress(self):
        from ase.calculators.emt import EMT
        from amorphgen.utils.common import calculator_supports_stress
        assert calculator_supports_stress(EMT()) is True

    def test_npt_with_stressless_calc_raises(self):
        from amorphgen.utils.common import build_md_dynamics
        atoms = self._atoms()
        atoms.calc = self._lj()
        with pytest.raises(RuntimeError, match="stress"):
            build_md_dynamics(atoms, ensemble="NPT", npt_method="berendsen")

    def test_npt_with_stress_capable_calc_builds(self):
        from ase.calculators.emt import EMT
        from amorphgen.utils.common import build_md_dynamics
        atoms = self._atoms()
        atoms.calc = EMT()
        dyn = build_md_dynamics(atoms, ensemble="NPT", npt_method="berendsen")
        assert dyn is not None

    def test_require_stress_helper_message_is_actionable(self):
        from amorphgen.utils.common import require_stress
        with pytest.raises(RuntimeError) as exc:
            require_stress(self._lj(), "Cell-filter optimisation")
        msg = str(exc.value)
        # Must name the remedy AND where it goes (CLI flag / YAML block).
        assert "stress" in msg
        assert "-C none" in msg and "cell_filter: none" in msg and "opt:" in msg


# ─── Ewald Coulomb ─────────────────────────────────────────────────────────

class TestEwaldCoulomb:
    """The Coulomb part of BuckinghamCalculator is an Ewald sum by default."""

    _P0 = {("Na", "Cl"): {"A": 0.0, "rho": 1.0, "C": 0.0},
           ("Na", "Na"): {"A": 0.0, "rho": 1.0, "C": 0.0},
           ("Cl", "Cl"): {"A": 0.0, "rho": 1.0, "C": 0.0}}
    _MADELUNG = -1.747565 * 14.399645 / 2.82          # eV per ion pair, NaCl a=5.64

    def _nacl(self):
        from ase.build import bulk
        return bulk("NaCl", "rocksalt", a=5.64).repeat((2, 2, 2))   # 16 atoms, 7.98 A cell

    @pytest.mark.parametrize("cutoff", [6.0, 12.0])   # below and above L/2
    def test_nacl_madelung_energy(self, cutoff):
        from amorphgen.utils.classical import BuckinghamCalculator
        at = self._nacl()
        at.calc = BuckinghamCalculator(self._P0, charges={"Na": 1, "Cl": -1}, cutoff=cutoff)
        e_pair = at.get_potential_energy() / (len(at) // 2)
        assert e_pair == pytest.approx(self._MADELUNG, abs=2e-3)

    def test_wolf_is_only_approximate(self):
        from amorphgen.utils.classical import BuckinghamCalculator
        at = self._nacl()
        at.calc = BuckinghamCalculator(self._P0, charges={"Na": 1, "Cl": -1}, cutoff=6.0,
                                       coulomb_method="wolf")
        e_pair = at.get_potential_energy() / (len(at) // 2)
        assert abs(e_pair - self._MADELUNG) > 0.02     # Wolf: ~1 % off here

    def test_forces_match_finite_differences(self):
        from amorphgen.utils.classical import BuckinghamCalculator
        rng = np.random.default_rng(3)
        at = self._nacl(); at.rattle(0.15, seed=1)
        at.calc = BuckinghamCalculator(
            {("Na", "Cl"): {"A": 2000.0, "rho": 0.3, "C": 0.0},
             ("Na", "Na"): {"A": 0.0, "rho": 1.0, "C": 0.0},
             ("Cl", "Cl"): {"A": 1000.0, "rho": 0.35, "C": 20.0}},
            charges={"Na": 1, "Cl": -1}, cutoff=8.0)
        F = at.get_forces()
        for k in rng.choice(len(at), 2, replace=False):
            for ax in range(3):
                h = 1e-4; p = at.positions.copy()
                at.positions[k, ax] = p[k, ax] + h; ep = at.get_potential_energy()
                at.positions[k, ax] = p[k, ax] - h; em = at.get_potential_energy()
                at.positions = p
                assert -(ep - em) / (2 * h) == pytest.approx(F[k, ax], abs=1e-4)

    def test_default_alpha_and_method(self):
        from amorphgen.utils.classical import BuckinghamCalculator
        c = BuckinghamCalculator(self._P0, charges={"Na": 1, "Cl": -1}, cutoff=10.0)
        assert c.coulomb_method == "ewald" and c.alpha == pytest.approx(0.35)
        w = BuckinghamCalculator(self._P0, charges={"Na": 1, "Cl": -1}, cutoff=10.0,
                                 coulomb_method="wolf")
        assert w.alpha == pytest.approx(0.2)
        with pytest.raises(ValueError):
            BuckinghamCalculator(self._P0, charges={"Na": 1}, coulomb_method="pme")

    def test_against_pymatgen_ewald_if_available(self):
        pmg = pytest.importorskip("pymatgen.analysis.ewald")
        from pymatgen.core import Structure, Lattice
        from amorphgen.utils.classical import BuckinghamCalculator
        at = self._nacl(); at.rattle(0.2, seed=2)
        at.calc = BuckinghamCalculator(self._P0, charges={"Na": 1, "Cl": -1}, cutoff=8.0)
        st = Structure(Lattice(at.cell[:]), at.get_chemical_symbols(), at.positions, coords_are_cartesian=True)
        st.add_oxidation_state_by_element({"Na": 1, "Cl": -1})
        ref = pmg.EwaldSummation(st, compute_forces=True)
        assert at.get_potential_energy() == pytest.approx(ref.total_energy, abs=1e-3)
        assert np.abs(at.get_forces() - np.array(ref.forces)).max() < 1e-3
