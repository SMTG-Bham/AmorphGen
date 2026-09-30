"""
tests/test_divergence_guard.py
-------------------------------
Tier 2 tests for the numerical-divergence guard (assert_finite / DivergenceError).

A foundation-model MLIP that goes out-of-distribution in the high-T liquid
regime returns NaN/Inf energies and forces. Without a guard those silently
propagate into a saved structure or trajectory — a wrong-but-plausible result.
These tests pin the guard that turns that into an eager, actionable failure on
both the MD path (attach_outputs observer) and the relaxation path
(opt_cell step loop).
"""

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from ase.md.langevin import Langevin

from amorphgen.utils.common import assert_finite, attach_outputs, DivergenceError


class NaNCalculator(Calculator):
    """Returns NaN energy/forces — stands in for a diverged / OOD MLIP."""

    implemented_properties = ["energy", "forces", "free_energy"]

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        n = len(self.atoms)
        self.results = {
            "energy": float("nan"),
            "free_energy": float("nan"),
            "forces": np.full((n, 3), np.nan),
        }


# ── assert_finite unit behaviour ──────────────────────────────────────────────

def test_assert_finite_passes_on_healthy_state():
    atoms = bulk("Cu", cubic=True)
    atoms.calc = EMT()
    # Should not raise on a normal, finite calculation.
    assert_finite(atoms, context="unit test")


@pytest.mark.parametrize("energy, force, message", [
    (np.nan, 0.0, "potential energy"),
    (np.inf, 0.0, "potential energy"),
    (-1.0, np.nan, "forces"),
    (-1.0, np.inf, "forces"),
    (np.nan, np.nan, "potential energy and forces"),
], ids=["nan-energy", "inf-energy", "nan-forces", "inf-forces", "both"])
def test_assert_finite_rejects_nonfinite_energy_or_forces(energy, force, message):
    atoms = bulk("Cu", cubic=True)
    atoms.calc = SinglePointCalculator(
        atoms, energy=energy, forces=np.full((len(atoms), 3), force),
    )
    with pytest.raises(DivergenceError, match=f"Non-finite {message}"):
        assert_finite(atoms)


def test_divergence_message_is_actionable():
    atoms = bulk("Cu", cubic=True)
    atoms.calc = NaNCalculator()
    with pytest.raises(DivergenceError) as exc:
        assert_finite(atoms, context="MD stage 'stage3_melt'", step=42)
    msg = str(exc.value)
    # Pinpoints location …
    assert "step 42" in msg
    assert "stage3_melt" in msg
    # … and gives the user a concrete way forward.
    assert "timestep" in msg
    assert "diverged" in msg


def test_assert_finite_does_not_mask_calculator_errors():
    # An Atoms with no calculator raises inside get_potential_energy(); the
    # guard swallows that (returns) so the real error can surface on its own
    # rather than being reported as a spurious divergence.
    atoms = bulk("Cu", cubic=True)
    assert_finite(atoms)  # no calc attached -> no DivergenceError


# ── MD path: the attach_outputs observer guards the trajectory ────────────────

def test_md_observer_raises_before_writing_nan(tmp_path):
    atoms = bulk("Cu", cubic=True) * (2, 2, 2)
    atoms.calc = NaNCalculator()
    dyn = Langevin(atoms, timestep=1.0 * units.fs,
                   temperature_K=300, friction=0.01, fixcm=False)
    traj = tmp_path / "stage_traj.xyz"
    logger, writer = attach_outputs(dyn, atoms, str(tmp_path / "stage.log"), str(traj))

    try:
        with pytest.raises(DivergenceError):
            dyn.run(5)
    finally:
        logger.close()
        writer.close()

    # Energy and forces are invalid before the first step, even though the
    # positions are finite. No frame should reach the trajectory writer.
    assert not traj.exists()


# ── Relax path: the optimiser step loop guards the output ─────────────────────

def test_optimizer_raises_on_nan(tmp_path, monkeypatch):
    from amorphgen.pipeline import opt_cell

    monkeypatch.chdir(tmp_path)  # opt_cell writes files to cwd
    atoms = bulk("Cu", cubic=True) * (2, 2, 2)
    cfg = {"device": "cpu",
           "opt": {"optimizer": "FIRE", "fmax": 0.001,
                   "max_steps": 5, "cell_filter": "none"}}
    with pytest.raises(DivergenceError):
        opt_cell.run(atoms, cfg_override=cfg, calc=NaNCalculator())
    assert not (tmp_path / "stage1_opt.xyz").exists()
    assert not (tmp_path / "stage1_opt.cif").exists()
