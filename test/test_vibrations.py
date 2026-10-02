"""Analytic force models check harmonic frequencies and DOS conventions."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from scipy.integrate import trapezoid

from amorphgen.analysis.vibrations import compute_vibrational_dos


class HarmonicCalculator(Calculator):
    implemented_properties = ["forces"]

    def __init__(self, hessian):
        super().__init__()
        self.hessian = np.asarray(hessian)
        self.calls = 0

    def calculate(self, atoms=None, properties=("forces",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        self.results["forces"] = -(
            self.hessian @ atoms.positions.ravel()
        ).reshape((-1, 3))


def test_analytic_oscillator_frequency_and_unit_area():
    atoms = Atoms("H", positions=[[0, 0, 0]], masses=[1.0])
    calc = HarmonicCalculator(np.eye(3))
    result = compute_vibrational_dos([atoms], calculator=calc, npoints=1000)
    # A spring of 1 eV/A^2 with mass 1 amu: sqrt(k/m) / (2 pi).
    assert result["mode_frequencies_thz"] == pytest.approx([15.633304] * 3, rel=1e-6)
    assert result["total_modes"] == 3
    assert result["imaginary_modes"] == 0
    assert result["force_evaluations"] == calc.calls == 6
    assert trapezoid(result["dos"], result["frequencies_thz"]) == pytest.approx(1)
    np.testing.assert_allclose(result["projected_dos"]["H"], result["dos"])
    assert result["frequency_units"] == "THz"
    assert "Gamma" in result["sampling"]


def test_coupled_diatomic_mass_weighting_and_projections():
    atoms = Atoms("HHe", positions=np.zeros((2, 3)), masses=[1, 4])
    hessian = np.zeros((6, 6))
    hessian[0, 0] = hessian[3, 3] = 2
    hessian[0, 3] = hessian[3, 0] = -2
    atoms.calc = HarmonicCalculator(hessian)
    result = compute_vibrational_dos([atoms])
    expected = 15.633304 * np.sqrt(2 * (1 + 1 / 4))
    assert result["mode_frequencies_thz"] == pytest.approx([0] * 5 + [expected], rel=1e-6)
    assert result["imaginary_modes"] == 0
    np.testing.assert_allclose(
        result["projected_dos"]["H"] + result["projected_dos"]["He"], result["dos"]
    )
    for element in ("H", "He"):
        assert trapezoid(result["projected_dos"][element], result["frequencies_thz"]) == pytest.approx(0.5)


def test_imaginary_mode_is_signed_and_counted():
    atoms = Atoms("H", masses=[1])
    result = compute_vibrational_dos(
        [atoms], calculator=HarmonicCalculator(np.diag([-1, 0, 4]))
    )
    assert result["mode_frequencies_thz"] == pytest.approx([-15.633304, 0, 31.266608], rel=1e-6)
    assert result["imaginary_modes"] == 1
    assert result["per_structure"][0]["imaginary_modes"] == 1
    assert result["frequencies_thz"][0] < -15


def test_pooling_equal_weight_per_mode_and_different_calculators():
    a = Atoms("H", masses=[1])
    b = Atoms("HeHe", masses=[4, 4])
    a.calc = HarmonicCalculator(np.eye(3))
    b.calc = HarmonicCalculator(np.eye(6))
    result = compute_vibrational_dos([a, b])
    assert result["total_modes"] == 9
    assert len(result["per_structure"]) == 2
    assert result["force_evaluations"] == 18
    assert trapezoid(result["projected_dos"]["H"], result["frequencies_thz"]) == pytest.approx(1 / 3)
    assert trapezoid(result["projected_dos"]["He"], result["frequencies_thz"]) == pytest.approx(2 / 3)


def test_original_atoms_and_calculator_attachment_are_preserved():
    atoms = Atoms("H", positions=[[1, 2, 3]], cell=[4, 5, 6], pbc=True)
    atoms.info["label"] = "original"
    original_calc = HarmonicCalculator(np.eye(3))
    atoms.calc = original_calc
    original = atoms.copy()
    compute_vibrational_dos([atoms], calculator=HarmonicCalculator(np.eye(3)))
    np.testing.assert_array_equal(atoms.positions, original.positions)
    np.testing.assert_array_equal(atoms.cell, original.cell)
    np.testing.assert_array_equal(atoms.pbc, original.pbc)
    assert atoms.info == original.info
    assert atoms.calc is original_calc
    assert original_calc.calls == 0


@pytest.mark.parametrize("parameter,value", [
    ("displacement", 0), ("displacement", -1), ("displacement", np.nan),
    ("sigma", 0), ("sigma", np.inf), ("sigma", True),
    ("npoints", 1), ("npoints", 2.5), ("npoints", True),
])
def test_invalid_parameters(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        compute_vibrational_dos([Atoms("H")], **{parameter: value})


def test_empty_inputs_and_missing_calculator():
    with pytest.raises(ValueError, match="At least one"):
        compute_vibrational_dos([])
    with pytest.raises(ValueError, match="no atoms"):
        compute_vibrational_dos([Atoms()])
    with pytest.raises(ValueError, match="force calculator"):
        compute_vibrational_dos([Atoms("H")])


def test_singlepoint_rejected_but_explicit_calculator_can_override():
    atoms = Atoms("H")
    atoms.calc = SinglePointCalculator(atoms, forces=np.zeros((1, 3)))
    with pytest.raises(ValueError, match="SinglePointCalculator"):
        compute_vibrational_dos([atoms])
    result = compute_vibrational_dos([atoms], calculator=HarmonicCalculator(np.eye(3)))
    assert result["total_modes"] == 3
    assert isinstance(atoms.calc, SinglePointCalculator)


@pytest.mark.parametrize("mass", [0, -1, np.nan, np.inf])
def test_invalid_masses(mass):
    atoms = Atoms("H", masses=[mass])
    with pytest.raises(ValueError, match="finite positive masses"):
        compute_vibrational_dos([atoms], calculator=HarmonicCalculator(np.eye(3)))


def test_all_inputs_validated_before_force_evaluations():
    atoms = Atoms("H")
    calc = HarmonicCalculator(np.eye(3))
    atoms.calc = calc
    with pytest.raises(ValueError, match="force calculator"):
        compute_vibrational_dos([atoms, Atoms("He")])
    assert calc.calls == 0


def test_constraints_rejected_explicitly():
    atoms = Atoms("H", constraint=FixAtoms(indices=[0]))
    with pytest.raises(ValueError, match="constraints"):
        compute_vibrational_dos([atoms], calculator=HarmonicCalculator(np.eye(3)))


def test_nonfinite_forces_rejected_without_mutation():
    atoms = Atoms("H")
    with pytest.raises(ValueError, match="Forces.*finite"):
        compute_vibrational_dos([atoms], calculator=HarmonicCalculator(np.full((3, 3), np.nan)))
    np.testing.assert_array_equal(atoms.positions, np.zeros((1, 3)))


def test_calculator_without_forces_rejected():
    atoms = Atoms("H")
    with pytest.raises(ValueError, match="does not support forces"):
        compute_vibrational_dos([atoms], calculator=Calculator())


def test_nonfinite_geometry_rejected():
    atoms = Atoms("H", positions=[[np.nan, 0, 0]])
    with pytest.raises(ValueError, match="finite positions"):
        compute_vibrational_dos([atoms], calculator=HarmonicCalculator(np.eye(3)))
    atoms.positions[:] = 0
    atoms.cell[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite cell"):
        compute_vibrational_dos([atoms], calculator=HarmonicCalculator(np.eye(3)))


def test_partial_periodic_cell_and_invalid_periodic_vectors():
    atoms = Atoms("H", cell=[2, 0, 0], pbc=[True, False, False])
    assert compute_vibrational_dos(
        [atoms], calculator=HarmonicCalculator(np.eye(3))
    )["total_modes"] == 3
    atoms.pbc = True
    with pytest.raises(ValueError, match="independent periodic cell vectors"):
        compute_vibrational_dos([atoms], calculator=HarmonicCalculator(np.eye(3)))


def test_hessian_symmetrisation_diagnostic():
    hessian = np.array([[2, 0.1, 0], [0, 2, 0], [0, 0, 2]])
    result = compute_vibrational_dos(
        [Atoms("H", masses=[1])], calculator=HarmonicCalculator(hessian)
    )
    assert result["per_structure"][0]["hessian_asymmetry"] == pytest.approx(0.1)
    assert result["mode_frequencies_thz"] == pytest.approx(
        15.633304 * np.sqrt([1.95, 2, 2.05]), rel=1e-6
    )
