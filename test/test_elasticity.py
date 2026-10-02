"""Analytic stress/strain checks for elastic descriptors, without an MLIP."""

import json

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms

from amorphgen.analysis.elasticity import compute_elastic_moduli


def isotropic_stiffness(bulk=80.0, shear=30.0):
    tensor = np.zeros((6, 6))
    tensor[:3, :3] = bulk - 2 * shear / 3
    np.fill_diagonal(tensor[:3, :3], bulk + 4 * shear / 3)
    np.fill_diagonal(tensor[3:, 3:], shear)
    return tensor


class LinearStress(Calculator):
    """Analytic Hooke-law stress, with an optional internal relaxation mode."""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, reference, stiffness, residual=None, coupling=0):
        super().__init__()
        self.reference_cell = reference.cell.array.copy()
        self.reference_fractional = reference.get_scaled_positions().copy()
        self.reference_volume = reference.get_volume()
        self.stiffness = np.array(stiffness) * units.GPa
        self.residual = np.zeros(6) if residual is None else np.array(residual) * units.GPa
        self.coupling = coupling
        self.evaluations = []

    def calculate(self, atoms=None, properties=("stress",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        deformation = np.linalg.solve(self.reference_cell, atoms.cell.array).T
        eps = deformation - np.eye(3)
        strain = np.array([eps[0, 0], eps[1, 1], eps[2, 2],
                           eps[1, 2] + eps[2, 1], eps[0, 2] + eps[2, 0],
                           eps[0, 1] + eps[1, 0]])
        displacement = atoms.positions - self.reference_fractional @ atoms.cell.array
        # Spring constant is 1 eV/Angstrom^2. The coupling softens C11 by
        # coupling^2 / reference_volume in eV/Angstrom^3 upon relaxation.
        force = -displacement
        force[0, 0] -= self.coupling * strain[0]
        stress = self.stiffness @ strain + self.residual
        stress[0] += self.coupling * displacement[0, 0] / self.reference_volume
        self.results = {
            "energy": (0.5 * self.reference_volume * strain @ self.stiffness @ strain
                       + 0.5 * np.sum(displacement**2)
                       + self.coupling * displacement[0, 0] * strain[0]),
            "forces": force,
            "stress": stress,
        }
        self.evaluations.append((atoms.cell.array.copy(), atoms.positions.copy()))


@pytest.fixture
def atoms():
    return Atoms("SiO", scaled_positions=[[0.2, 0.3, 0.4], [0.7, 0.8, 0.9]],
                 cell=[[4, 0.2, 0.1], [0.0, 4.3, 0.3], [0.2, 0.4, 5.0]], pbc=True)


def test_isotropic_moduli_shear_convention_and_units(atoms):
    expected = isotropic_stiffness()
    calc = LinearStress(atoms, expected, residual=[0.2, -0.3, 0.4, 0.01, 0.02, 0.03])
    result = compute_elastic_moduli([atoms], calculator=calc)
    structure = result["per_structure"][0]
    np.testing.assert_allclose(structure["stiffness_tensor_gpa"], expected, atol=1e-10)
    np.testing.assert_allclose(structure["residual_stress_gpa"],
                               [0.2, -0.3, 0.4, 0.01, 0.02, 0.03], atol=1e-12)
    for average in ("voigt", "reuss", "hill"):
        moduli = structure["moduli"][average]
        assert moduli["bulk_modulus_gpa"] == pytest.approx(80)
        assert moduli["shear_modulus_gpa"] == pytest.approx(30)
        assert moduli["young_modulus_gpa"] == pytest.approx(80)
        assert moduli["poisson_ratio"] == pytest.approx(1 / 3)
    assert result["voigt_order"] == ["xx", "yy", "zz", "yz", "xz", "xy"]
    assert structure["mechanically_stable"] is True
    assert structure["compliance_valid"] is True
    assert structure["max_residual_stress_gpa"] == pytest.approx(0.4)
    assert any("Residual stress" in warning for warning in structure["warnings"])
    assert len(calc.evaluations) == 13
    json.dumps(result, allow_nan=False)


def test_anisotropic_tensor_and_ensemble(atoms):
    generator = np.random.default_rng(100)
    matrix = generator.normal(size=(6, 6))
    expected = matrix.T @ matrix + np.eye(6) * 10
    second = atoms.copy()
    atoms.calc = LinearStress(atoms, expected)
    second.calc = LinearStress(second, 2 * expected)
    result = compute_elastic_moduli([atoms, second])
    first_result, second_result = result["per_structure"]
    np.testing.assert_allclose(first_result["stiffness_tensor_gpa"], expected, atol=1e-10)
    np.testing.assert_allclose(second_result["stiffness_tensor_gpa"], 2 * expected, atol=1e-10)
    np.testing.assert_allclose(result["ensemble"]["stiffness_tensor_mean_gpa"], 1.5 * expected)
    np.testing.assert_allclose(result["ensemble"]["stiffness_tensor_std_gpa"], 0.5 * np.abs(expected))
    for key in ("bulk_modulus_gpa", "shear_modulus_gpa"):
        voigt = first_result["moduli"]["voigt"][key]
        reuss = first_result["moduli"]["reuss"][key]
        hill = first_result["moduli"]["hill"][key]
        assert 0 < reuss <= hill <= voigt
        summary = result["ensemble"]["moduli"]["hill"][key]
        assert summary == pytest.approx({"mean": 1.5 * hill, "std": 0.5 * hill, "count": 2})


def test_input_arrays_and_calculator_attachment_unchanged(atoms):
    atoms.set_constraint(FixAtoms(indices=[1]))
    atoms.set_momenta(np.ones((2, 3)) * 100)
    atoms.info["tag"] = "original"
    attached = LinearStress(atoms, isotropic_stiffness())
    atoms.calc = attached
    positions, cell, momenta = atoms.positions.copy(), atoms.cell.array.copy(), atoms.get_momenta().copy()
    compute_elastic_moduli([atoms])
    np.testing.assert_array_equal(atoms.positions, positions)
    np.testing.assert_array_equal(atoms.cell, cell)
    np.testing.assert_array_equal(atoms.get_momenta(), momenta)
    assert atoms.calc is attached
    assert atoms.info == {"tag": "original"}
    assert atoms.constraints[0].get_indices().tolist() == [1]


def test_atomic_relaxation_softens_modulus_without_changing_cell_or_inputs(atoms):
    tensor = isotropic_stiffness()
    calc = LinearStress(atoms, tensor, coupling=2.0)
    atoms.positions[0, 0] += 0.2
    positions = atoms.positions.copy()
    result = compute_elastic_moduli([atoms], calculator=calc, relax=True, fmax=1e-8)
    tensor[0, 0] -= 4 / (atoms.get_volume() * units.GPa)
    structure = result["per_structure"][0]
    np.testing.assert_allclose(structure["stiffness_tensor_gpa"], tensor, atol=1e-7)
    np.testing.assert_allclose(structure["residual_stress_gpa"], 0, atol=1e-8)
    assert structure["volume_angstrom3"] == atoms.get_volume()
    np.testing.assert_array_equal(atoms.positions, positions)
    assert result["relaxed_ions"] is True


def test_unconverged_relaxation_raises(atoms):
    calc = LinearStress(atoms, isotropic_stiffness())
    atoms.positions[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="relaxation did not converge"):
        compute_elastic_moduli([atoms], calculator=calc, relax=True, steps=1, fmax=1e-12)


def test_raw_tensor_is_symmetrized(atoms):
    raw = isotropic_stiffness()
    raw[0, 1] += 20
    result = compute_elastic_moduli([atoms], calculator=LinearStress(atoms, raw))
    structure = result["per_structure"][0]
    np.testing.assert_allclose(structure["raw_stiffness_tensor_gpa"], raw, atol=1e-10)
    np.testing.assert_allclose(structure["stiffness_tensor_gpa"], (raw + raw.T) / 2, atol=1e-10)
    assert structure["symmetry_error_gpa"] == pytest.approx(20)
    assert any("asymmetry" in warning for warning in structure["warnings"])


@pytest.mark.parametrize("shear", [0, -10, 1e-13])
def test_unstable_or_ill_conditioned_moduli_are_unavailable(atoms, shear):
    result = compute_elastic_moduli([atoms], calculator=LinearStress(atoms, isotropic_stiffness(shear=shear)))
    structure = result["per_structure"][0]
    assert structure["compliance_valid"] is False
    assert structure["moduli"]["reuss"] is None
    assert structure["moduli"]["hill"] is None
    assert structure["warnings"]
    assert result["ensemble"]["moduli"]["hill"]["bulk_modulus_gpa"] == {
        "mean": None, "std": None, "count": 0,
    }
    json.dumps(result, allow_nan=False)


def test_missing_and_single_point_calculators_rejected(atoms):
    with pytest.raises(ValueError, match="live stress-capable calculator"):
        compute_elastic_moduli([atoms])
    stored = SinglePointCalculator(atoms, stress=np.zeros(6), energy=0)
    atoms.calc = stored
    with pytest.raises(ValueError, match="single-point"):
        compute_elastic_moduli([atoms])
    compute_elastic_moduli([atoms], calculator=LinearStress(atoms, isotropic_stiffness()))
    assert atoms.calc is stored


def test_nonfinite_stress_rejected(atoms):
    calc = LinearStress(atoms, isotropic_stiffness(), residual=[np.nan] * 6)
    with pytest.raises(ValueError, match="stress must contain six finite"):
        compute_elastic_moduli([atoms], calculator=calc)


@pytest.mark.parametrize("kwargs", [
    {"strain": 0}, {"strain": -0.1}, {"strain": np.nan}, {"strain": 1},
    {"strain": True}, {"fmax": 0}, {"fmax": np.inf}, {"steps": 0},
    {"steps": 1.2}, {"steps": True}, {"relax": "yes"},
])
def test_invalid_parameters_rejected(atoms, kwargs):
    with pytest.raises(ValueError):
        compute_elastic_moduli([atoms], calculator=LinearStress(atoms, isotropic_stiffness()), **kwargs)


def test_invalid_structures_rejected(atoms):
    calc = LinearStress(atoms, isotropic_stiffness())
    with pytest.raises(ValueError, match="at least one"):
        compute_elastic_moduli([], calculator=calc)
    with pytest.raises(TypeError, match="not a single"):
        compute_elastic_moduli(atoms, calculator=calc)
    with pytest.raises(TypeError, match="ASE Atoms"):
        compute_elastic_moduli(["structure.xyz"], calculator=calc)
    atoms.pbc = [True, True, False]
    with pytest.raises(ValueError, match="all three"):
        compute_elastic_moduli([atoms], calculator=calc)
    atoms.pbc = True
    atoms.cell[2] = 0
    with pytest.raises(ValueError, match="nonsingular"):
        compute_elastic_moduli([atoms], calculator=calc)
