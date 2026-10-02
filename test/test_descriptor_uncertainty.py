"""Independent structures, rather than sites or modes, set precision."""

import json

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from scipy.integrate import trapezoid

from amorphgen.analysis.bond_order import compute_bond_order
from amorphgen.analysis.elasticity import compute_elastic_moduli
from amorphgen.analysis.energy import compute_energy_ranking
from amorphgen.analysis.oxygen import compute_oxygen_speciation
from amorphgen.analysis.vibrations import compute_vibrational_dos
from amorphgen.analysis.voids import compute_void_distribution


def test_oxygen_site_fraction_and_structure_prevalence_are_distinct():
    bonded = Atoms("OSi", positions=[[0, 0, 0], [1.5, 0, 0]])
    free = Atoms("O2Si", positions=[[0, 0, 0], [0, 4, 0], [4, 4, 4]])
    result = compute_oxygen_speciation([bonded, free], cutoff=1.6)
    assert result["fraction_of_sites"]["free"] == pytest.approx(2 / 3)
    assert result["fraction_of_structures"]["free"] == pytest.approx(1 / 2)
    uncertainty = result["uncertainty"]["fraction_of_sites.free"]
    assert uncertainty["per_structure"] == [0, 1]
    assert uncertainty["mean"] == pytest.approx(0.5)
    assert uncertainty["sem"] == pytest.approx(0.5)
    assert uncertainty["n_structures"] == 2


def test_oxygen_absence_is_missing_site_fraction_not_zero():
    result = compute_oxygen_speciation([Atoms("Cu"), Atoms("Cu2")], cutoff=1.6)
    uncertainty = result["uncertainty"]["fraction_of_sites.free"]
    assert uncertainty["per_structure"] == [None, None]
    assert uncertainty["n_structures"] == 0
    assert result["fraction_of_structures"]["free"] == 0
    json.dumps(result, allow_nan=False)


def test_ordered_fraction_uses_structures_for_precision_and_sites_for_pooling():
    dimer = Atoms("Si2", positions=[[0, 0, 0], [1, 0, 0]])
    isolated = Atoms("Si", positions=[[0, 0, 0]])
    result = compute_bond_order([dimer, isolated, Atoms()], cutoff=1.1,
                                qbar6_threshold=0.1, min_neighbors=1)
    assert result["fraction_of_sites"] == pytest.approx(2 / 3)
    assert result["fraction_of_structures"] == pytest.approx(1 / 3)
    uncertainty = result["uncertainty"]["ordered_fraction"]
    assert uncertainty["per_structure"] == [1, 0, None]
    assert uncertainty["sem"] == pytest.approx(0.5)
    assert uncertainty["n_structures"] == 2
    assert result["uncertainty"]["fraction_of_structures"]["n_structures"] == 3


def test_energy_missing_and_nonfinite_values_preserve_structure_alignment():
    structures = [Atoms("Si"), Atoms("Si2"), Atoms("Si"), Atoms("Si")]
    structures[0].info["energy"] = -2.0
    structures[1].info["energy"] = -8.0
    structures[3].info["energy"] = float("nan")
    result = compute_energy_ranking(iter(structures))
    assert result["ranking"] == [1, 0]
    uncertainty = result["uncertainty"]["energy_per_atom"]
    assert uncertainty["per_structure"] == [-2, -4, None, None]
    assert uncertainty["mean"] == -3
    assert uncertainty["sem"] == pytest.approx(1)
    assert uncertainty["n_structures"] == 2
    json.dumps(result, allow_nan=False)


def test_no_energy_has_unavailable_uncertainty():
    result = compute_energy_ranking([Atoms("Si")])
    assert result["uncertainty"]["energy_per_atom"]["mean"] is None
    assert result["uncertainty"]["energy_per_atom"]["sem"] is None


def test_void_ensemble_sem_is_distinct_from_monte_carlo_error():
    structures = [Atoms("Si", cell=[length] * 3, pbc=True) for length in [3, 7]]
    result = compute_void_distribution(structures, n_samples=400, nbins=10, seed=2)
    values = [frame["accessible_fraction"] for frame in result["per_structure"]]
    uncertainty = result["uncertainty"]["accessible_fraction"]
    assert uncertainty["mean"] == pytest.approx(np.mean(values))
    assert uncertainty["sem"] == pytest.approx(abs(values[0] - values[1]) / 2)
    assert uncertainty["sem"] != pytest.approx(result["accessible_fraction_stderr"])
    bins = result["uncertainty"]["bin_volume_fraction"]
    assert np.shape(bins["per_structure"]) == (2, 10)
    assert len(bins["bootstrap_low"]) == 10
    json.dumps(result, allow_nan=False)


def test_void_no_accessible_points_retains_curve_shape_without_false_certainty():
    atoms = Atoms("Si", cell=[4] * 3, pbc=True)
    result = compute_void_distribution([atoms, atoms], n_samples=5,
                                       radii={"Si": 10}, nbins=7)
    density = result["uncertainty"]["probability_density"]
    assert density["per_structure"] == [[None] * 7] * 2
    assert density["mean"] == [None] * 7
    assert density["sem"] == [None] * 7
    assert density["n_structures"] == 0
    assert result["uncertainty"]["accessible_fraction"]["sem"] == 0


class _HarmonicForces(Calculator):
    implemented_properties = ["forces"]

    def calculate(self, atoms=None, properties=("forces",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["forces"] = -atoms.positions


def test_vibrational_uncertainty_normalizes_each_structure_before_averaging():
    a, b = Atoms("H", masses=[1]), Atoms("HeHe", masses=[4, 4])
    result = compute_vibrational_dos([a, b], calculator=_HarmonicForces(), npoints=100)
    grid = result["frequencies_thz"]
    curves = np.array([frame["dos"] for frame in result["per_structure"]])
    assert trapezoid(curves, grid, axis=1) == pytest.approx([1, 1])
    uncertainty = result["uncertainty"]["dos"]
    np.testing.assert_allclose(uncertainty["mean"], curves.mean(axis=0))
    np.testing.assert_allclose(uncertainty["sem"], np.abs(curves[0] - curves[1]) / 2,
                               atol=1e-14)
    assert not np.allclose(uncertainty["mean"], result["dos"])
    projected = result["uncertainty"]["projected_dos.H"]["mean"]
    assert trapezoid(projected, grid) == pytest.approx(0.5)
    assert trapezoid(result["projected_dos"]["H"], grid) == pytest.approx(1 / 3)


class _LinearStress(Calculator):
    implemented_properties = ["stress"]

    def __init__(self, stiffness):
        super().__init__()
        self.stiffness = stiffness

    def calculate(self, atoms=None, properties=("stress",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        eps = atoms.cell.array / 4 - np.eye(3)
        strain = [eps[0, 0], eps[1, 1], eps[2, 2],
                  eps[1, 2] + eps[2, 1], eps[0, 2] + eps[2, 0],
                  eps[0, 1] + eps[1, 0]]
        self.results["stress"] = self.stiffness @ strain * units.GPa


def test_elastic_missing_moduli_and_tensor_components_keep_structure_alignment():
    stable = Atoms("Si", cell=[4] * 3, pbc=True)
    unstable = stable.copy()
    stable.calc = _LinearStress(np.eye(6) * 100)
    unstable.calc = _LinearStress(-np.eye(6) * 10)
    result = compute_elastic_moduli([stable, unstable])
    hill = result["uncertainty"]["hill.bulk_modulus_gpa"]
    assert hill["per_structure"] == pytest.approx([100 / 3, None])
    assert hill["n_structures"] == 1
    assert hill["sem"] is None
    tensor = result["uncertainty"]["stiffness_tensor_gpa"]
    assert tensor["shape"] == [6, 6]
    assert np.shape(tensor["per_structure"]) == (2, 36)
    assert tensor["mean"][0] == pytest.approx(45)
    assert tensor["sem"][0] == pytest.approx(55)
    json.dumps(result, allow_nan=False)
