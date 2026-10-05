"""Geometric and statistical checks for periodic point-clearance sampling."""

import itertools
import json

import numpy as np
import pytest
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii

from amorphgen.analysis.voids import compute_void_distribution, _point_clearances


def single_atom(length=4.0):
    return Atoms("Si", positions=[[0, 0, 0]], cell=[length] * 3, pbc=True)


def test_isolated_sphere_volume_and_histogram_normalization():
    """A nonoverlapping periodic sphere has an analytic excluded volume."""
    atoms = single_atom()
    result = compute_void_distribution(
        [atoms], n_samples=20000, radii={"Si": 1.0}, probe_radius=0.4,
        nbins=30, seed=7,
    )
    expected = 1 - 4 * np.pi * 1.4**3 / (3 * atoms.get_volume())
    assert result["accessible_fraction"] == pytest.approx(
        expected, abs=5 * result["accessible_fraction_stderr"]
    )
    assert sum(result["bin_volume_fraction"]) == pytest.approx(
        result["accessible_fraction"]
    )
    assert np.dot(result["probability_density"],
                  np.diff(result["bin_edges"])) == pytest.approx(1.0)
    assert result["accessible_volume"] == pytest.approx(
        result["accessible_fraction"] * atoms.get_volume()
    )
    assert result["mean_clearance"] >= 0.4
    assert result["max_clearance"] <= 2 * np.sqrt(3) - 1
    assert result["bin_edges"][0] == 0.4
    assert result["n_structures"] == 1
    assert "not connected" in result["definition"]
    json.dumps(result, allow_nan=False)


def test_ensemble_weighted_by_cell_volume():
    atoms = [single_atom(3), single_atom(6)]
    result = compute_void_distribution(atoms, n_samples=400, radii={"Si": 1.0})
    entries = result["per_structure"]
    volumes = np.array([entry["cell_volume"] for entry in entries])
    fractions = np.array([entry["accessible_fraction"] for entry in entries])
    weights = volumes / sum(volumes)
    expected = weights @ fractions
    assert result["accessible_fraction"] == pytest.approx(expected)
    assert result["accessible_volume"] == pytest.approx(
        np.mean([entry["accessible_volume"] for entry in entries])
    )
    assert result["accessible_fraction_stderr"] == pytest.approx(
        np.sqrt(sum(weights**2 * fractions * (1 - fractions) / 400))
    )
    assert result["accessible_volume_stderr"] == pytest.approx(
        np.sqrt(sum(entry["accessible_volume_stderr"]**2 for entry in entries)) / 2
    )


def test_fully_blocked_samples_report_interval_and_finite_empty_histogram():
    result = compute_void_distribution(
        [single_atom()], radii={"Si": 4.0}, n_samples=100, probe_radius=0.3
    )
    assert result["accessible_fraction"] == 0
    assert result["accessible_volume"] == 0
    assert result["mean_clearance"] is None
    assert result["max_clearance"] is None
    assert result["clearance_quantiles"] == {"p10": None, "p50": None, "p90": None}
    assert result["per_structure"][0]["clearance_quantiles"] == result["clearance_quantiles"]
    assert result["probe_curve"]["accessible_fraction"] == [0] * 51
    assert result["probe_curve"]["radii"] == result["bin_edges"]
    assert all(bounds[1] > 0 for bounds in
               result["per_structure"][0]["probe_curve"]["accessible_fraction_interval_95"])
    assert sum(result["probability_density"]) == 0
    assert sum(result["bin_volume_fraction"]) == 0
    assert result["per_structure"][0]["accessible_fraction_interval_95"][1] > 0
    json.dumps(result, allow_nan=False)


def test_seed_reproducibility_and_no_global_rng_or_atoms_mutation():
    atoms = single_atom()
    original_positions = atoms.positions.copy()
    original_cell = atoms.cell.copy()
    np.random.seed(95)
    state = np.random.get_state()
    one = compute_void_distribution([atoms], n_samples=100, seed=33)
    two = compute_void_distribution([atoms], n_samples=100, seed=33)
    three = compute_void_distribution([atoms], n_samples=100, seed=34)
    assert one == two
    assert one["mean_clearance"] != three["mean_clearance"]
    assert np.random.get_state()[1].tolist() == state[1].tolist()
    np.testing.assert_array_equal(atoms.positions, original_positions)
    np.testing.assert_array_equal(atoms.cell, original_cell)


def test_periodic_translation_invariance():
    atoms = single_atom()
    translated = atoms.copy()
    translated.positions[0] += 7 * atoms.cell[0] - 4 * atoms.cell[1]
    one = compute_void_distribution([atoms], n_samples=300)
    two = compute_void_distribution([translated], n_samples=300)
    assert one["bin_volume_fraction"] == two["bin_volume_fraction"]
    assert one["mean_clearance"] == pytest.approx(two["mean_clearance"])


def test_repeated_cell_has_same_accessible_fraction():
    atoms = single_atom()
    one = compute_void_distribution([atoms], n_samples=3000, radii={"Si": 1.0})
    two = compute_void_distribution([atoms.repeat((2, 2, 2))], n_samples=3000,
                                    radii={"Si": 1.0})
    uncertainty = np.hypot(one["accessible_fraction_stderr"],
                           two["accessible_fraction_stderr"])
    assert one["accessible_fraction"] == pytest.approx(
        two["accessible_fraction"], abs=5 * uncertainty
    )


def test_skew_triclinic_clearance_matches_explicit_lattice_search():
    """Skew cells must not use fractional rounding as the MIC algorithm."""
    cell = np.array([[4.0, 0, 0], [3.9, 0.8, 0], [3.8, 0.7, 1.2]])
    rng = np.random.default_rng(15)
    positions = rng.random((2, 3)) @ cell
    points = rng.random((30, 3)) @ cell
    radii = np.array([0.2, 0.5])
    translations = np.array(list(itertools.product(range(-6, 7), repeat=3))) @ cell
    image_positions = positions[None, :, :] + translations[:, None, :]
    expected = np.min(
        np.linalg.norm(points[:, None, None, :] - image_positions[None, :, :, :],
                       axis=3) - radii[None, None, :], axis=(1, 2)
    )
    actual = _point_clearances(points, positions, cell, radii)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_nearest_surface_can_belong_to_more_distant_atom():
    clearance = _point_clearances(
        np.array([[0.0, 0, 0]]), np.array([[1.0, 0, 0], [2.0, 0, 0]]),
        np.eye(3) * 10, np.array([0.1, 1.5]),
    )
    assert clearance[0] == pytest.approx(0.5)


def test_partial_radii_mapping_uses_documented_defaults():
    atoms = Atoms("SiO", positions=[[0, 0, 0], [2, 2, 2]], cell=[4] * 3, pbc=True)
    result = compute_void_distribution([atoms], n_samples=20, radii={"Si": 1.3})
    assert result["radii"] == {"Si": 1.3, "O": covalent_radii[atomic_numbers["O"]]}
    assert result["radius_source"] == "ASE covalent radii with overrides"


def test_pair_arrays_are_chunked(monkeypatch):
    import amorphgen.analysis.voids as module
    original_find_mic = module.find_mic
    calls = []

    def tracked_find_mic(vectors, cell, pbc):
        calls.append(len(vectors))
        return original_find_mic(vectors, cell, pbc)

    monkeypatch.setattr(module, "_MAX_PAIRS", 256)
    monkeypatch.setattr(module, "find_mic", tracked_find_mic)
    atoms = single_atom().repeat((6, 6, 6))
    compute_void_distribution([atoms], n_samples=5)
    assert len(calls) > 1
    assert max(calls) <= 256


@pytest.mark.parametrize("keyword,value", [
    ("n_samples", 0), ("n_samples", -1), ("n_samples", 2.5), ("n_samples", True),
    ("nbins", 0), ("nbins", 1.5), ("nbins", False),
    ("probe_radius", -0.1), ("probe_radius", np.nan),
    ("probe_radius", np.inf), ("probe_radius", "1"),
    ("seed", -1), ("seed", 0.5), ("seed", True),
    ("radii", []), ("radii", {"Si": 0}), ("radii", {"Si": -1}),
    ("radii", {"Si": np.nan}), ("radii", {"Si": np.inf}),
    ("radii", {"Unknown": 1}), ("radii", {"Si": True}),
])
def test_invalid_parameters(keyword, value):
    with pytest.raises(ValueError):
        compute_void_distribution([single_atom()], **{keyword: value})


@pytest.mark.parametrize("atoms", [
    Atoms(),
    Atoms("Si", cell=[4, 4, 0], pbc=True),
    Atoms("Si", cell=[4, 4, 4], pbc=[True, True, False]),
    Atoms("Si", cell=[[4, 0, 0], [4, 0, 0], [0, 0, 4]], pbc=True),
    Atoms("Si", positions=[[np.nan, 0, 0]], cell=[4, 4, 4], pbc=True),
    Atoms("Si", cell=[np.inf, 4, 4], pbc=True),
])
def test_invalid_structures(atoms):
    with pytest.raises(ValueError):
        compute_void_distribution([atoms])


def test_empty_ensemble_rejected():
    with pytest.raises(ValueError, match="at least one"):
        compute_void_distribution([])


def test_one_sample_and_nondeterministic_seed_supported():
    result = compute_void_distribution([single_atom()], n_samples=1, nbins=1, seed=None)
    assert result["seed"] is None
    assert result["per_structure"][0]["n_samples"] == 1
    json.dumps(result, allow_nan=False)


def test_probe_curve_matches_analytic_excluded_sphere_volume():
    """Every probe radius inflates the periodic excluded sphere analytically."""
    radii = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    result = compute_void_distribution(
        [single_atom()], n_samples=12000, radii={"Si": 1.0},
        probe_radius=0.8, probe_radii=radii, seed=9,
    )
    curve = result["probe_curve"]
    expected = 1 - 4 * np.pi * (1 + np.array(radii))**3 / (3 * 4**3)
    observed = np.array(curve["accessible_fraction"])
    errors = np.array(curve["accessible_fraction_stderr"])
    assert np.all(abs(observed - expected) < 5 * errors)
    assert np.all(np.diff(observed) <= 0)
    assert observed[0] > result["accessible_fraction"]
    assert observed[4] == result["accessible_fraction"]
    assert curve["accessible_volume"][4] == pytest.approx(result["accessible_volume"])
    assert curve["accessible_fraction_stderr"][4] == pytest.approx(
        result["accessible_fraction_stderr"])


def test_probe_curves_and_clearance_quantiles_use_volume_weighted_draws(monkeypatch):
    """A larger cell contributes more represented volume per clearance draw."""
    import amorphgen.analysis.voids as module

    def known_clearances(points, positions, cell, atom_radii):
        return np.arange(4.0) + (0 if cell[0, 0] == 1 else 4)

    monkeypatch.setattr(module, "_point_clearances", known_clearances)
    result = compute_void_distribution(
        [single_atom(1), single_atom(2)], n_samples=4, probe_radius=2,
        probe_radii=[0, 2, 4, 6, 8],
    )
    curve = result["probe_curve"]
    frames = result["per_structure"]
    assert curve["accessible_fraction"] == pytest.approx([1, 8.5 / 9, 8 / 9, 4 / 9, 0])
    assert curve["accessible_volume"] == pytest.approx([4.5, 4.25, 4, 2, 0])
    assert frames[0]["probe_curve"]["n_accessible"] == [4, 2, 0, 0, 0]
    assert frames[1]["probe_curve"]["n_accessible"] == [4, 4, 4, 2, 0]
    assert result["clearance_quantiles"] == {"p10": 4, "p50": 5, "p90": 7}
    assert frames[0]["clearance_quantiles"] == {"p10": 2, "p50": 2, "p90": 3}
    assert frames[1]["clearance_quantiles"] == {"p10": 4, "p50": 5, "p90": 7}
    assert result["uncertainty"]["probe_curve"]["accessible_fraction"]["mean"] == [
        1, 0.75, 0.5, 0.25, 0]
    assert result["uncertainty"]["clearance_quantiles"]["p50"]["mean"] == 3.5
    assert curve["accessible_fraction_stderr"][1] == pytest.approx(0.25 / 9)
    assert curve["accessible_volume_stderr"][1] == pytest.approx(0.25 / 2)
    assert curve["accessible_fraction_stderr"][3] == pytest.approx(8 * 0.25 / 9)
    json.dumps(result, allow_nan=False)


def test_probe_curve_is_independent_of_base_probe_and_preserves_old_outputs():
    atoms = [single_atom(3), single_atom(4)]
    radii = [2.0, 0.0, 1.0, 0.0]
    original = list(radii)
    result = compute_void_distribution(atoms, n_samples=200, probe_radii=radii)
    default = compute_void_distribution(atoms, n_samples=200)
    inaccessible = compute_void_distribution(
        atoms, n_samples=200, probe_radius=10, probe_radii=radii)
    assert radii == original
    assert result["probe_curve"]["radii"] == [0, 1, 2]
    assert default["probe_curve"]["radii"] == default["bin_edges"]
    assert result["probe_curve"] == inaccessible["probe_curve"]
    assert inaccessible["accessible_fraction"] == 0
    assert inaccessible["clearance_quantiles"] == {"p10": None, "p50": None, "p90": None}
    # Only the requested curve and its uncertainty change when radii change.
    for output in (result, default):
        output.pop("probe_curve")
        output["uncertainty"].pop("probe_curve")
        for frame in output["per_structure"]:
            frame.pop("probe_curve")
    assert result == default
    json.dumps(inaccessible, allow_nan=False)


@pytest.mark.parametrize("radii", [
    [], (), np.array([]), 1, "1,2", [[0, 1]], [0, [1]],
    [-1], [np.nan], [np.inf], ["1"], [True], [0, False],
    {"radius": 1}, {0, 1}, np.array([False, True]),
])
def test_invalid_probe_radius_sequences(radii):
    with pytest.raises(ValueError, match="probe_radii"):
        compute_void_distribution([single_atom()], probe_radii=radii)


def test_numpy_probe_radii_supported_without_mutation():
    radii = np.array([2.0, 1.0, 1.0, 0.0])
    result = compute_void_distribution([single_atom()], n_samples=10, probe_radii=radii)
    assert result["probe_curve"]["radii"] == [0, 1, 2]
    np.testing.assert_array_equal(radii, [2, 1, 1, 0])
