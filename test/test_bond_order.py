"""Crystal references, periodic geometry and classification regressions."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.neighborlist import neighbor_list
from scipy.special import eval_legendre

from amorphgen.analysis import StructureAnalyser, compute_bond_order


@pytest.mark.parametrize("crystal,cutoff,expected,count", [
    ("fcc", 3.0, 0.5745242597140698, 12),
    ("sc", 4.1, np.sqrt(2) / 4, 6),
])
def test_known_crystal_q6_in_primitive_cell(crystal, cutoff, expected, count):
    atoms = bulk("Cu", crystal, a=4.0)
    frame = compute_bond_order([atoms], cutoff=cutoff)["per_structure"][0]
    assert frame["n_atoms"] == 1
    np.testing.assert_allclose(frame["q6"], expected, atol=1e-12)
    np.testing.assert_allclose(frame["qbar6"], expected, atol=1e-12)
    np.testing.assert_array_equal(frame["neighbor_counts"], count)
    assert frame["ordered_count"] == frame["largest_cluster_size"] == 1
    assert frame["largest_cluster_fraction"] == 1.0


def test_rotation_translation_wrapping_and_repetition():
    # FCC's primitive cell is triclinic and has one atom; all its neighbours
    # are periodic images. The result must equal a conventional supercell.
    primitive = bulk("Cu", "fcc", a=4.0)
    supercell = bulk("Cu", "fcc", a=4.0, cubic=True) * (2, 3, 2)
    rotated = supercell.copy()
    rotated.rotate(41, (1, 3, 2), rotate_cell=True)
    rng = np.random.default_rng(21)
    rotated.positions += rng.integers(-3, 4, (len(rotated), 3)) @ rotated.cell.array
    rotated.translate([1.273, -2.36, 7.13])
    result = compute_bond_order([primitive, supercell, rotated], cutoff=3.0)
    for frame in result["per_structure"]:
        np.testing.assert_allclose(frame["q6"], 0.5745242597140698, atol=1e-12)
        np.testing.assert_allclose(frame["qbar6"], frame["q6"], atol=1e-12)
        np.testing.assert_array_equal(frame["neighbor_counts"], 12)
        assert frame["largest_cluster_size"] == frame["n_atoms"]
        assert frame["ordered_fraction"] == 1.0
    assert result["largest_cluster_size"] == pytest.approx((1 + 48 + 48) / 3)


def test_rocksalt_gete_first_geometric_shell():
    atoms = bulk("GeTe", "rocksalt", a=6.0, cubic=True) * (2, 2, 2)
    frame = compute_bond_order([atoms], cutoff=3.5)["per_structure"][0]
    np.testing.assert_allclose(frame["q6"], np.sqrt(1 / 8), atol=1e-12)
    np.testing.assert_allclose(frame["qbar6"], np.sqrt(1 / 8), atol=1e-12)
    np.testing.assert_array_equal(frame["neighbor_counts"], 6)
    assert frame["ordered_count"] == frame["largest_cluster_size"] == len(atoms)


def test_disorder_reduces_neighbor_averaged_order():
    crystal = bulk("Cu", "fcc", a=4.0, cubic=True) * (4, 4, 4)
    liquid = crystal.copy()
    liquid.positions = np.random.default_rng(42).uniform(0, 16, (len(liquid), 3))
    ordered, disordered = compute_bond_order(
        [crystal, liquid], cutoff=3.0)["per_structure"]
    assert ordered["ordered_fraction"] == 1
    assert disordered["qbar6_mean"] < 0.2
    assert disordered["ordered_fraction"] < 0.2
    assert disordered["largest_cluster_size"] < len(liquid) / 5


def test_averages_complex_vectors_before_invariant():
    atoms = Atoms("Cu6", positions=[
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.2, 1],
        [1.5, 0.3, 0.2], [0.1, 1.4, 0.5],
    ])
    cutoff = 1.45
    frame = compute_bond_order([atoms], cutoff=cutoff, min_neighbors=1)["per_structure"][0]
    source, target, vectors = neighbor_list("ijD", atoms, cutoff)
    unit = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    expected_q6, expected_qbar6, scalar_average = [], [], []
    for atom in range(len(atoms)):
        directions = unit[source == atom]
        # Addition theorem gives the invariant independently of a harmonic
        # implementation: q_l^2 = mean(P_l(rhat_i . rhat_j)).
        expected_q6.append(np.sqrt(np.mean(eval_legendre(6, directions @ directions.T))))
        neighbourhood = [atom, *target[source == atom]]
        all_directions, weights = [], []
        for neighbor in neighbourhood:
            shell = unit[source == neighbor]
            all_directions.extend(shell)
            weights.extend([1 / len(neighbourhood) / len(shell)] * len(shell))
        all_directions = np.asarray(all_directions)
        weights = np.asarray(weights)
        kernel = eval_legendre(6, np.clip(all_directions @ all_directions.T, -1, 1))
        expected_qbar6.append(np.sqrt(weights @ kernel @ weights))
        scalar_average.append(np.mean(frame["q6"][neighbourhood]))
    np.testing.assert_allclose(frame["q6"], expected_q6, atol=1e-12)
    np.testing.assert_allclose(frame["qbar6"], expected_qbar6, atol=1e-12)
    assert np.max(np.abs(frame["qbar6"] - scalar_average)) > 0.1


def test_disconnected_ordered_clusters_and_isolated_atom():
    tetrahedron = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.8, 0], [0.5, 0.3, 0.8]])
    positions = np.vstack([tetrahedron, tetrahedron + [10, 0, 0], [20, 0, 0]])
    frame = compute_bond_order(
        [Atoms("Cu9", positions=positions)], cutoff=1.2,
        qbar6_threshold=0, min_neighbors=3)["per_structure"][0]
    assert frame["ordered_count"] == 8
    assert frame["ordered_fraction"] == pytest.approx(8 / 9)
    assert frame["largest_cluster_size"] == 4
    assert frame["largest_cluster_fraction"] == pytest.approx(4 / 9)
    np.testing.assert_array_equal(frame["cluster_ids"], [0, 0, 0, 0, 1, 1, 1, 1, -1])
    assert frame["q6"][-1] == frame["qbar6"][-1] == 0


def test_ordered_component_crosses_periodic_boundary():
    atoms = Atoms("Cu2", positions=[[0.1, 1, 1], [3.9, 1, 1]], cell=[4, 4, 4], pbc=True)
    frame = compute_bond_order([atoms], cutoff=0.3, min_neighbors=1)["per_structure"][0]
    assert frame["largest_cluster_size"] == 2
    np.testing.assert_array_equal(frame["cluster_ids"], [0, 0])


def test_partial_periodicity_does_not_add_nonperiodic_images():
    atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=[1, 1, 1], pbc=[True, False, False])
    frame = compute_bond_order([atoms], cutoff=1.1)["per_structure"][0]
    np.testing.assert_array_equal(frame["neighbor_counts"], [2])
    np.testing.assert_allclose(frame["q6"], 1, atol=1e-12)
    assert frame["ordered_count"] == 0  # Coordination guard excludes chains.


@pytest.mark.parametrize("frames", [[], [Atoms()], [Atoms("He", positions=[[0, 0, 0]])]])
def test_empty_and_no_neighbor_inputs(frames):
    result = compute_bond_order(frames, cutoff=1, qbar6_threshold=0, min_neighbors=1)
    assert result["n_structures"] == len(frames)
    assert result["q6_mean"] == result["qbar6_mean"] == 0
    assert result["ordered_fraction"] == result["largest_cluster_size"] == 0
    for frame in result["per_structure"]:
        np.testing.assert_array_equal(frame["q6"], np.zeros(frame["n_atoms"]))
        np.testing.assert_array_equal(frame["qbar6"], np.zeros(frame["n_atoms"]))
        np.testing.assert_array_equal(frame["cluster_ids"], np.full(frame["n_atoms"], -1))
        assert not frame["ordered"].any()


def test_empty_frames_with_automatic_cutoff():
    result = compute_bond_order([Atoms()])
    assert result["parameters"]["cutoff"] == {}
    assert result["per_structure"][0]["largest_cluster_fraction"] == 0


def test_pair_cutoffs_and_analyser_api():
    # A complete pair table must work even without a cell/RDF volume.
    atoms = Atoms("GeTeGe", positions=[[0, 0, 0], [1, 0, 0], [3, 0, 0]])
    pair_cutoffs = {"Ge-Ge": 0.5, "Te-Ge": 1.2, "Te-Te": 0.5}
    result = compute_bond_order([atoms], cutoff=pair_cutoffs, min_neighbors=1)
    frame = result["per_structure"][0]
    np.testing.assert_array_equal(frame["neighbor_counts"], [1, 1, 0])
    assert result["parameters"]["cutoff"] == {"Ge-Ge": 0.5, "Ge-Te": 1.2, "Te-Te": 0.5}
    analyser = StructureAnalyser([atoms], cutoff=1.2)
    via_analyser = analyser.bond_order(min_neighbors=1)
    np.testing.assert_allclose(via_analyser["per_structure"][0]["qbar6"], frame["qbar6"])
    assert analyser.bond_order(cutoff=0.5)["ordered_fraction"] == 0


def test_cutoff_spec_default_and_overrides():
    atoms = Atoms("GeTeGe", positions=[[0, 0, 0], [1, 0, 0], [3, 0, 0]])
    result = compute_bond_order([atoms], cutoff="0.5,Ge-Te=1.2", min_neighbors=1)
    np.testing.assert_array_equal(result["per_structure"][0]["neighbor_counts"], [1, 1, 0])
    assert result["parameters"]["cutoff"]["Ge-Ge"] == 0.5


@pytest.mark.parametrize("cutoff", [0, -1, np.nan, np.inf, True, "nan", {"Ge-Ge": -1}, {"default": 0}])
def test_invalid_cutoffs(cutoff):
    with pytest.raises(ValueError, match="cutoff.*finite.*positive"):
        compute_bond_order([Atoms("Ge")], cutoff=cutoff)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf, True, None])
def test_invalid_thresholds(threshold):
    with pytest.raises(ValueError, match="qbar6_threshold"):
        compute_bond_order([], cutoff=1, qbar6_threshold=threshold)


@pytest.mark.parametrize("count", [0, -1, 2.5, True, None])
def test_invalid_minimum_neighbors(count):
    with pytest.raises(ValueError, match="min_neighbors"):
        compute_bond_order([], cutoff=1, min_neighbors=count)


@pytest.mark.parametrize("positions,pbc", [
    ([[0, 0, 0], [0, 0, 0]], False),
    ([[0, 0, 0], [4, 0, 0]], True),
])
def test_overlaps_rejected_including_periodic_equivalence(positions, pbc):
    atoms = Atoms("Cu2", positions=positions, cell=[4, 4, 4], pbc=pbc)
    with pytest.raises(ValueError, match="overlapping atoms"):
        compute_bond_order([atoms], cutoff=1)


def test_invalid_coordinates_and_periodic_cell():
    with pytest.raises(ValueError, match="positions must be finite"):
        compute_bond_order([Atoms("Cu", positions=[[np.nan, 0, 0]])], cutoff=1)
    with pytest.raises(ValueError, match="periodic cell vectors"):
        compute_bond_order([Atoms("Cu", pbc=True)], cutoff=1)
