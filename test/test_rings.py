"""Known network topologies, bounded searches and ring-summary semantics."""

import json

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from amorphgen.analysis.rings import compute_ring_statistics


def polygon(size, symbol="Si", side=1.0):
    angles = np.arange(size) * 2 * np.pi / size
    radius = side / (2 * np.sin(np.pi / size))
    return Atoms([symbol] * size, positions=np.column_stack(
        (radius * np.cos(angles), radius * np.sin(angles), np.zeros(size))))


def tree():
    return Atoms("Si4", positions=[(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0)])


@pytest.mark.parametrize("size", [3, 4, 6, 9])
def test_isolated_polygon_counts_edges_and_network_nodes(size):
    result = compute_ring_statistics([polygon(size)], cutoff=1.05, max_ring=size)
    assert result["ring_sizes"] == [size]
    assert result["counts"] == [size]  # one cycle gives size edge observations
    assert result["fractions"] == [100.0]
    assert result["n_ring_edges"] == result["total_rings"] == size
    assert result["n_network_nodes"] == result["n_network_edges"] == size
    assert result["n_unresolved_edges"] == 0
    assert result["ring_edge_fraction"] == 1.0
    assert result["mean_ring_size"] == result["min_ring_size"] == size
    assert result["max_ring_size"] == size
    assert result["std_ring_size"] == 0.0
    assert result["per_structure"][0]["index"] == 0
    assert result["per_structure"][0]["counts"] == {size: size}
    assert result["cutoff"] == 1.05 and result["max_ring"] == size
    assert result["n_structures"] == 1
    assert "not unique cycles" in result["count_definition"]
    json.dumps(result, allow_nan=False)


def test_bridged_polygon_sizes_count_formers_not_bridge_atoms():
    atoms = polygon(6, side=2.0)
    bridges = (atoms.positions + np.roll(atoms.positions, -1, axis=0)) / 2
    atoms += Atoms("O6", positions=bridges)
    result = compute_ring_statistics([atoms], cutoff=1.05)
    assert result["bond_pair"] == ("Si", "O")
    assert result["ring_sizes"] == [6]
    assert result["counts"] == [6]
    assert result["n_network_nodes"] == 6
    assert result["n_network_edges"] == 6


def test_disconnected_ring_tree_and_isolated_atom_cover_all_edges():
    atoms = polygon(6)
    branch = tree()
    branch.translate((10, 0, 0))
    atoms += branch
    atoms += Atoms("Si", positions=[(30, 0, 0)])
    result = compute_ring_statistics([atoms], cutoff=1.05)
    assert result["n_network_nodes"] == 11
    assert result["n_network_edges"] == 9
    assert result["n_ring_edges"] == 6
    assert result["n_unresolved_edges"] == 3
    assert result["ring_edge_fraction"] == pytest.approx(2 / 3)
    assert result["ring_sizes"] == [6]


def test_unresolved_edges_include_rings_above_search_limit():
    too_small = compute_ring_statistics([polygon(6)], cutoff=1.05, max_ring=5)
    complete = compute_ring_statistics([polygon(6)], cutoff=1.05, max_ring=6)
    assert too_small["n_unresolved_edges"] == 6
    assert too_small["n_ring_edges"] == 0
    assert too_small["ring_edge_fraction"] == 0.0
    assert too_small["mean_ring_size"] is None
    assert complete["n_unresolved_edges"] == 0
    assert complete["n_ring_edges"] == 6


def test_tree_and_no_edge_frames_have_distinct_missing_semantics():
    result = compute_ring_statistics([tree(), Atoms("Si")], cutoff=1.05)
    assert result["ring_sizes"] == result["counts"] == result["fractions"] == []
    assert result["n_unresolved_edges"] == 3
    assert result["per_structure"][0]["ring_edge_fraction"] == 0.0
    assert result["per_structure"][1]["ring_edge_fraction"] is None
    for key in ("mean_ring_size", "std_ring_size", "min_ring_size", "max_ring_size"):
        assert result[key] is None
        assert result["uncertainty"][key]["n_structures"] == 0
    assert result["uncertainty"]["ring_edge_fraction"]["n_structures"] == 1
    json.dumps(result, allow_nan=False)


def test_pooled_statistics_and_structure_uncertainty_use_different_weights():
    result = compute_ring_statistics([polygon(3), polygon(6), tree()], cutoff=1.05)
    assert result["mean_ring_size"] == pytest.approx(5.0)
    assert result["std_ring_size"] == pytest.approx(np.sqrt(2))
    assert result["min_ring_size"] == 3 and result["max_ring_size"] == 6
    assert result["fractions"] == pytest.approx([100 / 3, 200 / 3])
    assert result["n_network_nodes"] == 13
    assert result["n_network_edges"] == 12
    assert result["n_ring_edges"] == 9
    assert result["ring_edge_fraction"] == 0.75
    uncertainty = result["uncertainty"]["mean_ring_size"]
    assert uncertainty["per_structure"] == [3.0, 6.0, None]
    assert uncertainty["mean"] == pytest.approx(4.5)
    assert uncertainty["n_structures"] == 2
    assert uncertainty["sem"] == pytest.approx(1.5)
    assert result["uncertainty"]["ring_edge_fraction"]["mean"] == pytest.approx(2 / 3)
    assert result["fraction_of_structures"] == {3: 1 / 3, 6: 1 / 3}
    assert result["uncertainty"]["fractions"][3]["per_structure"] == [1.0, 0.0, None]


def test_periodic_chain_does_not_close_through_a_different_image():
    atoms = Atoms("Si", cell=[1, 0, 0], pbc=[True, False, False])
    result = compute_ring_statistics([atoms], cutoff=1.05, max_ring=8)
    assert result["n_network_nodes"] == result["n_network_edges"] == 1
    assert result["n_unresolved_edges"] == 1
    assert result["n_ring_edges"] == 0


def test_periodic_diamond_graph_counts_scale_with_supercell():
    primitive = bulk("Si", "diamond", a=5.43)
    one = compute_ring_statistics([primitive], cutoff=2.6)
    many = compute_ring_statistics([primitive.repeat(2)], cutoff=2.6)
    assert one["ring_sizes"] == many["ring_sizes"] == [6]
    for key in ("n_network_nodes", "n_network_edges", "n_ring_edges", "total_rings"):
        assert many[key] == 8 * one[key]
    assert one["n_network_nodes"] == 2
    assert one["n_network_edges"] == 4
    assert one["n_unresolved_edges"] == many["n_unresolved_edges"] == 0


def test_auto_pair_uses_all_frames_and_accepts_generators():
    frames = [Atoms("O"), polygon(3)]
    first = compute_ring_statistics((atoms for atoms in frames), cutoff=1.05)
    second = compute_ring_statistics(reversed(frames), cutoff=1.05)
    assert first["bond_pair"] == second["bond_pair"] == ("Si", "O")
    assert first["n_network_nodes"] == second["n_network_nodes"] == 3
    assert first["counts"] == second["counts"] == []
    assert first["per_structure"][0]["n_network_nodes"] == 0


def test_equal_electronegativity_uses_distinct_deterministic_symbols():
    atoms = Atoms("SiCu", positions=[(0, 0, 0), (1, 0, 0)])
    result = compute_ring_statistics([atoms], cutoff=1.05)
    assert result["bond_pair"] == ("Cu", "Si")


def test_positional_callback_and_pair_cutoff_mapping_remain_supported():
    calls = []

    def cutoff(a, b):
        calls.append((a, b))
        return 1.05

    result = compute_ring_statistics([polygon(4)], ("Si", "Si"), None, 4, cutoff)
    assert result["counts"] == [4]
    assert result["cutoff"] == 1.05
    assert calls == [("Si", "Si")]
    mapped = compute_ring_statistics([polygon(4)], cutoff={(14, 14): 1.05})
    assert mapped["counts"] == result["counts"]
    assert mapped["cutoff"] == {"Si-Si": 1.05}
    json.dumps(mapped, allow_nan=False)


@pytest.mark.parametrize("cutoff", [0, -1, np.inf, np.nan, True, "1.0", {},
                                        {("Si", "Si"): -1}, {("Si", "Qq"): 1},
                                        {"Si-Si": 1}])
def test_invalid_cutoffs_are_named_value_errors(cutoff):
    with pytest.raises(ValueError, match="cutoff"):
        compute_ring_statistics([polygon(3)], cutoff=cutoff)


def test_invalid_callback_cutoff_is_validated():
    with pytest.raises(ValueError, match="cutoff"):
        compute_ring_statistics([polygon(3)], get_cutoff_fn=lambda *_: np.nan)


@pytest.mark.parametrize("max_ring", [2, 0, -1, 3.0, True, np.nan])
def test_invalid_ring_limits_are_named_value_errors(max_ring):
    with pytest.raises(ValueError, match="max_ring"):
        compute_ring_statistics([polygon(3)], max_ring=max_ring)


@pytest.mark.parametrize("pair", ["Si", "Si-O", (), ("Si",), ("Si", "O", "C"),
                                   ("Si", "Qq"), (14, 8), None])
def test_invalid_pairs_are_named_value_errors(pair):
    if pair is None:
        pair = 1
    with pytest.raises(ValueError, match="bond_pair"):
        compute_ring_statistics([polygon(3)], bond_pair=pair)


@pytest.mark.parametrize("frames, message", [([], "at least one"), ([Atoms()], "no atoms"),
                                             ([None], "ASE Atoms"), (None, "iterable")])
def test_invalid_structure_inputs(frames, message):
    with pytest.raises(ValueError, match=message):
        compute_ring_statistics(frames)


@pytest.mark.parametrize("defect, message", [("positions", "positions"),
                                             ("cell", "cell"),
                                             ("periodic_empty", "periodic axis"),
                                             ("dependent", "independent")])
def test_invalid_geometry(defect, message):
    atoms = polygon(3)
    if defect == "positions":
        atoms.positions[0, 0] = np.nan
    elif defect == "cell":
        atoms.cell[0, 0] = np.inf
    elif defect == "periodic_empty":
        atoms.pbc = True
    else:
        atoms.set_cell([(2, 0, 0), (1, 0, 0), (0, 0, 2)])
    with pytest.raises(ValueError, match=message):
        compute_ring_statistics([atoms])
