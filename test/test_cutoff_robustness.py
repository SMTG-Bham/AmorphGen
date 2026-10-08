"""Cutoff sensitivity against hand-counted neighbour shells."""

import copy
import csv
import json

import numpy as np
import pytest
from ase import Atoms

from amorphgen.analysis import StructureAnalyser


def silica_line(distances):
    return Atoms("Si" + "O" * len(distances),
                 positions=[[0, 0, 0], *[[d, 0, 0] for d in distances]],
                 cell=[40] * 3)


def test_crowded_shell_counts_pairs_once_and_keeps_directional_cn():
    sa = StructureAnalyser([silica_line([1.7, 1.95, 2.05, 2.3])], cutoff=2.0)
    original_cutoff = copy.deepcopy(sa.cutoff)
    report = sa.cutoff_robustness(window=.2, points=5)
    assert report["window"] == .2
    assert report["offsets"] == pytest.approx([-.2, -.1, 0, .1, .2])
    assert report["n_structures"] == 1
    assert "O-Si" in report["pairs"] and "Si-O" not in report["pairs"]
    pair = report["pairs"]["O-Si"]
    assert pair["cutoff"] == 2.0
    assert pair["cutoffs"] == pytest.approx([1.8, 1.9, 2, 2.1, 2.2])
    assert pair["n_pairs"] == 3
    assert pair["near_pairs"] == 2
    assert pair["near_fraction"] == pytest.approx(2 / 3)
    assert pair["per_structure"][0]["n_pairs"] == 3
    assert pair["per_structure"][0]["near_pairs"] == 2
    assert pair["per_structure"][0]["near_fraction"] == pytest.approx(2 / 3)
    forward = pair["coordination"]["Si-O"]
    reverse = pair["coordination"]["O-Si"]
    assert forward["mean"] == [1, 1, 2, 3, 3]
    assert reverse["mean"] == [.25, .25, .5, .75, .75]
    assert forward["delta"] == 2
    assert reverse["delta"] == .5
    assert forward["total_atoms"] == 1
    assert reverse["total_atoms"] == 4
    assert forward["per_structure"] == [[1, 1, 2, 3, 3]]
    assert sa.cutoff == original_cutoff
    assert forward["mean"][2] == sa.coordination()["Si-O"]["mean"]


def test_separated_shell_has_no_near_pairs_and_constant_coordination():
    sa = StructureAnalyser([silica_line([1.2, 1.5, 2.5])], cutoff=2.0)
    pair = sa.cutoff_robustness(window=.2)["pairs"]["O-Si"]
    assert pair["n_pairs"] == 2
    assert pair["near_pairs"] == 0
    assert pair["near_fraction"] == 0
    assert pair["coordination"]["Si-O"]["mean"] == [2] * 5
    assert pair["coordination"]["Si-O"]["delta"] == 0


def test_homonuclear_pair_contributes_two_central_site_neighbours():
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1.99, 0, 0]], cell=[20] * 3)
    pair = StructureAnalyser([atoms], cutoff=2).cutoff_robustness()["pairs"]["Si-Si"]
    assert pair["n_pairs"] == pair["near_pairs"] == 1
    assert pair["near_fraction"] == 1
    cn = pair["coordination"]["Si-Si"]
    assert cn["mean"] == [0, 0, 1, 1, 1]
    assert cn["total_atoms"] == 2


def test_periodic_self_images_are_contacts_and_reverse_images_count_once():
    atoms = Atoms("Si", cell=[2] * 3, pbc=True)
    sa = StructureAnalyser([atoms], cutoff=2)
    pair = sa.cutoff_robustness()["pairs"]["Si-Si"]
    # Six first neighbours at +/- each cell vector, or three undirected
    # contacts. The unshifted ASE cutoff is strict, so none are baseline bonds.
    assert pair["n_pairs"] == pair["near_pairs"] == 3
    assert pair["coordination"]["Si-Si"]["mean"] == [0, 0, 0, 6, 6]
    assert pair["coordination"]["Si-Si"]["mean"][2] == sa.coordination()["Si-Si"]["mean"]


def test_baseline_and_endpoints_preserve_existing_unequal_cutoff_boundaries():
    atoms = silica_line([1.9, 2.0, 2.1])
    sa = StructureAnalyser([atoms], cutoff={"default": 3.0, "Si-O": 2.0})
    original = copy.deepcopy(sa.cutoff)
    report = sa.cutoff_robustness(window=.1)
    pair = report["pairs"]["O-Si"]
    # Unlike the global maximum cutoff, a smaller pair cutoff is inclusive.
    assert pair["coordination"]["Si-O"]["mean"] == [1, 1, 2, 2, 3]
    assert pair["n_pairs"] == 3
    assert pair["near_pairs"] == 2
    for contacts in report["pairs"].values():
        for direction, curve in contacts["coordination"].items():
            assert curve["mean"][2] == sa.coordination()[direction]["mean"]
    assert sa.cutoff == original


def test_global_cutoff_excludes_exact_upper_endpoint_and_includes_lower_switch():
    sa = StructureAnalyser([silica_line([1.9, 2.0, 2.1])], cutoff=2.0)
    pair = sa.cutoff_robustness(window=.1)["pairs"]["O-Si"]
    assert pair["coordination"]["Si-O"]["mean"] == [0, 1, 1, 2, 2]
    assert pair["n_pairs"] == 2
    assert pair["near_pairs"] == 2


def test_pooled_sites_and_equal_structure_means_have_distinct_changes():
    small = Atoms("Si2", positions=[[0, 0, 0], [1.99, 0, 0]], cell=[40] * 3)
    large = Atoms("Si8", positions=[[x, 0, 0] for x in [0, 1, 5, 6, 10, 11, 15, 16]],
                  cell=[40] * 3)
    with pytest.warns(UserWarning, match="different compositions"):
        sa = StructureAnalyser([small, large], cutoff=2)
    pair = sa.cutoff_robustness()["pairs"]["Si-Si"]
    assert pair["n_pairs"] == 5
    assert pair["near_pairs"] == 1
    assert pair["near_fraction"] == .2
    assert [row["near_fraction"] for row in pair["per_structure"]] == [1, 0]
    cn = pair["coordination"]["Si-Si"]
    assert cn["mean"] == [.8, .8, 1, 1, 1]
    assert cn["ensemble_mean"] == [.5, .5, 1, 1, 1]
    assert cn["delta"] == pytest.approx(.2)
    assert cn["ensemble_delta"] == .5
    assert cn["total_atoms"] == 10


def test_missing_centre_is_missing_and_missing_neighbour_is_zero():
    structures = [silica_line([1.99]), Atoms("Si", cell=[40] * 3),
                  Atoms("O", cell=[40] * 3)]
    with pytest.warns(UserWarning, match="different compositions"):
        sa = StructureAnalyser(structures, cutoff=2)
    report = sa.cutoff_robustness()
    pair = report["pairs"]["O-Si"]
    cn = pair["coordination"]["Si-O"]
    assert cn["per_structure"] == [[0, 0, 1, 1, 1], [0] * 5, None]
    assert cn["mean"] == [0, 0, .5, .5, .5]
    assert cn["ensemble_mean"] == cn["mean"]
    assert cn["total_atoms"] == 2
    assert [row["near_fraction"] for row in pair["per_structure"]] == [1, None, None]
    json.dumps(report, allow_nan=False)


def test_no_candidate_contacts_has_undefined_share_and_zero_coordination():
    sa = StructureAnalyser([Atoms("Si", cell=[20] * 3)], cutoff=2)
    pair = sa.cutoff_robustness()["pairs"]["Si-Si"]
    assert pair["n_pairs"] == pair["near_pairs"] == 0
    assert pair["near_fraction"] is None
    assert pair["coordination"]["Si-Si"]["mean"] == [0] * 5


def test_disabled_pair_stays_disabled_and_positive_lower_cutoff_is_clipped():
    sa = StructureAnalyser([silica_line([.05])],
                           cutoff={"default": .05, "O-Si": 0})
    report = sa.cutoff_robustness(window=.1)
    disabled = report["pairs"]["O-Si"]
    assert disabled["cutoffs"] == [0] * 5
    assert disabled["n_pairs"] == 0
    assert disabled["coordination"]["Si-O"]["mean"] == [0] * 5
    assert report["pairs"]["O-O"]["cutoffs"][0] == 0


@pytest.mark.parametrize("window", [0, -.1, np.nan, np.inf, True])
def test_invalid_window_is_rejected(window):
    sa = StructureAnalyser([silica_line([1.5])], cutoff=2)
    with pytest.raises(ValueError, match="window"):
        sa.cutoff_robustness(window=window)


@pytest.mark.parametrize("points", [1, 2, 4, 2.5, True])
def test_sweep_requires_odd_number_of_at_least_three_points(points):
    sa = StructureAnalyser([silica_line([1.5])], cutoff=2)
    with pytest.raises(ValueError, match="points"):
        sa.cutoff_robustness(points=points)


def test_reporting_and_machine_readable_exports(tmp_path):
    from amorphgen.analysis.robustness import (
        format_cutoff_robustness, save_cutoff_robustness,
    )
    sa = StructureAnalyser([silica_line([1.7, 1.95, 2.05, 2.3])], cutoff=2)
    report = sa.cutoff_robustness(window=.2)
    text = format_cutoff_robustness(report)
    assert "cutoff robustness" in text.lower()
    assert "Si-O" in text and "O-Si" in text
    save_cutoff_robustness(report, tmp_path)
    saved = json.loads((tmp_path / "analysis_cutoff_robustness.json").read_text())
    assert saved == report
    tables = {}
    for suffix in ["pairs", "coordination"]:
        path = tmp_path / f"analysis_cutoff_robustness_{suffix}.csv"
        with path.open() as handle:
            tables[suffix] = list(csv.DictReader(handle))
    pair = next(row for row in tables["pairs"] if row["pair"] == "O-Si")
    assert int(pair["n_pairs_upper"]) == 3
    assert int(pair["near_pairs"]) == 2
    assert float(pair["near_fraction"]) == pytest.approx(2 / 3)
    curve = [row for row in tables["coordination"] if row["direction"] == "Si-O"]
    assert [float(row["offset_A"]) for row in curve] == pytest.approx([-.2, -.1, 0, .1, .2])
    assert [float(row["pooled_mean"]) for row in curve] == [1, 1, 2, 3, 3]
    assert [float(row["delta"]) for row in curve] == [2] * 5


def test_both_summaries_include_robustness_at_requested_window(capsys):
    sa = StructureAnalyser([silica_line([1.7, 1.95, 2.05, 2.3])], cutoff=2)
    for summary in [sa.summary, sa.per_structure_summary]:
        text = summary(cutoff_window=.2)
        assert "cutoff robustness" in text.lower()
        assert "0.200" in text or "0.20" in text
