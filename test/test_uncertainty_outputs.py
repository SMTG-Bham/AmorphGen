"""Uncertainty reaches user-facing APIs, reports and plot exports."""

import csv
import json

import numpy as np
import pytest
from ase import Atoms

from amorphgen.analysis import StructureAnalyser
from amorphgen.analysis.plotting import plot_sq, plot_tr


def _angle(degrees):
    radians = np.radians(degrees)
    return Atoms("Si3", positions=[[0, 0, 0], [1, 0, 0],
                                  [np.cos(radians), np.sin(radians), 0]],
                 cell=[8, 8, 8])


def test_angle_histograms_weight_structures_not_angles():
    small = _angle(91)
    large = _angle(121).repeat((3, 1, 1))
    with pytest.warns(UserWarning, match="different compositions"):
        sa = StructureAnalyser([small, large], cutoff=1.1)
    result = sa.angle_distribution(bins=90, seed=43)["Si-Si-Si"]
    rows = np.asarray(result["per_structure"])
    np.testing.assert_allclose(rows.sum(axis=1) * 2, 1)
    np.testing.assert_allclose(result["distribution"], rows.mean(axis=0))
    np.testing.assert_allclose(result["uncertainty"]["sem"], np.abs(rows[0] - rows[1]) / 2)
    assert np.count_nonzero(result["distribution"]) == 2
    np.testing.assert_allclose(np.asarray(result["distribution"])[np.nonzero(result["distribution"])], .25)
    # Enlarging only the second input does not create independent observations.
    original = StructureAnalyser([small, _angle(121)], cutoff=1.1)
    expected = original.angle_distribution(bins=90, seed=43)["Si-Si-Si"]
    assert result["uncertainty"] == expected["uncertainty"]


def test_missing_angles_are_missing_and_linear_angles_are_included():
    with pytest.warns(UserWarning, match="different compositions"):
        sa = StructureAnalyser([_angle(180), Atoms("Si", cell=[8] * 3)], cutoff=1.1)
    data = sa.angle_distribution()["Si-Si-Si"]
    assert data["distribution"][-1] == .5
    assert data["uncertainty"]["n_per_point"] == [1] * 90
    assert data["uncertainty"]["sem"] == [None] * 90
    assert data["per_structure"][1] == [None] * 90


def test_total_cn_and_reports_separate_spread_uncertainty_and_fractions():
    connected = Atoms("Si2", positions=[[0, 0, 0], [1, 0, 0]], cell=[8] * 3)
    disconnected = connected.copy()
    disconnected.positions[1, 0] = 3
    sa = StructureAnalyser([connected, disconnected], cutoff=1.1)
    result = sa.total_coordination()["Si"]
    assert result["uncertainty"]["mean"] == .5
    assert result["uncertainty"]["sem"] == .5
    assert result["fraction_of_sites"][1] == .5
    assert result["fraction_of_structures"][1] == .5
    text = sa.summary()
    for label in ("Fraction of sites", "Fraction of structures", "SEM=", "95% t CI", "site SD"):
        assert label in text


def test_curve_exports_include_uncertainty_and_structure_identity(tmp_path):
    sa = StructureAnalyser([_angle(90), _angle(120)], cutoff=1.1)
    sa.plot(output_dir=tmp_path, prefix="sample", rmax=3, dpi=40)
    for descriptor in ("rdf", "angles"):
        path = tmp_path / f"sample_{descriptor}_uncertainty.json"
        result = json.loads(path.read_text())
        assert result["descriptors"]
        for stats in result["descriptors"].values():
            assert stats["n_structures"] == 2
            assert stats["sampling_unit"] == "structure"
            assert stats["band_type"] == "pointwise"
        with (tmp_path / f"sample_{descriptor}_uncertainty.csv").open() as handle:
            reader = csv.DictReader(handle)
            assert {"sem", "ci_low", "ci_high", "bootstrap_low", "bootstrap_high"} <= set(reader.fieldnames)
            assert len(list(reader)) > 0
    with (tmp_path / "sample_angles.csv").open() as handle:
        assert {row["structure_index"] for row in csv.DictReader(handle)} == {"0", "1"}
    with (tmp_path / "sample_cn.csv").open() as handle:
        assert {"fraction_of_sites(%)", "fraction_of_structures(%)"} <= set(next(csv.reader(handle)))
    scalar = json.loads((tmp_path / "sample_statistics.json").read_text())
    assert scalar["density"]["uncertainty"]["n_structures"] == 2
    assert scalar["bond_angles"]["Si-Si-Si"]["uncertainty"]["sem"] == pytest.approx(15)


def test_sq_and_tr_exports_include_bands(tmp_path):
    frames = [_angle(90), _angle(120)]
    for frame in frames:
        frame.pbc = True
    sa = StructureAnalyser(frames, cutoff=1.1)
    sq = sa.structure_factor_direct(qmax=4, nq=20, partials=True, n_bootstrap=40)
    plot_sq(sq, output_dir=tmp_path, dpi=40)
    tr = sa.total_correlation(qmax=4, nq=20, rmax=3, nr=25, n_bootstrap=40)
    plot_tr(tr, output_dir=tmp_path, dpi=40)
    for key in ("sq", "tr"):
        with (tmp_path / f"analysis_{key}_uncertainty.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert any(row["bootstrap_low"] and row["ci_low"] for row in rows)


def test_per_structure_report_excludes_absent_centre_from_cn_uncertainty():
    first = Atoms("SiO", positions=[[0, 0, 0], [1, 0, 0]], cell=[8] * 3)
    second = Atoms("O", cell=[8] * 3)
    with pytest.warns(UserWarning, match="different compositions"):
        sa = StructureAnalyser([first, second], cutoff=1.1)
    line = next(line for line in sa.per_structure_summary().splitlines()
                if "CN(Si-O):" in line)
    assert "unavailable (n=1)" in line
