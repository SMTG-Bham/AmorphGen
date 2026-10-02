"""Structure-level sampling, normalisation and uncertainty of curve means."""

import numpy as np
import pytest
from ase import Atoms
from scipy.stats import t

from amorphgen.analysis.rdf import (
    compute_averaged_rdf,
    compute_rdf,
    compute_structure_factor,
    compute_structure_factor_direct,
    compute_total_correlation,
)


def _structure(seed, length=8.0, symbols="Si4O8"):
    rng = np.random.default_rng(seed)
    atoms = Atoms(symbols, cell=[length] * 3, pbc=True)
    atoms.positions = rng.uniform(0, length, (len(atoms), 3))
    return atoms


def _rows(result):
    return np.asarray(result["per_structure"], dtype=float)


def test_rdf_uncertainty_uses_structures_not_pair_counts():
    structures = [_structure(i, symbols=symbols) for i, symbols in
                  enumerate(("Si4O8", "Si8O16", "Si12O24"))]
    result = compute_rdf(structures, rmax=3.0, nbins=30, sigma=0,
                         n_bootstrap=200, seed=23)
    rows = _rows(result)
    expected = [compute_rdf([a], rmax=3.0, nbins=30, sigma=0)["g_r"]
                for a in structures]
    np.testing.assert_allclose(rows, expected)
    np.testing.assert_allclose(result["g_r"], rows.mean(axis=0))
    sem = rows.std(axis=0, ddof=1) / np.sqrt(3)
    stats = result["uncertainty"]
    np.testing.assert_allclose(stats["sem"], sem)
    np.testing.assert_allclose(stats["ci_low"], rows.mean(axis=0) - t.ppf(.975, 2) * sem)
    np.testing.assert_allclose(stats["ci_high"], rows.mean(axis=0) + t.ppf(.975, 2) * sem)
    repeat = compute_rdf(structures, rmax=3.0, nbins=30, sigma=0,
                         n_bootstrap=200, seed=23)
    assert stats["bootstrap_low"] == repeat["uncertainty"]["bootstrap_low"]
    assert stats["bootstrap_high"] == repeat["uncertainty"]["bootstrap_high"]


def test_missing_partial_excluded_but_observed_zero_curve_retained():
    pair = Atoms("SiO", positions=[[0, 0, 0], [1, 0, 0]], cell=[8] * 3, pbc=True)
    distant = pair.copy()
    distant.positions[1] = [3, 0, 0]
    missing = _structure(1, symbols="Si4")
    result = compute_rdf([pair, distant, missing], pair="Si-O", rmax=2, nbins=20,
                         sigma=0)
    rows = _rows(result)
    assert np.all(rows[1] == 0)
    assert np.all(np.isnan(rows[2]))
    np.testing.assert_allclose(result["g_r"], rows[:2].mean(axis=0))
    assert result["uncertainty"]["n_structures"] == 2
    assert result["uncertainty"]["n_per_point"] == [2] * 20
    only_one = compute_rdf([pair, missing], pair="Si-O", rmax=2, nbins=20)
    assert only_one["uncertainty"]["sem"] == [None] * 20
    assert only_one["uncertainty"]["ci_low"] == [None] * 20


def test_average_rdf_keeps_shared_grid_for_different_cells():
    structures = [_structure(0, 8), _structure(1, 10)]
    result = compute_averaged_rdf(structures, nbins=25)
    expected = [compute_rdf([a], rmax=4.0, nbins=25, sigma=0)["g_r"]
                for a in structures]
    np.testing.assert_allclose(_rows(result), expected)
    np.testing.assert_allclose(result["g_r_mean"], np.mean(expected, axis=0))
    np.testing.assert_allclose(result["g_r_std"], np.std(expected, axis=0))


@pytest.mark.parametrize("weighting", ["unweighted", "xray", "neutron"])
def test_ft_sq_transforms_each_density_and_composition(weighting):
    structures = [_structure(0, 8, "Si4O8"), _structure(1, 10, "Si8O4")]
    kwargs = dict(qmax=6, nq=25, rmax=4, weighting=weighting)
    result = compute_structure_factor(structures, **kwargs)
    expected = [compute_structure_factor([a], **kwargs)["s_q"] for a in structures]
    np.testing.assert_allclose(_rows(result), expected)
    np.testing.assert_allclose(result["s_q"], np.mean(expected, axis=0))
    np.testing.assert_allclose(result["uncertainty"]["sem"],
                               np.abs(np.subtract(*expected)) / 2)


def test_direct_sq_equal_structure_weight_and_missing_partials():
    structures = [_structure(0, 8, "Si4O8"), _structure(1, 10, "Si8")]
    kwargs = dict(qmax=5, nq=20, weighting="xray", partials=True, sigma_q=.1)
    result = compute_structure_factor_direct(structures, **kwargs)
    singles = [compute_structure_factor_direct([a], **kwargs) for a in structures]
    rows = np.asarray([s["s_q"] for s in singles])
    np.testing.assert_allclose(_rows(result), rows, equal_nan=True)
    valid_both = np.isfinite(rows).all(axis=0)
    assert valid_both.sum() > 5
    np.testing.assert_allclose(np.asarray(result["s_q"])[valid_both],
                               rows[:, valid_both].mean(axis=0))
    counts = np.asarray([s["n_per_bin"] for s in singles])
    assert np.any(counts[0, valid_both] != counts[1, valid_both])
    pooled = (np.nan_to_num(rows) * counts).sum(axis=0) / np.maximum(counts.sum(axis=0), 1)
    assert not np.allclose(np.asarray(result["s_q"])[valid_both], pooled[valid_both])
    partial = result["partials_uncertainty"]["O-Si"]
    assert partial["n_structures"] == 1
    assert all(value is None for value in partial["sem"])
    assert np.isnan(np.asarray(result["partials_per_structure"]["O-Si"][1], dtype=float)).all()


def test_weighted_ft_explains_unestimable_singleton_species_partial():
    atoms = _structure(0, symbols="Si8O")
    with pytest.raises(ValueError, match="structure_factor_direct"):
        compute_structure_factor([atoms], weighting="xray", qmax=5, nq=20)
    direct = compute_structure_factor_direct([atoms], qmax=5, nq=20, weighting="xray")
    assert np.isfinite(direct["s_q"]).any()


@pytest.mark.parametrize("window", [None, "lorch"])
def test_tr_transforms_per_structure_before_averaging(window):
    structures = [_structure(0, 8), _structure(1, 10)]
    kwargs = dict(qmax=6, nq=30, qmin=.3, rmax=4, nr=40, weighting="neutron", window=window)
    result = compute_total_correlation(structures, **kwargs)
    singles = [compute_total_correlation([a], **kwargs) for a in structures]
    for name in ("g_r", "T_r", "G_r"):
        expected = np.asarray([s[name] for s in singles])
        np.testing.assert_allclose(result["per_structure_curves"][name], expected)
        np.testing.assert_allclose(result[name], expected.mean(axis=0))
        np.testing.assert_allclose(result["curve_uncertainty"][name]["sem"],
                                   np.abs(expected[0] - expected[1]) / 2)
    # The ensemble mean density times the ensemble mean g is generally biased.
    wrong = 4 * np.pi * np.asarray(result["r"]) * result["rho"] * np.asarray(result["g_r"])
    assert not np.allclose(result["T_r"], wrong)
    assert result["uncertainty"] == result["curve_uncertainty"]["T_r"]


def test_comparison_exports_structure_uncertainty(tmp_path, monkeypatch):
    import csv
    import json
    from amorphgen.analysis import StructureAnalyser
    from amorphgen.analysis import comparison_plots as plots
    from amorphgen.analysis.structure import BondAngleData

    analyser = StructureAnalyser([_structure(0, 8), _structure(1, 10)], cutoff=2.0)
    spec = plots.EnsembleSpec.from_analyser("Example", analyser)
    # Two structures with very different angle counts must contribute equally.
    angles = BondAngleData(2)
    angles["O-Si-O"] = [60.0] * 9 + [120.0]
    angles.per_structure = [{"O-Si-O": [60.0] * 9}, {"O-Si-O": [120.0]}]
    monkeypatch.setattr(analyser, "_compute_all_angles", lambda *a, **k: angles)
    monkeypatch.setattr(plots, "_save", lambda *a, **k: None)
    plots.plot_bond_angles([spec], [("O-Si-O", "-")], str(tmp_path), save_pdf=False)
    with open(tmp_path / "angles.csv") as handle:
        distribution = {float(row["angle_deg"]): float(row["probability_density"])
                        for row in csv.DictReader(handle)}
    assert distribution[61] == pytest.approx(.25)
    assert distribution[121] == pytest.approx(.25)
    with open(tmp_path / "angles_uncertainty.json") as handle:
        summary = json.load(handle)["descriptors"]["Example: O-Si-O"]
    assert summary["n_structures"] == 2
    assert len(summary["per_structure"]) == 2

    plots.plot_partial_rdf([spec], [("Si-O", "-")], str(tmp_path), save_pdf=False)
    with open(tmp_path / "rdf_uncertainty.json") as handle:
        summary = json.load(handle)["descriptors"]["Example: Si-O"]
    assert summary["n_structures"] == 2
    assert len(summary["sem"]) == 200

    # Cached analysers have no file list; density must use their loaded atoms.
    plots.plot_density([spec], None, str(tmp_path), save_pdf=False)
    with open(tmp_path / "density_uncertainty.json") as handle:
        summary = json.load(handle)["descriptors"]["Example"]
    rho = analyser.density()["values"]
    assert summary["sem"][0] == pytest.approx(abs(rho[0] - rho[1]) / 2)
    assert (tmp_path / "density_per_structure.csv").is_file()
