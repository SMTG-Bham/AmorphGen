"""Experimental and XRD exports retain scientific support and uncertainty."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from amorphgen.analysis.scattering_output import (
    save_experiment_comparison, save_xrd_pattern,
)
from amorphgen.analysis.uncertainty import summarize_structures
from amorphgen.analysis.experiment import format_experiment_report


def _comparison(kind="sq"):
    uncertainty = summarize_structures([[1, 2, None], [3, 4, 5]], n_bootstrap=0)
    return {
        "kind": kind, "x": [.2, .4, .8], "observed": [1.5, 3.5, 5.5],
        "sigma": [.1, .2, .3], "calculated": uncertainty["mean"],
        "residual": [.5, -.5, -.5], "uncertainty": uncertainty,
        "included_indices": [0, 1, 3], "excluded_indices": [2],
        "per_structure": uncertainty["per_structure"],
        "metrics": {"n_points": 3, "rmse": .5, "mae": .5, "bias": -1 / 6,
                    "rw": .2, "chi_square": 10, "reduced_chi_square": 10 / 3,
                    "degrees_of_freedom": 3},
        "overlap": {"n_input_points": 4, "n_points": 3, "n_excluded": 1,
                    "n_excluded_x_range": 0, "n_excluded_outside_support": 0,
                    "n_excluded_missing_support": 1, "x_min": .2, "x_max": .8,
                    "model_x_min": 0, "model_x_max": 1},
        "fitting": {"scale": 1., "offset": 0., "n_parameters": 0},
        "units": {"x": "1/angstrom" if kind == "sq" else "angstrom",
                  "observed": "dimensionless" if kind == "sq" else "1/angstrom^2"},
        "definitions": {"rw_weighting": "inverse measurement variance"},
    }


def _xrd():
    uncertainty = summarize_structures([[10, None, 20], [12, None, 24]], n_bootstrap=0)
    return {
        "q": [1, 2, 3], "two_theta": [14, 28, 43],
        "intensity": uncertainty["mean"], "per_structure": uncertainty["per_structure"],
        "uncertainty": uncertainty, "wavelength": 1.5406,
        "method": "direct", "weighting": "xray", "n_per_bin": [4, 0, 12],
        "intensity_raw": uncertainty["mean"], "metadata": {"sigma_q": .1},
        "intensity_convention": "coherent intensity per atom",
        "intensity_units": "electron^2/atom",
    }


@pytest.mark.parametrize("kind", ["sq", "tr"])
def test_comparison_exports_statistics_intervals_and_exact_support(tmp_path, kind):
    result = _comparison(kind)
    paths = save_experiment_comparison(result, tmp_path, prefix="sample.v1", dpi=40,
                                       save_pdf=True)
    assert json.loads(Path(paths["json"]).read_text()) == result
    assert Path(paths["json"]).name == f"sample.v1_experiment_{kind}.json"
    with Path(paths["csv"]).open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert [int(row["input_index"]) for row in rows] == [0, 1, 3]
    assert [float(row["n_structures"]) for row in rows] == [2, 2, 1]
    assert rows[2]["ci_low"] == rows[2]["sem"] == ""
    assert float(rows[0]["residual"]) == .5
    assert float(rows[0]["ci_low"]) == result["uncertainty"]["ci_low"][0]
    report = Path(paths["report"]).read_text()
    assert report == format_experiment_report(result) + "\n"
    assert "95% pointwise Student-t" in report
    assert "not a simultaneous band" in report
    assert "1 without finite model support" in report
    assert "inverse measurement variance" in report
    assert "No scale or offset fitted" in report
    assert all(Path(paths[extension]).stat().st_size > 500 for extension in ("png", "pdf"))


def test_comparison_plot_does_not_connect_across_excluded_points(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from amorphgen.analysis import plotting

    captured = []
    monkeypatch.setattr(plotting, "_save_fig", lambda figure, *args, **kw: captured.append(figure))
    original = plt.get_fignums()
    save_experiment_comparison(_comparison(), tmp_path)
    assert plt.get_fignums() == original
    ax, residual_ax = captured[0].axes
    mean = next(line for line in ax.lines if line.get_label() == "Ensemble mean")
    np.testing.assert_allclose(mean.get_xdata(), [.2, .4, np.nan, .8], equal_nan=True)
    np.testing.assert_allclose(mean.get_ydata(), [2, 3, np.nan, 5], equal_nan=True)
    np.testing.assert_allclose(residual_ax.lines[1].get_ydata(),
                               [.5, -.5, np.nan, -.5], equal_nan=True)
    assert "95% pointwise t CI of mean" in ax.get_legend_handles_labels()[1]
    assert "Measured (±1σ)" in ax.get_legend_handles_labels()[1]
    assert ax.get_ylabel() == "S(q)"
    assert residual_ax.get_xlabel() == "q (Å⁻¹)"
    # An unavailable CI at the singleton point must not extend the first band.
    bands = next(collection for collection in ax.collections
                 if collection.get_label() == "95% pointwise t CI of mean")
    assert max(vertex[0] for path in bands.get_paths() for vertex in path.vertices) <= .4


def test_comparison_without_measurement_sigma_uses_unweighted_points(tmp_path, monkeypatch):
    from amorphgen.analysis import plotting

    result = _comparison("tr")
    result["sigma"] = None
    captured = []
    monkeypatch.setattr(plotting, "_save_fig", lambda figure, *args, **kw: captured.append(figure))
    paths = save_experiment_comparison(result, tmp_path)
    with Path(paths["csv"]).open() as handle:
        assert all(row["sigma"] == "" for row in csv.DictReader(handle))
    ax, residual_ax = captured[0].axes
    assert "Measured" in ax.get_legend_handles_labels()[1]
    assert ax.get_ylabel() == "T(r) (Å⁻²)"
    assert residual_ax.get_xlabel() == "r (Å)"


def test_comparison_breaks_source_gaps_even_with_contiguous_observation_indices(tmp_path, monkeypatch):
    from amorphgen.analysis import plotting

    result = _comparison()
    result["included_indices"] = [0, 1, 2]
    result["plot_break_before"] = [False, False, True]
    captured = []
    monkeypatch.setattr(plotting, "_save_fig", lambda figure, *args, **kw: captured.append(figure))
    paths = save_experiment_comparison(result, tmp_path)
    with Path(paths["csv"]).open() as handle:
        assert [row["plot_break_before"] for row in csv.DictReader(handle)] == ["False", "False", "True"]
    ax = captured[0].axes[0]
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [.2, .4, np.nan, .8], equal_nan=True)


@pytest.mark.parametrize("kind", ["sq", "xrd"])
def test_single_structure_figure_explicitly_reports_unavailable_intervals(tmp_path, monkeypatch, kind):
    from amorphgen.analysis import plotting

    result = _comparison() if kind == "sq" else _xrd()
    result["uncertainty"] = summarize_structures([result["calculated" if kind == "sq" else "intensity"]],
                                                 n_bootstrap=0)
    captured = []
    monkeypatch.setattr(plotting, "_save_fig", lambda figure, *args, **kw: captured.append(figure))
    (save_experiment_comparison if kind == "sq" else save_xrd_pattern)(result, tmp_path)
    assert "95% pointwise t CI of mean" not in captured[0].axes[0].get_legend_handles_labels()[1]
    assert any("Confidence intervals unavailable: fewer than two structures" in text.get_text()
               for text in captured[0].texts)


def test_xrd_exports_keep_missing_shells_counts_and_units(tmp_path, monkeypatch):
    from amorphgen.analysis import plotting

    result = _xrd()
    captured = []
    original_save = plotting._save_fig

    def save(figure, *args, **kwargs):
        captured.append(figure)
        original_save(figure, *args, **kwargs)

    monkeypatch.setattr(plotting, "_save_fig", save)
    paths = save_xrd_pattern(result, tmp_path, dpi=40, save_pdf=True)
    assert json.loads(Path(paths["json"]).read_text()) == result
    with Path(paths["csv"]).open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[1]["intensity_electron^2_per_atom"] == ""
    assert float(rows[1]["n_structures"]) == 0
    assert float(rows[1]["n_wavevectors"]) == 0
    assert rows[1]["intensity_raw_electron^2_per_atom"] == ""
    assert rows[1]["ci_low"] == rows[1]["std"] == ""
    ax = captured[0].axes[0]
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [11, np.nan, 22], equal_nan=True)
    assert ax.lines[0].get_marker() == "."  # Isolated shells remain visible.
    bars = [collection for collection in ax.collections if hasattr(collection, "get_segments")]
    assert len(bars) == 1
    assert len(bars[0].get_segments()) == 2
    assert ax.get_xlabel() == "2θ (degrees)"
    assert ax.get_ylabel() == "Coherent intensity per atom (electron²)"
    assert "1.5406" in ax.get_title()
    assert all(Path(paths[extension]).stat().st_size > 500 for extension in ("png", "pdf"))


@pytest.mark.parametrize("save,result", [(save_experiment_comparison, _comparison),
                                          (save_xrd_pattern, _xrd)])
def test_nonfinite_json_fails_before_creating_files(tmp_path, save, result):
    invalid = result()
    invalid["nonfinite"] = float("nan")
    with pytest.raises(ValueError, match="JSON"):
        save(invalid, tmp_path)
    assert not list(tmp_path.iterdir())
