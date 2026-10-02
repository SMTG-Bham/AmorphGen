"""Experimental-data validation and ensemble comparisons on measured grids."""

import json

import numpy as np
import pytest
from scipy.stats import t

from amorphgen.analysis.experiment import (
    compare_experiment,
    format_experiment_report,
    load_experiment,
)


def test_loader_sorts_values_and_uncertainties_together(tmp_path):
    path = tmp_path / "measured.csv"
    path.write_text("q,S(q),sigma\n3,6,0.3\n1,2,0.1\n2,4,0.2\n")
    result = load_experiment(path, skiprows=1)
    assert result["x"] == [1, 2, 3]
    assert result["observed"] == [2, 4, 6]
    assert result["sigma"] == [.1, .2, .3]
    assert result["units"] == {"x": "1/angstrom", "observed": "dimensionless"}
    json.dumps(result, allow_nan=False)
    comparison = compare_experiment({"q": [1, 3], "s_q": [2, 6]}, result)
    assert comparison["experiment_metadata"]["source"] == str(path)
    assert comparison["experiment_metadata"]["columns"] == [0, 1, 2]
    assert comparison["experiment_metadata"]["delimiter"] == ","
    assert comparison["experiment_metadata"]["skiprows"] == 1
    assert comparison["experiment_metadata"]["comments"] == "#"


def test_loader_whitespace_and_explicit_columns(tmp_path):
    path = tmp_path / "measured.dat"
    path.write_text("# row T r sigma\n10 20 2 0.2\n20 10 1 0.1\n")
    with pytest.raises(ValueError, match="two or three columns"):
        load_experiment(path)
    result = load_experiment(path, "T(r)", columns=(2, 1, 3))
    assert result["kind"] == "tr"
    assert result["x"] == [1, 2]
    assert result["observed"] == [10, 20]
    assert result["sigma"] == [.1, .2]
    assert result["units"] == {"x": "angstrom", "observed": "1/angstrom^2"}
    result = load_experiment(path, "tr", columns=(2, 1))
    assert result["sigma"] is None


@pytest.mark.parametrize("contents,match", [
    ("1 2\n1 3\n", "duplicate"),
    ("nan 2\n2 3\n", "finite"),
    ("1 inf\n2 3\n", "finite"),
    ("1 2 0\n2 3 .1\n", "positive"),
    ("1 2 -.1\n2 3 .1\n", "positive"),
    ("1 2 nan\n2 3 .1\n", "positive"),
    ("q sq\n1 2\n", "cannot read"),
    ("1 2\n2 wrong\n", "cannot read"),
    ("1 2\n2 3 .1\n", "cannot read"),
    ("# empty\n\n", "no data"),
])
def test_loader_rejects_bad_data_without_silent_row_dropping(tmp_path, contents, match):
    path = tmp_path / "measured.dat"
    path.write_text(contents)
    with pytest.raises(ValueError, match=match):
        load_experiment(path)


@pytest.mark.parametrize("kwargs", [
    {"kind": "pdf"}, {"skiprows": -1}, {"skiprows": True},
    {"columns": (0,)}, {"columns": (0, 0)}, {"columns": (-1, 1)},
    {"columns": (True, 1)}, {"columns": (0., 1.)},
    {"delimiter": "::"}, {"comments": ""},
])
def test_loader_rejects_invalid_options(tmp_path, kwargs):
    path = tmp_path / "measured.dat"
    path.write_text("1 2\n")
    with pytest.raises(ValueError):
        load_experiment(path, **kwargs)


def test_compare_known_metrics_without_fitting():
    model = {"q": [1, 2, 3], "per_structure": [[2, 3, 4], [4, 5, 6]]}
    experiment = {"kind": "sq", "x": [1, 2, 3], "observed": [2, 2, 6], "sigma": [1, 2, 1]}
    result = compare_experiment(model, experiment, n_bootstrap=0)
    assert result["calculated"] == [3, 4, 5]
    assert result["residual"] == [1, 2, -1]
    metrics = result["metrics"]
    assert metrics["n_points"] == 3
    assert metrics["rmse"] == pytest.approx(np.sqrt(2))
    assert metrics["mae"] == pytest.approx(4 / 3)
    assert metrics["bias"] == pytest.approx(2 / 3)
    assert metrics["rw"] == pytest.approx(np.sqrt(3 / 41))
    assert metrics["chi_square"] == pytest.approx(3)
    assert metrics["reduced_chi_square"] == pytest.approx(1)
    assert metrics["degrees_of_freedom"] == 3
    assert result["fitting"] == {"scale": 1, "offset": 0, "n_parameters": 0}
    assert result["uncertainty"]["sem"] == [1, 1, 1]
    np.testing.assert_allclose(result["uncertainty"]["ci_low"],
                               np.array([3, 4, 5]) - t.ppf(.975, 1))
    json.dumps(result, allow_nan=False)


def test_compare_interpolates_structures_before_recomputing_bands():
    # Endpoint variances are nonzero, but anticorrelated complete curves all
    # meet at the measured midpoint. Interpolating endpoint CIs is incorrect.
    model = {"q": [0, 2], "per_structure": [[0, 2], [2, 0], [1, 1]]}
    measured = {"kind": "sq", "x": [1], "observed": [1]}
    result = compare_experiment(model, measured, n_bootstrap=200, seed=29)
    assert result["per_structure"] == [[1], [1], [1]]
    assert result["uncertainty"]["sem"] == [0]
    assert result["uncertainty"]["ci_low"] == [1]
    assert result["uncertainty"]["ci_high"] == [1]
    assert result["uncertainty"]["bootstrap_low"] == [1]
    assert result["uncertainty"]["bootstrap_high"] == [1]
    assert result["metrics"]["rmse"] == 0


def test_compare_no_extrapolation_or_bridging_missing_bins():
    model = {"q": [0, 1, 2, 3, 4],
             "per_structure": [[0, None, 2, 3, None], [None, 1, None, 3, 4], None]}
    x = [-1, 0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5]
    measured = {"kind": "sq", "x": x, "observed": x}
    result = compare_experiment(model, measured, n_bootstrap=0)
    assert result["x"] == [0, 1, 2, 2.5, 3, 3.5, 4]
    assert result["included_indices"] == [1, 3, 5, 6, 7, 8, 9]
    assert result["excluded_indices"] == [0, 2, 4, 10]
    assert result["plot_break_before"] == [False, True, True, False, False, False, False]
    assert result["uncertainty"]["n_per_point"] == [1, 1, 1, 1, 2, 1, 1]
    assert result["uncertainty"]["n_total_structures"] == 3
    assert result["uncertainty"]["sem"] == [None, None, None, None, 0, None, None]
    assert result["overlap"]["n_excluded_outside_support"] == 2
    assert result["overlap"]["n_excluded_missing_support"] == 2
    assert result["metrics"]["rmse"] == 0
    json.dumps(result, allow_nan=False)
    clipped = compare_experiment(model, measured, x_range=(.5, 3.5), n_bootstrap=0)
    assert clipped["x"] == [1, 2, 2.5, 3, 3.5]
    assert clipped["overlap"]["n_excluded_x_range"] == 4
    assert clipped["overlap"]["n_excluded_outside_support"] == 0
    assert clipped["overlap"]["n_excluded_missing_support"] == 2


@pytest.mark.parametrize("rows,observed_x,expected_breaks", [
    ([[0, None, 2]], [0, 2], [False, True]),
    ([[0, None, 2], [None, 1, None]], [0, 1, 2], [False, True, True]),
    ([[0, 1, None], [None, 1, 2]], [0, 2], [False, False]),
    ([[0, 1, None]], [0, 1], [False, False]),
])
def test_plot_breaks_preserve_native_support_gaps_on_sparse_measured_grid(
        rows, observed_x, expected_breaks):
    result = compare_experiment(
        {"q": [0, 1, 2], "per_structure": rows},
        {"x": observed_x, "observed": observed_x}, n_bootstrap=0)
    # Every measured coordinate is supported, so excluded observations alone
    # cannot reveal the native interpolation gap in the first two examples.
    assert result["excluded_indices"] == []
    assert result["plot_break_before"] == expected_breaks
    assert result["metrics"]["rmse"] == 0


def test_mean_only_legacy_input_cannot_invent_ensemble_bands_or_measurement_errors():
    model = {"q": [1, 2], "s_q": [2, 3]}
    measured = {"kind": "sq", "x": [1, 1.5, 2], "observed": [1, 2, 3]}
    result = compare_experiment(model, measured)
    assert result["calculated"] == [2, 2.5, 3]
    assert result["uncertainty"]["n_structures"] == 1
    assert result["uncertainty"]["ci_low"] == [None, None, None]
    assert result["metrics"]["chi_square"] is None
    assert result["metrics"]["reduced_chi_square"] is None
    assert result["metrics"]["rw"] == pytest.approx(np.sqrt(1.25 / 14))
    assert result["definitions"]["rw_weighting"] == "unit weights"
    report = format_experiment_report(result)
    assert "no measurement sigma" in report
    assert "undefined with fewer than two" in report


def test_tr_uses_tr_curve_and_sorted_measurements():
    model = {"q": [1, 2], "s_q": [100, 100], "r": [2, 1],
             "per_structure_curves": {"T_r": [[4, 2], [6, 4]]}}
    measured = {"kind": "tr", "x": [2, 1], "observed": [5, 3], "sigma": [.2, .1]}
    result = compare_experiment(model, measured, n_bootstrap=0)
    assert result["x"] == [1, 2]
    assert result["calculated"] == [3, 5]
    assert result["sigma"] == [.1, .2]
    assert result["metrics"]["rmse"] == 0
    report = format_experiment_report(result)
    assert "Experimental T(r)" in report
    assert "dof: 2" in report
    assert "diagonal errors" in report


def test_zero_observed_norm_leaves_rw_undefined():
    result = compare_experiment({"q": [1], "s_q": [0]},
                                {"x": [1], "observed": [0]}, n_bootstrap=0)
    assert result["metrics"]["rw"] is None
    assert result["metrics"]["rmse"] == 0


@pytest.mark.parametrize("model,measured,kwargs,match", [
    ({"q": [1, 2], "s_q": [1, 2]}, {"x": [3], "observed": [3]}, {}, "no supported"),
    ({"q": [1, 2], "s_q": [None, None]}, {"x": [1], "observed": [1]}, {}, "no supported"),
    ({"q": [1, 1], "s_q": [1, 2]}, {"x": [1], "observed": [1]}, {}, "duplicate"),
    ({"q": [1, 2], "per_structure": [[1]]}, {"x": [1], "observed": [1]}, {}, "matching x"),
    ({"q": [1], "s_q": [1]}, {"x": [1], "observed": [1], "sigma": [0]}, {}, "positive"),
    ({"q": [1], "s_q": [1]}, {"kind": "tr", "x": [1], "observed": [1]}, {"kind": "sq"}, "does not match"),
    ({"q": [1], "s_q": [1]}, {"x": [1], "observed": [1]}, {"x_range": (2, 1)}, "bounds in order"),
    ({"q": [1], "s_q": [1]}, {"x": [1], "observed": [1]}, {"confidence": 1}, "confidence"),
])
def test_compare_rejects_undefined_or_invalid_comparisons(model, measured, kwargs, match):
    with pytest.raises(ValueError, match=match):
        compare_experiment(model, measured, **kwargs)
