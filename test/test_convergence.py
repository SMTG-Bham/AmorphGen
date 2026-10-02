"""Precision planning must be auditable and independent of generation order."""

import itertools
import json

import numpy as np
import pytest
from scipy.stats import t

from amorphgen.analysis.convergence import convergence_report
from amorphgen.analysis.uncertainty import summarize_structures


def test_scalar_endpoint_is_student_t_and_matches_uncertainty_summary():
    summary = summarize_structures([1, 3, 5], n_bootstrap=0)
    report = convergence_report({"density": summary}, {"density": 0.1})
    descriptor = report["descriptors"]["density"]
    expected = t.ppf(0.975, 2) * 2 / np.sqrt(3)
    assert descriptor["mean"] == 3
    assert descriptor["std"] == 2
    assert descriptor["half_width"] == pytest.approx(expected)
    assert descriptor["half_width"] == pytest.approx(summary["ci_high"] - summary["mean"])
    assert descriptor["curve"]["sizes"] == [1, 2, 3]
    assert descriptor["curve"]["half_width"][0] is None
    assert descriptor["curve"]["half_width"][-1] == descriptor["half_width"]
    assert descriptor["status"] == "not_met"
    assert report["sampling_unit"] == "structure"
    assert report["band_type"] == "pointwise"


@pytest.mark.parametrize("size", [2, 3, 4, 5])
def test_complete_data_curve_is_exact_all_subset_rms(size):
    values = [1, 2, 4, 9, 12]
    report = convergence_report({"value": values}, {"value": 1}, sizes=[size])
    subset_widths = [t.ppf(0.975, size - 1) * np.std(subset, ddof=1) / np.sqrt(size)
                     for subset in itertools.combinations(values, size)]
    curve = report["descriptors"]["value"]["curve"]
    assert curve["half_width"][curve["sizes"].index(size)] == pytest.approx(
        np.sqrt(np.mean(np.square(subset_widths))))


def test_exact_order_invariance_of_every_output_field_and_no_input_mutation():
    values = np.array([[1e10, -0., 2], [1e-8, 4, np.nan],
                       [-1e10, 4.1, 2.1], [9, 5, 2.2],
                       [7, np.inf, 2.3]])
    original = values.copy()
    baseline = convergence_report({"curve": values, "scalar": values[:, 0]},
                                  {"curve": 0.1, "scalar": 1}, sizes=[7, 3, 2, 3])
    for order in itertools.permutations(range(len(values))):
        permutation = values[list(order)]
        assert convergence_report({"scalar": permutation[:, 0], "curve": permutation},
                                  {"scalar": 1, "curve": 0.1}, sizes=[2, 7, 3]) == baseline
    np.testing.assert_equal(values, original)
    json.dumps(baseline, allow_nan=False)


def test_forecast_is_minimum_integer_total_not_a_rounded_normal_approximation():
    values = [1, 3, 5]
    tolerance = 0.7
    report = convergence_report({"value": values}, {"value": tolerance})
    total = report["estimated_total_structures"]
    std = np.std(values, ddof=1)
    assert total > len(values)
    assert t.ppf(0.975, total - 1) * std / np.sqrt(total) <= tolerance
    assert t.ppf(0.975, total - 2) * std / np.sqrt(total - 1) > tolerance
    assert report["estimated_additional_structures"] == total - len(values)
    assert report["estimate_status"] == "estimated"


def test_curve_uses_worst_bin_and_overall_forecast_uses_strictest_descriptor():
    report = convergence_report({"a": [[1, 10], [2, 20], [3, 30]], "b": [1, 2, 3]},
                                {"a": 1, "b": 5})
    curve = report["descriptors"]["a"]
    assert curve["half_width"] == max(curve["half_width_per_point"])
    assert curve["estimated_total_structures"] == report["estimated_total_structures"]
    assert curve["status"] == "not_met"
    assert report["descriptors"]["b"]["status"] == "met"
    assert report["descriptors"]["b"]["estimated_additional_structures"] == 0
    assert report["status"] == "not_met"


def test_missing_data_scales_counts_conservatively_and_has_exact_endpoint():
    values = [[1, 4], None, [3, None], [5, 8]]
    report = convergence_report({"curve": values}, {"curve": 2}, sizes=[2, 4, 8])
    descriptor = report["descriptors"]["curve"]
    assert descriptor["n_total_structures"] == 4
    assert descriptor["n_structures"] == 3
    assert descriptor["n_per_point"] == [3, 2]
    assert descriptor["curve"]["effective_counts"] == [[1, 1], [3, 2], [6, 4]]
    assert descriptor["curve"]["half_width"][0] is None
    assert descriptor["curve"]["half_width"][1] == descriptor["half_width"]
    assert descriptor["curve_method"] == "availability_adjusted_student_t_planning"
    total = descriptor["estimated_total_structures"]
    confirmation = convergence_report({"curve": values}, {"curve": 2}, sizes=[total - 1, total])
    curve = confirmation["descriptors"]["curve"]["curve"]
    assert curve["half_width"][curve["sizes"].index(total)] <= 2
    assert curve["half_width"][curve["sizes"].index(total - 1)] > 2


@pytest.mark.parametrize("values", [[], [3], [None, np.nan],
                                    [None, 3, np.inf], [[1, None], [2, None]],
                                    [[1, 3], [2, None]], [[], []],
                                    np.empty((0, 2))])
def test_missing_and_singleton_components_never_pass_or_invent_a_forecast(values):
    report = convergence_report({"x": values}, {"x": 1e9}, sizes=[10, 100])
    descriptor = report["descriptors"]["x"]
    assert descriptor["half_width"] is None
    assert descriptor["status"] == report["status"] == "insufficient_data"
    assert descriptor["estimate_status"] == "insufficient_data"
    assert descriptor["estimated_total_structures"] is None
    assert report["estimated_additional_structures"] is None
    assert all(width is None for width in descriptor["curve"]["half_width"])
    json.dumps(report, allow_nan=False)


def test_zero_observed_variance_is_explicitly_qualified():
    report = convergence_report({"x": [4, 4, 4]}, {"x": 0.01})
    descriptor = report["descriptors"]["x"]
    assert descriptor["zero_variance"] is True
    assert descriptor["half_width"] == 0
    assert report["status"] == report["estimate_status"] == "met"
    assert report["estimated_total_structures"] == 3
    assert report["estimated_additional_structures"] == 0
    assert any("does not establish" in text for text in report["assumptions"])


def test_declared_targets_only_determine_overall_status_and_units_are_preserved():
    report = convergence_report({"good": {"per_structure": [3, 3], "units": "eV/atom"},
                                 "missing": [None, 1]}, {"good": 0.1})
    assert report["descriptors"]["good"]["units"] == "eV/atom"
    assert report["descriptors"]["missing"]["units"] is None
    assert report["descriptors"]["missing"]["tolerance"] is None
    assert report["descriptors"]["missing"]["status"] == "undeclared"
    assert report["status"] == "met"
    undeclared = convergence_report({"x": [1, 2]})
    assert undeclared["status"] == "undeclared"
    assert undeclared["estimated_total_structures"] is None


def test_cap_exceeded_never_appears_met_or_reports_an_unattainable_target():
    report = convergence_report({"x": [1, 3, 5]}, {"x": 1e-6}, max_structures=10)
    descriptor = report["descriptors"]["x"]
    assert report["status"] == descriptor["status"] == "not_met"
    assert report["estimate_status"] == descriptor["estimate_status"] == "exceeds_max_structures"
    assert report["estimated_total_structures"] is None
    assert descriptor["estimated_additional_structures"] is None
    # The cap controls future forecasts, not current evidence.
    report = convergence_report({"x": [1, 1, 1]}, {"x": 1}, max_structures=1)
    assert report["estimated_total_structures"] == 3


def test_custom_sizes_are_sorted_deduplicated_and_include_current_endpoint():
    report = convergence_report({"x": [1, 2, 3]}, {"x": 1}, sizes=[10, 2, 10])
    assert report["descriptors"]["x"]["curve"]["sizes"] == [2, 3, 10]


def test_extreme_valid_confidence_does_not_round_quantile_to_infinity():
    report = convergence_report({"x": [1, 2, 3]}, {"x": 1},
                                confidence=np.nextafter(1., 0.))
    assert np.isfinite(report["descriptors"]["x"]["half_width"])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("tolerance", [0, -1, np.inf, np.nan, True, "0.1", None,
                                      10 ** 1000])
def test_invalid_tolerances_rejected(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        convergence_report({"x": [1, 2]}, {"x": tolerance})


@pytest.mark.parametrize("options", [
    {"confidence": 0}, {"confidence": 1}, {"confidence": np.nan},
    {"confidence": True}, {"confidence": "0.95"},
    {"max_structures": 0}, {"max_structures": -1}, {"max_structures": 2.5},
    {"max_structures": True}, {"sizes": [0]}, {"sizes": [-1]},
    {"sizes": [1.1]}, {"sizes": [True]}, {"sizes": 1},
])
def test_invalid_planning_options_rejected(options):
    with pytest.raises(ValueError):
        convergence_report({"x": [1, 2]}, {"x": 0.1}, **options)


@pytest.mark.parametrize("descriptors,tolerances", [
    ({"x": [1, 2]}, {"unknown": 1}),
    ({"x": [1, 2], "y": [1]}, {}),
    ({"x": [[1, 2], [1]]}, {}),
    ({"x": [[[[1]]]]}, {}),
    ({"x": {"mean": 1}}, {}),
    ({"x": np.array([1 + 2j, 3 + 4j])}, {}),
    ({"": [1, 2]}, {}),
    ({"x": [1, 2]}, {1: 0.1}),
    ({"x": [1, 2]}, []),
    ([1, 2], {}),
])
def test_invalid_descriptor_inputs_rejected(descriptors, tolerances):
    with pytest.raises(ValueError):
        convergence_report(descriptors, tolerances)


def test_extreme_inputs_cannot_overflow_to_fake_zero_variance():
    with pytest.raises(ValueError, match="too large"):
        convergence_report({"x": [-1.7e308, 1.7e308]}, {"x": 1})
