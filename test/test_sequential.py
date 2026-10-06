"""Finite-sample precision must account for every component and every look."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from amorphgen.analysis.sequential import sequential_convergence_report


def test_public_export_and_native_report_contract():
    from amorphgen.analysis import sequential_convergence_report as exported

    assert exported is sequential_convergence_report
    report = exported({"density": [0.4, 0.6]}, {"density": _target()})
    assert report["schema"] == "amorphgen.sequential_precision.v1"
    assert report["sampling_unit"] == "structure"
    assert report["n_structures"] == 2
    assert report["targets"]["density"]["n_structures"] == 2
    assert report["sequentially_valid"] is True
    assert "n_boxes" not in report


def test_planning_report_does_not_claim_optional_stopping_coverage():
    from amorphgen.analysis import convergence_report

    report = convergence_report({"density": [0.5, 0.5]}, {"density": 0.1})
    assert report["status"] == "met"
    assert report["sequentially_valid"] is False
    assert report["optional_stopping_coverage_guaranteed"] is False


def _target(tolerance=0.1, bounds=(0.0, 1.0), components=1):
    return {"bounds": list(bounds), "tolerance": tolerance, "components": components}


def test_two_sided_empirical_bernstein_matches_theorem_and_exact_spending():
    values = [0.1, 0.4, 0.7, 1.0]
    report = sequential_convergence_report({"x": values}, {"x": _target()}, confidence=0.9)
    delta = 0.1 / (4 * 3)
    log_factor = math.log(4 / delta)
    expected = math.sqrt(2 * np.var(values, ddof=1) * log_factor / 4) + 7 * log_factor / 9
    target = report["targets"]["x"]
    assert target["mean"] == pytest.approx(np.mean(values))
    assert target["std"] == pytest.approx(np.std(values, ddof=1))
    assert target["ci_halfwidth"] == pytest.approx(expected)
    spending = report["alpha_spending"]
    assert spending["alpha_per_component_at_n"] == pytest.approx(delta)
    assert spending["alpha_spent_upper_bound"] == pytest.approx(sum(0.1 / (n * (n - 1)) for n in range(2, 5)))
    assert spending["alpha_remaining_lower_bound"] == pytest.approx(0.1 / 4)
    assert report["inference_contract"]["sequentially_valid"] is True
    assert report["inference_contract"]["optional_stopping_coverage_guaranteed"] is True
    json.dumps(report, allow_nan=False)


def test_multiplicity_includes_every_component_and_never_clips_sample_variance():
    report = sequential_convergence_report(
        {"scalar": [0, 1], "vector": [[0, 0.5], [1, 0.5]]},
        {"scalar": _target(), "vector": _target(components=2)},
    )
    log_factor = math.log(4 * 3 * 2 / 0.05)
    expected = math.sqrt(2 * 0.5 * log_factor / 2) + 7 * log_factor / 3
    assert report["n_components"] == 3
    assert report["targets"]["scalar"]["ci_halfwidth"] == pytest.approx(expected)
    assert report["targets"]["vector"]["ci_halfwidth"][0] == pytest.approx(expected)
    assert report["targets"]["vector"]["ci_halfwidth"][1] == pytest.approx(7 * log_factor / 3)


def test_zero_variance_requires_evidence_and_eventually_passes():
    small = sequential_convergence_report({"x": [0.5, 0.5]}, {"x": _target()})
    assert small["targets"]["x"]["zero_observed_variance"] is True
    assert small["targets"]["x"]["ci_halfwidth"] > 0
    assert not small["passed"]
    enough = sequential_convergence_report({"x": [0.5] * 1000}, {"x": _target()})
    assert enough["passed"]
    assert 0 < enough["targets"]["x"]["ci_halfwidth"] <= 0.1


def test_constant_decimal_observations_have_exactly_zero_sample_variance():
    report = sequential_convergence_report(
        {"density": [0.976991] * 6},
        {"density": _target(tolerance=0.1, bounds=(0.1, 5.1))},
    )
    target = report["targets"]["density"]
    assert target["std"] == 0.0
    assert target["zero_observed_variance"] is True
    assert target["ci_halfwidth"] > 0


@pytest.mark.parametrize("rows", [[], [0.5]])
def test_empty_and_singleton_do_not_pass_and_are_strict_json(rows):
    report = sequential_convergence_report({"x": rows}, {"x": _target(tolerance=1000)})
    assert not report["converged"]
    assert report["status"] == "insufficient_data"
    assert report["targets"]["x"]["ci_halfwidth"] is None
    assert report["alpha_spending"]["alpha_spent_upper_bound"] == 0
    json.dumps(report, allow_nan=False)


def test_empty_vector_preserves_predeclared_multiplicity():
    report = sequential_convergence_report({"x": []}, {"x": _target(components=3)})
    assert report["n_components"] == 3
    assert report["targets"]["x"]["mean"] == [None, None, None]
    assert not report["passed"]


def test_resume_and_irregular_looks_use_original_cumulative_count():
    targets = {"x": _target()}
    values = np.linspace(0.2, 0.8, 50)
    for size in [2, 7, 19]:
        sequential_convergence_report({"x": values[:size]}, targets)
    resumed = sequential_convergence_report({"x": values}, targets)
    assert resumed == sequential_convergence_report({"x": values.tolist()}, targets)
    assert resumed["alpha_spending"]["alpha_per_component_at_n"] == pytest.approx(0.05 / (50 * 49))


def test_affine_scaling_and_all_targets_required():
    base = np.linspace(0, 1, 300)
    targets = {"a": _target(tolerance=10), "b": _target(tolerance=0.01, bounds=(-3, 7))}
    report = sequential_convergence_report({"a": base, "b": -3 + 10 * base}, targets)
    assert report["targets"]["b"]["ci_halfwidth"] == pytest.approx(10 * report["targets"]["a"]["ci_halfwidth"])
    assert report["targets"]["a"]["passed"]
    assert not report["targets"]["b"]["passed"]
    assert not report["converged"]


@pytest.mark.parametrize("value", [None, np.nan, np.inf, -np.inf, True, False, 1j, "0.5", -0.1, 1.1])
def test_invalid_observations_never_dropped_or_clipped(value):
    with pytest.raises(ValueError, match="observation"):
        sequential_convergence_report({"x": [0.5, value]}, {"x": _target()})


@pytest.mark.parametrize("observations,targets", [
    ({}, {}),
    ({"x": [0, 1]}, {"y": _target()}),
    ({"x": [0, 1], "extra": [0, 1]}, {"x": _target()}),
    ({"x": [0, 1], "y": [0]}, {"x": _target(), "y": _target()}),
    ({"x": [[0, 1], [0, 1]]}, {"x": _target()}),
    ({"x": [0, 1]}, {"x": _target(components=2)}),
    ({"x": [[0], [0, 1]]}, {"x": _target()}),
    ({"x": []}, {"x": _target(components=0)}),
    ({"x": []}, {"x": _target(components=True)}),
    ({"x": []}, {"x": _target(components=1.0)}),
    ({"x": []}, {"x": _target(bounds=(0, 0))}),
    ({"x": []}, {"x": _target(bounds=(1, 0))}),
    ({"x": []}, {"x": _target(bounds=(0, np.inf))}),
    ({"x": []}, {"x": _target(bounds=(-1e308, 1e308))}),
    ({"x": []}, {"x": {**_target(), "bounds": np.array(0.0)}}),
    ({"x": []}, {"x": _target(tolerance=0)}),
    ({"x": []}, {"x": _target(tolerance=np.nan)}),
    ({"x": []}, {"x": _target(tolerance=True)}),
    ({"x": []}, {"x": {**_target(), "auto_bounds": True}}),
    ({"": []}, {"": _target()}),
])
def test_invalid_target_family_and_shape_rejected(observations, targets):
    with pytest.raises(ValueError):
        sequential_convergence_report(observations, targets)


@pytest.mark.parametrize("confidence", [0, 1, -1, np.nan, np.inf, True, "0.95"])
def test_invalid_confidence_rejected(confidence):
    with pytest.raises(ValueError, match="confidence"):
        sequential_convergence_report({"x": [0, 1]}, {"x": _target()}, confidence=confidence)


def test_extreme_confidence_and_large_finite_observations_are_safe():
    report = sequential_convergence_report(
        {"x": [1e300, 1.1e300, 1.2e300]},
        {"x": _target(tolerance=1e290, bounds=(1e300, 1.2e300))},
        confidence=np.nextafter(1.0, 0.0),
    )
    assert math.isfinite(report["targets"]["x"]["std"])
    assert report["targets"]["x"]["ci_halfwidth"] > 0
    assert not report["passed"]
    json.dumps(report, allow_nan=False)


def test_unrepresentable_radius_is_null_and_cannot_pass():
    report = sequential_convergence_report(
        {"x": [0, 1e308]}, {"x": _target(tolerance=1e308, bounds=(0, 1e308))}
    )
    assert report["targets"]["x"]["ci_halfwidth"] is None
    assert not report["passed"]
    json.dumps(report, allow_nan=False)


def test_smallest_positive_support_range_never_claims_zero_uncertainty():
    smallest = float(np.nextafter(0.0, 1.0))
    report = sequential_convergence_report(
        {"x": [0.0] * 1000},
        {"x": _target(tolerance=smallest, bounds=(0, smallest))},
    )
    assert report["targets"]["x"]["ci_halfwidth"] > 0


@pytest.mark.parametrize("lower,upper,tolerance", [
    (1e16, 1e16 + 2, 0.5),
    (1.0, np.nextafter(1.0, 2.0), 1e-16),
])
def test_radius_accounts_for_rounding_the_reported_mean(lower, upper, tolerance):
    report = sequential_convergence_report(
        {"x": [lower, upper] * 500},
        {"x": _target(tolerance=tolerance, bounds=(lower, upper))},
    )
    target = report["targets"]["x"]
    # The true sample mean is halfway between adjacent representable floats.
    # Reporting either endpoint moves the center by half the support range.
    assert target["mean_rounding_allowance"] == pytest.approx((upper - lower) / 2, rel=1e-12, abs=0)
    assert target["ci_halfwidth"] > (upper - lower) / 2
    assert not target["passed"]
    assert not report["converged"]
