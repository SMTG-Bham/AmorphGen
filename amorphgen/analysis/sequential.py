"""Time-uniform precision for a fixed family of bounded structure means.

For every component and every cumulative sample size n >= 2, allocate
alpha / (K*n*(n-1)) to a two-sided empirical Bernstein interval.  The sum
over all components and all sample sizes is alpha.  Consequently the joint
coverage event survives arbitrary batch sizes, optional stopping, and resume
when the full original prefix and the same target specification are retained.

The fixed-time inequality is Maurer and Pontil (2009), Theorem 11, applied
to X and 1-X with half the component error budget for each tail:
https://www.cs.mcgill.ca/~colt2009/papers/012.pdf
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Integral, Real
from typing import Any

import numpy as np


_METHOD = "bounded_empirical_bernstein_alpha_spending"
_SOURCE = "https://www.cs.mcgill.ca/~colt2009/papers/012.pdf"
_ASSUMPTIONS = [
    "Each row is one independently generated structure from the same fixed "
    "generating distribution; correlated trajectory frames are not independent rows.",
    "The target names, component definitions, component counts, and support bounds "
    "are fixed before sampling. Bounds describe population support, not observed extrema.",
    "Every generated structure contributes to every targeted component. Missing, "
    "non-finite, rejected, or selectively omitted observations cannot be silently dropped.",
    "Resume retains the entire original sample prefix and immutable target specification; "
    "it does not reset the cumulative sample count or reuse structures.",
    "Coverage concerns equal-structure population means of the declared components. "
    "It does not establish physical equilibration or accuracy of the simulation model.",
]


def _finite_real(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite real number")
    return result


def _target_config(name: str, config: Any) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError(f"target {name!r} must be a mapping")
    unknown = set(config) - {"bounds", "tolerance", "components"}
    if unknown:
        raise ValueError(f"target {name!r} contains unsupported fields: {sorted(map(str, unknown))}")
    bounds = config.get("bounds")
    if (
        not isinstance(bounds, (list, tuple, np.ndarray))
        or np.ndim(bounds) != 1
        or len(bounds) != 2
    ):
        raise ValueError(f"target {name!r} bounds must be [lower, upper]")
    lower = _finite_real(bounds[0], f"target {name!r} lower bound")
    upper = _finite_real(bounds[1], f"target {name!r} upper bound")
    if not lower < upper or not math.isfinite(upper - lower):
        raise ValueError(f"target {name!r} bounds must have a positive finite range")
    tolerance = _finite_real(config.get("tolerance"), f"target {name!r} tolerance")
    if tolerance <= 0:
        raise ValueError(f"target {name!r} tolerance must be positive")
    components = config.get("components", 1)
    if (
        isinstance(components, (bool, np.bool_))
        or not isinstance(components, Integral)
        or components < 1
        or components > np.iinfo(np.intp).max
    ):
        raise ValueError(f"target {name!r} components must be a positive integer")
    return {"bounds": [lower, upper], "tolerance": tolerance, "components": int(components)}


def _observations(values: Any, name: str, config: Mapping[str, Any]) -> np.ndarray:
    components = int(config["components"])
    try:
        # Object conversion preserves booleans and complex values, including in
        # heterogeneous lists, so float coercion cannot conceal invalid input.
        raw = np.asarray(values, dtype=object)
        if raw.ndim == 1 and raw.size == 0:
            raw = raw.reshape(0, components)
        elif raw.ndim == 1 and components == 1:
            raw = raw.reshape(-1, 1)
        if raw.ndim != 2 or raw.shape[1] != components:
            raise ValueError(f"expected a structure-by-{components}-component matrix")
        array = np.empty(raw.shape, dtype=float)
        for index in np.ndindex(raw.shape):
            array[index] = _finite_real(raw[index], f"observation {index}")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid observations for target {name!r}: {exc}") from exc
    lower, upper = config["bounds"]
    if np.any(array < lower) or np.any(array > upper):
        raise ValueError(f"observations for target {name!r} violate declared support bounds")
    return array


def _native(values: np.ndarray, components: int) -> Any:
    def one(value: Any) -> float | None:
        value = float(value)
        return value if math.isfinite(value) else None

    result = [one(value) for value in np.asarray(values).reshape(-1)]
    return result[0] if components == 1 else result


def _native_radii(values: np.ndarray, components: int) -> Any:
    """Round a positive bound outward, including subnormal support ranges."""
    result: list[float | None] = []
    for value in np.asarray(values).reshape(-1):
        rounded = float(value)
        # A positive mathematical radius must never underflow to fake zero
        # uncertainty. One outward step also protects the final conversion.
        rounded = math.nextafter(rounded, math.inf) if value > 0 else rounded
        result.append(rounded if math.isfinite(rounded) else None)
    return result[0] if components == 1 else result


def sequential_convergence_report(
    descriptors: Mapping[str, Any],
    targets: Mapping[str, Mapping[str, Any]],
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Report simultaneous, anytime-valid absolute precision of bounded means.

    ``targets`` is a nonempty predeclared mapping from name to ``bounds``
    (one finite lower/upper pair shared by that target's components), positive
    absolute ``tolerance``, and positive integer ``components`` (default one).
    ``descriptors`` has exactly the same keys and holds cumulative independent
    structure rows, either scalar sequences or n-by-components arrays. All
    targets must contain the same number of rows. Missing values, booleans,
    complex values, and support violations raise ValueError.

    This function is stateless: callers must persist the immutable target
    specification and the full cumulative sample prefix across checkpoints.
    It deliberately does not discover targets, grids, bounds, or multiplicity
    from observed data. Undefined or unrepresentable values are JSON null and
    cannot cause a target to pass. With fewer than two rows, no target passes.
    """
    confidence = _finite_real(confidence, "confidence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("targets must be a nonempty mapping")
    if not isinstance(descriptors, Mapping):
        raise ValueError("descriptors must be a mapping")
    if any(not isinstance(name, str) or not name for name in targets):
        raise ValueError("target names must be nonempty strings")
    if set(descriptors) != set(targets):
        raise ValueError("observations must contain exactly the predeclared target names")
    configs = {name: _target_config(name, targets[name]) for name in sorted(targets)}
    matrices = {name: _observations(descriptors[name], name, configs[name]) for name in configs}
    lengths = {len(matrix) for matrix in matrices.values()}
    if len(lengths) != 1:
        raise ValueError("all targets must have the same number of structure rows")
    n = lengths.pop()
    components_total = sum(config["components"] for config in configs.values())
    alpha = 1.0 - confidence
    # Work in logs to avoid underflow in the error allocation for large n/K.
    log_factor = (
        math.log(4.0) - math.log(alpha) + math.log(components_total)
        + math.log(n) + math.log(n - 1)
        if n >= 2 else None
    )
    alpha_component = alpha / components_total / n / (n - 1) if n >= 2 else None
    output: dict[str, Any] = {}
    for name, config in configs.items():
        components = config["components"]
        lower, upper = config["bounds"]
        width = np.longdouble(upper) - np.longdouble(lower)
        normalized = (matrices[name].astype(np.longdouble) - np.longdouble(lower)) / width
        normalized_mean = (
            np.mean(normalized, axis=0)
            if n else np.full(components, np.nan, dtype=np.longdouble)
        )
        mean = np.longdouble(lower) + width * normalized_mean
        reported_mean = mean.astype(float)
        # JSON stores binary64 centers. For narrow support far from zero,
        # rounding the mean can move that center farther than the statistical
        # radius. Compute the displacement in support-relative coordinates,
        # avoiding cancellation in a large-offset mean, and enlarge the bound.
        mean_rounding = np.abs(
            reported_mean.astype(np.longdouble) - np.longdouble(lower)
            - width * normalized_mean
        )
        std = np.full(components, np.nan, dtype=np.longdouble)
        radius = np.full(components, np.inf, dtype=np.longdouble)
        if n >= 2:
            normalized_std = np.std(normalized, ddof=1, axis=0)
            # A finite constant column has exactly zero sample variance.
            # Reduction roundoff can otherwise manufacture a tiny nonzero
            # variance when its normalized decimal value is not exact.
            constant = np.all(matrices[name] == matrices[name][0], axis=0)
            normalized_std[constant] = 0.0
            std = width * normalized_std
            # The two tails each receive alpha_component/2, hence log(4/delta).
            # Sample variance is unbiased (ddof=1) and must NOT be capped at .25.
            radius = width * (
                np.sqrt(2.0 * normalized_std**2 * log_factor / n)
                + 7.0 * log_factor / (3.0 * (n - 1))
            )
            radius += mean_rounding
        native_radius = _native_radii(radius, components)
        radii = [native_radius] if components == 1 else native_radius
        # Comparing JSON-native widths prevents an overflowing bound from
        # accidentally being interpreted as finite by another consumer.
        passed = bool(n >= 2 and all(value is not None and value <= config["tolerance"] for value in radii))
        output[name] = {
            **config,
            "n_structures": n,
            "mean": _native(reported_mean, components),
            "std": _native(std, components),
            "ci_halfwidth": native_radius,
            "mean_rounding_allowance": _native(mean_rounding, components),
            "max_ci_halfwidth": max(radii) if all(value is not None for value in radii) else None,
            "passed": passed,
            "status": "met" if passed else "insufficient_data" if n < 2 else "not_met",
            "zero_observed_variance": bool(n >= 2 and np.any(std == 0)),
        }
    converged = all(result["passed"] for result in output.values())
    return {
        "schema": "amorphgen.sequential_precision.v1",
        "method": _METHOD,
        "confidence": confidence,
        "n_structures": n,
        "sampling_unit": "structure",
        "n_components": components_total,
        "sequentially_valid": True,
        "optional_stopping_coverage_guaranteed": True,
        "converged": converged,
        "passed": converged,
        "status": "met" if converged else "insufficient_data" if n < 2 else "not_met",
        "targets": output,
        "alpha_spending": {
            "index": "cumulative_structure_count",
            "schedule": "alpha / (K * n * (n - 1)), n >= 2",
            "alpha_family": alpha,
            "alpha_per_component_at_n": alpha_component,
            "alpha_spent_upper_bound": alpha * (1.0 - 1.0 / n) if n >= 2 else 0.0,
            "alpha_remaining_lower_bound": alpha / n if n >= 2 else alpha,
            "total_spending_bound": alpha,
        },
        "inference_contract": {
            "interval_method": _METHOD,
            "sequentially_valid": True,
            "optional_stopping_coverage_guaranteed": True,
            "coverage_scope": "simultaneous_over_declared_components_and_all_cumulative_sample_sizes",
            "coverage_conditional_on_assumptions": True,
            "interpretation": (
                "Under the stated independent-structure and predeclared-support assumptions, "
                "every declared population mean lies within its reported radius of its "
                "sample mean simultaneously at all sample sizes with probability at least "
                "confidence. Arbitrary batch looks and stopping preserve this guarantee."
            ),
            "reference": "Maurer and Pontil (2009), Theorem 11; two-sided union bound, then alpha spending",
            "reference_url": _SOURCE,
        },
        "assumptions": list(_ASSUMPTIONS),
    }
