"""Order-independent precision planning for means of independent structures.

These are variance-based planning curves, not a history of generated prefixes.
For complete data, each component's planning curve is the root-mean-square
Student-t interval half-width over *all* subsets for sizes 2 through the observed
ensemble size: every subset sample variance is unbiased for the full ensemble
sample variance. Vector
descriptors take the maximum of these componentwise RMS half-widths (not the
RMS of subset maxima). Larger sizes extrapolate the same variance model.
Missing data use a separately labelled availability-rate approximation.
No finite-population correction is applied:
the target is the generating distribution's mean, not the finite sample mean.
These fixed-sample planning intervals do not support adaptive stopping; use
``sequential_convergence_report`` with predeclared bounded targets for that.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

import numpy as np
from scipy.stats import t


_ASSUMPTIONS = [
    "Structures are independent draws from the same generating distribution; "
    "correlated trajectory frames are not independent structures.",
    "Uncertainty is the two-sided Student-t interval half-width for an "
    "equal-weight mean of available structures; normal-mean coverage is "
    "approximate for non-normal data.",
    "Planning holds the full ensemble sample variance fixed. For complete "
    "data, each component equals the RMS half-width over all subsets for sizes "
    "2 through the observed ensemble size; larger sizes extrapolate the same "
    "variance model. Vector descriptors use the maximum componentwise RMS, not the RMS "
    "of subset maxima or the half-width of a generation-order prefix.",
    "Missing values use floor(planned ensemble size * observed contributing "
    "count / observed ensemble size) per component. This availability-rate "
    "planning approximation assumes representative missingness and is not "
    "an exact all-subset RMS result.",
    "All curve components must satisfy the tolerance. Intervals are pointwise, "
    "not simultaneous confidence bands, and use no finite-population correction.",
    "Forecasts are conditional planning estimates, not guarantees; small "
    "ensembles, rare structures, or changing variance can invalidate them.",
    "Zero observed variance yields zero estimated half-width with at least "
    "two observations; it does not establish absence of population variability.",
]


def _positive_integer(value, name):
    if (not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_)) or value < 1):
        raise ValueError(f"{name} must be a positive integer")
    # Scipy's degrees of freedom and the plotting axis must be representable.
    if value > np.iinfo(np.int64).max:
        raise ValueError(f"{name} is too large")
    return int(value)


def _finite_real(value, message):
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(result):
        raise ValueError(message)
    return result


def _native(value):
    array = np.asarray(value)
    if array.ndim:
        return [_native(item) for item in array]
    value = float(array)
    return value if np.isfinite(value) else None


def _matrix(values, name):
    if isinstance(values, Mapping):
        if "per_structure" not in values:
            raise ValueError(f"descriptor {name!r} must contain per_structure")
        values = values["per_structure"]
    try:
        rows = list(values)
        if any(np.iscomplexobj(row) for row in rows if row is not None):
            raise ValueError("descriptor observations must be real numbers")
        shape = next((np.asarray(row).shape for row in rows
                      if row is not None and np.asarray(row).ndim), None)
        if shape is not None:
            if len(shape) != 1:
                raise ValueError("expected scalars or a structure-by-bin matrix")
            rows = [np.full(shape, np.nan) if row is None else row
                    for row in rows]
        array = np.asarray(rows, dtype=float)
        if not rows and isinstance(values, np.ndarray) and values.ndim == 2:
            array = np.empty(values.shape, dtype=float)
        if array.ndim not in (1, 2):
            raise ValueError("expected scalars or a structure-by-bin matrix")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid values for descriptor {name!r}: {exc}") from exc
    scalar = array.ndim == 1
    matrix = array[:, None] if scalar else array
    # Copy before canonicalization, including normalizing signed zeros. Sorting
    # entire rows preserves component alignment and makes reductions bitwise
    # independent of input generation order (not just statistically equivalent).
    matrix = np.where(np.isfinite(matrix), matrix, np.nan)
    matrix[matrix == 0] = 0.0
    if len(matrix) and matrix.shape[1]:
        keys = tuple(matrix[:, column]
                     for column in range(matrix.shape[1] - 1, -1, -1))
        matrix = matrix[np.lexsort(keys)]
    return matrix, scalar


def _statistics(matrix, name):
    counts = np.isfinite(matrix).sum(axis=0)
    means = np.full(matrix.shape[1], np.nan)
    stds = np.full(matrix.shape[1], np.nan)
    for column, count in enumerate(counts):
        if count:
            values = matrix[np.isfinite(matrix[:, column]), column].astype(np.longdouble)
            mean = np.mean(values)
            std = np.std(values, ddof=1) if count > 1 else np.longdouble(np.nan)
            if (abs(mean) > np.finfo(float).max
                    or (count > 1 and (not np.isfinite(std)
                                      or std > np.finfo(float).max))):
                raise ValueError(f"descriptor {name!r} has values too large for finite statistics")
            means[column] = float(mean)
            stds[column] = float(std)
    return counts, means, stds


def _half_widths(stds, counts, confidence):
    result = np.full(len(stds), np.nan)
    usable = (counts > 1) & np.isfinite(stds)
    # isf avoids rounding (1 + confidence) / 2 to one at high confidence.
    with np.errstate(over="ignore", invalid="ignore"):
        result[usable] = (t.isf((1 - confidence) / 2, counts[usable] - 1)
                          * (stds[usable] / np.sqrt(counts[usable])))
    return result


def _maximum_half_width(widths):
    return float(np.max(widths)) if len(widths) and np.all(np.isfinite(widths)) else None


def _planned_counts(counts, n_total, size):
    if not n_total:
        return np.zeros(len(counts), dtype=np.int64)
    # Python integer arithmetic avoids multiplication overflow and rounding at
    # exact availability boundaries, including the observed ensemble endpoint.
    return np.array([int(size) * int(count) // n_total for count in counts], dtype=np.int64)


def _planning_curve(std, n_per_point, n_total, sizes, confidence):
    """Evaluate the report's planning rule; shared with report plotting.

    ``std`` and ``n_per_point`` are either scalars or corresponding lists.
    Undefined standard deviations remain undefined at every planned size.
    """
    scalar = np.asarray(std).ndim == 0
    stds = np.atleast_1d(np.asarray(std, dtype=float))
    counts = np.atleast_1d(np.asarray(n_per_point, dtype=np.int64))
    result = {"sizes": [], "half_width": [], "half_width_per_point": [],
              "effective_counts": []}
    for size in sizes:
        effective = _planned_counts(counts, n_total, size)
        widths = _half_widths(stds, effective, confidence)
        result["sizes"].append(int(size))
        result["half_width"].append(_maximum_half_width(widths))
        result["half_width_per_point"].append(_native(widths[0] if scalar else widths))
        result["effective_counts"].append(int(effective[0]) if scalar else effective.tolist())
    return result


def _estimate(stds, counts, n_total, tolerance, confidence, max_structures, status):
    if status in ("undeclared", "insufficient_data"):
        return None, None, status
    if status == "met":
        return n_total, 0, "met"

    def meets(size):
        widths = _half_widths(stds, _planned_counts(counts, n_total, size), confidence)
        width = _maximum_half_width(widths)
        return width is not None and width <= tolerance

    if n_total >= max_structures or not meets(max_structures):
        return None, None, "exceeds_max_structures"
    # The t half-width decreases with contributing count. Integer bisection
    # finds the smallest additional ensemble size satisfying every component.
    low, high = n_total, max_structures
    while high - low > 1:
        middle = (low + high) // 2
        if meets(middle):
            high = middle
        else:
            low = middle
    return high, high - n_total, "estimated"


def convergence_report(descriptors, tolerances=None, confidence=0.95, sizes=None,
                       max_structures=1_000_000) -> dict:
    """Report absolute precision tolerances and ensemble-size planning curves.

    ``descriptors`` maps names to per-structure scalars, aligned structure-by-
    component matrices, or uncertainty summaries containing ``per_structure``.
    Every descriptor must have the same number of structure rows. ``None``,
    NaN and infinity are missing observations. Whole missing curve rows may be
    ``None``. Empty curve components and fewer than two finite observations in
    any component cannot satisfy a tolerance or produce a sample-size forecast.

    ``tolerances`` maps any subset of descriptor names to positive, finite,
    absolute CI half-widths in that descriptor's own units. Vector descriptors
    pass only when *every* pointwise half-width meets their tolerance; this is
    not a simultaneous-coverage claim. Undeclared descriptors are reported but
    do not participate in the overall tolerance status. Omit ``tolerances``
    for an exploratory report with no declared stopping criterion.

    Curves use the full ensemble sample variance, canonically sorted to ensure
    exact invariance to row permutations. With complete data each component
    equals the RMS Student-t half-width over all subsets of size n, for
    2 <= n <= observed ensemble size; larger sizes extrapolate the same model.
    Vector descriptors use the maximum componentwise RMS. With missing data,
    contributing counts are floor(n * observed_count / observed_total), a
    plug-in availability approximation. Forecasts keep both the
    observed variance and availability fixed and find the smallest total size
    at least as large as the current ensemble satisfying all declared bounds.

    ``sizes`` optionally specifies positive integer planning sizes (including
    future sizes); duplicates are removed, sizes sorted, and the observed
    endpoint always included. By default, at most 64 observed sizes are used.
    ``max_structures`` caps sample-size forecasts, not the observed ensemble.
    Results contain only JSON-native values, with undefined numbers as null.
    This planning report is explicitly not sequentially valid: repeated looks
    do not preserve its nominal confidence coverage.
    """
    confidence_message = "confidence must be a finite number between 0 and 1"
    confidence = _finite_real(confidence, confidence_message)
    if not 0 < confidence < 1:
        raise ValueError(confidence_message)
    max_structures = _positive_integer(max_structures, "max_structures")
    if tolerances is None:
        tolerances = {}
    if not isinstance(descriptors, Mapping) or not isinstance(tolerances, Mapping):
        raise ValueError("descriptors and tolerances must be mappings")
    if any(not isinstance(name, str) or not name for name in descriptors):
        raise ValueError("descriptor names must be nonempty strings")
    if any(not isinstance(name, str) or not name for name in tolerances):
        raise ValueError("tolerance names must be nonempty strings")
    unknown = set(tolerances) - set(descriptors)
    if unknown:
        raise ValueError(f"unknown descriptor tolerance(s): {', '.join(sorted(unknown))}; "
                         f"available descriptors: {', '.join(sorted(descriptors)) or '(none)'}")
    targets = {}
    for name, value in tolerances.items():
        message = f"tolerance for {name!r} must be positive and finite"
        value = _finite_real(value, message)
        if value <= 0:
            raise ValueError(message)
        targets[name] = value
    matrices = {name: _matrix(values, name) for name, values in descriptors.items()}
    lengths = {len(matrix) for matrix, _ in matrices.values()}
    if len(lengths) > 1:
        raise ValueError("all descriptors must have the same number of structure rows")
    n_total = next(iter(lengths), 0)
    if sizes is None:
        curve_sizes = (list(range(1, n_total + 1)) if n_total <= 64 else
                       np.unique(np.geomspace(1, n_total, 64).astype(np.int64)).tolist())
    else:
        try:
            curve_sizes = sorted({_positive_integer(size, "sizes") for size in sizes})
        except TypeError as exc:
            raise ValueError("sizes must be a sequence of positive integers") from exc
    if n_total:
        curve_sizes = sorted(set(curve_sizes) | {n_total})
    output = {}
    for name in sorted(matrices):
        matrix, scalar = matrices[name]
        source = descriptors[name]
        units = source.get("units") if isinstance(source, Mapping) else None
        if units is not None and not isinstance(units, str):
            raise ValueError(f"units for {name!r} must be a string or None")
        counts, means, stds = _statistics(matrix, name)
        widths = _half_widths(stds, counts, confidence)
        half_width = _maximum_half_width(widths)
        tolerance = targets.get(name)
        status = ("undeclared" if tolerance is None else
                  "insufficient_data" if half_width is None else
                  "met" if half_width <= tolerance else "not_met")
        total, additional, estimate_status = _estimate(
            stds, counts, n_total, tolerance, confidence, max_structures, status)

        def native(values):
            return _native(values[0] if scalar else values)

        native_std = native(stds)
        native_counts = int(counts[0]) if scalar else counts.tolist()
        output[name] = {
            "tolerance": tolerance, "status": status, "units": units,
            "n_total_structures": n_total,
            "n_structures": int(np.isfinite(matrix).any(axis=1).sum()),
            "n_per_point": native_counts, "mean": native(means),
            "std": native_std, "half_width": half_width,
            "half_width_per_point": native(widths),
            "zero_variance": bool(len(stds) and np.any(stds == 0)),
            "estimated_total_structures": total,
            "estimated_additional_structures": additional,
            "estimate_status": estimate_status,
            "curve": _planning_curve(native_std, native_counts, n_total,
                                     curve_sizes, confidence),
            "curve_method": ("all_subset_rms_student_t" if np.all(counts == n_total)
                             else "availability_adjusted_student_t_planning"),
        }
    declared = [output[name] for name in targets]
    status = ("undeclared" if not declared else
              "insufficient_data" if any(item["status"] == "insufficient_data" for item in declared) else
              "met" if all(item["status"] == "met" for item in declared) else "not_met")
    if status in ("undeclared", "insufficient_data"):
        total, additional, estimate_status = None, None, status
    elif any(item["estimate_status"] == "exceeds_max_structures" for item in declared):
        total, additional, estimate_status = None, None, "exceeds_max_structures"
    else:
        total = max(item["estimated_total_structures"] for item in declared)
        additional = total - n_total
        estimate_status = "met" if status == "met" else "estimated"
    return {
        "confidence": confidence, "method": "all_subset_rms_student_t",
        "sampling_unit": "structure", "band_type": "pointwise",
        "sequentially_valid": False,
        "optional_stopping_coverage_guaranteed": False,
        "tolerance_type": "absolute", "n_structures": n_total,
        "max_structures": max_structures, "status": status,
        "estimated_total_structures": total,
        "estimated_additional_structures": additional,
        "estimate_status": estimate_status, "descriptors": output,
        "assumptions": list(_ASSUMPTIONS),
    }
