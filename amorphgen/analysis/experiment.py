"""Measured scattering curves and comparisons on their native sampling grid.

Experimental standard deviations describe measurement error; confidence bands
describe uncertainty of the mean across independent simulated structures. They
are deliberately kept separate. No scale, offset, background, or resolution
parameters are fitted by these routines.
"""

from __future__ import annotations

from collections.abc import Mapping
from io import StringIO
from pathlib import Path

import numpy as np

from amorphgen.analysis.uncertainty import summarize_structures


def _kind(value):
    aliases = {"sq": "sq", "s(q)": "sq", "tr": "tr", "t(r)": "tr"}
    try:
        return aliases[str(value).strip().lower()]
    except KeyError:
        raise ValueError("kind must be 'sq' (S(q)) or 'tr' (T(r))") from None


def _units(kind):
    return {"x": "1/angstrom" if kind == "sq" else "angstrom",
            "observed": "dimensionless" if kind == "sq" else "1/angstrom^2"}


def _vector(values, name):
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain numeric values") from None
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    return result


def _axis(values, name):
    result = _vector(values, name)
    if not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be nonempty and finite")
    order = np.argsort(result, kind="stable")
    if np.any(np.diff(result[order]) == 0):
        raise ValueError(f"{name} must not contain duplicate coordinates")
    return result[order], order


def _measurement(experiment):
    if not isinstance(experiment, Mapping):
        raise ValueError("experiment must be a mapping returned by load_experiment")
    if "x" not in experiment or "observed" not in experiment:
        raise ValueError("experiment must contain x and observed arrays")
    x, order = _axis(experiment["x"], "experimental x")
    y = _vector(experiment["observed"], "experimental observations")
    if len(y) != len(x) or not np.isfinite(y).all():
        raise ValueError("experimental observations must match x and be finite")
    sigma = experiment.get("sigma")
    if sigma is not None:
        sigma = _vector(sigma, "experimental sigma")
        if (len(sigma) != len(x) or not np.isfinite(sigma).all()
                or np.any(sigma <= 0)):
            raise ValueError("experimental sigma must match x and be finite and positive")
        sigma = sigma[order]
    return x, y[order], sigma


def load_experiment(path, kind="sq", *, columns=None, delimiter=None,
                    skiprows=0, comments="#") -> dict:
    """Load a measured S(q) or T(r) curve from a numeric text or CSV file.

    Default input has exactly two columns (coordinate, observation), or three
    (coordinate, observation, positive one-sigma measurement uncertainty).
    ``columns=(x_column, y_column[, sigma_column])`` selects zero-based columns
    explicitly when other columns are present. ``delimiter=None`` detects a
    comma in the first data line, otherwise whitespace is used. Headers must
    be commented or skipped explicitly using ``skiprows``; malformed selected
    data are errors, never silently discarded. Blank/comment lines are ignored.

    Data are sorted by coordinate with observations and uncertainties kept
    together. Duplicate or nonfinite coordinates, nonfinite observations, and
    nonpositive/nonfinite uncertainties are rejected. No units, normalisation,
    density, or S(q)/T(r) convention conversion is performed: q is in inverse
    angstrom, r in angstrom, S(q) dimensionless, and T(r)=4*pi*r*rho*g(r) in
    inverse square angstrom. The return value is JSON-native.
    """
    kind = _kind(kind)
    if (not isinstance(skiprows, (int, np.integer))
            or isinstance(skiprows, (bool, np.bool_)) or skiprows < 0):
        raise ValueError("skiprows must be a non-negative integer")
    if columns is not None:
        try:
            columns = tuple(columns)
        except TypeError:
            raise ValueError("columns must select two or three distinct non-negative integers") from None
        if (len(columns) not in (2, 3) or any(
                not isinstance(c, (int, np.integer))
                or isinstance(c, (bool, np.bool_)) or c < 0 for c in columns)
                or len(set(columns)) != len(columns)):
            raise ValueError("columns must select two or three distinct non-negative integers")
    if comments is not None and (not isinstance(comments, str) or not comments):
        raise ValueError("comments must be a nonempty string or None")
    if delimiter is not None and (not isinstance(delimiter, str) or len(delimiter) != 1):
        raise ValueError("delimiter must be a single character or None")
    contents = Path(path).read_text(encoding="utf-8-sig")
    data_lines = contents.splitlines()[skiprows:]
    first = next((line.split(comments, 1)[0].strip() if comments else line.strip()
                  for line in data_lines
                  if (line.split(comments, 1)[0] if comments else line).strip()), None)
    if first is None:
        raise ValueError("experimental file contains no data")
    if delimiter is None and "," in first:
        delimiter = ","
    try:
        data = np.loadtxt(StringIO(contents), delimiter=delimiter,
                          skiprows=skiprows, comments=comments, usecols=columns,
                          ndmin=2)
    except ValueError as exc:
        raise ValueError(f"cannot read experimental data: {exc}") from exc
    if data.shape[1] not in (2, 3):
        raise ValueError("experimental data require two or three columns; use columns= to select them")
    x, y, sigma = _measurement({"x": data[:, 0], "observed": data[:, 1],
                               "sigma": data[:, 2] if data.shape[1] == 3 else None})
    return {"kind": kind, "x": x.tolist(), "observed": y.tolist(),
            "sigma": sigma.tolist() if sigma is not None else None,
            "source": str(path), "units": _units(kind),
            "columns": list(columns) if columns is not None else list(range(data.shape[1])),
            "delimiter": delimiter, "skiprows": int(skiprows), "comments": comments}


def _model(calculated, kind):
    if not isinstance(calculated, Mapping):
        raise ValueError("calculated must be a scattering result mapping")
    axis_name, value_name = ("q", "s_q") if kind == "sq" else ("r", "T_r")
    if axis_name not in calculated:
        raise ValueError(f"calculated {kind} result must contain {axis_name}")
    x, order = _axis(calculated[axis_name], "calculated x")
    curves = calculated.get("per_structure_curves", {}).get(value_name)
    if curves is None:
        curves = calculated.get("per_structure")
    if curves is None:
        curves = calculated.get("uncertainty", {}).get("per_structure")
    if curves is None:
        if value_name not in calculated:
            raise ValueError(f"calculated result must contain {value_name} or per_structure curves")
        # An old mean-only result is one curve, not an invented ensemble.
        curves = [calculated[value_name]]
    try:
        rows = [np.full(len(x), np.nan) if row is None else row for row in curves]
        matrix = np.asarray(rows, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("per_structure must be a rectangular structure-by-coordinate matrix") from None
    if matrix.ndim != 2 or matrix.shape[1] != len(x) or matrix.shape[0] == 0:
        raise ValueError("per_structure must be a nonempty structure-by-coordinate matrix matching x")
    return x, matrix[:, order]


def _interpolate_segments(source_x, source_y, target_x):
    """Interpolate only adjacent finite bins; missing bins split support."""
    result = np.full(len(target_x), np.nan)
    finite_indices = np.flatnonzero(np.isfinite(source_y))
    if not len(finite_indices):
        return result
    runs = np.split(finite_indices, np.flatnonzero(np.diff(finite_indices) != 1) + 1)
    for run in runs:
        inside = (target_x >= source_x[run[0]]) & (target_x <= source_x[run[-1]])
        if inside.any():
            result[inside] = np.interp(target_x[inside], source_x[run], source_y[run])
    return result


def compare_experiment(calculated, experiment, *, kind=None, x_range=None,
                       confidence=0.95, n_bootstrap=1000, seed=0) -> dict:
    """Compare an ensemble S(q) or T(r) result to measured observations.

    Interpolate *each structure* onto the measured coordinates, then recompute
    the equal-weight ensemble mean and pointwise Student t/bootstrap intervals.
    Interpolation never bridges missing bins or extrapolates. At each point,
    only structures with supported finite values contribute. Measurements with
    no model support, or outside optional inclusive ``x_range=(lo, hi)``, are
    excluded and counted. Indices refer to the full sorted measurement grid.
    ``plot_break_before`` marks breaks needed before a retained point, either
    because measured points were excluded or because the measured grid skips
    an unsupported interval of the native model grid entirely. Plotting lines
    and bands must respect these breaks.

    Residuals and bias are calculated minus observed. RMSE, MAE, and bias use
    equal point weights. Rw = sqrt(sum(w*residual**2)/sum(w*observed**2)), with
    w=1/sigma**2 if measured one-sigma uncertainties exist, otherwise w=1.
    Rw is a fraction, not a percentage, and is undefined for zero denominator.
    Its definition follows https://dictionary.iucr.org/R_factor.

    Chi-square = sum((residual/sigma)**2) is provided only with measurement
    sigma; reduced chi-square divides by N (no fitted parameters). These are
    descriptive diagonal-error metrics: transformed T(r) and sampled curves
    can have correlated errors. They do not establish statistical significance
    or a p-value. Ensemble SEM is never substituted for measurement sigma.
    All metrics share exactly the returned grid. No fit or unit conversion is
    performed; experimental and model conventions must already agree.
    """
    if not isinstance(experiment, Mapping):
        raise ValueError("experiment must be a mapping returned by load_experiment")
    kind = _kind(experiment.get("kind", "sq") if kind is None else kind)
    if "kind" in experiment and kind != _kind(experiment["kind"]):
        raise ValueError("kind does not match the experimental curve")
    x, observed, sigma = _measurement(experiment)
    model_x, rows = _model(calculated, kind)
    in_range = np.ones(len(x), dtype=bool)
    if x_range is not None:
        bounds = _vector(x_range, "x_range")
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError("x_range must contain finite lower and upper bounds in order")
        in_range = (x >= bounds[0]) & (x <= bounds[1])
    aligned = np.asarray([_interpolate_segments(model_x, row, x) for row in rows])
    available = np.isfinite(aligned).any(axis=0)
    included = in_range & available
    if not included.any():
        raise ValueError("no supported experimental points overlap the calculated curve and x_range")
    included_indices = np.flatnonzero(included)
    included_x = x[included]
    # Native grid points can each be supported without any structure spanning
    # the interval between them. Count unsupported elementary intervals, so
    # sparse measured coordinates cannot conceal a gap when plotted.
    finite = np.isfinite(rows)
    interval_support = np.any(finite[:, :-1] & finite[:, 1:], axis=0)
    missing_prefix = np.r_[0, np.cumsum(~interval_support)]
    first = np.searchsorted(model_x[1:], included_x[:-1], side="right")
    stop = np.searchsorted(model_x[:-1], included_x[1:], side="left")
    plot_break_before = np.r_[False, (np.diff(included_indices) > 1)
                              | (missing_prefix[stop] > missing_prefix[first])]
    summary = summarize_structures(aligned[:, included], confidence=confidence,
                                   n_bootstrap=n_bootstrap, seed=seed)
    mean = np.asarray(summary["mean"], dtype=float)
    obs = observed[included]
    measured_sigma = sigma[included] if sigma is not None else None
    residual = mean - obs
    count = int(included.sum())
    # Scaling weights by their largest value avoids needless overflow in Rw.
    weights = np.ones(count) if measured_sigma is None else (measured_sigma.min() / measured_sigma) ** 2
    denominator = float(np.sum(weights * obs ** 2))
    rw = float(np.sqrt(np.sum(weights * residual ** 2) / denominator)) if denominator > 0 else None
    chi_square = float(np.sum((residual / measured_sigma) ** 2)) if measured_sigma is not None else None
    metrics = {"n_points": count, "rmse": float(np.sqrt(np.mean(residual ** 2))),
               "mae": float(np.mean(np.abs(residual))), "bias": float(np.mean(residual)),
               "rw": rw, "chi_square": chi_square,
               "reduced_chi_square": chi_square / count if chi_square is not None else None,
               "degrees_of_freedom": count if measured_sigma is not None else None}
    outside_support = (x < model_x[0]) | (x > model_x[-1])
    overlap = {"n_input_points": len(x), "n_points": count,
               "n_excluded": int((~included).sum()),
               "n_excluded_x_range": int((~in_range).sum()),
               "n_excluded_outside_support": int((in_range & outside_support).sum()),
               "n_excluded_missing_support": int((in_range & ~outside_support & ~available).sum()),
               "x_min": float(x[included][0]), "x_max": float(x[included][-1]),
               "model_x_min": float(model_x[0]), "model_x_max": float(model_x[-1])}
    return {"kind": kind, "x": x[included].tolist(), "observed": obs.tolist(),
            "sigma": measured_sigma.tolist() if measured_sigma is not None else None,
            "calculated": summary["mean"], "residual": residual.tolist(),
            "per_structure": summary["per_structure"], "uncertainty": summary,
            "metrics": metrics, "overlap": overlap,
            "included_indices": included_indices.tolist(),
            "excluded_indices": np.flatnonzero(~included).tolist(),
            "plot_break_before": plot_break_before.tolist(),
            "units": _units(kind), "source": experiment.get("source"),
            "experiment_metadata": {key: experiment[key] for key in (
                "kind", "source", "units", "columns", "delimiter", "skiprows", "comments")
                if key in experiment},
            "x_range": np.asarray(x_range, dtype=float).tolist() if x_range is not None else None,
            "fitting": {"scale": 1.0, "offset": 0.0, "n_parameters": 0},
            "definitions": {
                "residual": "calculated minus observed",
                "rmse": "sqrt(mean(residual**2))",
                "mae": "mean(abs(residual))", "bias": "mean(residual)",
                "rw": "sqrt(sum(w*residual**2)/sum(w*observed**2)); fraction, not percent",
                "rw_weighting": "inverse measurement variance" if sigma is not None else "unit weights",
                "chi_square": "sum((residual/sigma)**2); measurement sigma only",
                "reduced_chi_square": "chi_square/N; no fitted parameters",
                "chi_square_assumption": "diagonal measurement errors; descriptive for correlated points",
                "ensemble_bands": "pointwise confidence intervals of the mean across independent structures",
            }}


def format_experiment_report(result) -> str:
    """Return a concise readable comparison report, including exclusions."""
    label = "S(q)" if result["kind"] == "sq" else "T(r)"
    metrics, overlap = result["metrics"], result["overlap"]
    stats = result["uncertainty"]
    number = lambda value: f"{value:.6g}" if value is not None else "undefined"
    lines = [f"Experimental {label} comparison",
             f"  Points: {overlap['n_points']} / {overlap['n_input_points']}; "
             f"range {overlap['x_min']:.6g} to {overlap['x_max']:.6g} {result['units']['x']}",
             f"  Excluded: {overlap['n_excluded_x_range']} outside requested range; "
             f"{overlap['n_excluded_outside_support']} outside model grid; "
             f"{overlap['n_excluded_missing_support']} without finite model support",
             f"  RMSE: {number(metrics['rmse'])}; MAE: {number(metrics['mae'])}; "
             f"bias (calculated-observed): {number(metrics['bias'])}",
             f"  Rw: {number(metrics['rw'])} (fraction; {result['definitions']['rw_weighting']})"]
    if metrics["chi_square"] is not None:
        lines.append(f"  Chi-square: {number(metrics['chi_square'])}; reduced chi-square: "
                     f"{number(metrics['reduced_chi_square'])}; dof: {metrics['degrees_of_freedom']}")
        lines.append("  Chi-square uses measurement sigma and diagonal errors; correlated points limit interpretation.")
    else:
        lines.append("  Chi-square: unavailable (no measurement sigma supplied)")
    counts = stats["n_per_point"]
    lines.extend([f"  Ensemble: {stats['n_structures']} structures; {min(counts)} to {max(counts)} per point; "
                  f"{100 * stats['confidence']:.6g}% pointwise Student-t confidence interval of the mean",
                  "  Each interval is pointwise, not a simultaneous band; undefined with fewer than two structures per point.",
                  "  All metrics use the same included points; residual = calculated minus observed.",
                  "  No scale or offset fitted."])
    return "\n".join(lines)
