"""Strict tabular exports and publication-style scattering comparisons.

Bands describe pointwise Student-t confidence intervals of an ensemble mean,
with structures as sampling units. Missing model support remains a gap in
both curves and bands; it is never drawn as an interpolated connection.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


_SUMMARY_FIELDS = ("std", "sem", "ci_low", "ci_high", "bootstrap_low", "bootstrap_high")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _number(value):
    return "unavailable" if value is None else f"{value:.6g}"


def _array(values, name, size=None):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or (size is not None and len(array) != size):
        raise ValueError(f"{name} must be a one-dimensional array of matching length")
    return array


def _summary_arrays(result, size):
    summary = result.get("uncertainty", {})
    return {
        name: _array(summary.get(name, [None] * size), name, size)
        for name in ("n_per_point", *_SUMMARY_FIELDS)
    }


def _break_at_gaps(x, indices, *curves, break_before=None):
    """Insert separators where excluded experimental points left grid holes."""
    gaps = np.flatnonzero(np.diff(indices) > 1) + 1
    if break_before is not None:
        gaps = np.union1d(gaps, np.flatnonzero(break_before)).astype(int)
        gaps = gaps[gaps > 0]
    return tuple(np.insert(values, gaps, np.nan) for values in (x, *curves))


def _band(ax, x, low, high, confidence, color):
    finite = np.isfinite(x) & np.isfinite(low) & np.isfinite(high)
    if finite.any():
        ax.fill_between(x, low, high, where=finite, color=color, alpha=0.2,
                        linewidth=0,
                        label=f"{100 * confidence:g}% pointwise t CI of mean")
        # Sparse reciprocal shells can leave isolated supported bins. A fill
        # has zero width there, so retain their intervals as vertical bars.
        isolated = finite & ~np.r_[False, finite[:-1]] & ~np.r_[finite[1:], False]
        if isolated.any():
            ax.vlines(x[isolated], low[isolated], high[isolated], color=color,
                      alpha=.45, linewidth=.9)


def _interval_note(summary):
    if (np.isfinite(summary["ci_low"]) & np.isfinite(summary["ci_high"])).any():
        return "Bands: pointwise t confidence intervals of the structure mean (n ≥ 2)."
    return "Confidence intervals unavailable: fewer than two structures contribute at each point."


def _paths(output_dir, prefix, descriptor, report=False, save_pdf=False):
    base = Path(output_dir) / (f"{prefix}_{descriptor}" if prefix else descriptor)
    extensions = {"json": "json", "csv": "csv", "png": "png"}
    if report:
        extensions["report"] = "txt"
    if save_pdf:
        extensions["pdf"] = "pdf"
    return base, {key: f"{base}.{suffix}" for key, suffix in extensions.items()}


def _cell(value):
    """Keep unavailable statistics as empty CSV cells, matching JSON null."""
    return None if value is None or not np.isfinite(value) else value


def save_experiment_comparison(result, output_dir=".", prefix="analysis",
                               dpi=300, save_pdf=False):
    """Save an S(q) or T(r) comparison as JSON, CSV, text and PNG/PDF.

    ``result`` is returned by ``compare_experiment``. CSV rows share the exact
    support used for the fit statistics and include source indices, counts,
    spread and confidence bounds. The plot separates measurement uncertainties
    from the ensemble mean confidence interval, with a residual panel below.
    Returns paths under keys ``json``, ``csv``, ``report``, ``png`` and (when
    requested) ``pdf``. Figures remain outside pyplot's global registry.
    """
    # Reject nonstandard JSON numbers before creating any partial exports.
    serialized = json.dumps(result, indent=2, allow_nan=False, default=_json_default)
    from .experiment import format_experiment_report

    if result["kind"] not in ("sq", "tr"):
        raise ValueError("comparison kind must be 'sq' or 'tr'")
    report = format_experiment_report(result) + "\n"
    x = _array(result["x"], "x")
    size = len(x)
    observed = _array(result["observed"], "observed", size)
    calculated = _array(result["calculated"], "calculated", size)
    residual = _array(result["residual"], "residual", size)
    sigma = (None if result.get("sigma") is None
             else _array(result["sigma"], "sigma", size))
    summary = _summary_arrays(result, size)
    indices = _array(result.get("included_indices", range(size)), "included_indices", size)
    break_before = _array(result.get("plot_break_before", [False] * size),
                          "plot_break_before", size).astype(bool)
    break_before[1:] |= np.diff(indices) > 1
    confidence = result["uncertainty"]["confidence"]
    kind = result["kind"]
    base, paths = _paths(output_dir, prefix, f"experiment_{kind}",
                         report=True, save_pdf=save_pdf)
    base.parent.mkdir(parents=True, exist_ok=True)
    Path(paths["json"]).write_text(serialized + "\n")
    Path(paths["report"]).write_text(report)
    coordinate = "q_A^-1" if kind == "sq" else "r_A"
    with Path(paths["csv"]).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["input_index", coordinate, "observed", "sigma", "calculated",
                         "residual", "n_structures", *_SUMMARY_FIELDS, "confidence",
                         "plot_break_before"])
        for i in range(size):
            writer.writerow([int(indices[i]), x[i], observed[i],
                             None if sigma is None else _cell(sigma[i]), calculated[i],
                             residual[i], _cell(summary["n_per_point"][i]),
                             *(_cell(summary[field][i]) for field in _SUMMARY_FIELDS),
                             confidence, bool(break_before[i])])
    _plot_experiment(result, x, indices, observed, sigma, calculated, residual,
                     summary, break_before, base, dpi, save_pdf)
    return paths


def _plot_experiment(result, x, indices, observed, sigma, calculated, residual,
                     summary, break_before, base, dpi, save_pdf):
    from .plotting import _apply_pub_style, _figure, _save_fig, _EXP_COLOR, _PALETTE

    fig, (ax, residual_ax) = _figure(2, 1, figsize=(7, 5.7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    gx, mean, diff, low, high = _break_at_gaps(
        x, indices, calculated, residual, summary["ci_low"], summary["ci_high"],
        break_before=break_before)
    ax.plot(gx, mean, color=_PALETTE[0], lw=1.5, marker=".", ms=3,
            label="Ensemble mean")
    _band(ax, gx, low, high, result["uncertainty"]["confidence"], _PALETTE[0])
    ax.errorbar(x, observed, yerr=sigma, color=_EXP_COLOR, fmt="o", ms=2.8,
                elinewidth=.7, capsize=1.5,
                label="Measured (±1σ)" if sigma is not None else "Measured")
    residual_ax.axhline(0, color=".5", ls=":", lw=.8)
    residual_ax.plot(gx, diff, color=_PALETTE[0], lw=1.2, marker=".", ms=3)
    kind = result["kind"]
    ax.set_ylabel("S(q)" if kind == "sq" else "T(r) (Å⁻²)")
    residual_ax.set_xlabel("q (Å⁻¹)" if kind == "sq" else "r (Å)")
    residual_ax.set_ylabel("Calc. − obs." if kind == "sq" else "Calc. − obs.\n(Å⁻²)")
    for axis in (ax, residual_ax):
        _apply_pub_style(axis)
    ax.legend(loc="best", fontsize=8, frameon=False)
    metrics = result["metrics"]
    ax.set_title(f"RMSE = {_number(metrics['rmse'])}   Rw = {_number(metrics['rw'])}"
                 f"   N = {metrics['n_points']}", fontsize=10)
    note = _interval_note(summary) + "\nGaps indicate excluded or unavailable data."
    fig.text(.12, .015, note, fontsize=7.5, color=".3")
    fig.tight_layout(rect=(0, .065, 1, 1), h_pad=.4)
    _save_fig(fig, str(base), dpi=dpi, save_pdf=save_pdf)


def save_xrd_pattern(result, output_dir=".", prefix="analysis", dpi=300,
                     save_pdf=False):
    """Save coherent X-ray intensity vs 2θ as strict JSON, CSV and PNG/PDF.

    ``result`` is returned by ``xrd_pattern``. CSV includes q (Å⁻¹), 2θ
    (degrees), coherent intensity per atom (electron²), the number of
    contributing structures, and pointwise uncertainty. Missing reciprocal
    shells remain empty in the table and gaps in the plot. Return a mapping
    of ``json``, ``csv``, ``png`` and optional ``pdf`` paths.
    """
    serialized = json.dumps(result, indent=2, allow_nan=False, default=_json_default)
    q = _array(result["q"], "q")
    size = len(q)
    angle = _array(result["two_theta"], "two_theta", size)
    intensity = _array(result["intensity"], "intensity", size)
    summary = _summary_arrays(result, size)
    raw = (None if "intensity_raw" not in result
           else _array(result["intensity_raw"], "intensity_raw", size))
    counts = (None if "n_per_bin" not in result
              else _array(result["n_per_bin"], "n_per_bin", size))
    confidence = result["uncertainty"]["confidence"]
    base, paths = _paths(output_dir, prefix, "xrd", save_pdf=save_pdf)
    base.parent.mkdir(parents=True, exist_ok=True)
    Path(paths["json"]).write_text(serialized + "\n")
    with Path(paths["csv"]).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["q_A^-1", "two_theta_deg", "intensity_electron^2_per_atom",
                         "n_structures", *_SUMMARY_FIELDS, "confidence"]
                        + (["intensity_raw_electron^2_per_atom"] if raw is not None else [])
                        + (["n_wavevectors"] if counts is not None else []))
        for i in range(size):
            writer.writerow([q[i], angle[i], _cell(intensity[i]),
                             _cell(summary["n_per_point"][i]),
                             *(_cell(summary[field][i]) for field in _SUMMARY_FIELDS),
                             confidence]
                            + ([_cell(raw[i])] if raw is not None else [])
                            + ([_cell(counts[i])] if counts is not None else []))
    _plot_xrd(result, angle, intensity, summary, base, dpi, save_pdf)
    return paths


def _plot_xrd(result, angle, intensity, summary, base, dpi, save_pdf):
    from .plotting import _apply_pub_style, _figure, _save_fig, _PALETTE

    fig, ax = _figure(figsize=(7, 4.6))
    ax.plot(angle, intensity, lw=1.5, color=_PALETTE[0], marker=".", ms=3,
            label="Ensemble mean")
    _band(ax, angle, summary["ci_low"], summary["ci_high"],
          result["uncertainty"]["confidence"], _PALETTE[0])
    ax.set_xlabel("2θ (degrees)")
    ax.set_ylabel("Coherent intensity per atom (electron²)")
    _apply_pub_style(ax)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"λ = {result['wavelength']:g} Å", fontsize=10)
    fig.text(.12, .025, _interval_note(summary), fontsize=8, color=".3")
    fig.tight_layout(rect=(0, .06, 1, 1))
    _save_fig(fig, str(base), dpi=dpi, save_pdf=save_pdf)
