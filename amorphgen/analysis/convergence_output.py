"""Text, tabular and figure exports for descriptor convergence reports."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from ._serialization import numpy_json_default as _json_default

def _number(value):
    return "unavailable" if value is None else f"{value:.6g}"


def _components(value):
    """Treat scalar and pointwise summaries consistently without dropping nulls."""
    return np.asarray(value, dtype=object).reshape(-1).tolist()


def _counts(value):
    values = _components(value)
    if not values:
        return "no components"
    if len(values) == 1:
        return str(values[0])
    return f"{min(values)}–{max(values)} per point"


def _estimate_text(row, maximum):
    status = row["estimate_status"]
    if status in ("estimated", "met"):
        return (f"estimated total={row['estimated_total_structures']}, "
                f"additional={row['estimated_additional_structures']}")
    if status == "exceeds_max_structures":
        return f"estimated total exceeds search limit {maximum}; additional unavailable"
    if status == "undeclared":
        return "estimate unavailable (no declared tolerance)"
    return "estimate unavailable (insufficient data)"


def format_convergence_report(report):
    """Return declared tolerances, observed uncertainty and sampling forecasts.

    Uncertainty is a Student-t interval half-width for the ensemble mean.
    Curve-valued descriptors use their largest pointwise half-width; this is
    not a simultaneous confidence band. Forecasts are conditional on the
    observed variance and availability remaining representative.
    """
    confidence = 100 * report["confidence"]
    maximum = report["max_structures"]
    lines = [
        "Descriptor convergence",
        f"Structures: {report['n_structures']}; confidence: {confidence:g}%; "
        "tolerances: absolute mean CI half-width in descriptor units.",
        f"Overall status: {report['status']}; {_estimate_text(report, maximum)}.",
        "Planning curves are independent of generation order.",
    ]
    for name, row in report["descriptors"].items():
        tolerance = ("undeclared" if row["tolerance"] is None
                     else _number(row["tolerance"]))
        units = f" {row['units']}" if row.get("units") else ""
        lines.append(
            f"{name}: tolerance={tolerance}{units}; {confidence:g}% t half-width="
            f"{_number(row['half_width'])}{units}; status={row['status']}; "
            f"available structures={row['n_structures']}/{row['n_total_structures']} "
            f"(n={_counts(row['n_per_point'])}); "
            f"{_estimate_text(row, maximum)}."
        )
    for assumption in report.get("assumptions", []):
        lines.append(f"Assumption: {assumption}")
    return "\n".join(lines) + "\n"


def _projection(report, row):
    """Use the same planning calculation as the numerical report."""
    from .convergence import _planning_curve

    current = row["n_total_structures"]
    target = row["estimated_total_structures"]
    if target is None or target <= current:
        return None
    sizes = np.unique(np.rint(np.geomspace(max(current, 1), target, 60)).astype(int))
    return _planning_curve(row["std"], row["n_per_point"], current,
                           sizes.tolist(), report["confidence"])


def _plot_descriptor(report, name, row, projection, base, dpi, save_pdf):
    from .plotting import _apply_pub_style, _figure, _save_fig, _PALETTE

    fig, ax = _figure(figsize=(7, 4.7))
    current = row["n_total_structures"]
    curve = row["curve"]
    sizes = np.asarray(curve["sizes"], dtype=float)
    widths = np.asarray(curve["half_width"], dtype=float)
    observed = sizes <= current
    complete = all(count == current for count in _components(row["n_per_point"]))
    vector = len(_components(row["n_per_point"])) > 1
    label = (("Maximum componentwise RMS planning" if vector else "All-subset RMS planning")
             if complete else "Availability-adjusted planning")
    ax.plot(sizes[observed], widths[observed], color=_PALETTE[0], lw=1.8, label=label)
    # Explicit future sizes supplied through the API are also forecasts.
    future = sizes >= current
    if np.any(sizes > current):
        ax.plot(sizes[future], widths[future], color=_PALETTE[0], ls="--",
                lw=1.5, label="Projected uncertainty")
    if projection:
        ax.plot(projection["sizes"], projection["half_width"], color=_PALETTE[0],
                ls="--", lw=1.5,
                label=None if np.any(sizes > current) else "Projected uncertainty")
    if row["half_width"] is not None:
        ax.scatter([current], [row["half_width"]], color=_PALETTE[1], s=35,
                   zorder=4, label="Full-ensemble observed half-width")
    else:
        ax.text(.98, .72, "Observed uncertainty unavailable\n(fewer than 2 at some points)",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)
    if row["tolerance"] is not None:
        ax.axhline(row["tolerance"], color=_PALETTE[2], ls=":", lw=1.5,
                   label=f"Declared tolerance: {_number(row['tolerance'])}")
    target = row["estimated_total_structures"]
    if target is not None and target > current:
        ax.axvline(target, color=".45", ls=":", lw=1,
                   label=f"Estimated total: {target} (+{row['estimated_additional_structures']})")
    maximum_size = max([current, *curve["sizes"], target or current, 2])
    if maximum_size > 100:
        ax.set_xscale("log")
    ax.set_xlim(left=1, right=maximum_size * 1.05)
    ax.set_ylim(bottom=0)
    confidence = 100 * report["confidence"]
    prefix = "Largest pointwise " if vector else ""
    units = f" ({row['units']})" if row.get("units") else ""
    ax.set_ylabel(f"{prefix}{confidence:g}% mean CI half-width{units}")
    ax.set_xlabel("Ensemble size (structures)")
    ax.set_title(f"{name} · {row['status']}", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    _apply_pub_style(ax, label_fs=10, tick_fs=9)
    note = (_estimate_text(row, report["max_structures"]) + ".\n"
            "Pointwise intervals; independent structures and unchanged variance/availability assumed.\n"
            "Solid curve: variance-based planning, not a generation-order history.")
    if row.get("zero_variance"):
        note += "\nZero observed variance does not guarantee zero population variance."
    fig.text(.12, .02, note, fontsize=8, color=".3", va="bottom")
    fig.tight_layout(rect=(0, .17, 1, 1))
    _save_fig(fig, str(base), dpi=dpi, save_pdf=save_pdf)


def _stems(names):
    slugs = {name: re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")[:120] or "descriptor"
             for name in names}
    for name, slug in slugs.items():
        # Collisions must not silently overwrite another descriptor's plot.
        duplicate = sum(other == slug for other in slugs.values()) > 1
        suffix = "_" + hashlib.sha256(name.encode()).hexdigest()[:8] if duplicate else ""
        yield name, "analysis_convergence_" + slug + suffix


def save_convergence_report(report, output_dir, dpi=300, save_pdf=False):
    """Save a report as text, strict JSON, two CSV tables and descriptor plots.

    Files are named ``analysis_convergence*``. The curves CSV has one row per
    descriptor, ensemble size and component, retaining unavailable values as
    empty cells. It includes dashed planning projections through each estimated
    target. The summary CSV stores component counts as a JSON scalar or list.
    Independent PNG figures (and optional PDFs) identify the observed endpoint,
    declared tolerance and projected required size. Return a mapping of paths.
    """
    # Validate JSON before creating a partial export with nonstandard NaN tokens.
    serialized = json.dumps(report, indent=2, allow_nan=False, default=_json_default)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    base = destination / "analysis_convergence"
    paths = {"json": str(base.with_suffix(".json")),
             "report": str(base.with_suffix(".txt")),
             "summary": str(destination / "analysis_convergence_summary.csv"),
             "curves": str(destination / "analysis_convergence_curves.csv"),
             "figures": {}}
    Path(paths["json"]).write_text(serialized + "\n")
    Path(paths["report"]).write_text(format_convergence_report(report))
    fields = ["descriptor", "units", "confidence", "tolerance", "status", "n_total_structures",
              "n_structures", "n_per_point", "half_width", "zero_variance",
              "estimated_total_structures", "estimated_additional_structures", "estimate_status"]
    projections = {name: _projection(report, row)
                   for name, row in report["descriptors"].items()}
    with Path(paths["summary"]).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, row in report["descriptors"].items():
            result = {key: row.get(key) for key in fields}
            result.update(descriptor=name, confidence=report["confidence"],
                          n_per_point=json.dumps(row["n_per_point"], default=_json_default))
            writer.writerow(result)
    with Path(paths["curves"]).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["descriptor", "ensemble_size", "point_index", "half_width",
                         "max_pointwise_half_width", "effective_count", "confidence",
                         "tolerance", "kind"])
        for name, row in report["descriptors"].items():
            original_sizes = set(row["curve"]["sizes"])
            for curve, additional in ((row["curve"], False), (projections[name], True)):
                if curve is None:
                    continue
                for index, size in enumerate(curve["sizes"]):
                    if additional and size in original_sizes:
                        continue
                    components = _components(curve["half_width_per_point"][index])
                    counts = _components(curve["effective_counts"][index])
                    kind = "projection" if size > row["n_total_structures"] else "planning"
                    for point, (width, count) in enumerate(zip(components, counts)):
                        writer.writerow([name, size, point, width, curve["half_width"][index],
                                         count, report["confidence"], row["tolerance"], kind])
    for name, stem in _stems(report["descriptors"]):
        _plot_descriptor(report, name, report["descriptors"][name], projections[name],
                         destination / stem, dpi, save_pdf)
        paths["figures"][name] = [str(destination / f"{stem}.png")]
        if save_pdf:
            paths["figures"][name].append(str(destination / f"{stem}.pdf"))
    return paths
