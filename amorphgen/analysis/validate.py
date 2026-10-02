"""Validate computed structural metrics against literature reference ranges.

A reference YAML lists expected ranges for density, bond distances, mean
coordination numbers, and bond angle means. Each metric is compared to the
analyser's structure-weighted mean and its confidence interval. Intervals
crossing a reference bound are "inconclusive"; otherwise a metric is labelled
"match" / "concern" / "fail". Legacy results without uncertainty metadata
are still compared using their point estimates.
"""

from __future__ import annotations

from math import isfinite


def _finite(value):
    """Whether a scalar is a usable finite number (including NumPy scalars)."""
    try:
        return value is not None and isfinite(value)
    except (TypeError, ValueError):
        return False


def _verdict(value, low, high, tol_frac=0.05):
    """Classify a value against an expected [low, high] range.

    Returns "match" if inside the range; "concern" if within tol_frac of the
    nearer bound; otherwise "fail".
    """
    if not _finite(value):
        return "n/a"
    if not _finite(low) or not _finite(high) or low > high:
        return "inconclusive"
    if low <= value <= high:
        return "match"
    width = max(abs(high - low), 1e-9)
    margin = max(low - value, value - high)
    if margin <= tol_frac * max(abs(low), abs(high), width):
        return "concern"
    return "fail"


def _entry(computed, key, either_order=False):
    """The descriptor entry, or ``None`` when the analyser has no entry.

    With ``either_order`` the key also matches reversed: the analyser writes
    a bond distance with its two elements in alphabetical order ("O-Si") and
    an angle with its end atoms in that order ("N-Si-O"), while a reference
    may name them "Si-O" or "O-Si-N".
    """
    entry = computed.get(key)
    if entry is None and either_order:
        entry = computed.get("-".join(reversed(key.split("-"))))
    return entry


def _mean(computed, key, either_order=False):
    """The structure-weighted mean when available, else the legacy mean."""
    entry = _entry(computed, key, either_order)
    if entry is None:
        return None
    uncertainty = entry.get("uncertainty")
    value = (uncertainty.get("mean") if uncertainty is not None
             else entry.get("mean"))
    return value if _finite(value) else None


def _interval_verdict(value, low, high, uncertainty):
    """Compare the mean only when its interval resolves the reference bounds."""
    verdict = _verdict(value, low, high)
    if verdict in ("n/a", "inconclusive") or uncertainty is None:
        return verdict
    ci_low, ci_high = uncertainty.get("ci_low"), uncertainty.get("ci_high")
    if not _finite(ci_low) or not _finite(ci_high) or ci_low > ci_high:
        return "inconclusive"
    # Reference endpoints are inclusive. An interval touching a bound from
    # outside still admits both in-range and out-of-range values.
    if ci_low < low <= ci_high or ci_low <= high < ci_high:
        return "inconclusive"
    return verdict


def validate_against_reference(analyser, reference):
    """Compare analyser output to a reference dict (loaded from YAML).

    Parameters
    ----------
    analyser : StructureAnalyser
    reference : dict
        Parsed YAML with optional keys: ``density``, ``bond_distances``,
        ``coordination``, ``bond_angles`` (see examples/reference_*.yaml).

    Returns
    -------
    dict
        ``{"system": str, "sources": list[str], "rows": list[tuple],
        "intervals": dict}``
        where each row is (descriptor, computed, expected_lo, expected_hi,
        units, verdict). A metric the structures do not have (an element
        pair absent, or with no contact inside its cutoff) keeps its row, with
        ``None`` and the verdict "n/a". ``intervals`` maps descriptor names to
        the analyser's uncertainty metadata. Means and t intervals use
        structures as independent sampling units, not pooled atoms. With
        uncertainty metadata but no finite interval (e.g. one structure),
        a finite descriptor is "inconclusive".
    """
    rows = []
    intervals = {}

    def add_row(descriptor, entry, lo, hi, units):
        uncertainty = None if entry is None else entry.get("uncertainty")
        value = _mean({"metric": entry}, "metric")
        if uncertainty is not None:
            intervals[descriptor] = dict(uncertainty)
        rows.append((descriptor, value, lo, hi, units,
                     _interval_verdict(value, lo, hi, uncertainty)))

    if "density" in reference:
        d = analyser.density()
        lo, hi = reference["density"]["expected"]
        add_row("Density", d, lo, hi,
                reference["density"].get("units", "g/cm³"))

    bd = analyser.bond_distances() if "bond_distances" in reference else {}
    for pair, spec in reference.get("bond_distances", {}).items():
        lo, hi = spec["expected"]
        add_row(f"Bond {pair}", _entry(bd, pair, either_order=True),
                lo, hi, spec.get("units", "Å"))

    # directional: CN Si-O counts O around Si, CN O-Si counts Si around O
    cn = analyser.coordination() if "coordination" in reference else {}
    for pair, spec in reference.get("coordination", {}).items():
        lo, hi = spec["mean_expected"]
        add_row(f"CN {pair}", _entry(cn, pair), lo, hi, "")

    ba = analyser.bond_angles() if "bond_angles" in reference else {}
    for triplet, spec in reference.get("bond_angles", {}).items():
        lo, hi = spec["expected"]
        add_row(f"Angle {triplet}", _entry(ba, triplet, either_order=True),
                lo, hi, spec.get("units", "°"))

    return {
        "system": reference.get("system", "(unspecified)"),
        "sources": reference.get("references", []),
        "rows": rows,
        "intervals": intervals,
    }


def format_validation_report(result):
    """Render the dict from validate_against_reference() as a printable table."""
    rows = result["rows"]
    if not rows:
        return "No validation rows produced (check reference YAML)."

    intervals = result.get("intervals", {})
    interval_width = 27 if intervals else 0
    bar = "=" * (78 + interval_width)
    lines = [f"\n{bar}", f"  Validation: {result['system']}", bar]

    if result["sources"]:
        lines.append("  Reference sources:")
        for s in result["sources"]:
            lines.append(f"    - {s}")
        lines.append("")

    if intervals:
        lines.append("  Computed: equal-weight mean over structures; t intervals "
                     "describe uncertainty of that mean.")
        lines.append("  Structures are treated as independent sampling units.")
        lines.append("")
    interval_heading = f"  {'Mean t interval':>25}" if intervals else ""
    lines.append(f"  {'Descriptor':<22}{'Computed':>10}{interval_heading}  "
                 f"{'Expected':>14}  {'Units':<6}  Verdict")
    lines.append("  " + "-" * (72 + interval_width))
    for descriptor, value, lo, hi, units, verdict in rows:
        if not _finite(value):
            cval = "n/a"
        elif abs(value) >= 100:
            cval = f"{value:>10.1f}"
        else:
            cval = f"{value:>10.3f}"
        expected = f"[{lo:.2f}, {hi:.2f}]"
        interval = ""
        if intervals:
            uncertainty = intervals.get(descriptor)
            if uncertainty is None:
                interval_text = "not supplied"
            else:
                ci_low = uncertainty.get("ci_low")
                ci_high = uncertainty.get("ci_high")
                if _finite(ci_low) and _finite(ci_high) and ci_low <= ci_high:
                    confidence = uncertainty.get("confidence")
                    level = f"{100 * confidence:g}% " if _finite(confidence) else ""
                    interval_text = f"{level}[{ci_low:.3f}, {ci_high:.3f}]"
                else:
                    interval_text = "unavailable"
            interval = f"  {interval_text:>25}"
        lines.append(f"  {descriptor:<22}{cval:>10}{interval}  "
                     f"{expected:>14}  {units:<6}  {verdict}")

    lines.append("  " + "-" * (72 + interval_width))
    n_match = sum(1 for r in rows if r[5] == "match")
    n_concern = sum(1 for r in rows if r[5] == "concern")
    n_fail = sum(1 for r in rows if r[5] == "fail")
    n_na = sum(1 for r in rows if r[5] == "n/a")
    n_inconclusive = sum(1 for r in rows if r[5] == "inconclusive")
    na = f", {n_na} n/a" if n_na else ""
    inconclusive = f", {n_inconclusive} inconclusive" if n_inconclusive else ""
    lines.append(f"  Summary: {n_match} match, {n_concern} concern, "
                 f"{n_fail} fail{inconclusive}{na} (out of {len(rows)} metrics)")
    if n_inconclusive:
        lines.append("  inconclusive: the mean's interval crosses a reference "
                     "bound, or an interval is unavailable.")
        lines.append("  At least two independent structures are needed to "
                     "estimate uncertainty of the ensemble mean.")
    if n_na:
        lines.append("  n/a: not found in the structures (element absent, "
                     "or no contact within the cutoff)")
    lines.append(bar)
    return "\n".join(lines)
