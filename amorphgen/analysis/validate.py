"""Validate computed structural metrics against literature reference ranges.

A reference YAML lists expected ranges for density, bond distances, mean
coordination numbers, and bond angle means. Each metric is compared to the
analyser's computed value and labelled "match" / "concern" / "fail" so the
user can defend an ensemble against published data.
"""

from __future__ import annotations


def _verdict(value, low, high, tol_frac=0.05):
    """Classify a value against an expected [low, high] range.

    Returns "match" if inside the range; "concern" if within tol_frac of the
    nearer bound; otherwise "fail".
    """
    if value is None:
        return "n/a"
    if low <= value <= high:
        return "match"
    width = max(abs(high - low), 1e-9)
    margin = max(low - value, value - high)
    if margin <= tol_frac * max(abs(low), abs(high), width):
        return "concern"
    return "fail"


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
        ``{"system": str, "sources": list[str], "rows": list[tuple]}``
        where each row is (descriptor, computed, expected_lo, expected_hi,
        units, verdict).
    """
    rows = []

    if "density" in reference:
        d = analyser.density()
        lo, hi = reference["density"]["expected"]
        rows.append(("Density", d["mean"], lo, hi,
                     reference["density"].get("units", "g/cm³"),
                     _verdict(d["mean"], lo, hi)))

    bd = analyser.bond_distances() if "bond_distances" in reference else {}
    for pair, spec in reference.get("bond_distances", {}).items():
        if pair not in bd:
            continue
        lo, hi = spec["expected"]
        v = bd[pair]["mean"]
        rows.append((f"Bond {pair}", v, lo, hi,
                     spec.get("units", "Å"), _verdict(v, lo, hi)))

    cn = analyser.coordination() if "coordination" in reference else {}
    for pair, spec in reference.get("coordination", {}).items():
        if pair not in cn:
            continue
        lo, hi = spec["mean_expected"]
        v = cn[pair]["mean"]
        rows.append((f"CN {pair}", v, lo, hi, "", _verdict(v, lo, hi)))

    ba = analyser.bond_angles() if "bond_angles" in reference else {}
    for triplet, spec in reference.get("bond_angles", {}).items():
        if triplet not in ba:
            continue
        lo, hi = spec["expected"]
        v = ba[triplet]["mean"]
        rows.append((f"Angle {triplet}", v, lo, hi,
                     spec.get("units", "°"), _verdict(v, lo, hi)))

    return {
        "system": reference.get("system", "(unspecified)"),
        "sources": reference.get("references", []),
        "rows": rows,
    }


def format_validation_report(result):
    """Render the dict from validate_against_reference() as a printable table."""
    rows = result["rows"]
    if not rows:
        return "No validation rows produced (check reference YAML)."

    bar = "=" * 78
    lines = [f"\n{bar}", f"  Validation: {result['system']}", bar]

    if result["sources"]:
        lines.append("  Reference sources:")
        for s in result["sources"]:
            lines.append(f"    - {s}")
        lines.append("")

    lines.append(f"  {'Descriptor':<22}{'Computed':>10}  "
                 f"{'Expected':>14}  {'Units':<6}  Verdict")
    lines.append("  " + "-" * 72)
    for descriptor, value, lo, hi, units, verdict in rows:
        if value is None:
            cval = "n/a"
        elif abs(value) >= 100:
            cval = f"{value:>10.1f}"
        else:
            cval = f"{value:>10.3f}"
        expected = f"[{lo:.2f}, {hi:.2f}]"
        lines.append(f"  {descriptor:<22}{cval:>10}  "
                     f"{expected:>14}  {units:<6}  {verdict}")

    lines.append("  " + "-" * 72)
    n_match = sum(1 for r in rows if r[5] == "match")
    n_concern = sum(1 for r in rows if r[5] == "concern")
    n_fail = sum(1 for r in rows if r[5] == "fail")
    lines.append(f"  Summary: {n_match} match, {n_concern} concern, "
                 f"{n_fail} fail (out of {len(rows)} metrics)")
    lines.append(bar)
    return "\n".join(lines)
