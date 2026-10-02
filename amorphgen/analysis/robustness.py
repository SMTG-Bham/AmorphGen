"""Sensitivity of neighbour counts to the resolved distance cutoffs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from ase.neighborlist import neighbor_list


def validate_cutoff_window(window):
    """Return a finite positive half-window in Angstrom."""
    if isinstance(window, (bool, np.bool_)):
        raise ValueError("cutoff window must be finite and positive")
    try:
        value = float(window)
    except (TypeError, ValueError) as exc:
        raise ValueError("cutoff window must be finite and positive") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("cutoff window must be finite and positive")
    return value


def compute_cutoff_robustness(atoms_list, max_cutoff, get_cutoff_fn,
                              window=0.1, points=5):
    """Sweep already resolved cutoffs without refitting the RDF.

    Each positive pair cutoff is shifted by the same offset in
    ``[-window, window]``, with negative radii clipped to zero. Zero cutoffs
    stay zero. ``points`` must be odd and at least three, so the baseline
    is always included. The distance predicate is exactly the one used by
    ``build_neighbour_dict``: ``d <= pair_cutoff`` and ``d < max_cutoff``.

    ``pairs`` contains unordered contact counts, counting reciprocal
    neighbour-list entries once and retaining distinct periodic images
    (including self images). ``near_pairs`` counts contacts whose inclusion
    changes between the window endpoints, and ``near_fraction`` divides by
    the number included at the upper endpoint. It is None if that count is
    zero. Counts and fractions are pooled across structures.

    For each direction, ``coordination.mean`` is the curve of pooled site
    means, ``ensemble_mean`` weights contributing structures equally, and
    ``delta`` / ``ensemble_delta`` give their upper-minus-lower changes.
    ``per_structure`` retains aligned curves, with None when the central
    element is absent. Absent neighbours contribute zero, as in coordination().
    This is a sensitivity diagnostic, not a statistical confidence interval.
    """
    window = validate_cutoff_window(window)
    if (isinstance(points, (bool, np.bool_))
            or not isinstance(points, (int, np.integer))
            or points < 3 or points % 2 != 1):
        raise ValueError("cutoff points must be an odd integer >= 3")
    frames = list(atoms_list)
    if not frames:
        raise ValueError("cutoff robustness requires at least one structure")
    offsets = np.linspace(-window, window, points)
    offsets[points // 2] = 0.0
    species = sorted({s for atoms in frames for s in atoms.get_chemical_symbols()})
    pairs = [(a, b) for i, a in enumerate(species) for b in species[i:]]
    resolved = {f"{a}-{b}": float(get_cutoff_fn(a, b)) for a, b in pairs}
    if any(not math.isfinite(c) or c < 0 for c in [max_cutoff, *resolved.values()]):
        raise ValueError("cutoffs must be finite and nonnegative")

    # Keep the original global search bound, including any configured pair
    # that is absent from this ensemble. This also reproduces baseline CN at
    # exact cutoffs rather than quietly changing the shared boundary rule.
    limits = (np.maximum(0.0, max_cutoff + offsets) if max_cutoff > 0
              else np.zeros(points))
    cutoffs = {p: np.maximum(0.0, c + offsets) if c > 0 else np.zeros(points)
               for p, c in resolved.items()}
    if not np.all(np.isfinite(limits)):
        raise ValueError("cutoff window produces nonfinite radii")

    pair_counts = {p: [] for p in resolved}
    centre_counts = {s: [] for s in species}
    for atoms in frames:
        syms = np.asarray(atoms.get_chemical_symbols())
        for s in species:
            centre_counts[s].append(int(np.count_nonzero(syms == s)))
        if limits[-1] > 0 and len(atoms):
            source, target, distances = neighbor_list("ijd", atoms, cutoff=limits[-1])
            sources, targets = syms[source], syms[target]
        else:
            distances = np.empty(0)
            sources = targets = np.empty(0, dtype=str)
        for a, b in pairs:
            pair = f"{a}-{b}"
            # Hetero contacts occur once in the A->B direction. Homo contacts
            # have two reciprocal entries even for i==j periodic self images.
            d = distances[(sources == a) & (targets == b)]
            counts = np.array([np.count_nonzero((d <= cut) & (d < limit))
                               for cut, limit in zip(cutoffs[pair], limits)], dtype=int)
            if a == b:
                counts //= 2
            pair_counts[pair].append(counts)

    result = {
        "window": window, "offsets": offsets.tolist(), "n_structures": len(frames),
        "global_cutoffs": limits.tolist(),
        "near_fraction_definition": (
            "pooled contacts added between lower and upper cutoff / contacts "
            "at upper cutoff; reciprocal entries counted once, periodic images retained"),
        "boundary_definition": "d <= pair cutoff and d < global maximum cutoff",
        "pairs": {},
    }
    for a, b in pairs:
        pair = f"{a}-{b}"
        counts = np.array(pair_counts[pair])
        pooled = counts.sum(axis=0)
        total, near = int(pooled[-1]), int(pooled[-1] - pooled[0])
        row = {
            "cutoff": resolved[pair], "cutoffs": cutoffs[pair].tolist(),
            "n_pairs": total, "near_pairs": near,
            "near_fraction": near / total if total else None,
            "per_structure": [
                {"n_pairs": int(c[-1]), "near_pairs": int(c[-1] - c[0]),
                 "near_fraction": float((c[-1] - c[0]) / c[-1]) if c[-1] else None}
                for c in counts],
            "coordination": {},
        }
        for centre, partner in ([(a, b)] if a == b else [(a, b), (b, a)]):
            ns = centre_counts[centre]
            total_atoms = sum(ns)
            # An A-A contact supplies two neighbours; an A-B contact supplies
            # one neighbour per direction. All central sites enter the mean.
            directed = counts * (2 if a == b else 1)
            curves = [(c / n).tolist() if n else None for c, n in zip(directed, ns)]
            mean = (directed.sum(axis=0) / total_atoms).tolist()
            available = [curve for curve in curves if curve is not None]
            ensemble = [math.fsum(c[k] for c in available) / len(available)
                        for k in range(points)]
            row["coordination"][f"{centre}-{partner}"] = {
                "mean": mean, "ensemble_mean": ensemble,
                "delta": mean[-1] - mean[0],
                "ensemble_delta": ensemble[-1] - ensemble[0],
                "per_structure": curves, "total_atoms": total_atoms,
            }
        result["pairs"][pair] = row
    return result


def format_cutoff_robustness(report):
    """Format pair shares and directional coordination across the window."""
    offsets = ", ".join(f"{offset:+.3f}" for offset in report["offsets"])
    lines = [
        f"\n  Cutoff robustness (+/- {report['window']:.3f} A):",
        "    Near share = contacts added across window / contacts at upper cutoff.",
        "    Pooled unordered contacts; periodic images retained; no contacts = n/a.",
        f"    CN offsets (A): {offsets}; means pool all central sites.",
        "    Lower radii clipped at zero; zero cutoffs stay zero.",
        "    Sensitivity to fixed cutoffs, not a confidence interval.",
    ]
    for pair, row in report["pairs"].items():
        share = "n/a" if row["near_fraction"] is None else f"{100 * row['near_fraction']:.1f}%"
        lines.append(
            f"    {pair}: cutoff={row['cutoff']:.3f} A; "
            f"window=[{row['cutoffs'][0]:.3f}, {row['cutoffs'][-1]:.3f}] A; "
            f"near={row['near_pairs']}/{row['n_pairs']} ({share})")
        for direction, data in row["coordination"].items():
            curve = " -> ".join(f"{value:.3f}" for value in data["mean"])
            lines.append(f"      CN {direction}: {curve}; delta={data['delta']:+.3f}")
    return "\n".join(lines)


def save_cutoff_robustness(report, output_dir=".", prefix="analysis"):
    """Save the full JSON report and pooled pair/coordination CSV tables.

    Returns a mapping of ``json``, ``pairs`` and ``coordination`` to paths.
    Aligned per-structure contact counts and CN curves are retained in JSON.
    Fractions use the 0..1 scale; missing fractions are empty CSV cells.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base = output / f"{prefix}_cutoff_robustness"
    paths = {"json": f"{base}.json", "pairs": f"{base}_pairs.csv",
             "coordination": f"{base}_coordination.csv"}
    with open(paths["json"], "w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    with open(paths["pairs"], "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair", "cutoff_A", "lower_cutoff_A", "upper_cutoff_A",
                         "near_pairs", "n_pairs_upper", "near_fraction"])
        for pair, row in report["pairs"].items():
            writer.writerow([pair, row["cutoff"], row["cutoffs"][0], row["cutoffs"][-1],
                             row["near_pairs"], row["n_pairs"], row["near_fraction"]])
    with open(paths["coordination"], "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair", "direction", "offset_A", "cutoff_A", "pooled_mean",
                         "ensemble_mean", "delta", "ensemble_delta", "total_atoms"])
        for pair, row in report["pairs"].items():
            for direction, data in row["coordination"].items():
                for k, offset in enumerate(report["offsets"]):
                    writer.writerow([pair, direction, offset, row["cutoffs"][k],
                                     data["mean"][k], data["ensemble_mean"][k],
                                     data["delta"], data["ensemble_delta"], data["total_atoms"]])
    return paths
