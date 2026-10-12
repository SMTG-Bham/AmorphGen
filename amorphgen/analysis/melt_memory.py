"""Endpoint retention of crystal-like atom order during a melt ensemble.

The atom indices classified as ordered in the original input are the reference
population. Retention at a later frame is not evidence of uninterrupted crystal
survival: an atom can lose its order and regain it between saved endpoints.
"""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from ase.io import read

from .bond_order import compute_bond_order
from ._serialization import numpy_json_default


_INTERPRETATION = (
    "Endpoint retention of initially ordered atom indices; atoms may have lost "
    "and regained order between frames. This does not establish continuous "
    "crystal survival or identify a crystalline phase."
)
_IDENTITY_BASIS = (
    "Original atom indices, validated by atom count and the complete element "
    "sequence. The pipeline preserves atom order; externally reordered atoms "
    "of the same element cannot be detected by this check."
)
_METRICS = (
    "n_atoms", "q6_mean", "qbar6_mean", "ordered_count", "ordered_fraction",
    "largest_cluster_size", "largest_cluster_fraction",
)


def _new_report(cutoff, qbar6_threshold, min_neighbors):
    return {
        "schema_version": 1,
        "metric": "initially_ordered_atom_index_retention",
        "interpretation": _INTERPRETATION,
        "identity_basis": _IDENTITY_BASIS,
        "cutoff_reference": "original_input",
        "requested_parameters": {
            "cutoff": cutoff, "qbar6_threshold": qbar6_threshold,
            "min_neighbors": min_neighbors,
        },
        "parameters": None,
        "initial": None,
        "comparisons": [],
    }


def _ordered_mask(summary, n_atoms):
    mask = np.asarray(summary["ordered"], dtype=bool)
    if mask.shape != (n_atoms,):
        raise ValueError("Bond-order mask does not match the number of atoms.")
    return mask


def _initialise(report, atoms):
    result = compute_bond_order([atoms], **report["requested_parameters"])
    summary = result["per_structure"][0]
    ordered = _ordered_mask(summary, len(atoms))
    report["parameters"] = result["parameters"]
    report["initial"] = {
        "status": "ok", "reason": None,
        **{key: summary[key] for key in _METRICS},
        "ordered_indices": np.flatnonzero(ordered).tolist(),
    }
    return ordered


def _compare(initial_atoms, atoms, initial_ordered, parameters):
    if len(atoms) != len(initial_atoms):
        raise ValueError(
            f"Atom count differs from original input: {len(atoms)} != "
            f"{len(initial_atoms)}; atom-index retention is unavailable."
        )
    if atoms.get_chemical_symbols() != initial_atoms.get_chemical_symbols():
        raise ValueError(
            "Element sequence differs from original input; atom-index "
            "correspondence cannot be established."
        )
    # The resolved initial cutoff is deliberately passed unchanged. Automatic
    # RDF cutoffs recalculated on a liquid would change the comparison rule.
    result = compute_bond_order(
        [atoms], cutoff=parameters["cutoff"],
        qbar6_threshold=parameters["qbar6_threshold"],
        min_neighbors=parameters["min_neighbors"],
    )
    summary = result["per_structure"][0]
    ordered = _ordered_mask(summary, len(atoms))
    retained = initial_ordered & ordered
    denominator = int(initial_ordered.sum())
    retained_count = int(retained.sum())
    return {
        "status": "ok" if denominator else "unavailable",
        "reason": None if denominator else (
            "No initially ordered atoms; survival fraction is undefined. "
            "Calibrate the cutoff and --qbar6-threshold on the starting crystal."
        ),
        **{key: summary[key] for key in _METRICS},
        "initial_ordered_count": denominator,
        "retained_ordered_count": retained_count,
        "survival_fraction": retained_count / denominator if denominator else None,
        "lost_initial_order_count": denominator - retained_count,
        "newly_ordered_count": int((~initial_ordered & ordered).sum()),
        "retained_ordered_indices": np.flatnonzero(retained).tolist(),
    }


def compute_melt_memory(initial_atoms, frames, cutoff="auto-rdf",
                        qbar6_threshold=0.3, min_neighbors=4):
    """Compare a mapping of named ASE frames to the original crystal.

    Cutoffs are resolved using only ``initial_atoms`` and reused for every
    frame, which retains its own cell and PBC. ``survival_fraction`` is the
    number of initially ordered atom indices ordered at the endpoint divided
    by the initial ordered count. It is ``None`` if no initial atoms qualify.
    Count or element-sequence mismatches raise ``ValueError``; this function
    never truncates or remaps atoms to manufacture correspondence.
    """
    report = _new_report(cutoff, qbar6_threshold, min_neighbors)
    ordered = _initialise(report, initial_atoms)
    for label, atoms in frames.items():
        report["comparisons"].append({
            "label": str(label),
            **_compare(initial_atoms, atoms, ordered, report["parameters"]),
        })
    return report


def _unavailable(reason, initial_count=None):
    return {
        "status": "unavailable", "reason": reason,
        **{key: None for key in _METRICS},
        "initial_ordered_count": initial_count,
        "retained_ordered_count": None,
        "survival_fraction": None,
        "lost_initial_order_count": None,
        "newly_ordered_count": None,
        "retained_ordered_indices": None,
    }


def format_melt_memory(report):
    """Return a compact text report with the denominator and interpretation."""
    lines = ["Melt memory: crystal-like order retention", _INTERPRETATION,
             _IDENTITY_BASIS]
    initial = report["initial"]
    if initial["status"] == "ok":
        lines.append(
            f"Original input: {initial['ordered_count']}/{initial['n_atoms']} "
            f"ordered atoms; mean qbar6 {initial['qbar6_mean']:.4f}; "
            f"largest ordered cluster {initial['largest_cluster_size']}."
        )
        lines.append(
            "Fixed comparison parameters (resolved from original input): "
            + json.dumps(report["parameters"], sort_keys=True, default=_json_default)
        )
    else:
        lines.append(f"Original input: unavailable ({initial['reason']}).")
    for row in report["comparisons"]:
        if row["survival_fraction"] is None:
            lines.append(f"{row['label']}: survival unavailable ({row['reason']}).")
            if row["ordered_fraction"] is not None:
                lines.append(
                    f"  Ordered fraction {100 * row['ordered_fraction']:.1f}%; "
                    f"largest ordered cluster {row['largest_cluster_size']}/{row['n_atoms']}."
                )
        else:
            lines.append(
                f"{row['label']}: {100 * row['survival_fraction']:.1f}% "
                f"({row['retained_ordered_count']}/{row['initial_ordered_count']}) "
                f"of initially ordered atoms remain ordered; "
                f"ordered fraction {100 * row['ordered_fraction']:.1f}%; "
                f"largest ordered cluster {row['largest_cluster_size']}/{row['n_atoms']}."
            )
    return "\n".join(lines) + "\n"


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    return numpy_json_default(value)


def prepare_melt_memory(input_file, cutoff="auto-rdf", qbar6_threshold=0.3,
                        min_neighbors=4):
    """Validate settings and analyse the original input before expensive MD.

    Missing/unreadable original inputs are retained as explicit unavailable
    diagnostics. Invalid descriptor settings or input geometry raise before
    MD begins. The returned context can be reused by ``report_melt_memory``.
    """
    parameters = dict(cutoff=cutoff, qbar6_threshold=qbar6_threshold,
                      min_neighbors=min_neighbors)
    compute_bond_order([], **parameters)  # Validate even without an input file.
    report = _new_report(**parameters)
    atoms, ordered = None, None
    try:
        atoms = read(input_file, index=-1)
    except (OSError, ValueError, RuntimeError, EOFError, IndexError) as exc:
        report["initial"] = _unavailable(f"Cannot read original input: {exc}")
    else:
        ordered = _initialise(report, atoms)
    report["initial"]["path"] = str(input_file)
    return {"report": report, "atoms": atoms, "ordered": ordered}


def report_melt_memory(input_file, shared_dir, snapshot_files, output_dir,
                       cutoff="auto-rdf", qbar6_threshold=0.3, min_neighbors=4,
                       cfg_override=None, prepared=None):
    """Write ``melt_memory.json/.csv/.txt`` for an MQ ensemble before quenching.

    Read the original input and the last frames of the stage 3 and stage 4
    checkpoint files, including old multi-frame ``stage4_eq.xyz`` files.
    Respect configured checkpoint names. Do not substitute a trajectory for
    an absent checkpoint: its last saved frame may precede the stage endpoint.
    Snapshots are compared in exactly the order supplied by the caller.

    Missing/unreadable files and invalid atom correspondence are explicit
    unavailable rows, also on resumed or legacy runs. Such diagnostic failures
    do not discard the remaining valid comparisons or block quenches.
    """
    cfg = cfg_override or {}
    shared = Path(shared_dir)
    files = [
        ("stage3_melt", "stage3", shared / cfg.get("melt", {}).get("output_xyz", "stage3_melted.xyz")),
        ("stage4_equilibration", "stage4", shared / cfg.get("eq_high", {}).get("output_xyz", "stage4_eq.xyz")),
    ]
    files.extend((Path(path).stem, "snapshot", Path(path)) for path in snapshot_files)
    if prepared is None:
        prepared = prepare_melt_memory(input_file, cutoff, qbar6_threshold,
                                       min_neighbors)
    report = deepcopy(prepared["report"])
    initial_atoms, ordered = prepared["atoms"], prepared["ordered"]
    read_errors = (OSError, ValueError, RuntimeError, EOFError, IndexError)

    for label, kind, path in files:
        row = {"label": label, "kind": kind, "path": str(path), "frame_index": -1}
        if ordered is None:
            row.update(_unavailable(report["initial"]["reason"]))
        else:
            try:
                atoms = read(path, index=-1)
                row.update(_compare(initial_atoms, atoms, ordered, report["parameters"]))
            except read_errors as exc:
                row.update(_unavailable(str(exc), int(ordered.sum())))
        report["comparisons"].append(row)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "melt_memory.json").write_text(
        json.dumps(report, indent=2, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    columns = [
        "label", "kind", "path", "frame_index", "status", "reason",
        "initial_ordered_count", "retained_ordered_count", "survival_fraction",
        "lost_initial_order_count", "newly_ordered_count", *_METRICS,
    ]
    with (output / "melt_memory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["comparisons"])
    formatted = format_melt_memory(report)
    (output / "melt_memory.txt").write_text(formatted, encoding="utf-8")
    print("\n" + formatted)
    print(f"Melt-memory reports saved to {output}/melt_memory.{{json,csv,txt}}")
    return report
