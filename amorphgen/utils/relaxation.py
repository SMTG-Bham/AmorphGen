"""Persist the actual relaxation stopping criterion for later screening.

Extended XYZ preserves ``Atoms.info``; CIF and VASP do not. A companion JSON
file records the same values and the structure file's hash so a replaced
structure never acquires a stale convergence result. Legacy files without
metadata have unknown convergence.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


_FIELDS = {
    "relaxation_attempted", "relaxation_converged", "relaxation_fmax",
    "relaxation_max_force", "relaxation_steps", "relaxation_max_steps",
    "relaxation_engine", "relaxation_force_criterion", "relaxation_cell_filter",
    "relaxation_pressure_tol_gpa", "relaxation_pressure_gpa",
}


def clear_relaxation_metadata(atoms):
    """Discard a previous geometry's relaxation result before new dynamics."""
    for key in _FIELDS:
        atoms.info.pop(key, None)
    atoms.info.pop("relax_converged", None)  # historical screening alias


def record_relaxation_metadata(atoms, *, converged, fmax, max_force, steps,
                               max_steps, engine, force_criterion, cell_filter,
                               pressure_tol_gpa=None, pressure_gpa=None):
    """Replace any previous relaxation result with the evaluated final result.

    ``max_force`` must be the engine's stopping metric, including cell forces
    where applicable. A missing result stays unknown instead of becoming a
    success or failure inferred from the existence of an output file.
    """
    clear_relaxation_metadata(atoms)
    atoms.info.update({
        "relaxation_attempted": True,
        "relaxation_fmax": float(fmax),
        "relaxation_steps": int(steps),
        "relaxation_max_steps": int(max_steps),
        "relaxation_engine": str(engine),
        "relaxation_force_criterion": str(force_criterion),
        "relaxation_cell_filter": str(cell_filter or "none"),
    })
    if converged is not None:
        atoms.info["relaxation_converged"] = bool(converged)
    if max_force is not None:
        atoms.info["relaxation_max_force"] = float(max_force)
    if pressure_tol_gpa is not None:
        atoms.info["relaxation_pressure_tol_gpa"] = float(pressure_tol_gpa)
    if pressure_gpa is not None:
        atoms.info["relaxation_pressure_gpa"] = float(pressure_gpa)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_relaxation_metadata(path, atoms):
    """Atomically write a sidecar for an already-written structure file."""
    # ASE's extxyz reader returns NumPy scalars for numerical info fields.
    metadata = {key: (value.item() if isinstance(value, np.generic) else value)
                for key, value in atoms.info.items() if key in _FIELDS}
    if not metadata:
        return
    destination = Path(str(path) + ".relaxation.json")
    payload = {"schema_version": 1, "sha256": _sha256(path), "metadata": metadata}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=destination.parent,
                                         prefix=destination.name + ".",
                                         suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)


def read_relaxation_metadata(path, atoms=None):
    """Return valid sidecar metadata and optionally merge it into ``atoms.info``.

    Missing, unreadable, invalid or stale sidecars contribute no information;
    callers can still use metadata embedded in the structure itself.
    """
    try:
        with open(str(path) + ".relaxation.json", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            return {}
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict) or payload.get("sha256") != _sha256(path):
            return {}
    except (OSError, ValueError, TypeError):
        return {}
    metadata = {key: value for key, value in metadata.items() if key in _FIELDS}
    if atoms is not None:
        atoms.info.update(metadata)
    return metadata
