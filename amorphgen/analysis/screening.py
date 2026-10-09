"""Per-structure diagnostics with independent labelling and exclusion policies.

Screen thresholds are diagnostics to calibrate for a material, not universal
definitions of an amorphous structure. All screens are label-only by default.
Missing measurements have an explicit ``unavailable`` status and cannot pass.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers
from ase.neighborlist import neighbor_list

from .bond_order import _resolve_cutoff, _validate_geometry, compute_bond_order
from .cutoff import parse_cutoff_spec
from .energy import stored_info_energy
from .structure import is_bonding_pair
from ..utils.common import compute_density_gcm3
from ..utils.radii import auto_target_cn, default_minsep


_DEFAULTS = {
    "coordination": {"allowed": "auto", "max_fraction": 0.0},
    "crystal_like": {"qbar6_threshold": 0.3, "min_neighbors": 4,
                     "max_fraction": 0.0},
    "close_contacts": {"threshold_frac": 0.7},
    "density": {},
    "energy": {},
    "unconverged": {},
}
_OPTIONS = {
    "coordination": {"allowed", "max_fraction"},
    "crystal_like": {"qbar6_threshold", "min_neighbors", "max_fraction",
                     "max_cluster_fraction", "cutoff"},
    "close_contacts": {"threshold_frac", "min_distance"},
    "density": {"min", "max"},
    "energy": {"min", "max"},
    "unconverged": set(),
}


def _number(value, name, *, positive=False, fraction=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if fraction and not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _pair_table(value, name):
    if not isinstance(value, Mapping):
        return _number(value, name, positive=True)
    if not value:
        raise ValueError(f"{name} pair table must not be empty")
    result = {}
    for pair, threshold in value.items():
        symbols = pair.split("-") if isinstance(pair, str) else []
        if len(symbols) != 2 or any(s not in atomic_numbers for s in symbols):
            raise ValueError(f"{name} pair keys must be element pairs such as Si-O")
        key = "-".join(sorted(symbols))
        if key in result:
            raise ValueError(f"{name} repeats pair {key}")
        result[key] = _number(threshold, f"{name}.{pair}", positive=True)
    return result


def validate_screening_config(config):
    """Validate and normalize a screening mapping without mutating it.

    ``True`` enables coordination, crystal-like order, close contacts and
    relaxation diagnostics. Density and energy need explicit ``min`` and/or
    ``max`` bounds. ``False``, ``None`` and ``{}`` enable no screens. Individual
    screens accept a mapping or a boolean; omitted ``exclude`` is always false.
    Unknown keys and invalid thresholds raise ``ValueError`` before evaluation.
    """
    if config is None or config is False:
        return {}
    if config is True:
        config = {name: {} for name in (
            "coordination", "crystal_like", "close_contacts", "unconverged")}
    if not isinstance(config, Mapping):
        raise ValueError("screening must be a boolean or a mapping of screens")
    unknown = set(config) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown screening screen(s): {', '.join(sorted(map(str, unknown)))}")
    normalized = {}
    for name in _DEFAULTS:
        if name not in config or config[name] is False:
            continue
        spec = {} if config[name] is True else config[name]
        if not isinstance(spec, Mapping):
            raise ValueError(f"screening.{name} must be a boolean or mapping")
        unknown = set(spec) - (_OPTIONS[name] | {"enabled", "exclude"})
        if unknown:
            raise ValueError(f"unknown screening.{name} option(s): "
                             f"{', '.join(sorted(map(str, unknown)))}")
        for key in ("enabled", "exclude"):
            if key in spec and not isinstance(spec[key], bool):
                raise ValueError(f"screening.{name}.{key} must be a boolean")
        if not spec.get("enabled", True):
            continue
        opts = {"enabled": True, "exclude": False, **_DEFAULTS[name], **spec}
        if name == "coordination":
            allowed = opts["allowed"]
            if not isinstance(allowed, str) or allowed != "auto":
                if not isinstance(allowed, Mapping) or not allowed:
                    raise ValueError("coordination.allowed must be 'auto' or a nonempty element mapping")
                sets = {}
                for symbol, numbers in allowed.items():
                    if symbol not in atomic_numbers:
                        raise ValueError(f"coordination.allowed contains unknown element {symbol!r}")
                    if (not isinstance(numbers, (list, tuple, set)) or not numbers
                            or any(isinstance(n, (bool, np.bool_))
                                   or not isinstance(n, Integral) or n < 0 for n in numbers)):
                        raise ValueError(f"coordination.allowed.{symbol} must be a nonempty set of nonnegative integers")
                    sets[symbol] = sorted({int(n) for n in numbers})
                opts["allowed"] = sets
        if name in ("coordination", "crystal_like"):
            opts["max_fraction"] = _number(opts["max_fraction"], f"{name}.max_fraction", fraction=True)
        if name == "crystal_like":
            opts["qbar6_threshold"] = _number(opts["qbar6_threshold"], "crystal_like.qbar6_threshold", fraction=True)
            n = opts["min_neighbors"]
            if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 1:
                raise ValueError("crystal_like.min_neighbors must be a positive integer")
            opts["min_neighbors"] = int(n)
            if "max_cluster_fraction" in opts:
                opts["max_cluster_fraction"] = _number(opts["max_cluster_fraction"], "crystal_like.max_cluster_fraction", fraction=True)
            if "cutoff" in opts:
                cutoff_values = (opts["cutoff"].values()
                                 if isinstance(opts["cutoff"], Mapping)
                                 else [opts["cutoff"]])
                if any(isinstance(value, (bool, np.bool_)) for value in cutoff_values):
                    raise ValueError("crystal_like.cutoff must be positive")
                opts["cutoff"] = parse_cutoff_spec(opts["cutoff"])
                _resolve_cutoff([], opts["cutoff"])
                if isinstance(opts["cutoff"], dict):
                    pairs = {key: value for key, value in opts["cutoff"].items()
                             if key != "default"}
                    if pairs:
                        _pair_table(pairs, "crystal_like.cutoff")
        if name == "close_contacts":
            opts["threshold_frac"] = _number(opts["threshold_frac"], "close_contacts.threshold_frac", positive=True)
            if "min_distance" in opts:
                opts["min_distance"] = _pair_table(opts["min_distance"], "close_contacts.min_distance")
        if name in ("density", "energy"):
            if "min" not in opts and "max" not in opts:
                raise ValueError(f"{name} requires min and/or max")
            for key in ("min", "max"):
                if key in opts:
                    opts[key] = _number(opts[key], f"{name}.{key}")
                    if name == "density" and opts[key] < 0:
                        raise ValueError(f"density.{key} must be nonnegative")
            if opts.get("min", -np.inf) > opts.get("max", np.inf):
                raise ValueError(f"{name}.min must not exceed max")
        normalized[name] = opts
    return normalized


def _coordination(atoms, opts, cutoff):
    symbols = atoms.get_chemical_symbols()
    composition = Counter(symbols)
    allowed = opts["allowed"]
    if allowed == "auto":
        targets, tolerance = auto_target_cn(composition)
        allowed = {s: list(range(max(0, n - tolerance), n + tolerance + 1))
                   for s, n in (targets or {}).items()}
    checked = [i for i, symbol in enumerate(symbols) if symbol in allowed]
    if not checked:
        raise ValueError("no coordination sets available for the elements in this structure")
    ase_cutoff = ({tuple(pair.split("-")): value for pair, value in cutoff.items()}
                  if isinstance(cutoff, dict) else cutoff)
    source, target = neighbor_list("ij", atoms, ase_cutoff)
    bonds = {(a, b): is_bonding_pair(a, b, composition)
             for a in composition for b in composition}
    bonded = [int(i) for i, j in zip(source, target) if bonds[symbols[i], symbols[j]]]
    counts = np.bincount(bonded, minlength=len(atoms))
    unexpected = [i for i in checked if counts[i] not in allowed[symbols[i]]]
    fraction = len(unexpected) / len(checked)
    return fraction > opts["max_fraction"], {
        "allowed": allowed, "cutoff": cutoff, "checked_sites": len(checked),
        "unassessed_elements": sorted(set(symbols) - set(allowed)),
        "coordination_numbers": counts.tolist(), "unexpected_sites": unexpected,
        "unexpected_fraction": fraction,
    }


def _close_contacts(atoms, opts):
    symbols = atoms.get_chemical_symbols()
    unique = sorted(set(symbols))
    pairs = [f"{a}-{b}" for i, a in enumerate(unique) for b in unique[i:]]
    explicit = opts.get("min_distance")
    if isinstance(explicit, dict):
        missing = set(pairs) - set(explicit)
        if missing:
            raise ValueError(f"min_distance lacks element pair(s): {', '.join(sorted(missing))}")
        thresholds = {pair: explicit[pair] for pair in pairs}
    elif explicit is not None:
        thresholds = dict.fromkeys(pairs, explicit)
    else:
        thresholds = {pair: value * opts["threshold_frac"]
                      for pair, value in default_minsep(symbols).items()}
    source, target, shifts, distances = neighbor_list(
        "ijSd", atoms, {tuple(pair.split("-")): value for pair, value in thresholds.items()})
    contacts = []
    for i, j, shift, distance in zip(source, target, shifts, distances):
        # Keep one of each directed edge, including periodic self-images.
        if i > j or (i == j and tuple(shift) <= (0, 0, 0)):
            continue
        key = "-".join(sorted((symbols[i], symbols[j])))
        contacts.append({"atoms": [int(i), int(j)], "shift": shift.tolist(),
                         "pair": key, "distance": float(distance),
                         "threshold": thresholds[key]})
    return bool(contacts), {"thresholds": thresholds, "count": len(contacts),
                            "contacts": contacts}


def _stored_energy(atoms):
    """Read existing energy only; screening must not run a calculator."""
    key, value = stored_info_energy(atoms)
    if key is not None:
        return _number(value, f"stored {key}") / len(atoms)
    results = getattr(atoms.calc, "results", {})
    if "energy" in results:
        if atoms.calc.check_state(atoms):
            raise ValueError("stored calculator energy is stale for this structure")
        return _number(results["energy"], "stored energy") / len(atoms)
    raise ValueError("no stored potential energy")


def _unconverged(atoms):
    for key in ("relaxation_converged", "relax_converged"):
        if key in atoms.info:
            value = atoms.info[key]
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{key} must be an explicit boolean")
            return not bool(value), {"converged": bool(value), "metadata_key": key}
    raise ValueError("no explicit relaxation convergence metadata")


def _update_summary(result):
    rows = result["per_structure"]
    result["summary"] = {"generated": len(rows),
                         "passed": sum(row["passed"] for row in rows),
                         "labelled": sum(bool(row["labels"]) for row in rows),
                         "analysed": sum(row["analysed"] for row in rows),
                         "excluded": sum(row["excluded"] for row in rows)}
    result["passed_indices"] = [row["index"] for row in rows if row["passed"]]
    result["retained_indices"] = [row["index"] for row in rows if not row["excluded"]]
    result["analysed_indices"] = [row["index"] for row in rows if row["analysed"]]


def screen_structures(atoms_list, config, *, cutoff="auto-rdf", source_names=None):
    """Evaluate configured screens and return JSON-safe records in input order.

    Coordination uses chemically bonding neighbour-image counts and accepts
    ``auto_target_cn`` targets plus/minus its tolerance by default. Crystal-like
    order uses the existing neighbour-averaged qbar6 diagnostic. Close contacts
    include all pair types and periodic self-images; thresholds are strict lower
    bounds. Density (g/cm3) and energy (eV/atom) bounds are inclusive. Relaxation
    status comes only from explicit ``relaxation_converged`` or
    ``relax_converged`` booleans. Calculators are never evaluated.

    Every nonpassing screen adds a label independently of its ``exclude`` flag.
    Missing measurements add ``<screen>_unavailable`` and follow that same flag.
    ``passed`` means no labels, ``retained_indices`` means not excluded, and
    ``analysed`` starts false until :func:`mark_screening_analysed` is called.
    Resolved neighbour cutoffs are shared over the input ensemble. An optional
    ``crystal_like.cutoff`` specifies its own neighbour definition, also resolved
    over the complete input ensemble before selection.
    """
    config = validate_screening_config(config)
    frames = list(atoms_list)
    sources = ([None] * len(frames) if source_names is None else list(source_names))
    if len(sources) != len(frames):
        raise ValueError("source_names must match the number of structures")
    geometry_errors = {}
    for index, atoms in enumerate(frames):
        try:
            if not len(atoms):
                raise ValueError("structure has no atoms")
            _validate_geometry(atoms, index)
        except ValueError as exc:
            geometry_errors[index] = str(exc)
    resolved, cutoff_error = None, None
    crystal_cutoff, crystal_cutoff_error = None, None
    if set(config) & {"coordination", "crystal_like"}:
        valid = [atoms for i, atoms in enumerate(frames) if i not in geometry_errors]
        valid.sort(key=lambda atoms: (
            atoms.numbers.tobytes(), np.asarray(atoms.cell).tobytes(),
            atoms.positions.tobytes(), atoms.pbc.tobytes()))
        if "coordination" in config or "cutoff" not in config.get("crystal_like", {}):
            try:
                resolved = _resolve_cutoff(valid, cutoff)
            except (ValueError, TypeError, RuntimeError, ZeroDivisionError) as exc:
                cutoff_error = str(exc)
        crystal_cutoff, crystal_cutoff_error = resolved, cutoff_error
        if "cutoff" in config.get("crystal_like", {}):
            try:
                crystal_cutoff = _resolve_cutoff(valid, config["crystal_like"]["cutoff"])
                crystal_cutoff_error = None
            except (ValueError, TypeError, RuntimeError, ZeroDivisionError) as exc:
                crystal_cutoff_error = str(exc)
    result = {"parameters": {"screens": config, "cutoff": resolved},
              "definitions": {
                  "generated": "number of supplied candidate structures; upstream generation failures are not included",
                  "passed": "structures with no triggered or unavailable configured screens",
                  "labelled": "structures with at least one diagnostic or unavailable label, whether retained or excluded",
                  "analysed": "retained structures whose requested analysis completed",
                  "excluded": "structures rejected by at least one enabled screen with exclude=true",
              },
              "per_structure": []}
    if "crystal_like" in config:
        result["parameters"]["crystal_like_cutoff"] = crystal_cutoff
    for index, (atoms, source) in enumerate(zip(frames, sources)):
        row = {"index": index, "source": None if source is None else str(source),
               "labels": [], "excluded": False, "passed": True,
               "analysed": False, "screens": {}}
        for name, opts in config.items():
            try:
                if not len(atoms):
                    raise ValueError("structure has no atoms")
                if name in ("coordination", "crystal_like", "close_contacts", "density"):
                    if index in geometry_errors:
                        raise ValueError(geometry_errors[index])
                if name == "coordination" and cutoff_error:
                    raise ValueError(cutoff_error)
                if name == "crystal_like" and crystal_cutoff_error:
                    raise ValueError(crystal_cutoff_error)
                if name == "coordination":
                    triggered, metrics = _coordination(atoms, opts, resolved)
                elif name == "crystal_like":
                    order = compute_bond_order([atoms], cutoff=crystal_cutoff,
                        qbar6_threshold=opts["qbar6_threshold"],
                        min_neighbors=opts["min_neighbors"])["per_structure"][0]
                    metrics = {key: order[key] for key in (
                        "ordered_count", "ordered_fraction", "largest_cluster_size",
                        "largest_cluster_fraction", "q6_mean", "qbar6_mean")}
                    metrics["cutoff"] = crystal_cutoff
                    triggered = (metrics["ordered_fraction"] > opts["max_fraction"]
                                 or metrics["largest_cluster_fraction"] > opts.get("max_cluster_fraction", 1.0))
                elif name == "close_contacts":
                    triggered, metrics = _close_contacts(atoms, opts)
                elif name in ("density", "energy"):
                    if name == "density":
                        volume = float(atoms.get_volume())
                        if not np.isfinite(volume) or volume <= 0:
                            raise ValueError("density requires a finite positive cell volume")
                        value = _number(compute_density_gcm3(atoms), "density")
                    else:
                        value = _stored_energy(atoms)
                    metrics = {"value": value, "units": "g/cm3" if name == "density" else "eV/atom"}
                    triggered = value < opts.get("min", -np.inf) or value > opts.get("max", np.inf)
                else:
                    triggered, metrics = _unconverged(atoms)
                status = "labelled" if triggered else "passed"
            except (ValueError, TypeError, RuntimeError, FloatingPointError) as exc:
                triggered, status, metrics = None, "unavailable", {"reason": str(exc)}
            excluded = opts["exclude"] and status != "passed"
            row["screens"][name] = {"status": status, "triggered": triggered,
                                     "excluded": excluded, "metrics": metrics}
            if status != "passed":
                row["labels"].append(f"{name}_unavailable" if status == "unavailable" else name)
            row["excluded"] = row["excluded"] or excluded
        row["passed"] = not row["labels"]
        result["per_structure"].append(row)
    _update_summary(result)
    return result


def mark_screening_analysed(result, indices):
    """Mark retained original input indices analysed after successful analysis."""
    indices = list(indices)
    rows = result["per_structure"]
    for index in indices:
        if (isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral)
                or not 0 <= index < len(rows)):
            raise ValueError("analysed indices must be valid original structure indices")
        if rows[index]["excluded"]:
            raise ValueError("excluded structures cannot be marked analysed")
    for index in indices:
        rows[index]["analysed"] = True
    _update_summary(result)
    return result


def format_screening_report(result):
    """Render population accounting; columns overlap rather than form a funnel."""
    summary = result["summary"]
    keys = ("generated", "passed", "labelled", "analysed", "excluded")
    lines = ["Screening", "  " + "  ".join(f"{key:>10}" for key in keys),
             "  " + "  ".join(f"{summary[key]:>10}" for key in keys),
             "Generated: supplied candidates; upstream generation failures are not included.",
             "Passed: no labels; labelled: at least one diagnostic or unavailable result.",
             "Analysed: retained structures whose requested analysis completed."]
    for row in result["per_structure"]:
        if row["labels"]:
            source = row["source"] or f"structure {row['index']}"
            lines.append(f"  {source}: {', '.join(row['labels'])}"
                         f" ({'excluded' if row['excluded'] else 'retained'})")
    return "\n".join(lines)


def write_screening_outputs(result, prefix):
    """Write ``.json``, ``_structures.csv`` and ``_summary.csv``; return paths."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {"json": Path(f"{prefix}.json"),
             "structures": Path(f"{prefix}_structures.csv"),
             "summary": Path(f"{prefix}_summary.csv")}
    serialized = json.dumps(result, indent=2, allow_nan=False)
    paths["json"].write_text(serialized + "\n", encoding="utf-8")
    with paths["structures"].open("w", encoding="utf-8", newline="") as handle:
        fields = ["index", "source", "generated", "passed", "labelled", "analysed", "excluded", "labels"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["per_structure"]:
            writer.writerow({"index": row["index"], "source": row["source"],
                             "generated": True, "passed": row["passed"],
                             "labelled": bool(row["labels"]), "analysed": row["analysed"],
                             "excluded": row["excluded"], "labels": ";".join(row["labels"])})
    with paths["summary"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["summary"]))
        writer.writeheader()
        writer.writerow(result["summary"])
    return {name: str(path) for name, path in paths.items()}
