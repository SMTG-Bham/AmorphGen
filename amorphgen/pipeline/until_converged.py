"""Generate independent random structures in torch-sim batches until precise.

The generating procedure, descriptor definitions and support bounds are fixed
before sampling. Every generated structure contributes, or the whole batch
fails. Confidence sequences control error over all components and all sample
sizes; a resource cap is never reported as statistical convergence.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers
from ase.io import read, write

from ..analysis import StructureAnalyser
from ..analysis.sequential import sequential_convergence_report
from ..configs.default_config import DEFAULT_CONFIG
from ..utils.common import merge_config
from ..utils.repulsion import validate_repulsive_core_config
from ..utils.run_lock import run_lock
from ..utils.run_provenance import calculator_provenance
from ..utils.safety import validate_safety_config
from ..utils.torchsim_engine import build_model, batch_relax
from .random_gen import (
    generate_random, _GENERATION_DEFAULTS, _derive_structure_seed,
    _resume_value, _write_random_metadata,
)


CHECKPOINT = "adaptive_convergence.json"
_SCHEMA = "amorphgen.adaptive_random.v1"


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positive(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _model_identity(model, cfg):
    identity = calculator_provenance(cfg, model)
    identity["model"].pop("hash_unavailable_reason", None)
    if (identity["model"]["sha256"] is None
            and str(cfg["model"]).lower() not in {"lj", "lennard-jones"}):
        raise ValueError("Sequential production requires a fingerprint of loaded model weights")
    # Native LJ has no weights; its complete parameters are in the contract.
    return identity


def _source_identity():
    root = Path(__file__).resolve().parents[1]
    paths = ["pipeline/until_converged.py", "pipeline/random_gen.py",
             "utils/torchsim_engine.py", "utils/radii.py", "utils/safety.py",
             "utils/repulsion.py", "analysis/analyser.py", "analysis/structure.py",
             "analysis/energy.py", "analysis/cutoff.py", "analysis/uncertainty.py",
             "analysis/sequential.py"]
    versions = {}
    for package in ("numpy", "scipy", "ase", "torch", "torch-sim-atomistic",
                    "mace-torch", "sevenn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"sources": {name: _hash_file(root / name) for name in paths}, "versions": versions}


def _model_files(cfg):
    paths = [cfg.get("model_path")]
    reference = (cfg.get("safety") or {}).get("reference")
    if isinstance(reference, dict):
        paths.append(reference.get("model_path"))
    return {str(Path(path).resolve()): _hash_file(path) for path in paths if path is not None}


def _validate_targets(targets, composition, cutoff, confidence):
    sequential_convergence_report({name: [] for name in targets}, targets, confidence)
    structural = False
    for name, target in targets.items():
        if target.get("components", 1) != 1:
            raise ValueError("The generation controller currently accepts scalar descriptors only")
        if name in {"density", "energy.per_atom", "energy.total"}:
            continue
        prefix, separator, suffix = name.partition(".")
        count = {"coordination": 2, "total_coordination": 1,
                 "bond_distance": 2, "bond_angle": 3}.get(prefix)
        symbols = suffix.split("-")
        if (not separator or count is None or len(symbols) != count
                or any(symbol not in composition for symbol in symbols)):
            raise ValueError(f"Unsupported sequential descriptor {name!r}; use density, "
                             "energy.per_atom, energy.total, coordination.X-Y, "
                             "total_coordination.X, bond_distance.X-Y or bond_angle.X-Y-Z")
        structural = True
    if structural:
        _positive(cutoff, "A fixed numeric cutoff for sequential structural descriptors")
    elif cutoff is not None:
        _positive(cutoff, "cutoff")


def _observations(atoms, targets, cutoff):
    if (not len(atoms) or not np.isfinite(atoms.positions).all()
            or not np.isfinite(atoms.cell.array).all() or atoms.get_volume() <= 0):
        raise ValueError("Sequential evidence requires finite structures with positive volume")
    analyser = StructureAnalyser([atoms], cutoff=cutoff if cutoff is not None else 1.0)
    families = {}
    result = {}
    for name in targets:
        if name == "density":
            value = analyser.density()["per_structure"][0]
        elif name.startswith("energy."):
            value = float(atoms.get_potential_energy())
            if name == "energy.per_atom":
                value /= len(atoms)
        else:
            family, key = name.split(".", 1)
            # Distances and the two outer angle species are unordered in the
            # native analyser. Coordination remains directional.
            if family == "bond_distance":
                key = "-".join(sorted(key.split("-")))
            elif family == "bond_angle":
                first, center, last = key.split("-")
                first, last = sorted((first, last))
                key = f"{first}-{center}-{last}"
            if family not in families:
                method = {"coordination": analyser.coordination,
                          "total_coordination": analyser.total_coordination,
                          "bond_distance": analyser.bond_distances,
                          "bond_angle": analyser.bond_angles}[family]
                families[family] = method()
            row = families[family].get(key)
            if row is None:
                raise ValueError(f"Descriptor {name!r} is undefined for a generated structure")
            value = row["per_structure"][0]
        if value is None or not math.isfinite(value):
            raise ValueError(f"Descriptor {name!r} is missing or nonfinite; no structure may be dropped")
        result[name] = float(value)
    return result


def _report(structures, targets, confidence):
    return sequential_convergence_report(
        {name: [item["observations"][name] for item in structures] for name in targets},
        targets, confidence)


def _check_relaxed(atoms, opt):
    forces = np.asarray(atoms.get_forces(apply_constraint=False))
    if not np.isfinite(forces).all() or np.max(np.linalg.norm(forces, axis=1)) >= opt["fmax"]:
        raise RuntimeError("A torch-sim structure did not reach the force tolerance; batch not accepted")
    if str(opt.get("cell_filter") or "none").lower() != "none":
        stress = atoms.get_stress(voigt=False)
        pressure = abs(float(np.trace(stress)) / 3) * 160.21766208
        if not math.isfinite(pressure) or pressure >= opt["pressure_tol_gpa"]:
            raise RuntimeError("A torch-sim structure did not reach the pressure tolerance; batch not accepted")


def _save(path, state):
    state["state_sha256"] = _digest({key: value for key, value in state.items() if key != "state_sha256"})
    _write_random_metadata(str(path), state)


def _write_atoms(path, atoms):
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("Refusing symlink sequential structure output")
    path.parent.mkdir(exist_ok=True)
    temporary = path.with_name("." + path.name + ".pending")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    write(temporary, atoms, format="extxyz")
    temporary.replace(path)


def _restore(path, contract, seeds, targets, confidence, cutoff, batch_size, cap):
    if path.is_symlink():
        raise ValueError("Cannot resume a symlink checkpoint")
    state = json.loads(path.read_text())
    if not isinstance(state, dict) or state.get("schema") != _SCHEMA:
        raise ValueError("Invalid adaptive convergence checkpoint")
    digest = _digest({k: v for k, v in state.items() if k != "state_sha256"})
    if state.get("state_sha256") != digest:
        raise ValueError("Adaptive checkpoint integrity checksum mismatch")
    if state.get("contract") != contract or state.get("contract_sha256") != _digest(contract):
        raise ValueError("Cannot resume: frozen generation, targets, model or software changed")
    structures = state.get("structures")
    if not isinstance(structures, list) or state.get("n_structures") != len(structures):
        raise ValueError("Adaptive checkpoint structure count is inconsistent")
    if len(structures) > cap or (len(structures) != cap and len(structures) % batch_size):
        raise ValueError("Adaptive checkpoint contains an incomplete or oversized batch")
    for index, item in enumerate(structures):
        if item.get("index") != index or item.get("seed") != seeds[index]:
            raise ValueError("Adaptive checkpoint must retain the original contiguous seed prefix")
        for kind, expected in (("initial", f"random_initial/random_{index:04d}.xyz"),
                               ("final", f"random_opt/random_{index:04d}_opt.xyz")):
            artifact = path.parent / expected
            if item.get(kind + "_file") != expected or artifact.is_symlink() or artifact.parent.is_symlink():
                raise ValueError("Invalid adaptive checkpoint structure path")
            if not artifact.is_file() or _hash_file(artifact) != item.get(kind + "_sha256"):
                raise ValueError(f"Adaptive structure integrity mismatch: {expected}")
        atoms = read(path.parent / item["final_file"])
        if (atoms.info.get("sequential_index") != index or atoms.info.get("sequential_seed") != seeds[index]
                or _observations(atoms, targets, cutoff) != item.get("observations")):
            raise ValueError("Adaptive checkpoint observations or identities disagree with structures")
    if state.get("convergence") != _report(structures, targets, confidence):
        raise ValueError("Adaptive checkpoint confidence sequence disagrees with observations")
    return state


def run_until_converged(composition, output_dir, *, targets, batch_size=8,
                        min_structures=2, max_structures=1000, confidence=.95,
                        cutoff=None, generation=None, cfg_override=None, resume=False):
    """Run random placement and torch-sim relaxation to declared precision.

    Supports the native scalar density, energy, coordination, distance and
    angle descriptors. ``targets`` maps fixed names to ``bounds``, ``tolerance``
    and optional ``components=1``. ``generation`` holds generate_random settings
    including an explicit base ``seed``; independent index seeds use the same
    derivation as ordinary random generation. Settings are freshly copied for
    each draw: no batch-level retry ladder, resampling, filtering or discarded
    failures. A failed batch can only replay its original seed prefix.

    Outputs use the standard random_initial/random_opt layout (extxyz) and an
    atomic adaptive_convergence.json ledger. Resume requires the same complete
    settings, software, model and verified initial/final structure artifacts.
    """
    batch_size = _integer(batch_size, "batch_size")
    min_structures = _integer(min_structures, "min_structures", 2)
    max_structures = _integer(max_structures, "max_structures", min_structures)
    if not isinstance(composition, Mapping) or not composition:
        raise ValueError("composition must be a nonempty mapping")
    for symbol, count in composition.items():
        if symbol not in atomic_numbers:
            raise ValueError(f"Unknown composition species {symbol!r}")
        _integer(count, f"composition.{symbol}")
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("targets must be a nonempty mapping")
    targets = copy.deepcopy(dict(targets))
    _validate_targets(targets, composition, cutoff, confidence)
    if not isinstance(generation, Mapping):
        raise ValueError("generation must specify an explicit nonnegative seed")
    supplied = copy.deepcopy(dict(generation))
    if set(supplied) - (set(_GENERATION_DEFAULTS) - {"_expand_attempt", "_soft_pack"}):
        raise ValueError("Unsupported or private random generation setting")
    seed = _integer(supplied.get("seed"), "generation.seed", 0)
    generation = {**_GENERATION_DEFAULTS, **supplied}
    seeds = [_derive_structure_seed(seed, index, 0) for index in range(max_structures)]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Derived structure seeds collide within max_structures; choose another base seed")
    cfg = merge_config(DEFAULT_CONFIG, cfg_override or {})
    if (cfg_override or {}).get("engine", "torchsim") != "torchsim":
        raise ValueError("Until-converged production requires engine='torchsim'")
    cfg["engine"] = "torchsim"
    cfg["safety"] = validate_safety_config(cfg.get("safety"))
    cfg["repulsive_core"] = validate_repulsive_core_config(cfg.get("repulsive_core"))
    opt = {"fmax": .05, "max_steps": 1000, "cell_filter": "cubic",
           "optimizer": "lbfgs", "pressure_tol_gpa": .02, **cfg.get("opt", {})}
    opt["fmax"] = _positive(opt["fmax"], "opt.fmax")
    opt["max_steps"] = _integer(opt["max_steps"], "opt.max_steps")
    opt["pressure_tol_gpa"] = _positive(opt["pressure_tol_gpa"], "opt.pressure_tol_gpa")
    chunk_size = opt.get("batch_size")
    if chunk_size is None or chunk_size == "auto":
        chunk_size = batch_size
    if isinstance(chunk_size, str) and chunk_size.isdigit():
        chunk_size = int(chunk_size)
    chunk_size = _integer(chunk_size, "opt.batch_size")
    cfg["opt"] = opt
    dtype = cfg.get("default_dtype") or "auto"
    dtype = "float64" if dtype == "auto" else dtype
    if dtype not in {"float32", "float64"}:
        raise ValueError("default_dtype must be auto, float32 or float64")
    output_dir = Path(output_dir).resolve()
    with run_lock(output_dir):
        path = output_dir / CHECKPOINT
        exists = path.exists() or path.is_symlink()
        if resume and not exists:
            raise ValueError("Cannot resume without adaptive_convergence.json")
        if not resume and any(p.name != ".amorphgen.lock" for p in output_dir.iterdir()):
            raise ValueError("Fresh sequential production requires an empty output directory; use --resume")
        model_files = _model_files(cfg)
        model = build_model(cfg["model"], device=cfg["device"], model_path=cfg.get("model_path"),
                            classical_params=cfg.get("classical_params"), dtype=dtype)
        model.eval()
        if _model_files(cfg) != model_files:
            raise ValueError("Model files changed during loading")
        contract = _resume_value({"composition": dict(composition), "composition_order": list(composition),
            "generation": generation, "cfg": cfg, "targets": targets, "cutoff": cutoff,
            "confidence": confidence, "batch_size": batch_size, "min_structures": min_structures,
            "max_structures": max_structures, "model": _model_identity(model, cfg),
            "model_files": model_files, "software": _source_identity(),
            "estimand": "equal_structure_mean_of_fixed_random_placement_and_relaxation"})
        if exists:
            state = _restore(path, contract, seeds, targets, confidence, cutoff, batch_size, max_structures)
        else:
            state = {"schema": _SCHEMA, "contract": contract, "contract_sha256": _digest(contract),
                     "structures": [], "n_structures": 0, "history": [], "status": "running",
                     "converged": False, "convergence": _report([], targets, confidence)}
            _save(path, state)
        while True:
            n = len(state["structures"])
            met = n >= min_structures and state["convergence"]["converged"]
            if met or n >= max_structures:
                state.update(status="converged" if met else "max_structures_reached",
                             converged=bool(met), stopping_reason="precision_met" if met else "max_structures_reached")
                state.pop("error", None)
                _save(path, state)
                return state
            indices = list(range(n, min(n + batch_size, max_structures)))
            state.update(status="running", converged=False)
            state.pop("error", None)
            state.pop("stopping_reason", None)
            try:
                from ..utils.preemption import stop_if_requested
                stop_if_requested()
                initial = []
                for index in indices:
                    settings = copy.deepcopy(generation)
                    settings["seed"] = seeds[index]
                    atoms = generate_random(dict(composition), **settings)
                    atoms.info.update(sequential_index=index, sequential_seed=seeds[index])
                    initial_file = output_dir / f"random_initial/random_{index:04d}.xyz"
                    _write_atoms(initial_file, atoms)
                    initial.append(read(initial_file))
                relaxed = []
                for start in range(0, len(initial), chunk_size):
                    chunk = initial[start:start + chunk_size]
                    part = batch_relax(chunk, model, fmax=opt["fmax"], max_steps=opt["max_steps"],
                                       cell_filter=opt["cell_filter"], optimizer=opt["optimizer"],
                                       pressure_tol_gpa=opt["pressure_tol_gpa"], autobatch=False,
                                       safety=cfg["safety"], repulsive_core=cfg["repulsive_core"])
                    if len(part) != len(chunk):
                        raise RuntimeError("torch-sim returned an incomplete relaxation batch")
                    relaxed.extend(part)
                pending = []
                for index, atoms in zip(indices, relaxed):
                    if (atoms.info.get("sequential_index") != index
                            or atoms.info.get("sequential_seed") != seeds[index]):
                        raise RuntimeError("torch-sim changed structure identity/order in the batch")
                    _check_relaxed(atoms, opt)
                    initial_file = f"random_initial/random_{index:04d}.xyz"
                    final_file = f"random_opt/random_{index:04d}_opt.xyz"
                    _write_atoms(output_dir / final_file, atoms)
                    observed = _observations(read(output_dir / final_file), targets, cutoff)
                    pending.append({"index": index, "seed": seeds[index], "initial_file": initial_file,
                                    "initial_sha256": _hash_file(output_dir / initial_file),
                                    "final_file": final_file, "final_sha256": _hash_file(output_dir / final_file),
                                    "observations": observed})
                structures = state["structures"] + pending
                convergence = _report(structures, targets, confidence)
                state.update(structures=structures, n_structures=len(structures), convergence=convergence)
                met = len(structures) >= min_structures and convergence["converged"]
                state["history"].append({"n_structures": len(structures), "converged": bool(met)})
                _save(path, state)
                print(f"[convergence] {len(structures)} structures: "
                      f"{'precision met' if met else 'continue'} (sequentially valid)")
            except Exception as exc:
                state.update(status="failed", converged=False, stopping_reason="batch_failed",
                             error=f"{type(exc).__name__}: {exc}")
                _save(path, state)
                raise
