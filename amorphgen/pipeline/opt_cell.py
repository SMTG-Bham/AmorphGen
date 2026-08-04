"""
amorphgen.pipeline.opt_cell
----------------------------
Stage 1 – Structural optimisation of the crystalline input cell.
Stage 7 – Final optimisation of the quenched amorphous structure.

Supported optimisers (set via cfg["opt"]["optimizer"]):
  "LBFGS"          – default, fast quasi-Newton (recommended)
  "FIRE"           – molecular dynamics based, good for difficult cases
  "BFGSLineSearch" – robust line-search BFGS
  "BFGS"           – classic BFGS
  "MDMin"          – simple MD minimiser
"""

from __future__ import annotations

import importlib
import os
from copy import deepcopy

from ase.io import read, write
from ase.filters import UnitCellFilter
from ase.geometry import cell_to_cellpar

from ..utils import get_calculator, merge_config
from ..utils.common import assert_finite
from ..configs import DEFAULT_CONFIG

OPTIMIZERS = {
    "LBFGS":          ("ase.optimize", "LBFGS"),
    "FIRE":           ("ase.optimize", "FIRE"),
    "BFGSLineSearch": ("ase.optimize", "BFGSLineSearch"),
    "BFGS":           ("ase.optimize", "BFGS"),
    "MDMin":          ("ase.optimize", "MDMin"),
}

# Map --format choices to ASE write format strings and file extensions
FORMAT_MAP = {
    "extxyz": ("extxyz", ".xyz"),
    "vasp":   ("vasp",   ".vasp"),
    "cif":    ("cif",    ".cif"),
}


def _get_optimizer(name: str):
    """Import and return an ASE optimizer class by name."""
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from: {', '.join(OPTIMIZERS)}")
    module_path, cls_name = OPTIMIZERS[name]
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def _log(msg, lf=None):
    print(msg)
    if lf is not None:
        lf.write(msg + "\n")
        lf.flush()


def run(atoms_or_file, cfg_override=None, calc=None, stage_key="opt", **kwargs):
    """
    Optimise a structure using a chosen optimizer + cell filter.

    Parameters
    ----------
    atoms_or_file : str or ase.Atoms
    cfg_override : dict, optional
    calc : ASE calculator, optional
    stage_key : str
        Config section to read ("opt").

    Returns
    -------
    ase.Atoms
    """
    global_cfg = merge_config(DEFAULT_CONFIG, cfg_override)
    cfg = global_cfg.get(stage_key, global_cfg["opt"])

    if isinstance(atoms_or_file, str):
        atoms = read(atoms_or_file)
        input_path = atoms_or_file
        print(f"[Opt] Loaded from {atoms_or_file}")
    else:
        atoms = deepcopy(atoms_or_file)
        input_path = None
        print("[Opt] Using provided Atoms object")

    if calc is None:
        from ..utils.common import resolve_device
        device = resolve_device(global_cfg.get("device", "cuda"))
        calc = get_calculator(
            model=global_cfg.get("model", "mace-mpa-0"),
            device=device,
            model_path=global_cfg.get("model_path"),
        )
    atoms.calc = calc

    formula = atoms.get_chemical_formula(mode="hill")
    n_atoms = len(atoms)
    opt_name = cfg.get("optimizer", "LBFGS")
    fmax = cfg.get("fmax", 0.01)
    max_steps = cfg.get("max_steps", 1000)

    # Derive log/traj filenames from input file or stage key
    if input_path is not None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        default_log = f"{base}_opt.log"
        default_traj = f"{base}_opt.traj"
    else:
        # Use stage_key to differentiate Stage 1 vs Stage 7
        stage_prefix = "stage1" if stage_key == "opt" else "stage7"
        default_log = f"{stage_prefix}_opt.log"
        default_traj = f"{stage_prefix}_opt.traj"

    logfile = cfg.get("logfile", default_log)
    trajfile = cfg.get("traj_file", default_traj)

    with open(logfile, "w") as lf:
        from ..utils.common import compute_density_gcm3
        cp = cell_to_cellpar(atoms.cell)
        a, b, c, al, be, ga = cp
        vol = atoms.get_volume()
        density = compute_density_gcm3(atoms)
        _log(f"\n  Composition: {formula} ({n_atoms} atoms)", lf)
        _log(f"  Initial cell: a={a:.4f}  b={b:.4f}  c={c:.4f}", lf)
        _log(f"  Volume: {vol:.2f} A^3  Density: {density:.2f} g/cm3", lf)
        _log(f"  Optimizer: {opt_name}  fmax={fmax}  max_steps={max_steps}", lf)

        OptimizerClass = _get_optimizer(opt_name)

        # Cell filter: "FrechetCellFilter" (default), "UnitCellFilter",
        #              "ExpCellFilter", "StrainFilter", "cubic",
        #              "none" (positions only — cell stays fixed)
        filter_name = cfg.get("cell_filter", "FrechetCellFilter")
        _log(f"  Cell filter: {filter_name}", lf)

        if filter_name == "none" or filter_name is None:
            # Positions only — cell stays fixed
            target = atoms
        elif filter_name == "cubic":
            # Keep cubic shape (a=b=c, 90 deg) but allow volume to change
            from ase.filters import ExpCellFilter
            target = ExpCellFilter(atoms, hydrostatic_strain=True)
            _log("  [cell] Cubic: isotropic volume only, shape fixed", lf)
        elif filter_name == "ExpCellFilter":
            from ase.filters import ExpCellFilter
            target = ExpCellFilter(atoms)
        elif filter_name == "StrainFilter":
            from ase.filters import StrainFilter
            target = StrainFilter(atoms)
        elif filter_name == "UnitCellFilter":
            target = UnitCellFilter(atoms)
        else:
            # Default: FrechetCellFilter (better convergence for non-cubic)
            from ase.filters import FrechetCellFilter
            target = FrechetCellFilter(atoms)

        optimizer = OptimizerClass(target, logfile=None, trajectory=trajfile)

        header = (f"\n  {'Step':>5}  {'Energy(eV)':>14}  {'Fmax(eV/A)':>11}  "
                  f"{'a(A)':>10}  {'b(A)':>10}  {'c(A)':>10}  {'Vol(A3)':>10}")
        sep = "  " + "-" * 85
        _log(header, lf)
        _log(sep, lf)

        for step in range(max_steps):
            optimizer.step()
            energy = atoms.get_potential_energy()
            forces = target.get_forces()
            # Eager divergence guard: stop before a NaN/Inf is written to disk.
            assert_finite(atoms, context=f"optimisation of {formula}",
                          step=step + 1)
            max_f = float((forces ** 2).sum(axis=1).max() ** 0.5)
            cp = cell_to_cellpar(atoms.cell)
            a, b, c = cp[:3]
            vol = atoms.get_volume()
            line = (f"  {step+1:5d}  {energy:14.6f}  {max_f:11.6f}  "
                    f"{a:10.6f}  {b:10.6f}  {c:10.6f}  {vol:10.4f}")
            _log(line, lf)
            if max_f < fmax:
                _log(sep, lf)
                _log(f"\n  Converged after {step+1} steps!  Fmax = {max_f:.6f} eV/A", lf)
                break
        else:
            _log(sep, lf)
            _log(f"\n  WARNING: did not converge in {max_steps} steps.", lf)

    # ── Write output files ────────────────────────────────────────────────────
    # Derive base name from input file (if provided) for unique outputs
    if input_path is not None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        default_cif = f"{base}_opt.cif"
        default_xyz = f"{base}_opt.xyz"
    else:
        stage_prefix = "stage1" if stage_key == "opt" else "stage7"
        default_cif = f"{stage_prefix}_opt.cif"
        default_xyz = f"{stage_prefix}_opt.xyz"

    out_cif = cfg.get("output_cif", default_cif)
    out_xyz = cfg.get("output_xyz", default_xyz)
    write(out_cif, atoms)
    write(out_xyz, atoms, format="extxyz")
    final_density = compute_density_gcm3(atoms)
    print(f"[Opt] Final density: {final_density:.2f} g/cm3")
    print(f"[Opt] Saved -> {out_cif}, {out_xyz}")

    # Write additional format if requested via --format
    output_format = cfg.get("output_format", "extxyz")
    if output_format != "extxyz":
        fmt_str, fmt_ext = FORMAT_MAP.get(output_format, ("extxyz", ".xyz"))
        if input_path is not None:
            out_fmt = f"{base}_opt{fmt_ext}"
        else:
            out_fmt = f"{stage_prefix}_opt{fmt_ext}"
        # Don't overwrite if we already wrote this extension
        if out_fmt not in (out_cif, out_xyz):
            if fmt_str == "vasp":
                sorted_atoms = atoms[atoms.numbers.argsort()]
                write(out_fmt, sorted_atoms, format=fmt_str, sort=True)
            else:
                write(out_fmt, atoms, format=fmt_str)
            print(f"[Opt] Saved -> {out_fmt}")

    return atoms


def batch_optimize(
    input_dir: str,
    output_dir: str | None = None,
    cfg_override: dict | None = None,
    calc=None,
    pattern: str = "*.xyz",
    **kwargs,
) -> list[str]:
    """
    Optimise all structures in a directory using opt_cell.run().

    Parameters
    ----------
    input_dir : str
        Directory containing input structure files.
    output_dir : str, optional
        Directory for output files.  If None, a subdirectory
        ``input_dir + "_opt"`` is created.
    cfg_override : dict, optional
        Config overrides (passed to run()).
    calc : ASE calculator, optional
        Shared calculator.  If None, one is created from cfg.
    pattern : str
        Glob pattern for input files (default: ``*.xyz``).
    **kwargs
        Forwarded to run().

    Returns
    -------
    list of str
        Paths to optimised output files (.xyz).
    """
    import glob as _glob

    files = sorted(_glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        # Try other common formats as fallback (.extxyz kept for back-compat)
        for fallback in ["*.extxyz", "*.vasp", "*.cif"]:
            files = sorted(_glob.glob(os.path.join(input_dir, fallback)))
            if files:
                break
    if not files:
        print(f"[BatchOpt] No structure files found in {input_dir}/")
        print(f"  Searched: {pattern}, *.extxyz, *.vasp, *.cif")
        return []

    if output_dir is None:
        output_dir = input_dir.rstrip("/") + "_opt"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 65}")
    print(f"  AmorphGen - Batch Optimisation")
    print(f"  Input:  {input_dir}/  ({len(files)} structures)")
    print(f"  Output: {output_dir}/")
    print(f"{'=' * 65}\n")

    orig_dir = os.getcwd()
    os.chdir(output_dir)

    output_paths = []
    try:
        for i, fpath in enumerate(files):
            abs_path = os.path.join(orig_dir, fpath) if not os.path.isabs(fpath) else fpath
            print(f"\n  [{i+1}/{len(files)}] {os.path.basename(fpath)}")
            print(f"  {'-' * 60}")
            atoms = run(abs_path, cfg_override=cfg_override, calc=calc, **kwargs)
            output_paths.append(
                os.path.join(output_dir, os.path.splitext(os.path.basename(fpath))[0] + "_opt.xyz")
            )
    finally:
        os.chdir(orig_dir)

    print(f"\n{'=' * 65}")
    print(f"  Batch optimisation complete - {len(output_paths)}/{len(files)} structures")
    print(f"  Output: {output_dir}/")
    print(f"{'=' * 65}\n")

    return output_paths
