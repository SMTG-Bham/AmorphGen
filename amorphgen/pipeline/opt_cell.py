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

        # Every cell filter relaxes the cell, which needs a stress tensor.
        # Guard classical (stress-less) calculators with a clear error.
        if filter_name not in ("none", None):
            from ..utils.common import require_stress
            require_stress(calc, f"Cell-filter optimisation (cell_filter={filter_name!r})")

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
            # The manual step() loop bypasses ASE's irun(), so fire the
            # observers ourselves or the .traj file is never written.
            optimizer.nsteps += 1
            optimizer.call_observers()
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
    engine: str = "ase",
    indices=None,
    **kwargs,
) -> list[str]:
    """
    Optimise all structures in a directory using opt_cell.run().

    ``engine="torchsim"`` relaxes every structure in ONE batched call through
    torch-sim (optional extra ``amorphgen[torchsim]``; MACE / SevenNet / LJ,
    CUDA or CPU) instead of one after another through ASE. Output files and
    names are identical to the ASE path.

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

    if indices:
        # keep files whose stem ends in a number inside the selection
        import re as _re
        from ..utils.common import parse_index_spec
        sel = parse_index_spec(indices)
        keep = []
        for f in files:
            m = _re.search(r"(\d+)(?:_opt)?$", os.path.splitext(os.path.basename(f))[0])
            if m and int(m.group(1)) in sel:
                keep.append(f)
        print(f"[BatchOpt] index selection {indices}: {len(keep)} of {len(files)} files")
        files = keep
        if not files:
            return []
    if output_dir is None:
        output_dir = input_dir.rstrip("/") + "_opt"
    os.makedirs(output_dir, exist_ok=True)
    if str(engine).lower() == "torchsim":
        return _batch_optimize_torchsim(files, output_dir, cfg_override or {}, **kwargs)

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


def _batch_optimize_torchsim(files, output_dir, cfg, **kwargs):
    """Batched torch-sim relaxation of *files*; same outputs as the ASE path.

    Runs in chunks of ``batch_size`` structures (``opt: batch_size``, default
    16) and writes each chunk's outputs before starting the next, so a
    walltime kill loses at most one chunk. With ``resume=True`` inputs whose
    ``<stem>_opt.<ext>`` already exists are skipped.
    """
    from ase.io import read, write
    from ..utils.torchsim_engine import build_model, batch_relax
    from ..utils.common import compute_density_gcm3, merge_config
    from ..configs.default_config import DEFAULT_CONFIG
    full = merge_config(DEFAULT_CONFIG, cfg)
    stage_key = kwargs.get("stage_key", "opt")
    ocfg = full.get(stage_key, full["opt"])
    fmax = kwargs.get("fmax", ocfg.get("fmax", 0.01))
    max_steps = kwargs.get("max_steps", ocfg.get("max_steps", 1000))
    cell_filter = kwargs.get("cell_filter", ocfg.get("cell_filter", "cubic"))
    optimizer = kwargs.get("optimizer", ocfg.get("optimizer", "LBFGS"))
    pressure_tol = kwargs.get("pressure_tol_gpa", ocfg.get("pressure_tol_gpa", 0.02))
    batch_size = kwargs.get("batch_size") or ocfg.get("batch_size") or "auto"
    resume = bool(kwargs.get("resume", False))
    out_fmt = ocfg.get("output_format", "xyz")
    ext = {"xyz": ".xyz", "extxyz": ".xyz", "vasp": ".vasp", "cif": ".cif"}.get(out_fmt, ".xyz")
    ase_fmt = {"xyz": "extxyz", "extxyz": "extxyz", "vasp": "vasp", "cif": "cif"}.get(out_fmt, "extxyz")
    dtype = full.get("default_dtype")
    dtype = "float64" if dtype in (None, "auto") else dtype
    print(f"\n{'=' * 65}\n  AmorphGen - Batch Optimisation (torch-sim engine)\n"
          f"  Input: {len(files)} structures  Output: {output_dir}/\n{'=' * 65}")
    def _dest(f):
        return os.path.join(output_dir, f"{os.path.splitext(os.path.basename(f))[0]}_opt{ext}")

    todo = list(files)
    done_paths = []
    if resume:
        todo = [f for f in files if not os.path.exists(_dest(f))]
        done_paths = [_dest(f) for f in files if os.path.exists(_dest(f))]
        if done_paths:
            print(f"  [Resume] {len(done_paths)} already relaxed, {len(todo)} to do")
    if not todo:
        return done_paths

    model = build_model(full.get("model", "mace-mpa-0"), device=full.get("device", "auto"),
                        model_path=full.get("model_path"),
                        classical_params=full.get("classical_params"), dtype=dtype)
    paths = list(done_paths)
    if str(batch_size).lower() == "auto":
        from ..utils.torchsim_engine import estimate_batch_size
        batch_size = estimate_batch_size(model, [read(f) for f in todo[:8]], fraction=0.4, fallback=16)
    batch_size = int(batch_size)
    n_chunks = (len(todo) + batch_size - 1) // batch_size
    print(f"  [torch-sim] batch size {batch_size} -> {n_chunks} chunk(s) for {len(todo)} structures")

    def _free_gpu():
        import gc
        gc.collect()                      # drop tensors held by dead frames first
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _is_oom(exc):
        return "out of memory" in str(exc).lower() or "OutOfMemoryError" in str(exc)

    def _relax_chunk(chunk):
        """Relax *chunk*; on a GPU out-of-memory error split it in half and retry.

        The retry happens OUTSIDE the except block: inside it the traceback
        keeps the failed attempt's tensors alive and the retry inherits a full
        GPU. torch-sim's autobatcher is off because the chunking is done here
        (its memory-estimation probe is itself the largest allocation).
        """
        _free_gpu()
        oom = False
        try:
            return batch_relax([read(f) for f in chunk], model, fmax=fmax, max_steps=max_steps,
                               cell_filter=cell_filter, optimizer=optimizer,
                               pressure_tol_gpa=pressure_tol, autobatch=False)
        except RuntimeError as exc:
            if not _is_oom(exc) or len(chunk) == 1:
                raise
            oom = True
        if oom:
            _free_gpu()
            half = len(chunk) // 2
            print(f"  [torch-sim] GPU out of memory with {len(chunk)} structures; "
                  f"retrying as {half} + {len(chunk) - half}")
            first = _relax_chunk(chunk[:half])
            _free_gpu()
            return first + _relax_chunk(chunk[half:])

    for ci in range(n_chunks):
        chunk = todo[ci * batch_size:(ci + 1) * batch_size]
        if n_chunks > 1:
            print(f"  [torch-sim] chunk {ci + 1}/{n_chunks}: {len(chunk)} structures")
        relaxed = _relax_chunk(chunk)
        paths.extend(_write_torchsim_outputs(chunk, relaxed, output_dir, ext, ase_fmt))
    _free_gpu()
    return paths


def _write_torchsim_outputs(files, relaxed, output_dir, ext, ase_fmt):
    from ase.io import write
    from ..utils.common import compute_density_gcm3
    paths = []
    for f, a in zip(files, relaxed):
        stem = os.path.splitext(os.path.basename(f))[0]
        dest = os.path.join(output_dir, f"{stem}_opt{ext}")
        if ase_fmt == "vasp":
            write(dest, a, format="vasp", sort=True, direct=True)
        else:
            write(dest, a, format=ase_fmt)
        if ase_fmt != "cif":               # same convenience copy the ASE path writes
            write(os.path.join(output_dir, f"{stem}_opt.cif"), a, format="cif")
        with open(os.path.join(output_dir, f"{stem}_opt.log"), "w") as lf:
            lf.write(f"torch-sim FIRE batch relaxation\nE = {a.get_potential_energy():.6f} eV  "
                     f"max|F| = {a.info.get('max_force', float('nan')):.4f} eV/A  "
                     f"density = {compute_density_gcm3(a):.3f} g/cm3  cell = {a.cell.lengths().round(4).tolist()}\n")
        print(f"  {os.path.basename(dest):32s} E/atom = {a.get_potential_energy()/len(a):10.4f} eV  "
              f"max|F| = {a.info.get('max_force', float('nan')):.3f}  rho = {compute_density_gcm3(a):.3f}")
        paths.append(dest)
    return paths
