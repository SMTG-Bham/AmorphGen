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
from copy import deepcopy
from collections import Counter

from ase.io import read, write
from ase.filters import UnitCellFilter
from ase.geometry import cell_to_cellpar

from ..utils import get_calculator, merge_config
from ..configs import DEFAULT_CONFIG

OPTIMIZERS = {
    "LBFGS":          ("ase.optimize", "LBFGS"),
    "FIRE":           ("ase.optimize", "FIRE"),
    "BFGSLineSearch": ("ase.optimize", "BFGSLineSearch"),
    "BFGS":           ("ase.optimize", "BFGS"),
    "MDMin":          ("ase.optimize", "MDMin"),
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
        print(f"[Opt] Loaded from {atoms_or_file}")
    else:
        atoms = deepcopy(atoms_or_file)
        print("[Opt] Using provided Atoms object")

    if calc is None:
        device = global_cfg.get("device", "cuda")
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
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
    logfile = cfg.get("logfile", "opt_stage.log")
    trajfile = cfg.get("traj_file", "opt_stage.traj")

    with open(logfile, "w") as lf:
        cp = cell_to_cellpar(atoms.cell)
        a, b, c, al, be, ga = cp
        vol = atoms.get_volume()
        _log(f"\n  Composition: {formula} ({n_atoms} atoms)", lf)
        _log(f"  Initial cell: a={a:.4f}  b={b:.4f}  c={c:.4f}", lf)
        _log(f"  Volume: {vol:.2f} A^3", lf)
        _log(f"  Optimizer: {opt_name}  fmax={fmax}  max_steps={max_steps}", lf)

        OptimizerClass = _get_optimizer(opt_name)

        # Cell filter: "UnitCellFilter" (default), "ExpCellFilter",
        #              "StrainFilter", "cubic", "none" (positions only)
        filter_name = cfg.get("cell_filter", "UnitCellFilter")
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
        else:
            target = UnitCellFilter(atoms)

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

    out_cif = cfg.get("output_cif", "opt_final.cif")
    out_xyz = cfg.get("output_xyz", "opt_final.xyz")
    write(out_cif, atoms)
    write(out_xyz, atoms, format="xyz")
    print(f"[Opt] Saved -> {out_cif}, {out_xyz}")
    return atoms
