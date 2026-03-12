"""
pipeline/opt_cell.py
--------------------
Stage 1 – Structural optimisation of the crystalline input cell.
Stage 7 – Final optimisation of the quenched amorphous structure.

Supported optimisers (set via cfg["opt"]["optimizer"]):
  "LBFGS"          – default, fast quasi-Newton (recommended)
  "FIRE"           – molecular dynamics based, good for difficult cases
  "BFGSLineSearch" – robust line-search BFGS
  "BFGS"           – classic BFGS
  "MDMin"          – simple MD minimiser

Cell filter options (set via cfg["opt"]["cell_filter"]):
  "UnitCellFilter"  – default, relaxes all cell DOF simultaneously
  "ExpCellFilter"   – better volume/shape convergence for soft cells
  "StrainFilter"    – only relaxes cell strain, not atomic positions
  "cubic"           – reshapes to cubic before optimisation, keeps angles=90 deg
"""

from copy import deepcopy
from collections import Counter

from ase.io import read, write
from ase.geometry import cell_to_cellpar

from ..utils import get_mace_calculator, merge_config
from ..configs import DEFAULT_CONFIG


# ── Supported optimisers ──────────────────────────────────────────────────────
OPTIMIZERS = {
    "LBFGS":          ("ase.optimize",      "LBFGS"),
    "FIRE":           ("ase.optimize",      "FIRE"),
    "BFGSLineSearch": ("ase.optimize",      "BFGSLineSearch"),
    "BFGS":           ("ase.optimize",      "BFGS"),
    "MDMin":          ("ase.optimize",      "MDMin"),
}


# ── Supported cell filters ────────────────────────────────────────────────────
CELL_FILTERS = {
    "UnitCellFilter": ("ase.filters", "UnitCellFilter"),
    "ExpCellFilter":  ("ase.filters", "ExpCellFilter"),
    "StrainFilter":   ("ase.filters", "StrainFilter"),
    "cubic":          None,  # handled specially
}


def _get_cell_filter(name: str):
    """Import and return an ASE cell filter class by name."""
    name = name.strip()
    if name not in CELL_FILTERS:
        raise ValueError(
            f"Unknown cell_filter '{name}'. "
            f"Choose from: {', '.join(CELL_FILTERS)}"
        )
    if CELL_FILTERS[name] is None:
        return None  # cubic handled separately
    import importlib
    module = importlib.import_module(CELL_FILTERS[name][0])
    return getattr(module, CELL_FILTERS[name][1])


def _make_cubic(atoms):
    """
    Reshape the cell to a cube with the same volume, then fix angles at 90 deg.
    Uses FixedAngles constraint so the optimiser only moves atoms + isotropic vol.
    """
    import numpy as np
    vol  = atoms.get_volume()
    a    = vol ** (1.0 / 3.0)
    atoms.set_cell([a, a, a, 90, 90, 90], scale_atoms=True)
    try:
        from ase.constraints import FixedAngles
        existing = list(atoms.constraints)
        atoms.set_constraint(existing + [FixedAngles(atoms)])
        print("  [cell] Reshaped to cubic + FixedAngles constraint applied.")
    except ImportError:
        print("  [cell] Reshaped to cubic (FixedAngles not available — "
              "angles may drift slightly).")
    return atoms


def _get_optimizer(name: str):
    """Import and return an ASE optimizer class by name."""
    name = name.strip()
    if name not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{name}'. "
            f"Choose from: {', '.join(OPTIMIZERS)}"
        )
    module_path, cls_name = OPTIMIZERS[name]
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def run(atoms_or_file,
        stage_key: str = "opt",
        cfg_override: dict | None = None,
        calc=None) -> object:
    """
    Optimise a structure using a chosen optimizer + UnitCellFilter.

    Called for Stage 1 (stage_key="opt") and Stage 7 (stage_key="final_opt").

    Parameters
    ----------
    atoms_or_file : str or ase.Atoms
    stage_key     : "opt" for Stage 1, "final_opt" for Stage 7
    cfg_override  : dict, optional
    calc          : pre-built MACE calculator, optional

    Returns
    -------
    ase.Atoms  –  relaxed structure
    """
    global_cfg = merge_config(DEFAULT_CONFIG, cfg_override)
    cfg        = global_cfg[stage_key]

    stage_num  = "1" if stage_key == "opt" else "7"
    stage_name = ("Crystalline Optimisation"
                  if stage_key == "opt" else "Final Optimisation (amorphous)")

    opt_name   = cfg.get("optimizer", "LBFGS")
    OptimizerClass = _get_optimizer(opt_name)

    # ── load structure ─────────────────────────────────────────────────────────
    if isinstance(atoms_or_file, str):
        atoms = read(atoms_or_file)
        src   = atoms_or_file
    else:
        atoms = deepcopy(atoms_or_file)
        src   = "Atoms object"

    syms     = atoms.get_chemical_symbols()
    comp     = Counter(syms)
    comp_str = " ".join(f"{el}{n}" for el, n in sorted(comp.items()))

    def _log(msg, lf=None):
        print(msg)
        if lf:
            lf.write(msg + "\n")
            lf.flush()

    with open(cfg["logfile"], "w") as lf:

        _log("=" * 68, lf)
        _log(f"  amorphMD  –  Stage {stage_num}: {stage_name}", lf)
        _log("=" * 68, lf)
        _log(f"  Input        : {src}", lf)
        _log(f"  Atoms        : {len(atoms)}", lf)
        _log(f"  Composition  : {comp_str}", lf)
        _log(f"  Model        : {global_cfg['mace_model']}", lf)
        _log(f"  Device       : {global_cfg['device']}", lf)
        _log(f"  Optimizer    : {opt_name}", lf)
        _log(f"  fmax         : {cfg['fmax']} eV/Å", lf)
        _log(f"  max_steps    : {cfg['max_steps']}", lf)
        _log("", lf)

        # ── initial cell ───────────────────────────────────────────────────────
        _print_cell(atoms, "Initial", lf, _log)

        # ── calculator ────────────────────────────────────────────────────────
        if calc is None:
            calc = get_mace_calculator(
                model=global_cfg["mace_model"],
                device=global_cfg["device"],
                model_path=global_cfg.get("model_path"),
            )
        atoms.calc = calc

        # ── symmetry constraint ───────────────────────────────────────────────
        if cfg.get("fix_symmetry", False):
            from ase.constraints import FixSymmetry
            atoms.set_constraint(FixSymmetry(atoms))
            _log("  FixSymmetry constraint applied.", lf)
        else:
            atoms.set_constraint([])
            _log("  No symmetry constraint.", lf)

        # ── cell filter ───────────────────────────────────────────────────────
        filter_name = cfg.get("cell_filter", "UnitCellFilter")
        _log(f"  Cell filter  : {filter_name}", lf)

        if filter_name == "cubic":
            atoms = _make_cubic(atoms)
            from ase.filters import UnitCellFilter
            ucf = UnitCellFilter(atoms)
        else:
            FilterClass = _get_cell_filter(filter_name)
            ucf = FilterClass(atoms)

        # ── optimiser ─────────────────────────────────────────────────────────
        optimizer = OptimizerClass(ucf, logfile=None, trajectory=cfg["traj_file"])

        fmax      = cfg["fmax"]
        max_steps = cfg["max_steps"]

        header = (
            "\n  {:>5s}  {:>14s}  {:>11s}  {:>10s}  {:>10s}  {:>10s}"
            "  {:>8s}  {:>8s}  {:>8s}  {:>10s}".format(
                "Step", "Energy(eV)", "Fmax(eV/A)",
                "a(A)", "b(A)", "c(A)",
                "alpha", "beta", "gamma", "Vol(A3)"))
        sep = "  " + "-" * 110
        _log(header, lf)
        _log(sep, lf)

        energy = max_f = 0.0
        for step in range(max_steps):
            optimizer.step()

            energy = atoms.get_potential_energy()
            forces = ucf.get_forces()
            max_f  = float((forces ** 2).sum(axis=1).max() ** 0.5)
            cp     = cell_to_cellpar(atoms.cell)
            a, b, c, al, be, ga = cp
            vol    = atoms.get_volume()

            line = (
                "  {:5d}  {:14.6f}  {:11.6f}  {:10.6f}  {:10.6f}  {:10.6f}"
                "  {:8.4f}  {:8.4f}  {:8.4f}  {:10.4f}".format(
                    step + 1, energy, max_f, a, b, c, al, be, ga, vol))
            _log(line, lf)

            if max_f < fmax:
                _log(sep, lf)
                _log(f"\n  Converged after {step+1} steps!  "
                     f"Fmax = {max_f:.6f} eV/Å", lf)
                break
        else:
            _log(sep, lf)
            _log(f"\n  WARNING: did not converge in {max_steps} steps.", lf)

        # ── final summary ──────────────────────────────────────────────────────
        _log("", lf)
        _print_cell(atoms, "Final", lf, _log)
        _log(f"  Energy         = {energy:.6f} eV", lf)
        _log(f"  Energy/atom    = {energy/len(atoms):.6f} eV/atom", lf)
        _log(f"  Final Fmax     = {max_f:.6f} eV/Å", lf)

        # ── symmetry check ────────────────────────────────────────────────────
        try:
            from pymatgen.io.ase import AseAtomsAdaptor
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            pmg = AseAtomsAdaptor.get_structure(atoms)
            sga = SpacegroupAnalyzer(pmg, symprec=0.1)
            sg  = (f"{sga.get_space_group_symbol()} "
                   f"({sga.get_space_group_number()})")
            _log(f"\n  Space group: {sg}", lf)
        except Exception:
            pass

        # ── save ──────────────────────────────────────────────────────────────
        write(cfg["output_cif"], atoms)
        write(cfg["output_xyz"], atoms, format="extxyz")
        _log(f"\n  Saved → {cfg['output_cif']}  (CIF)", lf)
        _log(f"  Saved → {cfg['output_xyz']}  (extxyz)", lf)
        _log("=" * 68 + "\n", lf)

    return atoms


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_cell(atoms, label, lf, _log):
    cp = cell_to_cellpar(atoms.cell)
    a, b, c, al, be, ga = cp
    vol = atoms.get_volume()
    _log(f"  {label} cell parameters:", lf)
    _log(f"    a = {a:.6f} Å    b = {b:.6f} Å    c = {c:.6f} Å", lf)
    _log(f"    α = {al:.6f}°   β = {be:.6f}°   γ = {ga:.6f}°", lf)
    _log(f"    Volume = {vol:.4f} Å³", lf)
