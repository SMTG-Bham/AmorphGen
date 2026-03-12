"""
pipeline/final_opt.py
---------------------
Stage 6 – Final structural optimisation of the quenched amorphous structure.

Unlike Stage 1, FixSymmetry is OFF by default (amorphous material has no
crystallographic symmetry).  UnitCellFilter still allows cell relaxation.
"""

from ase.io import read, write
from ase.optimize import LBFGS
from ase.filters import UnitCellFilter
from ase.geometry import cell_to_cellpar

from ..utils import get_mace_calculator, merge_config
from ..configs import DEFAULT_CONFIG


def run(atoms_or_file, cfg_override: dict | None = None, calc=None) -> object:
    """
    Optimise the amorphous structure (no symmetry constraints).

    Parameters
    ----------
    atoms_or_file : str or ase.Atoms
    cfg_override  : dict, optional
    calc          : ASE calculator, optional

    Returns
    -------
    ase.Atoms
        Energy-minimised amorphous structure.
    """
    # ── config ───────────────────────────────────────────────────────────────
    global_cfg = merge_config(DEFAULT_CONFIG, cfg_override)
    cfg = global_cfg["final_opt"]
    mace_cfg = {"mace_model": global_cfg["mace_model"], "device": global_cfg["device"]}

    # ── load ─────────────────────────────────────────────────────────────────
    if isinstance(atoms_or_file, str):
        atoms = read(atoms_or_file)
        print(f"[Stage 6] Loaded from {atoms_or_file}")
    else:
        from copy import deepcopy
        atoms = deepcopy(atoms_or_file)
        print("[Stage 6] Using provided Atoms object")

    print(f"[Stage 6] {len(atoms)} atoms – amorphous optimisation")

    # ── calculator ────────────────────────────────────────────────────────────
    if calc is None:
        calc = get_mace_calculator(**mace_cfg)
    atoms.calc = calc

    # ── symmetry constraint (off for amorphous) ───────────────────────────────
    if cfg.get("fix_symmetry", False):
        from ase.constraints import FixSymmetry
        atoms.set_constraint(FixSymmetry(atoms))
        print("[Stage 6] FixSymmetry ON")
    else:
        atoms.set_constraint([])
        print("[Stage 6] No symmetry constraint (amorphous)")

    # ── optimiser ─────────────────────────────────────────────────────────────
    ucf = UnitCellFilter(atoms)
    optimizer = LBFGS(ucf,
                      logfile=cfg["logfile"],
                      trajectory=cfg["traj_file"])

    fmax = cfg["fmax"]
    max_steps = cfg["max_steps"]

    print(f"[Stage 6] LBFGS  fmax={fmax} eV/Å, max_steps={max_steps}")

    for step in range(max_steps):
        optimizer.step()
        energy = atoms.get_potential_energy()
        forces = ucf.get_forces()
        max_f = float((forces ** 2).sum(axis=1).max() ** 0.5)

        if (step + 1) % 10 == 0 or max_f < fmax:
            cp = cell_to_cellpar(atoms.cell)
            vol = atoms.get_volume()
            print(f"  step {step+1:4d}  E={energy:.4f} eV  "
                  f"Fmax={max_f:.4f} eV/Å  V={vol:.2f} Å³  "
                  f"a={cp[0]:.3f} b={cp[1]:.3f} c={cp[2]:.3f} Å")

        if max_f < fmax:
            print(f"[Stage 6] Converged after {step+1} steps!")
            break
    else:
        print(f"[Stage 6] WARNING: did not converge in {max_steps} steps.")

    # ── summary ───────────────────────────────────────────────────────────────
    n = len(atoms)
    cp = cell_to_cellpar(atoms.cell)
    vol = atoms.get_volume()
    energy = atoms.get_potential_energy()
    print(f"\n[Stage 6] Final energy          : {energy:.6f} eV")
    print(f"[Stage 6] Final energy per atom : {energy/n:.6f} eV/atom")
    print(f"[Stage 6] Final cell            : "
          f"a={cp[0]:.4f} b={cp[1]:.4f} c={cp[2]:.4f} Å  V={vol:.2f} Å³")

    # ── save ──────────────────────────────────────────────────────────────────
    write(cfg["output_cif"], atoms)
    write(cfg["output_xyz"], atoms)
    print(f"[Stage 6] Saved → {cfg['output_cif']}, {cfg['output_xyz']}\n")

    return atoms
