"""
amorphgen.pipeline.melt_cell
-----------------------------
Stage 2 – Heat the structure from T_start to T_end via a temperature ramp.

Ensemble is configurable: NPT (default) or NVT.
"""

from __future__ import annotations

from copy import deepcopy
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from ..utils import (get_calculator, make_cubic,
                     build_md_dynamics, attach_outputs, merge_config)
from ..configs import DEFAULT_CONFIG


def run(atoms_or_file, cfg_override=None, calc=None, **kwargs):
    """
    Heat the structure from T_start -> T_end.

    Parameters
    ----------
    atoms_or_file : str or ase.Atoms
    cfg_override : dict, optional
    calc : ASE calculator, optional

    Returns
    -------
    ase.Atoms — melted structure at T_end
    """
    global_cfg = merge_config(DEFAULT_CONFIG, cfg_override)
    cfg = global_cfg["melt"]
    ensemble = cfg.get("ensemble", "NPT").upper()

    if isinstance(atoms_or_file, str):
        atoms = read(atoms_or_file)
        print(f"[Stage 2] Loaded from {atoms_or_file}")
    else:
        atoms = deepcopy(atoms_or_file)
        print("[Stage 2] Using provided Atoms object")

    # Optional cubic reshape
    if cfg.get("make_cubic", True):
        atoms = make_cubic(atoms)
        print("[Stage 2] Cell reshaped to cubic")

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

    T_start = cfg["T_start"]
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_start)

    dyn = build_md_dynamics(
        atoms, ensemble=ensemble, T=T_start,
        timestep=cfg.get("timestep", 1.0),
        friction=cfg.get("friction", 0.01),
        ttime=cfg.get("ttime", 25.0),
    )

    logfile = cfg.get("log_file", "stage2_melt.log")
    trajfile = cfg.get("traj_file", "stage2_melt.extxyz")
    logger, traj = attach_outputs(dyn, atoms, logfile, trajfile,
                                  fmt=global_cfg.get("traj_format", "extxyz"))

    # Temperature ramp
    T_end = cfg["T_end"]
    T_step = abs(cfg.get("T_step", 100))
    steps = cfg.get("steps_per_T", 1000)
    target_temps = list(range(T_start, T_end + T_step, T_step))

    print(f"[Stage 2] {ensemble} heat ramp: {T_start} -> {T_end} K  "
          f"(+{T_step} K, {steps} steps each)")

    for T in target_temps:
        dyn.set_temperature(temperature_K=T)
        print(f"  -> T = {T:5d} K")
        dyn.run(steps)

    logger.close()
    traj.close()

    out_xyz = cfg.get("output_xyz", "stage2_melted.extxyz")
    write(out_xyz, atoms, format="extxyz")
    print(f"[Stage 2] Saved -> {out_xyz}\n")
    return atoms
