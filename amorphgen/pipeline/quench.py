"""
amorphgen.pipeline.quench
--------------------------
Stage 4 – Cool the melt from T_start down to T_end via a temperature ramp.

Ensemble is configurable: NVT (default) or NPT.
"""

from __future__ import annotations

from copy import deepcopy
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from ..utils import (get_calculator, build_md_dynamics,
                     attach_outputs, merge_config)
from ..configs import DEFAULT_CONFIG


def run(atoms_or_file, cfg_override=None, calc=None, **kwargs):
    """
    Cool the structure from T_start -> T_end.

    Parameters
    ----------
    atoms_or_file : str or ase.Atoms
    cfg_override : dict, optional
    calc : ASE calculator, optional

    Returns
    -------
    ase.Atoms — quenched structure at T_end
    """
    global_cfg = merge_config(DEFAULT_CONFIG, cfg_override)
    cfg = global_cfg["quench"]
    ensemble = cfg.get("ensemble", "NVT").upper()

    if isinstance(atoms_or_file, str):
        atoms = read(atoms_or_file)
        print(f"[Stage 4] Loaded from {atoms_or_file}")
    else:
        atoms = deepcopy(atoms_or_file)
        print("[Stage 4] Using provided Atoms object")

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

    logfile = cfg.get("log_file", "stage4_quench.log")
    trajfile = cfg.get("traj_file", "stage4_quench.extxyz")
    logger, traj = attach_outputs(dyn, atoms, logfile, trajfile,
                                  fmt=global_cfg.get("traj_format", "extxyz"))

    T_end = cfg["T_end"]
    T_step = cfg.get("T_step", -100)
    steps = cfg.get("steps_per_T", 1000)

    # Build temperature list (cooling)
    temps = []
    T = T_start
    while T >= T_end - 1e-6:
        temps.append(int(round(T)))
        T += T_step

    print(f"[Stage 4] {ensemble} quench: {T_start} -> {T_end} K  "
          f"({T_step} K/step, {steps} steps each)")

    for T in temps:
        dyn.set_temperature(temperature_K=T)
        print(f"  -> T = {T:5d} K")
        dyn.run(steps)

    logger.close()
    traj.close()

    out_xyz = cfg.get("output_xyz", "stage4_quenched.extxyz")
    write(out_xyz, atoms, format="extxyz")
    print(f"[Stage 4] Saved -> {out_xyz}\n")
    return atoms
