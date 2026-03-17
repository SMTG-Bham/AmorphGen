"""
amorphgen.pipeline.equilibrate
-------------------------------
Stages 2, 4 & 6 – Constant-temperature equilibration.

Stage 2 (stage="premelt"): pre-melt equilibration at 300 K
Stage 4 (stage="high"):    high-T equilibration after melting
Stage 6 (stage="low"):     low-T equilibration after quenching
"""

from __future__ import annotations

from copy import deepcopy
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from ..utils import (get_calculator, build_md_dynamics,
                     attach_outputs, merge_config)
from ..configs import DEFAULT_CONFIG


def run(atoms_or_file, cfg_override=None, calc=None, stage="high", **kwargs):
    """
    Equilibrate the structure at a fixed temperature.

    Parameters
    ----------
    atoms_or_file : str or ase.Atoms
    cfg_override : dict, optional
    calc : ASE calculator, optional
    stage : str
        "premelt" for Stage 2 (eq_premelt), "high" for Stage 4 (eq_high),
        or "low" for Stage 6 (eq_low).

    Returns
    -------
    ase.Atoms
    """
    stage_map = {
        "premelt": ("eq_premelt", "2"),
        "high":    ("eq_high",    "4"),
        "low":     ("eq_low",     "6"),
    }
    if stage not in stage_map:
        raise ValueError(f"Unknown equilibration stage '{stage}'. "
                         f"Expected one of: {list(stage_map.keys())}")
    stage_key, stage_label = stage_map[stage]

    global_cfg = merge_config(DEFAULT_CONFIG, cfg_override)
    cfg = global_cfg[stage_key]
    ensemble = cfg.get("ensemble", "NVT").upper()

    if isinstance(atoms_or_file, str):
        atoms = read(atoms_or_file)
        print(f"[Stage {stage_label}] Loaded from {atoms_or_file}")
    else:
        atoms = deepcopy(atoms_or_file)
        print(f"[Stage {stage_label}] Using provided Atoms object")

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

    default_T = {"premelt": 300, "high": 3000, "low": 300}
    T = cfg.get("T", default_T.get(stage, 300))
    MaxwellBoltzmannDistribution(atoms, temperature_K=T)

    dyn = build_md_dynamics(
        atoms, ensemble=ensemble, T=T,
        timestep=cfg.get("timestep", 1.0),
        friction=cfg.get("friction", 0.01),
        ttime=cfg.get("ttime", 25.0),
    )

    logfile = cfg.get("log_file", f"stage{stage_label}_eq.log")
    trajfile = cfg.get("traj_file", f"stage{stage_label}_eq.extxyz")
    logger, traj = attach_outputs(dyn, atoms, logfile, trajfile,
                                  fmt=global_cfg.get("traj_format", "extxyz"))

    steps = cfg.get("steps", 10000)
    total_ps = steps * cfg.get("timestep", 1.0) / 1000
    print(f"[Stage {stage_label}] {ensemble} equilibration  T={T} K  "
          f"{steps} steps ({total_ps:.1f} ps)")

    dyn.run(steps)

    logger.close()
    traj.close()

    out_xyz = cfg.get("output_xyz", f"stage{stage_label}_eq.extxyz")
    write(out_xyz, atoms, format="extxyz")
    print(f"[Stage {stage_label}] Saved -> {out_xyz}\n")
    return atoms
