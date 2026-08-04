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

from ..utils import (get_calculator, make_cubic, build_md_dynamics,
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

    # Optional cell cubification (default: off — preserve the input shape).
    # Useful at the high-T equilibration plateau when the heat ramp has
    # left the cell anisotropic and the user wants to restart from an
    # isotropic supercell.
    if cfg.get("make_cubic", False):
        atoms = make_cubic(atoms)
        print(f"[Stage {stage_label}] Reshaped cell to cubic (make_cubic)")

    logfile = cfg.get("log_file", f"stage{stage_label}_eq.log")
    trajfile = cfg.get("traj_file", f"stage{stage_label}_eq_traj.xyz")
    steps = cfg.get("steps", 10000)

    # Frame-level resume: pick a walltime-killed stage up from the last
    # trajectory frame instead of rerunning from step 0. The frame carries
    # the MD momenta, so only thermostat RNG / barostat scaling state is
    # lost (negligible in equilibrium MD). Opt-in via --resume; stages that
    # never started have no trajectory and start fresh as before.
    from ..utils.common import resume_md_stage, needs_velocity_init
    ck_atoms, elapsed = resume_md_stage(trajfile, kwargs.get("resume"),
                                        stage_label)
    if ck_atoms is not None:
        atoms = ck_atoms
        elapsed = min(elapsed, steps)

    if calc is None:
        from ..utils.common import resolve_device
        device = resolve_device(global_cfg.get("device", "cuda"))
        calc = get_calculator(
            model=global_cfg.get("model", "mace-mpa-0"),
            device=device,
            model_path=global_cfg.get("model_path"),
        )
    atoms.calc = calc

    default_T = {"premelt": 300, "high": 3000, "low": 300}
    T = cfg.get("T", default_T.get(stage, 300))
    if needs_velocity_init(atoms, elapsed):
        MaxwellBoltzmannDistribution(atoms, temperature_K=T)

    dyn = build_md_dynamics(
        atoms, ensemble=ensemble, T=T,
        timestep=cfg.get("timestep", 1.0),
        friction=cfg.get("friction", 0.01),
        ttime=cfg.get("ttime", 25.0),
        npt_method=cfg.get("npt_method", "berendsen"),
        taup_factor=cfg.get("taup_factor", 10.0),
        compressibility_GPa=cfg.get("compressibility_GPa", 100.0),
    )

    logger, traj = attach_outputs(dyn, atoms, logfile, trajfile,
                                  fmt=global_cfg.get("traj_format", "extxyz"),
                                  append=elapsed > 0)

    from ..utils.common import compute_density_gcm3
    density = compute_density_gcm3(atoms)
    total_ps = steps * cfg.get("timestep", 1.0) / 1000
    print(f"[Stage {stage_label}] {ensemble} equilibration  T={T} K  "
          f"{steps - elapsed} steps ({total_ps:.1f} ps total)  "
          f"density={density:.2f} g/cm3")

    dyn.run(steps - elapsed)

    logger.close()
    traj.close()

    out_xyz = cfg.get("output_xyz", f"stage{stage_label}_eq.xyz")
    write(out_xyz, atoms, format="extxyz")
    print(f"[Stage {stage_label}] Saved -> {out_xyz}\n")
    return atoms
