"""
amorphgen.pipeline.quench
--------------------------
Stage 5 – Cool the melt from T_start down to T_end via a temperature ramp.

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
        print(f"[Stage 5] Loaded from {atoms_or_file}")
    else:
        atoms = deepcopy(atoms_or_file)
        print("[Stage 5] Using provided Atoms object")

    logfile = cfg.get("log_file", "stage5_quench.log")
    trajfile = cfg.get("traj_file", "stage5_quench_traj.xyz")

    # Frame-level resume: continue a walltime-killed quench from the last
    # trajectory frame (momenta included); ramp position recovered via
    # ramp_resume_position. The legacy filename lets a run interrupted
    # under a pre-rename AmorphGen still resume after upgrade.
    from ..utils.common import (resume_md_stage, needs_velocity_init,
                                ramp_resume_position)
    ck_atoms, elapsed = resume_md_stage(trajfile, kwargs.get("resume"), "5",
                                        legacy_trajfile="stage5_quench.xyz")
    if ck_atoms is not None:
        atoms = ck_atoms

    if calc is None:
        from ..utils.common import resolve_device
        device = resolve_device(global_cfg.get("device", "cuda"))
        calc = get_calculator(
            model=global_cfg.get("model", "mace-mpa-0"),
            device=device,
            model_path=global_cfg.get("model_path"),
        )
    atoms.calc = calc

    T_start = cfg["T_start"]
    if needs_velocity_init(atoms, elapsed):
        MaxwellBoltzmannDistribution(atoms, temperature_K=T_start)

    dyn = build_md_dynamics(
        atoms, ensemble=ensemble, T=T_start,
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

    T_end = cfg["T_end"]
    T_step = cfg.get("T_step", -100)
    timestep_fs = cfg.get("timestep", 1.0)

    # Allow rate (K/ps) to auto-calculate steps_per_T
    rate = cfg.get("rate")
    if rate is not None:
        steps = int(round(abs(T_step) / (rate * timestep_fs / 1000)))
        steps = max(steps, 1)
    else:
        steps = cfg.get("steps_per_T", 1000)

    # Build temperature list (cooling)
    temps = []
    T = T_start
    while T >= T_end - 1e-6:
        temps.append(int(round(T)))
        T += T_step

    from ..utils.common import compute_density_gcm3
    density = compute_density_gcm3(atoms)
    actual_rate = abs(T_step) / (steps * timestep_fs / 1000)
    print(f"[Stage 5] {ensemble} quench: {T_start} -> {T_end} K  "
          f"({T_step} K/step, {steps} steps each, {actual_rate:.1f} K/ps)  "
          f"density={density:.2f} g/cm3")

    # Recover the ramp position on resume: k0 full segments done, offset
    # steps into segment k0 (see ramp_resume_position for the
    # elapsed==total edge semantics).
    k0, offset = ramp_resume_position(elapsed, steps, len(temps))
    for idx, T in enumerate(temps):
        if idx < k0:
            continue
        dyn.set_temperature(temperature_K=T)
        run_steps = steps - offset if idx == k0 else steps
        note = f"  (resumed, {run_steps} steps left)" if (idx == k0 and offset) else ""
        print(f"  -> T = {T:5d} K{note}")
        dyn.run(run_steps)

    logger.close()
    traj.close()

    out_xyz = cfg.get("output_xyz", "stage5_quenched.xyz")
    write(out_xyz, atoms, format="extxyz")
    print(f"[Stage 5] Saved -> {out_xyz}\n")
    return atoms
