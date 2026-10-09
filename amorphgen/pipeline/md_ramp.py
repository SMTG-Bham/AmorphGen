"""Shared ASE execution of the heating and cooling stages."""

from __future__ import annotations

from copy import deepcopy

from ase.io import read, write

from ..configs import DEFAULT_CONFIG
from ..utils.calculators import calculator_kwargs
from ..utils.common import (
    attach_outputs, build_md_dynamics, compute_density_gcm3, make_cubic,
    merge_config, needs_velocity_init, ramp_resume_position, resolve_device,
    resolve_ramp_schedule, resume_md_stage, run_index_for, set_md_temperature,
    stage_file, stage_rng, thermalize_momenta,
)
from ..utils.repulsion import with_repulsive_core


def run_ramp(atoms_or_file, cfg_override, calc, work_dir, *, stage,
             calculator_factory, resume=False):
    """Run stage 3 or 5, keeping their distinct configuration and seed streams."""
    section, default_ensemble, default_step, output = {
        3: ("melt", "NPT", 100, "stage3_melted.xyz"),
        5: ("quench", "NVT", -100, "stage5_quenched.xyz"),
    }[stage]
    global_cfg = merge_config(DEFAULT_CONFIG, cfg_override)
    cfg = global_cfg[section]
    rng = stage_rng(global_cfg.get("seed"), stage, run_index_for(global_cfg))
    ensemble = cfg.get("ensemble", default_ensemble).upper()
    if isinstance(atoms_or_file, str):
        atoms = read(atoms_or_file)
        print(f"[Stage {stage}] Loaded from {atoms_or_file}")
    else:
        atoms = deepcopy(atoms_or_file)
        print(f"[Stage {stage}] Using provided Atoms object")

    # Only heating offers cubification. Preserve the input shape otherwise.
    if stage == 3 and cfg.get("make_cubic", False):
        atoms = make_cubic(atoms)
        print("[Stage 3] Cell reshaped to cubic")
    stem = f"stage{stage}_{section}"
    logfile = stage_file(cfg.get("log_file", stem + ".log"), work_dir)
    trajfile = stage_file(cfg.get("traj_file", stem + "_traj.xyz"), work_dir)
    ck_atoms, elapsed = resume_md_stage(
        trajfile, resume, str(stage),
        legacy_trajfile=stage_file(stem + ".xyz", work_dir))
    if ck_atoms is not None:
        atoms = ck_atoms

    if calc is None:
        arguments = calculator_kwargs(global_cfg)
        arguments["device"] = resolve_device(arguments["device"])
        calc = calculator_factory(**arguments)
    atoms.calc = with_repulsive_core(calc, global_cfg.get("repulsive_core"))
    T_start, T_end = cfg["T_start"], cfg["T_end"]
    if needs_velocity_init(atoms, elapsed):
        thermalize_momenta(atoms, temperature_K=T_start, rng=rng)
    dyn = build_md_dynamics(
        atoms, ensemble=ensemble, T=T_start,
        timestep=cfg.get("timestep", 1.0),
        friction=cfg.get("friction", 0.01),
        ttime=cfg.get("ttime", 25.0),
        npt_method=cfg.get("npt_method", "berendsen"),
        taup_factor=cfg.get("taup_factor", 10.0),
        compressibility_GPa=cfg.get("compressibility_GPa", 100.0), rng=rng,
    )

    # Resolve before attaching outputs, so invalid settings cannot truncate
    # an existing trajectory or log. Units remain K, fs and K/ps.
    T_step = cfg.get("T_step", default_step)
    if stage == 3:
        T_step = abs(T_step)
        if cfg.get("rate") is not None and T_step == 0:
            raise ValueError("T_step cannot be 0 when rate is specified.")
    timestep_fs = cfg.get("timestep", 1.0)
    temps, steps = resolve_ramp_schedule(
        T_start, T_end, T_step, timestep_fs=timestep_fs,
        rate=cfg.get("rate"), steps_per_T=cfg.get("steps_per_T", 1000))
    actual_rate = abs(T_step) / (steps * timestep_fs / 1000)
    logger, traj = attach_outputs(
        dyn, atoms, logfile, trajfile,
        fmt=global_cfg.get("traj_format", "extxyz"),
        append=elapsed > 0, step_offset=elapsed, safety=global_cfg.get("safety"))

    density = compute_density_gcm3(atoms)
    label = "heat ramp" if stage == 3 else "quench"
    increment = f"+{T_step} K" if stage == 3 else f"{T_step} K/step"
    print(f"[Stage {stage}] {ensemble} {label}: {T_start} -> {T_end} K  "
          f"({increment}, {steps} steps each, {actual_rate:.1f} K/ps)  "
          f"density={density:.2f} g/cm3")
    try:
        ran = False
        k0, offset = ramp_resume_position(elapsed, steps, len(temps))
        for idx, T in enumerate(temps):
            if idx < k0:
                continue
            set_md_temperature(dyn, T)
            run_steps = steps - offset if idx == k0 else steps
            note = f"  (resumed, {run_steps} steps left)" if (idx == k0 and offset) else ""
            print(f"  -> T = {T:7.1f} K{note}")
            dyn.run(run_steps)
            ran = True
        if not ran:
            # Completed resumes must still validate their final state.
            dyn.run(0)
    finally:
        logger.close()
        traj.close()
    out_xyz = stage_file(cfg.get("output_xyz", output), work_dir)
    write(out_xyz, atoms, format="extxyz")
    print(f"[Stage {stage}] Saved -> {out_xyz}\n")
    return atoms
