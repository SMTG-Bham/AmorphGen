"""
pipeline/melt_cell.py
---------------------
Stage 3 – Heat the structure from T_start to T_end via a temperature ramp.

Uses NPT ensemble by default so the cell volume can expand freely as the
crystal melts (e.g. In2O3: ~4.2 g/cm³ → ~3 g/cm³ at 2500 K).

Two ways to specify the heating rate:

  (a) Explicit (original API):
        "T_step": 100,         # K per ramp increment
        "steps_per_T": 1000,   # MD steps at each temperature
        → rate = 100 K / (1000 × 0.001 ps) = 100 K/ps

  (b) Rate-based (convenience):
        "rate_K_per_ps": 100   # K/ps  (overrides T_step / steps_per_T)
        "rate_K_per_ps": 50    # slower melt
"""

from copy import deepcopy
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from ..utils import (get_mace_calculator, make_cubic, MDLogger,
                     TrajectoryWriter, attach_outputs, merge_config,
                     build_md_dynamics, resolve_ramp)
from ..configs import DEFAULT_CONFIG


def run(atoms_or_file, cfg_override: dict | None = None, calc=None) -> object:
    """
    Heat the structure from T_start → T_end  (Stage 3).

    Parameters
    ----------
    atoms_or_file : str or ase.Atoms
    cfg_override  : dict, optional
    calc          : pre-built MACE calculator, optional

    Returns
    -------
    ase.Atoms  –  melted structure at T_end
    """
    global_cfg = merge_config(DEFAULT_CONFIG, cfg_override)
    cfg        = global_cfg["melt"]
    ensemble   = cfg.get("ensemble", "NPT").upper()

    print(f"\n{'='*62}")
    print(f"  Stage 3: Melt  –  NPT heat ramp to {cfg['T_end']} K")
    print(f"{'='*62}")

    # ── load ──────────────────────────────────────────────────────────────────
    if isinstance(atoms_or_file, str):
        atoms = read(atoms_or_file)
        print(f"[Stage 3] Loaded from {atoms_or_file}")
    else:
        atoms = deepcopy(atoms_or_file)
        print("[Stage 3] Using provided Atoms object")

    # ── optional cubic reshape ─────────────────────────────────────────────────
    if cfg.get("make_cubic", True):
        atoms = make_cubic(atoms)
        print("[Stage 3] Cell reshaped to cubic (equal volume)")
        write("stage3_cubic_initial.cif", atoms)

    # ── calculator ────────────────────────────────────────────────────────────
    if calc is None:
        calc = get_mace_calculator(
            model=global_cfg["mace_model"],
            device=global_cfg["device"],
            model_path=global_cfg.get("model_path"),
        )
    atoms.calc = calc

    T_start = cfg["T_start"]
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_start)

    # ── dynamics ──────────────────────────────────────────────────────────────
    dyn = build_md_dynamics(
        atoms,
        ensemble=ensemble,
        temperature_K=T_start,
        timestep_fs=cfg["timestep_fs"],
        taut_fs=cfg["taut_fs"],
        pressure_bar=cfg.get("pressure_bar", 1.0),
        taup_fs=cfg.get("taup_fs", 1000.0),
        compressibility=cfg.get("compressibility", 4.5e-5),
    )

    # ── outputs ───────────────────────────────────────────────────────────────
    fmt    = global_cfg.get("traj_format", "extxyz")
    traj   = TrajectoryWriter(cfg["traj_file"], fmt=fmt, atoms=atoms)
    logger = MDLogger(cfg["log_file"], ensemble=ensemble)
    attach_outputs(dyn, atoms, logger, traj, interval=cfg.get("log_interval", 10))

    # ── ramp ──────────────────────────────────────────────────────────────────
    temps, steps_per_T, rate = resolve_ramp(cfg, heating=True)
    dt_ps    = cfg["timestep_fs"] / 1000.0
    total_ps = len(temps) * steps_per_T * dt_ps

    print(f"[Stage 3] {ensemble} heat ramp:  "
          f"{cfg['T_start']} → {cfg['T_end']} K")
    print(f"          Heating rate  : {rate:.1f} K/ps")
    print(f"          Steps/T       : {steps_per_T}  ({steps_per_T*dt_ps:.2f} ps per step)")
    print(f"          Total MD time : {total_ps:.1f} ps  ({len(temps)} steps)")

    for T in temps:
        dyn.set_temperature(temperature_K=T)
        print(f"  → T = {T:5d} K")
        dyn.run(steps_per_T)

    logger.close()
    traj.close()

    write(cfg["output_cif"], atoms)
    write(cfg["output_xyz"], atoms, format="extxyz")
    print(f"[Stage 3] Saved → {cfg['output_cif']}, {cfg['output_xyz']}\n")
    return atoms
