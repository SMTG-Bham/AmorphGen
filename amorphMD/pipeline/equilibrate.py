"""
pipeline/equilibrate.py
-----------------------
Constant-temperature MD equilibration. Called for three pipeline stages:

  Stage 2  (stage_key="eq_premelt") : pre-melt equilibration  at 300 K
  Stage 4  (stage_key="eq_high")    : high-T equilibration    at T_melt
  Stage 6  (stage_key="eq_low")     : low-T equilibration     at 300 K

Cell control options (set in cfg or via CLI, mainly useful for Stage 4):
  cell_mode = "free"           – default, cell evolves with ensemble
  cell_mode = "fix_volume"     – NVT only, cell shape and volume frozen
  cell_mode = "keep_cubic"     – NVT only, angles fixed at 90°, volume free
  cell_mode = "target_density" – rescale cell to target_density_g_cm3 before MD

Snapshot sampling (Stage 4):
  Set sample_interval_ps to save a frame every N ps.
  These become independent starting points for batch quench (stages 5→6→7).
"""

import os
from copy import deepcopy

import numpy as np
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from ..utils import (get_mace_calculator, MDLogger, TrajectoryWriter,
                     attach_outputs, merge_config, build_md_dynamics)
from ..configs import DEFAULT_CONFIG

# Maps stage_key → (stage number, human label)
_STAGE_INFO = {
    "eq_premelt": ("2", "Pre-melt Equilibration  300 K"),
    "eq_high":    ("4", "High-T Equilibration    at T_melt"),
    "eq_low":     ("6", "Low-T Equilibration     300 K"),
}

VALID_CELL_MODES = ("free", "fix_volume", "keep_cubic", "target_density")


# ─────────────────────────────────────────────────────────────────────────────
# Cell control helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_masses(atoms):
    """Total mass of the cell in grams."""
    from ase.data import atomic_masses
    from ase.data import atomic_numbers
    syms = atoms.get_chemical_symbols()
    return sum(atomic_masses[atomic_numbers[s]] for s in syms) / 6.02214076e23


def _current_density(atoms):
    """Density in g/cm³."""
    mass_g = _get_masses(atoms)
    vol_cm3 = atoms.get_volume() * 1e-24  # Å³ → cm³
    return mass_g / vol_cm3


def _rescale_to_density(atoms, target_g_cm3: float):
    """Rescale cell volume uniformly to reach target density (g/cm³)."""
    rho_now = _current_density(atoms)
    scale   = (rho_now / target_g_cm3) ** (1.0 / 3.0)
    new_cell = atoms.get_cell() * scale
    atoms.set_cell(new_cell, scale_atoms=True)
    rho_new = _current_density(atoms)
    print(f"  [cell] Density rescaled: {rho_now:.4f} → {rho_new:.4f} g/cm³  "
          f"(target: {target_g_cm3:.4f} g/cm³)")
    return atoms


def _apply_cubic_constraint(atoms):
    """
    Fix cell angles to 90° using FixedAngles (keeps volume free).
    Falls back to FixCellFilter mask if FixedAngles not available.
    """
    try:
        from ase.constraints import FixedAngles
        c = FixedAngles(atoms)
        existing = list(atoms.constraints)
        atoms.set_constraint(existing + [c])
        print("  [cell] FixedAngles constraint applied — cubic cell enforced.")
    except ImportError:
        # Fallback: fix off-diagonal strain components via ExpCellFilter mask
        print("  [cell] FixedAngles not available — using ExpCellFilter mask "
              "to keep cubic cell.")
        # Will be handled at dynamics level — just flag it
        atoms._keep_cubic_fallback = True


def _apply_fix_volume_constraint(atoms):
    """Fix cell shape and volume — pure NVT with frozen cell."""
    from ase.constraints import FixedPlane
    # Simplest: just ensure ensemble=NVT (no barostat) — no extra constraint needed.
    # We document this and warn if NPT was requested.
    print("  [cell] fix_volume mode: cell is frozen (NVT only, no barostat).")


# ─────────────────────────────────────────────────────────────────────────────
# Main equilibration function
# ─────────────────────────────────────────────────────────────────────────────

def run(atoms_or_file,
        stage_key: str = "eq_high",
        cfg_override: dict | None = None,
        calc=None):
    """
    Equilibrate the structure at a fixed temperature.

    Parameters
    ----------
    atoms_or_file : str or ase.Atoms
    stage_key     : "eq_premelt", "eq_high", or "eq_low"
    cfg_override  : dict, optional
    calc          : pre-built MACE calculator, optional

    Returns
    -------
    ase.Atoms
        Final structure after equilibration.
    tuple (ase.Atoms, list[str])
        When sample_interval_ps is set: (atoms, snapshot_paths).
    """
    global_cfg = merge_config(DEFAULT_CONFIG, cfg_override)
    cfg        = global_cfg[stage_key]
    stage_num, stage_name = _STAGE_INFO.get(stage_key, (stage_key, stage_key))

    # ── cell control mode ─────────────────────────────────────────────────────
    cell_mode = cfg.get("cell_mode", "free").lower()
    if cell_mode not in VALID_CELL_MODES:
        raise ValueError(f"Unknown cell_mode '{cell_mode}'. "
                         f"Choose from: {VALID_CELL_MODES}")

    target_density = cfg.get("target_density_g_cm3", None)
    if cell_mode == "target_density" and target_density is None:
        raise ValueError("cell_mode='target_density' requires "
                         "cfg['target_density_g_cm3'] to be set.")

    # fix_volume and keep_cubic force NVT
    ensemble = cfg.get("ensemble", "NVT").upper()
    if cell_mode in ("fix_volume", "keep_cubic") and ensemble == "NPT":
        print(f"  [cell] WARNING: cell_mode='{cell_mode}' is incompatible with NPT. "
              f"Switching to NVT.")
        ensemble = "NVT"

    print(f"\n{'='*65}")
    print(f"  Stage {stage_num}: {stage_name}")
    print(f"{'='*65}")

    # ── load ──────────────────────────────────────────────────────────────────
    if isinstance(atoms_or_file, str):
        atoms = read(atoms_or_file)
        print(f"[Stage {stage_num}] Loaded from {atoms_or_file}")
    else:
        atoms = deepcopy(atoms_or_file)
        print(f"[Stage {stage_num}] Using provided Atoms object")

    # ── print initial density ─────────────────────────────────────────────────
    rho = _current_density(atoms)
    print(f"[Stage {stage_num}] Initial density : {rho:.4f} g/cm³")
    print(f"[Stage {stage_num}] Cell mode       : {cell_mode}")
    print(f"[Stage {stage_num}] Ensemble        : {ensemble}")

    # ── apply cell control ────────────────────────────────────────────────────
    if cell_mode == "target_density":
        atoms = _rescale_to_density(atoms, target_density)

    elif cell_mode == "keep_cubic":
        _apply_cubic_constraint(atoms)

    elif cell_mode == "fix_volume":
        _apply_fix_volume_constraint(atoms)
    # "free" → no action

    # ── calculator ────────────────────────────────────────────────────────────
    if calc is None:
        calc = get_mace_calculator(
            model=global_cfg["mace_model"],
            device=global_cfg["device"],
            model_path=global_cfg.get("model_path"),
        )
    atoms.calc = calc

    T = cfg["temperature_K"]
    MaxwellBoltzmannDistribution(atoms, temperature_K=T)

    # ── dynamics ──────────────────────────────────────────────────────────────
    dyn = build_md_dynamics(
        atoms,
        ensemble=ensemble,
        temperature_K=T,
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

    # ── periodic density logging ──────────────────────────────────────────────
    log_interval  = cfg.get("log_interval", 10)
    density_interval = cfg.get("density_log_interval", 1000)  # steps

    def _log_density():
        rho = _current_density(atoms)
        vol = atoms.get_volume()
        t_ps = dyn.nsteps * cfg["timestep_fs"] / 1000.0
        print(f"  [density]  t={t_ps:8.2f} ps  ρ={rho:.4f} g/cm³  "
              f"V={vol:.2f} Å³")

    dyn.attach(_log_density, interval=density_interval)

    # ── snapshot sampling ─────────────────────────────────────────────────────
    sample_interval_ps = cfg.get("sample_interval_ps", None)
    snapshot_paths: list[str] = []

    if sample_interval_ps is not None:
        snap_dir       = cfg.get("snapshot_dir", "snapshots")
        timestep_fs    = cfg["timestep_fs"]
        steps_per_snap = max(1, int(sample_interval_ps * 1000 / timestep_fs))
        _counter       = {"n": 0}
        os.makedirs(snap_dir, exist_ok=True)

        def _write_snapshot():
            atoms.wrap()
            idx  = _counter["n"]
            t_ps = dyn.nsteps * timestep_fs / 1000.0
            rho  = _current_density(atoms)
            fname = os.path.join(
                snap_dir, f"snapshot_{idx:04d}_t{t_ps:.1f}ps.extxyz")
            write(fname, atoms, format="extxyz")
            snapshot_paths.append(fname)
            print(f"  [snapshot {idx:04d}]  t={t_ps:.1f} ps  "
                  f"ρ={rho:.4f} g/cm³  →  {fname}")
            _counter["n"] += 1

        dyn.attach(_write_snapshot, interval=steps_per_snap)
        total_ps   = cfg["steps"] * cfg["timestep_fs"] / 1000.0
        n_expected = int(total_ps / sample_interval_ps)
        print(f"[Stage {stage_num}] Snapshot every {sample_interval_ps} ps"
              f"  →  ~{n_expected} snapshots in '{snap_dir}/'")

    # ── run ───────────────────────────────────────────────────────────────────
    steps    = cfg["steps"]
    total_ps = steps * cfg["timestep_fs"] / 1000.0
    print(f"[Stage {stage_num}] Running {ensemble}  T={T} K  |  "
          f"{steps:,} steps  ({total_ps:.1f} ps)")

    dyn.run(steps)
    logger.close()
    traj.close()

    # ── final density report ──────────────────────────────────────────────────
    rho_final = _current_density(atoms)
    print(f"[Stage {stage_num}] Final density   : {rho_final:.4f} g/cm³")

    write(cfg["output_cif"], atoms)
    write(cfg["output_xyz"], atoms, format="extxyz")
    print(f"[Stage {stage_num}] Saved → {cfg['output_cif']}")

    if snapshot_paths:
        print(f"[Stage {stage_num}] {len(snapshot_paths)} snapshots → "
              f"'{cfg.get('snapshot_dir', 'snapshots')}/'")
        return atoms, snapshot_paths

    return atoms
