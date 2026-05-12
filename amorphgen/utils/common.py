"""
amorphgen.utils.common
-----------------------
Shared helpers used across all pipeline stages:
cell manipulation, MD dynamics builder, temperature ramps,
logging, trajectory I/O, config merging, and snapshot extraction.

Calculator-related functions are in :mod:`amorphgen.utils.calculators`.
"""

from __future__ import annotations

import os
import copy
import numpy as np
from ase import units
from ase.io import read, write


# ═════════════════════════════════════════════════════════════════════════════
# Density helper
# ═════════════════════════════════════════════════════════════════════════════

def compute_density_gcm3(atoms) -> float:
    """Compute density of an Atoms object in g/cm3."""
    mass_g = sum(atoms.get_masses()) / 6.022e23
    vol_cm3 = atoms.get_volume() * 1e-24
    return mass_g / vol_cm3


# ═════════════════════════════════════════════════════════════════════════════
# Cell helpers
# ═════════════════════════════════════════════════════════════════════════════

def make_cubic(atoms):
    """Reshape the cell to a cube of equal volume, rescaling atom positions."""
    vol = atoms.get_volume()
    L = vol ** (1.0 / 3.0)
    old_cell = atoms.get_cell()
    new_cell = np.eye(3) * L
    # Scale fractional coordinates
    frac = atoms.get_scaled_positions()
    atoms.set_cell(new_cell, scale_atoms=False)
    atoms.set_scaled_positions(frac)
    atoms.wrap()
    return atoms


# ═════════════════════════════════════════════════════════════════════════════
# MD dynamics builder
# ═════════════════════════════════════════════════════════════════════════════

def build_md_dynamics(atoms, ensemble: str = "NVT", T: float = 300.0,
                      timestep: float = 1.0, friction: float = 0.01,
                      ttime: float = 25.0, pfactor: float | None = None,
                      external_stress: float = 0.0, **kwargs):
    """
    Create an NVT or NPT ASE dynamics object.

    Parameters
    ----------
    atoms : ase.Atoms
        Must already have a calculator attached.
    ensemble : str
        ``"NVT"`` or ``"NPT"``.
    T : float
        Temperature in Kelvin.
    timestep : float
        Time step in fs.
    friction : float
        Langevin friction coefficient (for NVT).
    ttime : float
        Thermostat time constant in fs (for NPT Nose-Hoover).
    pfactor : float, optional
        Barostat coupling (for NPT).  If None, auto-calculated.
    external_stress : float
        External pressure in GPa (for NPT).
    **kwargs
        Extra arguments forwarded to the ASE dynamics class.

    Returns
    -------
    ASE dynamics object
    """
    from ase.md.langevin import Langevin

    dt = timestep * units.fs

    if ensemble.upper() == "NVT":
        dyn = Langevin(atoms, timestep=dt, temperature_K=T,
                       friction=friction / units.fs, **kwargs)
    elif ensemble.upper() == "NPT":
        # Use NPTBerendsen — more stable than Nose-Hoover for melt-quench
        from ase.md.nptberendsen import NPTBerendsen
        dyn = NPTBerendsen(
            atoms,
            timestep=dt,
            temperature_K=T,
            taut=ttime * units.fs,
            pressure_au=external_stress * units.GPa,
            taup=ttime * 10 * units.fs,
            compressibility_au=1.0 / (100 * units.GPa),
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown ensemble '{ensemble}'. Use 'NVT' or 'NPT'.")
    return dyn


# ═════════════════════════════════════════════════════════════════════════════
# Temperature ramp
# ═════════════════════════════════════════════════════════════════════════════

def resolve_ramp(T_start: float, T_end: float, T_step: float) -> list[float]:
    """
    Generate a list of temperatures for a ramp.

    Works for both heating (T_step > 0) and cooling (T_step < 0).
    Always includes T_end.
    """
    if T_step == 0:
        raise ValueError("T_step cannot be zero.")

    temps = []
    T = T_start
    if T_step > 0:
        while T <= T_end + 1e-6:
            temps.append(round(T, 2))
            T += T_step
    else:
        while T >= T_end - 1e-6:
            temps.append(round(T, 2))
            T += T_step

    # Ensure T_end is included
    if abs(temps[-1] - T_end) > 1e-6:
        temps.append(round(T_end, 2))

    return temps


# ═════════════════════════════════════════════════════════════════════════════
# Trajectory formats
# ═════════════════════════════════════════════════════════════════════════════

TRAJ_FORMATS = {"extxyz", "xyz", "traj", "lammps-dump"}


# ═════════════════════════════════════════════════════════════════════════════
# MD Logger
# ═════════════════════════════════════════════════════════════════════════════

class MDLogger:
    """
    Per-step MD logger that writes to both a file and stdout.

    Logs step number, time (ps), temperature (K), potential energy (eV),
    kinetic energy (eV), total energy (eV), and volume (Å³).
    """

    def __init__(self, logfile: str, mode: str = "w"):
        self._fh = open(logfile, mode)
        header = (f"{'Step':>8s}  {'Time_ps':>10s}  {'T_K':>8s}  "
                  f"{'Epot_eV':>12s}  {'Ekin_eV':>12s}  "
                  f"{'Etot_eV':>12s}  {'Vol_A3':>10s}")
        self._fh.write(header + "\n")
        self._fh.write("-" * len(header) + "\n")
        self._fh.flush()

    def log(self, dyn, atoms):
        step = dyn.nsteps
        t_ps = dyn.get_time() / units.fs / 1000.0
        T = atoms.get_temperature()
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        etot = epot + ekin
        vol = atoms.get_volume()
        line = (f"{step:8d}  {t_ps:10.4f}  {T:8.1f}  "
                f"{epot:12.4f}  {ekin:12.4f}  "
                f"{etot:12.4f}  {vol:10.2f}")
        self._fh.write(line + "\n")
        self._fh.flush()
        print(line)

    def close(self):
        self._fh.close()


# ═════════════════════════════════════════════════════════════════════════════
# Trajectory writer
# ═════════════════════════════════════════════════════════════════════════════

class TrajectoryWriter:
    """
    Unified trajectory output supporting multiple formats.

    Wraps ASE's write() for extxyz/xyz/lammps-dump and ASE's Trajectory
    for .traj binary format.
    """

    def __init__(self, filename: str, fmt: str = "extxyz"):
        self.filename = filename
        self.fmt = fmt.lower()
        if self.fmt not in TRAJ_FORMATS:
            raise ValueError(
                f"Unknown trajectory format '{fmt}'. "
                f"Choose from: {', '.join(sorted(TRAJ_FORMATS))}"
            )
        self._traj = None
        if self.fmt == "traj":
            from ase.io.trajectory import Trajectory
            self._traj = Trajectory(filename, "w")

    def write(self, atoms=None):
        if self._traj is not None:
            self._traj.write(atoms)
        else:
            write(self.filename, atoms, format=self.fmt, append=True)

    def close(self):
        if self._traj is not None:
            self._traj.close()


# ═════════════════════════════════════════════════════════════════════════════
# Attach logger + trajectory to dynamics
# ═════════════════════════════════════════════════════════════════════════════

def attach_outputs(dyn, atoms, logfile: str, trajfile: str,
                   fmt: str = "extxyz", interval: int = 100):
    """
    Attach an MDLogger and TrajectoryWriter to *dyn*.

    Returns (logger, traj_writer) so they can be closed later.
    """
    logger = MDLogger(logfile)
    traj = TrajectoryWriter(trajfile, fmt=fmt)

    dyn.attach(lambda: logger.log(dyn, atoms), interval=interval)
    dyn.attach(lambda: (atoms.wrap(), traj.write(atoms)), interval=interval)

    return logger, traj


# ═════════════════════════════════════════════════════════════════════════════
# Config merging
# ═════════════════════════════════════════════════════════════════════════════

def merge_config(defaults: dict, overrides: dict | None) -> dict:
    """Deep-merge *overrides* into a copy of *defaults*."""
    cfg = copy.deepcopy(defaults)
    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k] = merge_config(cfg[k], v)
            else:
                cfg[k] = v
    return cfg


# ═════════════════════════════════════════════════════════════════════════════
# Snapshot extraction
# ═════════════════════════════════════════════════════════════════════════════

_FORMAT_EXT = {"xyz": "xyz", "extxyz": "xyz",
               "vasp": "vasp", "cif": "cif"}


def extract_snapshots(traj_file: str, n_snapshots: int = 20,
                      select: str = "uniform",
                      output_dir: str = "snapshots",
                      burn_in_frames: int = 0,
                      output_format: str = "xyz") -> list[str]:
    """
    Extract snapshot frames from a trajectory file.

    Parameters
    ----------
    traj_file : str
        Path to the trajectory file.
    n_snapshots : int
        Number of snapshots to extract.
    select : str
        Selection strategy: ``"uniform"`` (evenly spaced) or
        ``"last"`` (final *n* frames).
    output_dir : str
        Directory for output files.
    burn_in_frames : int, default 0
        Number of leading frames to discard before sampling. Useful for
        skipping the non-equilibrated portion of an MD trajectory (e.g.
        the first ~50 ps of a 100 ps high-T equilibration). The reported
        frame indices in the output filenames remain absolute (relative
        to the original trajectory) so the provenance of each snapshot
        is preserved.
    output_format : str, default ``"xyz"``
        File format for written snapshots. One of ``xyz`` / ``extxyz``
        (extxyz with full metadata), ``vasp`` (POSCAR, sorted by species),
        or ``cif``.  All formats are ASE-readable round-trip.

    Returns
    -------
    list of str
        Paths to extracted snapshot files.
    """
    if output_format not in _FORMAT_EXT:
        raise ValueError(
            f"Unknown output_format '{output_format}'. "
            f"Supported: {sorted(_FORMAT_EXT)}"
        )
    ext = _FORMAT_EXT[output_format]
    ase_fmt = "extxyz" if output_format in ("xyz", "extxyz") else output_format
    frames = read(traj_file, index=":")
    n_frames_total = len(frames)

    if burn_in_frames < 0:
        raise ValueError(f"burn_in_frames must be >= 0, got {burn_in_frames}.")
    if burn_in_frames >= n_frames_total:
        raise ValueError(
            f"burn_in_frames={burn_in_frames} >= trajectory length "
            f"{n_frames_total}; nothing left to sample.")
    n_frames = n_frames_total - burn_in_frames

    if n_snapshots > n_frames:
        print(f"Warning: requested {n_snapshots} snapshots but only "
              f"{n_frames} frames available after burn-in. Using all.")
        n_snapshots = n_frames

    if select == "uniform":
        indices = np.linspace(burn_in_frames, n_frames_total - 1,
                              n_snapshots, dtype=int)
    elif select == "last":
        indices = list(range(max(burn_in_frames, n_frames_total - n_snapshots),
                             n_frames_total))
    else:
        raise ValueError(f"Unknown selection strategy '{select}'.")

    if burn_in_frames > 0:
        print(f"  Burn-in: skipping first {burn_in_frames} frames "
              f"(sampling from frame {burn_in_frames}-{n_frames_total - 1})")

    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, idx in enumerate(indices):
        fname = os.path.join(output_dir,
                             f"snapshot_{i:04d}_frame{idx:05d}.{ext}")
        if ase_fmt == "vasp":
            write(fname, frames[idx], format=ase_fmt, sort=True)
        else:
            write(fname, frames[idx], format=ase_fmt)
        paths.append(fname)

    print(f"Extracted {len(paths)} snapshots → {output_dir}/")
    return paths
