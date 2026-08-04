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
# Numerical-divergence guard
# ═════════════════════════════════════════════════════════════════════════════

class DivergenceError(RuntimeError):
    """Non-finite energy/forces during MD or relaxation — the run diverged.

    Almost always a foundation-model MLIP going out-of-distribution in the
    high-temperature liquid regime, or too large a timestep. Raised eagerly
    (see :func:`assert_finite`) so a NaN/Inf never silently propagates into a
    saved structure or trajectory — a wrong-but-plausible result is worse than
    a clear failure.
    """


def assert_finite(atoms, context: str = "", step=None) -> None:
    """Raise :class:`DivergenceError` if the current energy/forces are non-finite.

    Reads the energy/forces already computed for this step (the MD integrator
    and the optimiser both evaluate them every step, and ASE caches the result
    until the atoms change), so it adds no calculator call and is cheap enough
    to run every step. ``context`` and ``step`` are woven into the message to
    pinpoint where the divergence happened.

    A calculator that *raises* (rather than returning NaN) is left alone — that
    is a different failure and must surface on its own, not be masked here.
    """
    try:
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
    except Exception:
        return

    bad = []
    if not np.isfinite(energy):
        bad.append("potential energy")
    if forces is not None and not np.isfinite(forces).all():
        bad.append("forces")
    if not bad:
        return

    where = f" at step {step}" if step is not None else ""
    during = f" during {context}" if context else ""
    raise DivergenceError(
        f"Non-finite {' and '.join(bad)}{where}{during} — the calculation has "
        f"diverged. Most often the MLIP is out-of-distribution in the "
        f"high-temperature liquid regime, or the timestep is too large.\n"
        f"  Remedies: lower the melt temperature / heating-rate / cooling-rate, "
        f"reduce the timestep, or use --random-gen followed by a low-temperature "
        f"anneal instead of a full high-T melt-quench (universal MLIPs are "
        f"unreliable in the high-T liquid regime)."
    )


def resolve_device(device: str) -> str:
    """Resolve ``device="auto"`` to ``"cuda"`` or ``"cpu"``.

    Torch is an *optional* dependency (pulled in by the MLIP extras), so a
    torch-free install — random generation, analysis, or classical-potential
    pipelines — resolves ``auto`` to ``"cpu"`` instead of crashing on the
    import. Any explicit device string is passed through unchanged.
    """
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


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

_VALID_NPT_METHODS = ("berendsen", "mtk", "parrinello-rahman")


def build_md_dynamics(atoms, ensemble: str = "NVT", T: float = 300.0,
                      timestep: float = 1.0, friction: float = 0.01,
                      ttime: float = 25.0, pfactor: float | None = None,
                      external_stress: float = 0.0,
                      npt_method: str = "berendsen",
                      taup_factor: float = 10.0,
                      compressibility_GPa: float = 100.0,
                      **kwargs):
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
        Thermostat time constant in fs.  For ``"berendsen"`` it is
        ``taut``; for ``"mtk"`` and ``"parrinello-rahman"`` it is the
        Nose-Hoover-chain time constant (``ttime`` in the ASE API).
    pfactor : float, optional
        Barostat coupling factor for ``"mtk"`` and ``"parrinello-rahman"``.
        If ``None``, defaults to ``(ttime * taup_factor fs)**2 *
        compressibility_GPa GPa``, giving a barostat ``taup_factor``
        times slower than the thermostat (same spirit as the Berendsen
        ``taup``).  Ignored by ``"berendsen"``.
    external_stress : float
        External pressure in GPa (for NPT). For ``"mtk"`` and
        ``"parrinello-rahman"`` this is converted to an isotropic
        stress tensor.
    npt_method : {"berendsen", "mtk", "parrinello-rahman"}
        NPT integrator to use when ``ensemble == "NPT"``.

        * ``"berendsen"`` (default) — weak-coupling Berendsen barostat
          and thermostat (``ase.md.nptberendsen.NPTBerendsen``).
          Robust during 300 K -> 3000 K melt ramps; does **not**
          produce true canonical fluctuations, so heat capacities and
          isothermal compressibilities derived from fluctuations are
          incorrect.  Averages are correct.

        * ``"mtk"`` — Martyna-Tobias-Klein Nose-Hoover-chain NPT
          (``ase.md.nose_hoover_chain.IsotropicMTKNPT``).  Produces
          true canonical fluctuations.  Recommended for the
          equilibration stages (2, 4, 6); may become unstable during
          rapid temperature ramps (stages 3, 5).

        * ``"parrinello-rahman"`` — Nose-Hoover + Parrinello-Rahman
          flexible-cell NPT (``ase.md.npt.NPT``).  Allows the cell
          shape (not just volume) to change; useful for anisotropic
          glasses but requires upper-triangular cell.

        Ignored when ``ensemble == "NVT"``.
    taup_factor : float, default 10.0
        Ratio of barostat coupling time to thermostat coupling time,
        i.e. ``taup = taup_factor * ttime``.  Larger values give a
        slower, more stable barostat — useful for damping cell-volume
        excursions during the 300 K -> 3000 K melt ramp.  Applied to
        the Berendsen ``taup`` and to the MTK / Parrinello-Rahman
        barostat-time defaults.
    compressibility_GPa : float, default 100.0
        Reference isothermal compressibility used by the Berendsen
        barostat as ``1/(compressibility_GPa * GPa)``.  The default
        (100 GPa) is intentionally soft and gives liquid-like
        responsiveness; for stiffer oxides (a-In2O3, a-Ga2O3, a-HfO2,
        bulk modulus ~150-300 GPa) using 200 GPa gives more realistic
        and more stable volume control.  Ignored by ``"mtk"`` and
        ``"parrinello-rahman"``.
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
        return dyn

    if ensemble.upper() != "NPT":
        raise ValueError(f"Unknown ensemble '{ensemble}'. Use 'NVT' or 'NPT'.")

    method = npt_method.lower()
    if method not in _VALID_NPT_METHODS:
        raise ValueError(
            f"Unknown npt_method '{npt_method}'. "
            f"Choose from: {', '.join(_VALID_NPT_METHODS)}."
        )

    if method == "berendsen":
        # Weak-coupling Berendsen — more stable than Nose-Hoover for
        # the 300 K -> 3000 K melt-quench ramp; this is the default.
        from ase.md.nptberendsen import NPTBerendsen
        dyn = NPTBerendsen(
            atoms,
            timestep=dt,
            temperature_K=T,
            taut=ttime * units.fs,
            pressure_au=external_stress * units.GPa,
            taup=ttime * taup_factor * units.fs,
            compressibility_au=1.0 / (compressibility_GPa * units.GPa),
            **kwargs,
        )
    elif method == "mtk":
        # Martyna-Tobias-Klein Nose-Hoover-chain NPT (isotropic cell).
        # True canonical fluctuations; recommended for equilibration
        # stages, can be unstable in rapid temperature ramps.
        from ase.md.nose_hoover_chain import IsotropicMTKNPT
        if pfactor is None:
            pfactor = (ttime * taup_factor * units.fs) ** 2 * compressibility_GPa * units.GPa
        dyn = IsotropicMTKNPT(
            atoms,
            timestep=dt,
            temperature_K=T,
            pressure_au=external_stress * units.GPa,
            tdamp=ttime * units.fs,
            pdamp=ttime * taup_factor * units.fs,
            **kwargs,
        )
    else:  # parrinello-rahman
        # Nose-Hoover + Parrinello-Rahman flexible-cell NPT
        # (Melchionna integrator).  Requires an upper-triangular
        # cell; ASE will raise if not.
        try:
            from ase.md.melchionna import MelchionnaNPT as _NPT
        except ImportError:  # pragma: no cover — older ASE
            from ase.md.npt import NPT as _NPT
        if pfactor is None:
            pfactor = (ttime * taup_factor * units.fs) ** 2 * compressibility_GPa * units.GPa
        dyn = _NPT(
            atoms,
            timestep=dt,
            temperature_K=T,
            externalstress=external_stress * units.GPa,
            ttime=ttime * units.fs,
            pfactor=pfactor,
            **kwargs,
        )
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
        if mode != "a":     # resumed runs continue the existing table
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

    def __init__(self, filename: str, fmt: str = "extxyz",
                 append: bool = False):
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
            self._traj = Trajectory(filename, "a" if append else "w")
        elif not append and os.path.exists(filename):
            # File formats write with append=True per frame, so a fresh run
            # must truncate any stale trajectory from a previous attempt —
            # otherwise frames accumulate across reruns and frame-level
            # resume (read_md_checkpoint) miscounts the elapsed steps.
            os.remove(filename)

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

# Trajectory/log write interval (MD steps). ONE constant shared by
# attach_outputs and read_md_checkpoint: the frame-level-resume arithmetic
# ((n_frames - 1) * interval == elapsed steps) is only correct when the
# writer and the reader agree on this value — never change one without the
# other.
TRAJ_LOG_INTERVAL = 100


def attach_outputs(dyn, atoms, logfile: str, trajfile: str,
                   fmt: str = "extxyz", interval: int = TRAJ_LOG_INTERVAL,
                   append: bool = False):
    """
    Attach an MDLogger and TrajectoryWriter to *dyn*.

    Parameters
    ----------
    append : bool
        Continue existing log/trajectory files instead of truncating them
        (frame-level resume). The step-0 observer call of the resumed run is
        suppressed so the resume point is not written twice — the trajectory
        stays one-frame-per-``interval``-steps, which is what
        :func:`read_md_checkpoint` relies on to count elapsed steps.

    Returns (logger, traj_writer) so they can be closed later.
    """
    logger = MDLogger(logfile, mode="a" if append else "w")
    traj = TrajectoryWriter(trajfile, fmt=fmt, append=append)

    state = {"skip": append}   # skip the duplicate step-0 write on resume

    # Eager divergence guard. Attached FIRST and every step so it raises before
    # the trajectory writer below can persist a NaN/Inf frame. Uses the forces
    # the integrator already computed this step, so it costs no calculator call.
    stage_label = os.path.splitext(os.path.basename(trajfile))[0]

    def _finite_guard():
        assert_finite(atoms, context=f"MD stage '{stage_label}'",
                      step=getattr(dyn, "nsteps", None))

    dyn.attach(_finite_guard, interval=1)

    def _observe():
        if state["skip"]:
            state["skip"] = False
            return
        logger.log(dyn, atoms)
        atoms.wrap()
        traj.write(atoms)

    dyn.attach(_observe, interval=interval)

    return logger, traj


def read_md_checkpoint(trajfile: str, interval: int = TRAJ_LOG_INTERVAL):
    """Last frame + elapsed MD steps from a partial stage trajectory.

    Frames are written every *interval* steps starting at step 0, so a file
    with ``n`` frames corresponds to ``(n - 1) * interval`` completed steps.
    The returned frame carries the MD momenta (extxyz stores them), so the
    resumed run continues with the same velocities; only the thermostat RNG
    state / barostat scaling state is lost — negligible in equilibrium MD.

    Returns
    -------
    (ase.Atoms, int) or None
        ``(last_frame, elapsed_steps)``, or ``None`` when there is no usable
        checkpoint (missing, unreadable, or single-frame file — a torn file
        from a mid-write kill falls back to a fresh stage run).
    """
    if not os.path.exists(trajfile):
        return None
    try:
        frames = read(trajfile, index=":")
    except Exception:
        return None
    if not isinstance(frames, list) or len(frames) < 2:
        return None
    return frames[-1], (len(frames) - 1) * interval


def resume_md_stage(trajfile: str, resume, stage_label: str,
                    legacy_trajfile: str | None = None):
    """Shared frame-level-resume entry point for the MD stages (2-6).

    Returns ``(checkpoint_atoms_or_None, elapsed_steps)``. Holds the resume
    invariants in ONE place (see also :func:`needs_velocity_init` and
    :func:`ramp_resume_position`) so the three stage modules cannot drift.
    ``legacy_trajfile`` lets ramp stages pick up a trajectory written under
    the pre-rename default name by an older AmorphGen version.
    """
    if not resume:
        return None, 0
    ck = read_md_checkpoint(trajfile)
    if ck is None and legacy_trajfile is not None:
        ck = read_md_checkpoint(legacy_trajfile)
        if ck is not None:
            print(f"[Stage {stage_label}] Using legacy trajectory "
                  f"{legacy_trajfile} for resume")
    if ck is None:
        return None, 0
    atoms, elapsed = ck
    print(f"[Stage {stage_label}] Frame-level resume: {elapsed} steps "
          f"already completed")
    return atoms, elapsed


def needs_velocity_init(atoms, elapsed: int) -> bool:
    """Should the stage (re)draw Maxwell-Boltzmann velocities?

    Fresh runs always do. Resumed runs keep the checkpoint's momenta —
    unless the frame carries none (all-zero momenta cannot occur mid-MD, so
    zeros mean the trajectory format dropped them) and re-initialisation is
    the only option.
    """
    if not elapsed:
        return True
    return not np.abs(atoms.get_momenta()).sum() > 0


def ramp_resume_position(elapsed: int, steps_per_T: int, n_temps: int):
    """Position in a temperature ramp after *elapsed* completed steps.

    Returns ``(k0, offset)``: ``k0`` full segments are done and ``offset``
    steps of segment ``k0`` — the caller skips segments ``< k0`` and runs
    ``steps_per_T - offset`` for segment ``k0``. When ``elapsed`` equals the
    ramp total, ``k0 == n_temps`` and the loop runs nothing: the stage
    output is then written from the checkpoint frame, which can lag the true
    final state by up to ``TRAJ_LOG_INTERVAL - 1`` steps (the frames between
    write intervals are not recoverable) — physically negligible for
    equilibrium MD, but a resumed run is not byte-identical to an
    uninterrupted one.
    """
    return divmod(min(elapsed, steps_per_T * n_temps), steps_per_T)


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

# Map output_format -> (file extension, ASE write format).
_SNAPSHOT_FORMAT_MAP = {
    "extxyz": (".xyz", "extxyz"),
    "xyz":    (".xyz", "extxyz"),
    "vasp":   (".vasp", "vasp"),
    "cif":    (".cif", "cif"),
    "traj":   (".traj", "traj"),
}


def extract_snapshots(traj_file: str, n_snapshots: int = 20,
                      select: str = "uniform",
                      output_dir: str = "snapshots",
                      burn_in_frames: int = 0,
                      output_format: str = "extxyz") -> list[str]:
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
        Number of leading frames to skip before sampling.  Useful for
        discarding the equilibration period at the start of an MD
        trajectory.  Sampling indices run over the closed interval
        ``[burn_in_frames, n_frames - 1]``.  Raises ``ValueError`` if
        ``burn_in_frames >= n_frames``.
    output_format : str, default ``"extxyz"``
        Output file format.  Accepted values: ``"extxyz"``, ``"xyz"``
        (both write extended XYZ with a ``.xyz`` extension), ``"vasp"``
        (POSCAR-style), ``"cif"``, ``"traj"``.

    Returns
    -------
    list of str
        Paths to extracted snapshot files.
    """
    frames = read(traj_file, index=":")
    n_frames = len(frames)

    if burn_in_frames < 0:
        raise ValueError(
            f"burn_in_frames must be >= 0, got {burn_in_frames}."
        )
    if burn_in_frames >= n_frames:
        raise ValueError(
            f"burn_in_frames ({burn_in_frames}) must be smaller than the "
            f"trajectory length ({n_frames})."
        )

    available = n_frames - burn_in_frames
    if n_snapshots > available:
        print(f"Warning: requested {n_snapshots} snapshots but only "
              f"{available} frames are available after burn-in. "
              f"Using all available frames.")
        n_snapshots = available

    if select == "uniform":
        indices = np.linspace(burn_in_frames, n_frames - 1, n_snapshots,
                              dtype=int)
    elif select == "last":
        indices = list(range(max(burn_in_frames, n_frames - n_snapshots),
                             n_frames))
    else:
        raise ValueError(f"Unknown selection strategy '{select}'.")

    if output_format not in _SNAPSHOT_FORMAT_MAP:
        raise ValueError(
            f"Unknown output_format '{output_format}'. "
            f"Choose from: {', '.join(sorted(_SNAPSHOT_FORMAT_MAP))}."
        )
    ext, ase_fmt = _SNAPSHOT_FORMAT_MAP[output_format]

    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, idx in enumerate(indices):
        fname = os.path.join(output_dir,
                             f"snapshot_{i:04d}_frame{idx:05d}{ext}")
        write(fname, frames[idx], format=ase_fmt)
        paths.append(fname)

    print(f"Extracted {len(paths)} snapshots → {output_dir}/")
    return paths
