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


def calculator_supports_stress(calc) -> bool:
    """Return True if *calc* advertises a stress tensor.

    Variable-cell operations (NPT barostats, cell-filter optimisation) need
    the stress. The classical pair potentials (Lennard-Jones, Buckingham)
    implement only energy + forces, so this returns False for them.
    """
    props = getattr(calc, "implemented_properties", None) or []
    return "stress" in props


def require_stress(calc, context: str) -> None:
    """Raise a clear error if *calc* cannot provide stress for *context*.

    Prevents an opaque ``PropertyNotImplementedError`` from surfacing deep
    inside ASE when a stress-less calculator is used with a barostat or a
    cell filter.
    """
    if not calculator_supports_stress(calc):
        name = type(calc).__name__ if calc is not None else "the calculator"
        props = getattr(calc, "implemented_properties", []) if calc is not None else []
        raise RuntimeError(
            f"{context} requires a stress tensor, but {name} implements only "
            f"{list(props)}. Either use a stress-capable MLIP backend "
            f"(mace / chgnet / sevennet), or run with a fixed cell when using "
            f"a classical pair potential (lennard-jones / buckingham): pass "
            f"-C none on the CLI, or set cell_filter: none under opt: (or "
            f"random_gen:) in the YAML, and use an NVT ensemble for MD stages."
        )


def build_md_dynamics(atoms, ensemble: str = "NVT", T: float = 300.0,
                      timestep: float = 1.0, friction: float = 0.01,
                      ttime: float = 25.0, pfactor: float | None = None,
                      external_stress: float = 0.0,
                      npt_method: str = "berendsen",
                      taup_factor: float = 10.0,
                      compressibility_GPa: float = 100.0,
                      rng=None,
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
        if rng is not None:
            kwargs["rng"] = rng          # seeded thermostat noise
        dyn = Langevin(atoms, timestep=dt, temperature_K=T,
                       friction=friction / units.fs, **kwargs)
        return dyn

    if ensemble.upper() != "NPT":
        raise ValueError(f"Unknown ensemble '{ensemble}'. Use 'NVT' or 'NPT'.")

    # NPT barostats need the stress tensor; fail early and clearly for
    # stress-less calculators (classical LJ / Buckingham) rather than deep
    # inside the ASE integrator.
    require_stress(getattr(atoms, "calc", None), f"NPT ({npt_method}) dynamics")

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

def parse_index_spec(spec, n_total: int | None = None) -> set[int]:
    """``"80-90"``, ``"0,5,7-9"`` or a sequence of ints -> set of indices.

    Ranges are inclusive. ``n_total`` (if given) bounds the result.
    """
    if spec is None:
        return set()
    if not isinstance(spec, str):
        idx = {int(i) for i in spec}
    else:
        idx = set()
        for part in spec.replace(" ", "").split(","):
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                if not (a.isdigit() and b.isdigit()):
                    raise ValueError(f"bad index range '{part}' in '{spec}'")
                a, b = int(a), int(b)
                if b < a:
                    raise ValueError(f"index range '{part}' runs backwards")
                idx.update(range(a, b + 1))
            elif part.isdigit():
                idx.add(int(part))
            else:
                raise ValueError(f"bad index '{part}' in '{spec}'")
    if n_total is not None:
        idx = {i for i in idx if 0 <= i < n_total}
    return idx


def stage_rng(seed, stage: int, run_index: int = 0):
    """Per-stage, per-run NumPy Generator derived from the global ``seed``.

    ``None`` seed -> ``None`` (ASE's default, unseeded, generator). The stream
    depends only on (seed, stage, run_index), so stage 4 of run 7 draws the
    same thermostat noise whatever ran before it or on another machine.
    """
    if seed is None:
        return None
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(stage), int(run_index)]))


def run_index_from_cwd() -> int:
    """Index of a ``run_NNNN`` working directory (batch / ensemble modes), else 0."""
    import re
    m = re.search(r"run_(\d+)", os.path.basename(os.getcwd()))
    return int(m.group(1)) if m else 0


def resolve_ramp(T_start: float, T_end: float, T_step: float) -> list[float]:
    """
    Generate the list of temperatures for a ramp from ``T_start`` to ``T_end``.

    The ramp direction is taken from the endpoints, so only the *magnitude*
    of ``T_step`` matters — a mis-signed step (e.g. a positive step for a
    cooling ramp) can no longer produce an empty list or an infinite loop.
    Float steps are supported. ``T_end`` is always the final entry, even when
    the span is not an integer multiple of the step, and the ramp never
    overshoots past ``T_end``. ``T_start`` itself is NOT in the list: the
    system already sits at ``T_start`` when the ramp begins, so the segments
    are the *targets* T_start+step, T_start+2*step, ..., T_end. With
    ``steps_per_T`` MD steps per segment the realised rate then equals the
    configured one (an extra segment at T_start used to lower it by
    n/(n+1)).

    Raises
    ------
    ValueError
        If ``T_step`` has zero magnitude.
    """
    T_start = float(T_start)
    T_end = float(T_end)
    step = abs(float(T_step))
    if step == 0:
        raise ValueError("T_step magnitude cannot be zero.")

    span = abs(T_end - T_start)
    # Number of full steps that fit strictly inside the span (the -1e-9 keeps
    # an exactly-divisible span from emitting a duplicate endpoint below).
    n = int(np.ceil(span / step - 1e-9))
    sign = 1.0 if T_end >= T_start else -1.0

    temps = [round(T_start + sign * step * k, 2) for k in range(1, n)]
    temps.append(round(T_end, 2))
    # Near-divisible spans can round the last interior point onto T_end.
    if len(temps) >= 2 and abs(temps[-1] - temps[-2]) < 1e-9:
        temps.pop(-2)
    return temps


# ═════════════════════════════════════════════════════════════════════════════
# Trajectory formats
# ═════════════════════════════════════════════════════════════════════════════

# Only formats that store cell, pbc AND momenta (needed for frame-level
# resume) and that ASE can both write and read.  "xyz" is accepted as an
# alias of extxyz; lammps-dump is read-only in ASE and was removed.
TRAJ_FORMATS = {"extxyz", "traj"}
_TRAJ_ALIASES = {"xyz": "extxyz"}


# ═════════════════════════════════════════════════════════════════════════════
# MD Logger
# ═════════════════════════════════════════════════════════════════════════════

class MDLogger:
    """
    Per-step MD logger that writes to both a file and stdout.

    Logs step number, time (ps), temperature (K), potential energy (eV),
    kinetic energy (eV), total energy (eV), and volume (Å³).
    """

    def __init__(self, logfile: str, mode: str = "w", step_offset: int = 0):
        self.step_offset = int(step_offset)
        self._fh = open(logfile, mode)
        if mode != "a":     # resumed runs continue the existing table
            header = (f"{'Step':>8s}  {'Time_ps':>10s}  {'T_K':>8s}  "
                      f"{'Epot_eV':>12s}  {'Ekin_eV':>12s}  "
                      f"{'Etot_eV':>12s}  {'Vol_A3':>10s}")
            self._fh.write(header + "\n")
            self._fh.write("-" * len(header) + "\n")
            self._fh.flush()

    def log(self, dyn, atoms):
        step = dyn.nsteps + self.step_offset
        t_ps = (dyn.get_time() + self.step_offset * dyn.dt) / units.fs / 1000.0
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
        self.fmt = _TRAJ_ALIASES.get(self.fmt, self.fmt)
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
                   append: bool = False, step_offset: int = 0):
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
    logger = MDLogger(logfile, mode="a" if append else "w", step_offset=step_offset)
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
        # Write a wrapped COPY: wrapping the live atoms between run()
        # segments makes ASE's NPT (parrinello-rahman) integrator refuse
        # to continue ("modified the atoms"), and it is not needed for
        # the integration.  Energies/forces are carried over so the
        # trajectory frames stay self-contained.
        img = atoms.copy()
        img.wrap()
        calc = getattr(atoms, "calc", None)
        if calc is not None and getattr(calc, "results", None):
            from ase.calculators.singlepoint import SinglePointCalculator
            res = {k: calc.results[k] for k in ("energy", "forces", "stress")
                   if k in calc.results}
            if res:
                img.calc = SinglePointCalculator(img, **res)
        traj.write(img)

    dyn.attach(_observe, interval=interval)

    return logger, traj


def read_md_checkpoint(trajfile: str, interval: int = TRAJ_LOG_INTERVAL):
    """Last complete frame of an MD trajectory and the MD steps it represents.

    The returned frame carries the MD momenta (extxyz stores them), so the
    stage can continue from it. A trajectory whose LAST frame was torn by a
    walltime kill is truncated to its complete frames (which are kept) rather
    than discarded whole.

    Returns ``None`` when the file is missing, empty or unreadable.
    """
    if not (trajfile and os.path.isfile(trajfile)):
        return None
    try:
        frames = read(trajfile, index=":")
    except Exception:
        from ase.io import iread
        frames = []
        try:
            for fr in iread(trajfile, index=":"):
                frames.append(fr)
        except Exception:
            pass
        if not frames:
            return None
        import warnings
        warnings.warn(f"{trajfile}: last frame is incomplete (interrupted "
                      f"write); keeping the {len(frames)} complete frame(s).")
        try:
            write(trajfile, frames)
        except Exception:
            pass
    if len(frames) < 2:      # frame 0 is the starting structure: nothing done
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

def set_md_temperature(dyn, T: float) -> None:
    """Change the target temperature of a running ASE dynamics object.

    Langevin, NPTBerendsen and the Melchionna/NPT integrators expose
    ``set_temperature``; ``IsotropicMTKNPT`` (npt_method "mtk") does not,
    so its thermostat and barostat kT are updated directly.  Used by the
    heating and cooling ramps.
    """
    if hasattr(dyn, "set_temperature"):
        dyn.set_temperature(temperature_K=T)
        return
    kT = T * units.kB
    hit = False
    for obj in (dyn, getattr(dyn, "_thermostat", None), getattr(dyn, "_barostat", None)):
        if obj is None:
            continue
        if hasattr(obj, "_kT"):
            obj._kT = kT; hit = True
        if hasattr(obj, "_temperature_K"):
            obj._temperature_K = T; hit = True
    if not hit:
        raise AttributeError(
            f"{type(dyn).__name__} has no set_temperature and no known "
            f"temperature attribute; cannot ramp its temperature.")


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
