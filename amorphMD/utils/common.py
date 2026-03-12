"""
utils/common.py
---------------
Shared helpers used across all pipeline stages:

  get_mace_calculator   Build a MACE calculator (foundation or custom model)
  list_models           Print the full MACE model registry
  make_cubic            Reshape cell to a cube of equal volume
  build_md_dynamics     Construct NVT or NPT Berendsen dynamics
  resolve_ramp          Resolve T-ramp params (rate_K_per_ps or T_step style)
  MDLogger              Per-step thermodynamic logger
  TrajectoryWriter      Unified trajectory writer (extxyz/xyz/traj/lammps-dump)
  attach_outputs        Attach logger + trajectory to an ASE dynamics object
  merge_config          Deep-merge config overrides
"""

import os
from ase import units
from ase.io import write


# ─────────────────────────────────────────────────────────────────────────────
# MACE model registry
# ─────────────────────────────────────────────────────────────────────────────

MACE_FOUNDATION_MODELS = {
    # ── MACE-MP-0a ────────────────────────────────────────────────────────────
    "mace-mp-0a-small":    "small",
    "mace-mp-0a-medium":   "medium",
    "mace-mp-0a-large":    "large",
    # ── MACE-MP-0b ────────────────────────────────────────────────────────────
    "mace-mp-0b-small":    "small-0b",
    "mace-mp-0b-medium":   "medium-0b",
    "mace-mp-0b-large":    "large-0b",
    # ── MACE-MP-0b2 ───────────────────────────────────────────────────────────
    "mace-mp-0b2-small":   "small-0b2",
    "mace-mp-0b2-medium":  "medium-0b2",
    "mace-mp-0b2-large":   "large-0b2",
    # ── MACE-MP-0b3 ───────────────────────────────────────────────────────────
    "mace-mp-0b3-small":   "small-0b3",
    "mace-mp-0b3-medium":  "medium-0b3",
    "mace-mp-0b3-large":   "large-0b3",
    # ── MACE-MPA-0  (MPTrj + sAlex — recommended default) ────────────────────
    "mace-mpa-0":          "medium-mpa-0",
    "mace-mpa-0-medium":   "medium-mpa-0",
    # ── MACE-OMAT-0 ───────────────────────────────────────────────────────────
    "mace-omat-0-small":   "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-small.model",
    "mace-omat-0-medium":  "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model",
    "mace-omat-0":         "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model",
    # ── MACE-MATPES ───────────────────────────────────────────────────────────
    "mace-matpes-pbe":     "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model",
    "mace-matpes-r2scan":  "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-r2scan-omat-ft.model",
    # ── MACE-MH  (multi-domain: bulk + surface + molecule) ───────────────────
    "mace-mh-0":           "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-0.model",
    "mace-mh-1":           "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model",
    # ── MACE-OMOL (molecules) ─────────────────────────────────────────────────
    "mace-omol":           "https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/mace-omol-0-medium.model",
}

MODEL_DESCRIPTIONS = {
    "mace-mp-0a-small":    "MACE-MP-0a  small   | MPTrj | DFT PBE+U | initial release",
    "mace-mp-0a-medium":   "MACE-MP-0a  medium  | MPTrj | DFT PBE+U | initial release",
    "mace-mp-0a-large":    "MACE-MP-0a  large   | MPTrj | DFT PBE+U | initial release",
    "mace-mp-0b-small":    "MACE-MP-0b  small   | MPTrj | improved pair repulsion",
    "mace-mp-0b-medium":   "MACE-MP-0b  medium  | MPTrj | improved pair repulsion",
    "mace-mp-0b-large":    "MACE-MP-0b  large   | MPTrj | improved pair repulsion",
    "mace-mp-0b2-small":   "MACE-MP-0b2 small   | MPTrj | improved high-pressure stability",
    "mace-mp-0b2-medium":  "MACE-MP-0b2 medium  | MPTrj | improved high-pressure stability",
    "mace-mp-0b2-large":   "MACE-MP-0b2 large   | MPTrj | improved high-pressure stability",
    "mace-mp-0b3-small":   "MACE-MP-0b3 small   | MPTrj | fixed phonons vs 0b2",
    "mace-mp-0b3-medium":  "MACE-MP-0b3 medium  | MPTrj | fixed phonons vs 0b2",
    "mace-mp-0b3-large":   "MACE-MP-0b3 large   | MPTrj | fixed phonons vs 0b2",
    "mace-mpa-0":          "MACE-MPA-0  medium  | MPTrj+sAlex | ★ recommended default",
    "mace-mpa-0-medium":   "MACE-MPA-0  medium  | MPTrj+sAlex | ★ recommended default",
    "mace-omat-0-small":   "MACE-OMAT-0 small   | OMAT | excellent phonons | ASL license",
    "mace-omat-0-medium":  "MACE-OMAT-0 medium  | OMAT | excellent phonons | ASL license",
    "mace-matpes-pbe":     "MACE-MATPES-PBE     | MATPES-PBE  | DFT PBE, no +U | ASL license",
    "mace-matpes-r2scan":  "MACE-MATPES-r2SCAN  | MATPES-r2SCAN | better functional | ASL license",
    "mace-mh-0":           "MACE-MH-0           | multi-domain bulk/surface/molecule",
    "mace-mh-1":           "MACE-MH-1           | multi-domain | ★ best cross-domain accuracy",
    "mace-omol":           "MACE-OMOL-0         | OMOL | optimised for molecules",
}


def list_models():
    """Print all available MACE foundation models to stdout."""
    bar = "-" * 74
    print(f"\n{bar}")
    print("  Available MACE foundation models  (pass with --model NAME)")
    print(bar)
    for name, desc in MODEL_DESCRIPTIONS.items():
        print(f"  {name:<26s}  {desc}")
    print(bar)
    print("  Custom model:  --model-path /path/to/my_finetuned.model\n")


# ─────────────────────────────────────────────────────────────────────────────
# Calculator factory
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_device(device: str) -> str:
    """Resolve 'auto' → 'cuda' if a GPU is available, else 'cpu'."""
    if device != "auto":
        return device
    try:
        import torch
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        resolved = "cpu"
    print(f"[MACE] device auto-detected → {resolved}")
    return resolved


def get_mace_calculator(model: str = "mace-mpa-0",
                        device: str = "auto",
                        model_path: str | None = None,
                        **kwargs):
    """
    Build and return a MACE ASE calculator.

    Parameters
    ----------
    model      : str   Short name from MACE_FOUNDATION_MODELS, or a raw mace_mp
                       string (e.g. "medium-mpa-0"), or a direct HTTPS URL.
                       Ignored when model_path is set.
    device     : str   "cuda", "cpu", or "auto" (auto-detects GPU).
    model_path : str   Path to a local .model file (fine-tuned / custom model).
                       Takes priority over model.
    **kwargs          Extra keyword arguments forwarded to the calculator.

    Returns
    -------
    ASE calculator (MACECalculator or mace_mp wrapper)
    """
    from mace.calculators import mace_mp, MACECalculator
    device = _resolve_device(device)

    # ── local / custom model ─────────────────────────────────────────────────
    if model_path is not None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Custom model file not found: {model_path}\n"
                "Please check the path to your .model file."
            )
        print(f"[MACE] Loading custom model: {model_path}")
        return MACECalculator(model_paths=model_path, device=device, **kwargs)

    # ── resolve short-name ───────────────────────────────────────────────────
    resolved = MACE_FOUNDATION_MODELS.get(model, model)

    if resolved.startswith("https://") or os.path.isfile(resolved):
        print(f"[MACE] Loading from URL/path: {resolved[:80]}")
        return MACECalculator(model_paths=resolved, device=device, **kwargs)
    else:
        print(f"[MACE] Loading '{model}' → mace_mp(model='{resolved}', device='{device}')")
        return mace_mp(model=resolved, device=device, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Cell helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_cubic(atoms):
    """Reshape the cell to a cube of equal volume, rescaling atom positions."""
    volume = atoms.get_volume()
    L = volume ** (1.0 / 3.0)
    atoms.set_cell([[L, 0, 0], [0, L, 0], [0, 0, L]], scale_atoms=True)
    atoms.wrap()
    return atoms


# ─────────────────────────────────────────────────────────────────────────────
# MD ensemble builder  (shared by melt, quench, equilibrate)
# ─────────────────────────────────────────────────────────────────────────────

def build_md_dynamics(atoms, ensemble: str, temperature_K: float,
                      timestep_fs: float = 1.0,
                      taut_fs: float = 100.0,
                      pressure_bar: float = 1.0,
                      taup_fs: float = 1000.0,
                      compressibility: float = 4.5e-5):
    """
    Construct NVT or NPT Berendsen dynamics.

    Parameters
    ----------
    atoms           : ase.Atoms with calculator attached
    ensemble        : "NVT" or "NPT"
    temperature_K   : target temperature in K
    timestep_fs     : MD timestep in femtoseconds
    taut_fs         : thermostat coupling time (fs)
    pressure_bar    : target pressure in bar (NPT only)
    taup_fs         : barostat coupling time (fs, NPT only)
    compressibility : isothermal compressibility 1/bar (NPT only)

    Returns
    -------
    NVTBerendsen or NPTBerendsen dynamics object
    """
    from ase.md.nvtberendsen import NVTBerendsen
    from ase.md.nptberendsen import NPTBerendsen

    dt = timestep_fs * units.fs
    ensemble = ensemble.upper()

    if ensemble == "NVT":
        return NVTBerendsen(
            atoms,
            timestep=dt,
            temperature_K=temperature_K,
            taut=taut_fs * units.fs,
        )
    elif ensemble == "NPT":
        return NPTBerendsen(
            atoms,
            timestep=dt,
            temperature_K=temperature_K,
            taut=taut_fs * units.fs,
            pressure=pressure_bar,
            taup=taup_fs * units.fs,
            compressibility=compressibility,
        )
    else:
        raise ValueError(f"Unknown ensemble '{ensemble}'. Use 'NVT' or 'NPT'.")


# ─────────────────────────────────────────────────────────────────────────────
# Temperature ramp resolver  (shared by melt_cell and quench)
# ─────────────────────────────────────────────────────────────────────────────

def resolve_ramp(cfg: dict, heating: bool = True) -> tuple[list[int], int, float]:
    """
    Build a temperature ramp from stage config.

    Supports two APIs:
      (a) Explicit:   T_step + steps_per_T
      (b) Rate-based: rate_K_per_ps  (overrides T_step / steps_per_T)

    Parameters
    ----------
    cfg     : stage config dict  (melt or quench sub-dict)
    heating : True → heating ramp, False → cooling ramp

    Returns
    -------
    temps       : list[int]   temperature sequence [T_start, ..., T_end]
    steps_per_T : int         MD steps at each temperature
    rate        : float       actual heating/cooling rate in K/ps

    Examples
    --------
    # 100 K/ps heating (explicit)
    cfg = {"T_start": 300, "T_end": 2500, "T_step": 100,
           "steps_per_T": 1000, "timestep_fs": 1.0}

    # 50 K/ps heating (rate-based)
    cfg = {"T_start": 300, "T_end": 2500,
           "rate_K_per_ps": 50, "timestep_fs": 1.0}
    """
    T_start    = cfg["T_start"]
    T_end      = cfg["T_end"]
    timestep   = cfg.get("timestep_fs", 1.0)
    rate_given = cfg.get("rate_K_per_ps", None)

    if rate_given is not None:
        rate = float(rate_given)
        if rate <= 0:
            raise ValueError(f"rate_K_per_ps must be positive, got {rate}")
        dT          = abs(T_end - T_start)
        T_step      = max(1, min(100, dT // 10))
        dt_ps       = timestep / 1000.0
        steps_per_T = max(1, round(T_step / (rate * dt_ps)))
        actual_rate = T_step / (steps_per_T * dt_ps)
    else:
        T_step      = abs(cfg.get("T_step", 100))
        steps_per_T = cfg.get("steps_per_T", 1000)
        dt_ps       = timestep / 1000.0
        actual_rate = T_step / (steps_per_T * dt_ps)

    if heating:
        if T_end <= T_start:
            raise ValueError(
                f"Heating ramp requires T_end ({T_end}) > T_start ({T_start})")
        temps = list(range(T_start, T_end + T_step, T_step))
        if temps[-1] > T_end:
            temps[-1] = T_end
    else:
        if T_end >= T_start:
            raise ValueError(
                f"Cooling ramp requires T_end ({T_end}) < T_start ({T_start})")
        temps = list(range(T_start, T_end - T_step, -T_step))
        if temps[-1] < T_end:
            temps[-1] = T_end

    return temps, steps_per_T, actual_rate


# ─────────────────────────────────────────────────────────────────────────────
# Thermodynamic logger
# ─────────────────────────────────────────────────────────────────────────────

class MDLogger:
    """Per-step thermodynamic logger for NVT and NPT MD runs."""

    HEADER_NVT = ("  {:>8s}  {:>14s}  {:>14s}  {:>10s}  {:>14s}\n".format(
        "Step", "Epot/atom(eV)", "Ekin/atom(eV)", "Temp(K)", "Density(g/cm3)"))
    HEADER_NPT = ("  {:>8s}  {:>14s}  {:>14s}  {:>10s}  {:>13s}  {:>14s}\n".format(
        "Step", "Epot/atom(eV)", "Ekin/atom(eV)", "Temp(K)",
        "Press(bar)", "Density(g/cm3)"))

    def __init__(self, path: str, ensemble: str = "NVT"):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._f = open(path, "w")
        self.ensemble = ensemble.upper()
        header = self.HEADER_NPT if self.ensemble == "NPT" else self.HEADER_NVT
        self._f.write(header)
        self._f.write("  " + "-" * (len(header) - 3) + "\n")
        self._f.flush()

    def log(self, dyn, atoms):
        epot    = atoms.get_potential_energy()
        ekin    = atoms.get_kinetic_energy()
        temp    = atoms.get_temperature()
        n       = len(atoms)
        mass_g  = sum(a.mass for a in atoms) * 1.66053906660e-24
        vol_cm3 = atoms.get_volume() * 1.0e-24
        density = mass_g / vol_cm3

        if self.ensemble == "NPT":
            stress = atoms.get_stress()
            p_bar  = -stress[:3].sum() / 3.0 * 1602.1766208
            line = ("  {:8d}  {:14.6f}  {:14.6f}  {:10.2f}  {:13.3f}  {:14.6f}\n".format(
                dyn.nsteps, epot / n, ekin / n, temp, p_bar, density))
        else:
            line = ("  {:8d}  {:14.6f}  {:14.6f}  {:10.2f}  {:14.6f}\n".format(
                dyn.nsteps, epot / n, ekin / n, temp, density))

        print(line.strip())
        self._f.write(line)
        self._f.flush()

    def close(self):
        self._f.close()


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory writer
# ─────────────────────────────────────────────────────────────────────────────

TRAJ_FORMATS = {
    "extxyz":      (".extxyz", "extxyz"),
    "xyz":         (".xyz",    "xyz"),
    "traj":        (".traj",   None),
    "lammps":      (".dump",   "lammps-dump-text"),
    "lammps-dump": (".dump",   "lammps-dump-text"),
}


class TrajectoryWriter:
    """
    Unified trajectory writer for MD stages.

    Supports extxyz (default), xyz, ASE binary traj, and lammps-dump.
    extxyz stores cell, PBC, energy, and forces in every frame — recommended
    for post-processing with OVITO, VESTA, or custom analysis scripts.

    Parameters
    ----------
    path  : str        Output file path. Extension auto-added if absent.
    fmt   : str        "extxyz" (default), "xyz", "traj", or "lammps-dump"
    atoms : ase.Atoms  Required only for "traj" format.
    """

    def __init__(self, path: str, fmt: str = "extxyz", atoms=None):
        fmt = fmt.lower().strip()
        if fmt not in TRAJ_FORMATS:
            raise ValueError(
                f"Unknown trajectory format '{fmt}'. "
                f"Choose from: {list(TRAJ_FORMATS.keys())}"
            )
        self.fmt = fmt
        ext, self._ase_fmt = TRAJ_FORMATS[fmt]

        base, existing_ext = os.path.splitext(path)
        if existing_ext.lower() not in (".extxyz", ".xyz", ".traj", ".dump"):
            path = base + ext
        self.path = path

        if fmt == "traj":
            from ase.io.trajectory import Trajectory as _AseTraj
            if atoms is None:
                raise ValueError("atoms must be provided for 'traj' format")
            self._writer = _AseTraj(path, "w", atoms)
        else:
            self._writer = None

    def write(self, atoms):
        """Write one frame."""
        atoms.wrap()
        if self.fmt == "traj":
            self._writer.write()
        else:
            write(self.path, atoms, format=self._ase_fmt, append=True)

    def close(self):
        if self.fmt == "traj" and self._writer is not None:
            self._writer.close()


# ─────────────────────────────────────────────────────────────────────────────
# MD attachment helper
# ─────────────────────────────────────────────────────────────────────────────

def attach_outputs(dyn, atoms, logger: MDLogger,
                   traj_writer: TrajectoryWriter,
                   interval: int = 10):
    """
    Attach logging and trajectory writing to a dynamics object.

    Parameters
    ----------
    dyn          : ASE dynamics object
    atoms        : ase.Atoms
    logger       : MDLogger
    traj_writer  : TrajectoryWriter
    interval     : int  steps between outputs
    """
    dyn.attach(lambda: logger.log(dyn, atoms), interval=interval)
    dyn.attach(lambda: traj_writer.write(atoms), interval=interval)


# ─────────────────────────────────────────────────────────────────────────────
# Config deep-merge
# ─────────────────────────────────────────────────────────────────────────────

def merge_config(defaults: dict, overrides: dict | None) -> dict:
    """Deep-merge *overrides* into a copy of *defaults*."""
    import copy
    cfg = copy.deepcopy(defaults)
    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k] = merge_config(cfg[k], v)
            else:
                cfg[k] = v
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot extraction from trajectory
# ─────────────────────────────────────────────────────────────────────────────

def extract_snapshots(
    traj_file: str,
    n: int = 10,
    select: str = "uniform",
    out_dir: str = "snapshots",
) -> list[str]:
    """
    Extract N snapshots from a trajectory file and save as individual extxyz files.

    Useful for post-processing a Stage 4 trajectory that was run without
    --sample-interval, or for re-sampling an existing trajectory.

    Parameters
    ----------
    traj_file : str
        Path to trajectory file (extxyz, xyz, traj, lammps-dump).
    n         : int
        Number of snapshots to extract (default: 10).
    select    : str
        Frame selection strategy:
          "uniform" – evenly spaced across full trajectory (default)
          "first"   – first N frames
          "last"    – last N frames
          "random"  – N random frames
    out_dir   : str
        Directory to write snapshot files (created if absent).

    Returns
    -------
    list[str]
        Paths to all written snapshot files.
    """
    import os
    import numpy as np
    from ase.io import read, write

    print(f"[extract_snapshots] Loading: {traj_file}")
    frames   = read(traj_file, index=":")
    n_frames = len(frames)
    print(f"[extract_snapshots] Total frames: {n_frames}")

    if n > n_frames:
        print(f"[extract_snapshots] WARNING: requested {n} but only "
              f"{n_frames} frames available — using all.")
        n = n_frames

    # ── select indices ────────────────────────────────────────────────────────
    if select == "uniform":
        indices = np.linspace(0, n_frames - 1, n, dtype=int)
    elif select == "first":
        indices = np.arange(n)
    elif select == "last":
        indices = np.arange(n_frames - n, n_frames)
    elif select == "random":
        indices = np.sort(np.random.choice(n_frames, n, replace=False))
    else:
        raise ValueError(f"Unknown select strategy '{select}'. "
                         "Choose: uniform, first, last, random.")

    print(f"[extract_snapshots] Strategy: {select}  →  frames {indices.tolist()}")

    # ── write snapshots ───────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    for snap_idx, frame_idx in enumerate(indices):
        atoms = frames[int(frame_idx)]
        fname = os.path.join(
            out_dir,
            f"snapshot_{snap_idx:04d}_frame{frame_idx:05d}.extxyz"
        )
        write(fname, atoms, format="extxyz")
        paths.append(fname)
        print(f"  snapshot_{snap_idx:04d}  ←  frame {frame_idx:5d}  →  {fname}")

    print(f"[extract_snapshots] Done. {n} snapshots saved to '{out_dir}/'")
    return paths
