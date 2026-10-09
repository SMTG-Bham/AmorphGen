"""
amorphgen.utils.equilibration
-------------------------------
Equilibration convergence analysis for AmorphGen MD trajectories.

Provides functions to assess whether an MD equilibration run has converged:
  1. Energy vs time (running average + drift detection)
  2. Block averaging (quantitative equilibration test)
  3. RDF in time windows (structural convergence)
  4. Mean Square Displacement (diffusion / liquid vs glass)
  5. Coordination number vs time
  6. Temperature vs time

Usage
-----
::

    from amorphgen.utils.equilibration import convergence_report

    # Quick all-in-one convergence report
    report = convergence_report(
        "stage4_eq.xyz",
        timestep_fs=1.0,
        T_target=3000,
    )

    # Or individual analyses
    fig_energy, drift = plot_energy_convergence(traj, timestep_fs=1.0)
    is_eq, block_data = block_average_test(traj, n_blocks=4)
    fig_msd, D_dict = plot_msd(traj, timestep_fs=1.0)

Notes
-----
- Two input modes: trajectory file (.xyz/.traj) or log file (.log).
  Log files are faster (no atomic positions to load) but only support
  energy, temperature, and block-average analyses.
  Trajectory files are needed for MSD, RDF, and CN analyses.
- Default timestep is taken from AmorphGen's default_config.py (0.5 fs), so
  the time axes are right for a default run. Trajectories are written every
  TRAJ_LOG_INTERVAL MD steps; pass ``frame_stride`` if yours differ.
"""

from __future__ import annotations

import numpy as np
from os import PathLike
from ase.io import read
from ase.neighborlist import neighbor_list

from .common import TRAJ_LOG_INTERVAL
from ..configs.default_config import DEFAULT_CONFIG

# Default MD timestep for the diagnostics, sourced from DEFAULT_CONFIG so the
# time axes are right out of the box for a default AmorphGen run and cannot
# drift from the pipeline. Always pass your run's actual timestep if it
# differs.
DEFAULT_TIMESTEP_FS: float = float(DEFAULT_CONFIG["melt"]["timestep"])


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_trajectory(source) -> list:
    """Load trajectory from file path or list of Atoms."""
    from ase import Atoms

    if isinstance(source, list):
        if source and isinstance(source[0], Atoms):
            return source
        return [read(f) for f in source]
    if isinstance(source, (str, PathLike)):
        return read(source, index=":")
    # ASE Trajectory object or other iterable
    return list(source)


def parse_md_log(logfile: str) -> dict:
    """
    Parse an AmorphGen MD stage log file.

    Expects whitespace-separated columns:
        Step Time(ps) T(K) Epot(eV) Ekin(eV) Etot(eV) Vol(A^3)
        P_GPa density_g_cm3

    Lines starting with 'Step', '-', or '->' are skipped as headers/markers.

    Returns
    -------
    dict with keys: 'step', 'time_ps', 'T_K', 'Epot_eV', 'Ekin_eV',
    'Etot_eV', 'Vol_A3', 'P_GPa', 'density_g_cm3' — each a numpy array.
    Legacy logs without pressure/density receive NaN in those columns.
    """
    data = {k: [] for k in ['step', 'time_ps', 'T_K', 'Epot_eV',
                             'Ekin_eV', 'Etot_eV', 'Vol_A3', 'P_GPa',
                             'density_g_cm3']}
    with open(logfile) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Step') or line.startswith('-'):
                continue
            if line.startswith('->'):
                continue
            parts = line.split()
            if len(parts) >= 7:
                try:
                    # Convert the entire row before appending: a malformed
                    # later column must not leave arrays of different sizes.
                    row = [int(parts[0]), *(float(v) for v in parts[1:7])]
                    row.extend(float(parts[i]) if len(parts) > i else np.nan
                               for i in (7, 8))
                except (ValueError, IndexError):
                    continue
                for key, value in zip(data, row):
                    data[key].append(value)
    return {k: np.array(v) for k, v in data.items()}


def _infer_log_interval_ps(log_data: dict) -> float | None:
    """Infer time interval between log entries (ps), NOT the MD timestep.

    The log interval = MD_timestep × print_interval (typically 100 steps).
    This should NOT be used as the MD timestep.
    """
    t = log_data['time_ps']
    if len(t) >= 2:
        dt_ps = t[1] - t[0]
        if dt_ps > 0:
            return dt_ps
    return None


def running_average(data: np.ndarray, window: int) -> np.ndarray:
    """Compute running average with given window size."""
    if window >= len(data):
        return np.full_like(data, np.mean(data))
    kernel = np.ones(window) / window
    avg = np.convolve(data, kernel, mode="valid")
    pad_left = (len(data) - len(avg)) // 2
    pad_right = len(data) - len(avg) - pad_left
    return np.pad(avg, (pad_left, pad_right), mode="edge")


def _sample_times(n_frames, timestep_fs, frame_stride, time_ps=None):
    """Validate actual saved-frame times, including irregular final samples."""
    if time_ps is None:
        if timestep_fs <= 0 or frame_stride <= 0:
            raise ValueError("timestep_fs and frame_stride must be positive")
        return np.arange(n_frames) * timestep_fs * frame_stride / 1000.0
    times = np.asarray(time_ps, dtype=float)
    if (times.shape != (n_frames,) or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)):
        raise ValueError("time_ps must contain one finite, strictly increasing time per frame")
    return times


# ══════════════════════════════════════════════════════════════════════════════
# 1. Energy convergence
# ══════════════════════════════════════════════════════════════════════════════

def extract_energies(source, n_atoms: int | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract potential energies (eV) and per-atom energies.

    Parameters
    ----------
    source : str or list
        Trajectory file, log file (.log), or list of Atoms.
    n_atoms : int, optional
        Required if source is a .log file.

    Returns
    -------
    energies : np.ndarray  — total potential energy (eV)
    energies_per_atom : np.ndarray  — energy per atom (eV/atom)
    """
    if isinstance(source, str) and source.endswith('.log'):
        log_data = parse_md_log(source)
        energies = log_data['Epot_eV']
        if n_atoms is None:
            raise ValueError("n_atoms required when reading from log file")
        return energies, energies / n_atoms

    frames = _load_trajectory(source)
    if not frames:
        return np.array([]), np.array([])
    n_atoms = len(frames[0])

    try:
        energies = np.array([atoms.get_potential_energy()
                             for atoms in frames])
    except RuntimeError:
        # Atoms don't have a calculator — try info dict
        energies = []
        for atoms in frames:
            e = atoms.info.get('energy', atoms.info.get('Energy', None))
            if e is None:
                raise RuntimeError(
                    "Trajectory frames have no energy data. "
                    "Use the .log file instead: "
                    "convergence_report('stage4_eq.log', n_atoms=72)")
            energies.append(e)
        energies = np.array(energies)

    return energies, energies / n_atoms


def plot_energy_convergence(source, timestep_fs: float = DEFAULT_TIMESTEP_FS,
                            window_ps: float = 0.5,
                            per_atom: bool = True,
                            n_atoms: int | None = None,
                            ax=None,
                            frame_stride: int = TRAJ_LOG_INTERVAL):
    """
    Plot potential energy vs time with running average and linear drift.

    Parameters
    ----------
    source : str or list
        Trajectory file, log file (.log), or list of Atoms.
    timestep_fs : float
        MD timestep in femtoseconds (defaults to AmorphGen's
        ``DEFAULT_CONFIG`` value, 0.5 fs). Pass your run's actual timestep
        if it differs.
    window_ps : float
        Running average window in picoseconds.
    n_atoms : int, optional
        Required if source is a .log file.

    Returns
    -------
    fig : matplotlib Figure
    drift_eV_per_ps : float
        Linear drift in energy. Should be ~0 if equilibrated.
    """
    import matplotlib.pyplot as plt

    # Get time array from log or compute from timestep
    if isinstance(source, str) and source.endswith('.log'):
        log_data = parse_md_log(source)
        energies_total = log_data['Epot_eV']
        if n_atoms is None:
            raise ValueError("n_atoms required for .log file")
        energies_per_atom = energies_total / n_atoms
        time_ps = log_data['time_ps']
        # Auto-detect timestep from log
        # Note: log time column is already in ps, no timestep inference needed
        e = energies_per_atom if per_atom else energies_total
    else:
        energies_total, energies_per_atom = extract_energies(source,
                                                              n_atoms)
        e = energies_per_atom if per_atom else energies_total
        n_steps = len(e)
        # One frame every frame_stride MD steps: real time = index*dt*stride.
        time_ps = np.arange(n_steps) * timestep_fs * frame_stride / 1000.0

    # window_steps counts FRAMES: one frame spans timestep_fs*frame_stride fs.
    window_steps = max(1, int(window_ps * 1000 / (timestep_fs * frame_stride)))
    e_avg = running_average(e, window_steps)

    # Linear fit to detect drift
    coeffs = np.polyfit(time_ps, e, 1)
    drift_eV_per_ps = coeffs[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    ax.plot(time_ps, e, alpha=0.3, lw=0.5, color="steelblue", label="Raw")
    ax.plot(time_ps, e_avg, color="darkblue", lw=1.5,
            label=f"Running avg ({window_ps} ps)")
    ax.plot(time_ps, np.polyval(coeffs, time_ps), "--", color="red", lw=1,
            label=f"Drift: {drift_eV_per_ps:.4f} eV/atom/ps")

    unit = "eV/atom" if per_atom else "eV"
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(f"Potential energy ({unit})")
    ax.set_title("Energy convergence")
    ax.legend(fontsize=9)
    fig.tight_layout()

    return fig, drift_eV_per_ps


# ══════════════════════════════════════════════════════════════════════════════
# 2. Block averaging
# ══════════════════════════════════════════════════════════════════════════════

def block_average_test(source, n_blocks: int = 4,
                       discard_fraction: float = 0.1,
                       n_atoms: int | None = None
                       ) -> tuple[bool | None, dict]:
    """
    Split trajectory into blocks and compare mean energies.

    A system is considered equilibrated if the block means agree
    within 2x the standard error of the mean (SEM).

    Parameters
    ----------
    source : str or list
        Trajectory file, log file (.log), or list of Atoms.
    n_blocks : int
        Number of equal blocks to split the production phase into.
    discard_fraction : float
        Fraction of initial trajectory to discard (thermalisation).
    n_atoms : int, optional
        Required if source is a .log file.

    Returns
    -------
    is_equilibrated : bool
    block_data : dict
        Keys: block_means, overall_mean, overall_std, sem,
        max_deviation, threshold, is_equilibrated.
    """
    if n_blocks < 2 or not 0 <= discard_fraction < 1:
        raise ValueError("n_blocks must be >= 2 and discard_fraction in [0, 1)")
    _, e_per_atom = extract_energies(source, n_atoms)

    n_discard = int(len(e_per_atom) * discard_fraction)
    e_prod = e_per_atom[n_discard:]

    block_size = len(e_prod) // n_blocks
    if block_size < 2 or not np.all(np.isfinite(e_prod)):
        return None, {
            "status": "insufficient_data", "block_means": [],
            "overall_mean": None, "overall_std": None, "sem": None,
            "max_deviation": None, "threshold": None,
            "is_equilibrated": None,
        }
    block_means = np.array([
        np.mean(e_prod[i * block_size:(i + 1) * block_size])
        for i in range(n_blocks)
    ])

    overall_mean = np.mean(e_prod)
    overall_std = np.std(e_prod)
    sem = overall_std / np.sqrt(len(e_prod))

    max_deviation = np.max(np.abs(block_means - overall_mean))
    # A block mean scatters by std/sqrt(block_size), not by the whole-run
    # SEM (std/sqrt(N)); the old threshold was sqrt(n_blocks) too strict and
    # flagged genuinely equilibrated runs as "NOT EQUILIBRATED".
    threshold = 2 * overall_std / np.sqrt(max(block_size, 1))

    is_equilibrated = bool(max_deviation <= threshold)

    return is_equilibrated, {
        "status": "ok",
        "block_means": block_means,
        "overall_mean": overall_mean,
        "overall_std": overall_std,
        "sem": sem,
        "max_deviation": max_deviation,
        "threshold": threshold,
        "is_equilibrated": is_equilibrated,
    }


def plot_block_averages(source, n_blocks: int = 4,
                        discard_fraction: float = 0.1,
                        timestep_fs: float = DEFAULT_TIMESTEP_FS,
                        n_atoms: int | None = None, ax=None,
                        frame_stride: int = TRAJ_LOG_INTERVAL):
    """Visualise block averaging: block means vs overall mean +/- 2*SEM."""
    import matplotlib.pyplot as plt

    is_eq, bd = block_average_test(source, n_blocks, discard_fraction,
                                   n_atoms=n_atoms)

    if isinstance(source, str) and source.endswith('.log'):
        log_data = parse_md_log(source)
        n_total = len(log_data['Epot_eV'])
        # Auto-detect timestep
        # Note: log time column is already in ps, no timestep inference needed
    else:
        _, e_per_atom = extract_energies(source, n_atoms)
        n_total = len(e_per_atom)

    n_discard = int(n_total * discard_fraction)
    n_prod = n_total - n_discard
    block_size = n_prod // n_blocks

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    dt_frame = timestep_fs * frame_stride / 1000   # ps per frame
    for i in range(n_blocks):
        t_start = (n_discard + i * block_size) * dt_frame
        t_end = (n_discard + (i + 1) * block_size) * dt_frame
        ax.hlines(bd["block_means"][i], t_start, t_end,
                  colors="steelblue", linewidths=2.5,
                  label="Block mean" if i == 0 else None)

    t_prod_start = n_discard * dt_frame
    ax.axhline(bd["overall_mean"], color="black", ls="--", lw=1,
               label=f"Mean: {bd['overall_mean']:.4f} eV/atom")
    ax.axhspan(bd["overall_mean"] - bd["threshold"],
               bd["overall_mean"] + bd["threshold"],
               alpha=0.15, color="green",
               label=f"+/-2 SEM ({bd['threshold']:.4f} eV)")
    ax.axvspan(0, t_prod_start, alpha=0.1, color="red",
               label="Discarded")

    status = "EQUILIBRATED" if is_eq else "NOT EQUILIBRATED"
    color = "green" if is_eq else "red"
    ax.set_title(f"Block average test: {status}", color=color,
                 fontweight="bold")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("E_pot (eV/atom)")
    ax.legend(fontsize=8)
    fig.tight_layout()

    return fig, is_eq, bd


# ══════════════════════════════════════════════════════════════════════════════
# 3. Mean Square Displacement
# ══════════════════════════════════════════════════════════════════════════════

def compute_msd(traj, timestep_fs: float = DEFAULT_TIMESTEP_FS,
                by_element: bool = True,
                frame_stride: int = TRAJ_LOG_INTERVAL,
                time_ps=None) -> tuple[np.ndarray, dict]:
    """
    Compute MSD from trajectory using unwrapped positions.

    Handles both orthorhombic and non-orthorhombic cells via
    fractional coordinate unwrapping on periodic axes only. Changes in cell
    size are removed before accumulating non-affine displacements, so uniform
    NPT expansion is not interpreted as self diffusion. Displacements are measured relative
    to the centre of mass, so a drift of the whole system does not count
    as diffusion.

    ``timestep_fs`` is the MD integration timestep. AmorphGen writes one
    trajectory frame every ``frame_stride`` MD steps (``TRAJ_LOG_INTERVAL``,
    default 100), so the real time between consecutive frames is
    ``timestep_fs * frame_stride``. Pass ``frame_stride=1`` for a trajectory
    that stores every step. Getting this wrong rescales the time axis (and
    hence the fitted diffusion coefficient) by ``frame_stride``.

    Parameters
    ----------
    traj : str, list of Atoms, or Trajectory
    timestep_fs : float
    by_element : bool
        If True, return MSD per element type in addition to total.

    Returns
    -------
    time_ps : np.ndarray
    msd_dict : dict — keys are element symbols (+ "all"),
                values are MSD arrays (A^2).
    """
    frames = _load_trajectory(traj)
    n_frames = len(frames)
    time_ps = _sample_times(n_frames, timestep_fs, frame_stride, time_ps)
    if not frames or not len(frames[0]):
        return time_ps, {}
    n_atoms = len(frames[0])
    symbols = frames[0].get_chemical_symbols()

    # Unwrap positions using fractional coordinate jumps
    positions = np.zeros((n_frames, n_atoms, 3))
    positions[0] = frames[0].get_positions()

    for i in range(1, n_frames):
        if frames[i].get_chemical_symbols() != symbols:
            raise ValueError("MSD requires unchanged atom order and composition")
        pbc = frames[i].get_pbc()
        if not np.array_equal(pbc, frames[0].get_pbc()):
            raise ValueError("MSD requires unchanged periodic boundary conditions")
        if np.any(pbc):
            cell = np.asarray(frames[i].cell.complete())
            previous_cell = np.asarray(frames[i - 1].cell.complete())
            frac_delta = (frames[i].get_scaled_positions(wrap=False)
                          - frames[i - 1].get_scaled_positions(wrap=False))
            frac_delta[:, pbc] -= np.round(frac_delta[:, pbc])
            delta = frac_delta @ ((previous_cell + cell) / 2)
        else:
            delta = frames[i].get_positions() - frames[i - 1].get_positions()

        positions[i] = positions[i - 1] + delta

    # Frames are stored every frame_stride MD steps, so the wall time between
    # frames is timestep_fs * frame_stride (fs). Dividing by 1000 -> ps.
    msd_dict = {}

    # Displacements relative to the centre of mass: its drift (the Langevin
    # thermostat leaves it free, and the Nose-Hoover NPT integrators keep the
    # net momentum of the drawn velocities) is a rigid translation, not
    # diffusion.
    masses = frames[0].get_masses()
    disp_all = positions - positions[0]
    disp_all -= ((disp_all * masses[None, :, None]).sum(axis=1)
                 / masses.sum())[:, None, :]

    if by_element:
        for elem in sorted(set(symbols)):
            mask = np.array([s == elem for s in symbols])
            disp = disp_all[:, mask, :]
            msd_dict[elem] = np.mean(np.sum(disp ** 2, axis=2), axis=1)

    msd_dict["all"] = np.mean(np.sum(disp_all ** 2, axis=2), axis=1)

    return time_ps, msd_dict


def plot_msd(traj, timestep_fs: float = DEFAULT_TIMESTEP_FS, ax=None,
             frame_stride: int = TRAJ_LOG_INTERVAL):
    """
    Plot MSD vs time per element. Fits diffusion coefficient D.

    D is fitted from the linear regime (last 50% of trajectory)
    using MSD = 6*D*t (3D diffusion).

    Returns
    -------
    fig : matplotlib Figure
    D_dict : dict — diffusion coefficients (cm^2/s) per element.
        D > 1e-6 cm^2/s indicates liquid/diffusive behaviour.
        D < 1e-6 cm^2/s indicates frozen/glass behaviour.
    """
    import matplotlib.pyplot as plt

    time_ps, msd_dict = compute_msd(traj, timestep_fs=timestep_fs,
                                    frame_stride=frame_stride)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    D_dict = {}
    for elem, msd in msd_dict.items():
        ax.plot(time_ps, msd, lw=1.5, label=elem)

        n_half = len(time_ps) // 2
        if n_half > 10:
            coeffs = np.polyfit(time_ps[n_half:], msd[n_half:], 1)
            D = coeffs[0] / 6.0       # A^2/ps
            D_cm2_s = D * 1e-4        # cm^2/s
            D_dict[elem] = D_cm2_s

    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(r"MSD ($\AA^2$)")
    ax.set_title("Mean Square Displacement")
    ax.legend()

    text_lines = [f"D({e}) = {D:.2e} cm$^2$/s"
                  for e, D in D_dict.items() if e != "all"]
    if text_lines:
        ax.text(0.02, 0.98, "\n".join(text_lines),
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    return fig, D_dict


# ══════════════════════════════════════════════════════════════════════════════
# 4. Temperature vs time
# ══════════════════════════════════════════════════════════════════════════════

def plot_temperature(source, timestep_fs: float = DEFAULT_TIMESTEP_FS,
                     T_target: float | None = None, ax=None,
                     frame_stride: int = TRAJ_LOG_INTERVAL):
    """
    Plot instantaneous temperature vs time.

    Parameters
    ----------
    source : str or list
        Log file (.log) or trajectory / list of Atoms.
    T_target : float, optional
        Target temperature (K). If given, shows expected fluctuation band.
    """
    import matplotlib.pyplot as plt

    if isinstance(source, str) and source.endswith('.log'):
        log_data = parse_md_log(source)
        temps = log_data['T_K']
        time_ps = log_data['time_ps']
        # Note: log time column is already in ps, no timestep inference needed
        n_atoms = None
    else:
        frames = _load_trajectory(source)
        temps = np.array([atoms.get_temperature() for atoms in frames])
        time_ps = np.arange(len(temps)) * timestep_fs * frame_stride / 1000.0
        n_atoms = len(frames[0])

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig = ax.get_figure()

    ax.plot(time_ps, temps, alpha=0.4, lw=0.5, color="orangered")

    # 0.5 ps window in FRAMES: one frame spans timestep_fs*frame_stride fs.
    window = max(1, int(0.5 * 1000 / (timestep_fs * frame_stride)))
    t_avg = running_average(temps, window)
    ax.plot(time_ps, t_avg, color="darkred", lw=1.5,
            label="Running avg (0.5 ps)")

    if T_target is not None:
        ax.axhline(T_target, ls="--", color="black", lw=1,
                   label=f"Target: {T_target} K")
        if n_atoms is not None and n_atoms > 0:
            sigma_T = T_target * np.sqrt(2.0 / (3 * n_atoms))
            ax.axhspan(T_target - 2 * sigma_T, T_target + 2 * sigma_T,
                       alpha=0.1, color="green",
                       label=f"Expected +/-2s ({2 * sigma_T:.0f} K)")

    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Temperature stability")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. RDF in time windows
# ══════════════════════════════════════════════════════════════════════════════

def _compute_partial_rdf_frame(atoms, p1: str, p2: str,
                                rmax: float, nbins: int) -> np.ndarray:
    """
    Compute partial RDF g(r) for one frame using neighbor_list.

    Same-species pairs count both directions (i->j and j->i) and normalise
    with the directed pair-density (n-1)/V, so g_AA(r) -> 1 at large r
    (consistent with analysis.rdf.compute_rdf).
    """
    dr = rmax / nbins
    r_centres = np.linspace(dr / 2, rmax - dr / 2, nbins)
    shell_vols = 4 * np.pi * r_centres ** 2 * dr

    idx_i, idx_j, dists = neighbor_list('ijd', atoms, cutoff=rmax)
    syms = np.array(atoms.get_chemical_symbols())
    vol = atoms.get_volume()

    n_source = int(np.sum(syms == p1))
    n_target = int(np.sum(syms == p2))

    if n_source == 0 or n_target == 0:
        return np.zeros(nbins)

    mask = (syms[idx_i] == p1) & (syms[idx_j] == p2)

    if p1 == p2:
        # Count both directions (i->j and j->i, as neighbor_list returns
        # them) and normalise with the directed pair-density (n-1)/V. This
        # matches analysis.rdf.compute_rdf and makes g_AA(r) -> 1 at large r.
        # (Keeping i<j — undirected — while normalising with (n-1)/V would
        # halve g_AA to ~0.5.)
        rho_target = (n_target - 1) / vol
    else:
        rho_target = n_target / vol

    pair_dists = dists[mask]
    if len(pair_dists) == 0:
        return np.zeros(nbins)

    bin_idx = np.clip((pair_dists / dr).astype(int), 0, nbins - 1)
    hist = np.zeros(nbins)
    np.add.at(hist, bin_idx, 1)

    g_r = np.zeros(nbins)
    valid = shell_vols > 0
    g_r[valid] = hist[valid] / (n_source * rho_target * shell_vols[valid])

    return g_r


def plot_rdf_time_windows(traj, pairs: list[tuple[str, str]] | None = None,
                          n_windows: int = 4, rmax: float = 4.0,
                          nbins: int = 100, timestep_fs: float = DEFAULT_TIMESTEP_FS,
                          frame_stride: int = TRAJ_LOG_INTERVAL):
    """
    Overlay partial RDFs from different time windows.

    If the RDFs from all windows overlap, the structure is equilibrated.
    """
    import matplotlib.pyplot as plt

    frames = _load_trajectory(traj)
    n_frames = len(frames)
    window_size = n_frames // n_windows

    if pairs is None:
        elements = sorted(set(frames[0].get_chemical_symbols()))
        pairs = [(e1, e2) for i, e1 in enumerate(elements)
                 for e2 in elements[i:]]

    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 4),
                             squeeze=False)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_windows))
    dr = rmax / nbins
    r_centres = np.linspace(dr / 2, rmax - dr / 2, nbins)

    for col, (p1, p2) in enumerate(pairs):
        ax = axes[0, col]
        for w in range(n_windows):
            start = w * window_size
            end = start + window_size
            window_frames = frames[start:end]

            t_start = start * timestep_fs * frame_stride / 1000
            t_end = end * timestep_fs * frame_stride / 1000

            g_r = np.zeros(nbins)
            for atoms in window_frames:
                g_r += _compute_partial_rdf_frame(atoms, p1, p2,
                                                   rmax, nbins)
            g_r /= len(window_frames)

            ax.plot(r_centres, g_r, color=colors[w], lw=1.5,
                    label=f"{t_start:.1f}-{t_end:.1f} ps")

        ax.set_xlabel(r"r ($\AA$)")
        ax.set_ylabel("g(r)")
        ax.set_title(f"{p1}-{p2}")
        ax.legend(fontsize=8)

    fig.suptitle("RDF convergence across time windows", fontweight="bold")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 6. Coordination number vs time
# ══════════════════════════════════════════════════════════════════════════════

def compute_cn_vs_time(traj, centre: str, neighbour: str,
                       cutoff: float | None = None,
                       window_size: int = 50,
                       timestep_fs: float = DEFAULT_TIMESTEP_FS,
                       frame_stride: int = TRAJ_LOG_INTERVAL
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute average coordination number vs time using a sliding window.
    """
    frames = _load_trajectory(traj)
    n_frames = len(frames)

    if cutoff is None:
        from ase.data import covalent_radii, atomic_numbers
        z1 = atomic_numbers[centre]
        z2 = atomic_numbers[neighbour]
        cutoff = 1.3 * (covalent_radii[z1] + covalent_radii[z2])

    cn_per_frame = []
    for atoms in frames:
        syms = np.array(atoms.get_chemical_symbols())
        idx_i, idx_j = neighbor_list('ij', atoms, cutoff=cutoff)

        centre_mask = (syms == centre)
        n_centre = int(np.sum(centre_mask))

        if n_centre == 0:
            cn_per_frame.append(0.0)
            continue

        pair_mask = (syms[idx_i] == centre) & (syms[idx_j] == neighbour)
        centre_indices = np.where(centre_mask)[0]

        cns = []
        filtered_i = idx_i[pair_mask]
        for ci in centre_indices:
            cns.append(int(np.sum(filtered_i == ci)))
        cn_per_frame.append(np.mean(cns))

    cn_per_frame = np.array(cn_per_frame)

    n_windows = max(1, n_frames // window_size)
    time_centres = []
    cn_avg = []
    cn_std = []

    for i in range(n_windows):
        start = i * window_size
        end = min(start + window_size, n_frames)
        block = cn_per_frame[start:end]
        time_centres.append((start + end) / 2)
        cn_avg.append(np.mean(block))
        cn_std.append(np.std(block))

    time_centres_ps = np.array(time_centres) * timestep_fs * frame_stride / 1000.0

    return time_centres_ps, np.array(cn_avg), np.array(cn_std)


def plot_cn_vs_time(traj, pairs: list[tuple[str, str, float]],
                    cutoffs: list[float] | None = None,
                    window_size: int = 50,
                    timestep_fs: float = DEFAULT_TIMESTEP_FS, ax=None,
                    frame_stride: int = TRAJ_LOG_INTERVAL):
    """
    Plot CN vs time for multiple centre-neighbour pairs.

    Parameters
    ----------
    pairs : list of (centre, neighbour, expected_cn)
        e.g. [("Si", "O", 4.0), ("O", "Si", 2.0)]
    cutoffs : list of float, optional
        Bond cutoffs per pair. None = auto from covalent radii.
    """
    import matplotlib.pyplot as plt

    if cutoffs is None:
        cutoffs = [None] * len(pairs)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    for (centre, neigh, expected), cutoff in zip(pairs, cutoffs):
        t, cn_avg, cn_std = compute_cn_vs_time(
            traj, centre, neigh, cutoff=cutoff,
            window_size=window_size, timestep_fs=timestep_fs,
            frame_stride=frame_stride,
        )
        ax.errorbar(t, cn_avg, yerr=cn_std, fmt="o-", ms=4, capsize=3,
                    label=f"{centre}-{neigh} (expect {expected:.1f})")
        ax.axhline(expected, ls="--", alpha=0.4)

    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Coordination number")
    ax.set_title("CN convergence")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 7. All-in-one convergence report
# ══════════════════════════════════════════════════════════════════════════════

DIFFUSION_THRESHOLD_CM2_S = 1e-6
MIN_DIFFUSION_FRAMES = 8


def _json_values(value):
    """Convert numerical results to strict JSON values (no NaN/Infinity)."""
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_values(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _energy_stationarity(energies):
    """Compare early/late means using a heuristic nominal two-SEM tolerance."""
    if len(energies) < 4 or not np.all(np.isfinite(energies)):
        return {"status": "insufficient_data", "passed": None}
    early, late = np.array_split(np.asarray(energies), 2)
    difference = float(np.mean(late) - np.mean(early))
    threshold = float(2 * np.sqrt(np.var(early, ddof=1) / len(early)
                                  + np.var(late, ddof=1) / len(late)))
    return {
        "status": "ok", "early_mean_eV_per_atom": float(np.mean(early)),
        "late_mean_eV_per_atom": float(np.mean(late)),
        "difference_eV_per_atom": difference,
        "threshold_eV_per_atom": threshold,
        "passed": bool(abs(difference) <= threshold),
        "criterion": "early/late means differ by at most two combined SEMs; heuristic, no autocorrelation correction",
    }


def _diffusion_window(frames, times):
    """Fit only the late half of a single temperature window."""
    result = {
        "status": "insufficient_data", "n_frames": len(frames),
        "diffusion_coefficients_cm2_s": {}, "msd_final_A2": {},
        "liquid_test": {"status": "insufficient_data", "is_liquid": None,
                        "threshold_cm2_s": DIFFUSION_THRESHOLD_CM2_S},
    }
    if not frames or not len(frames[0]):
        return result
    times, msd = compute_msd(frames, time_ps=times)
    result["msd_final_A2"] = {key: float(values[-1]) for key, values in msd.items()}
    if (len(frames) < MIN_DIFFUSION_FRAMES
            or not all(np.all(np.isfinite(values)) for values in msd.values())):
        return result
    start = len(times) // 2
    slopes = {key: float(np.polyfit(times[start:] - times[start], values[start:], 1)[0])
              for key, values in msd.items()}
    # A negative finite-window slope is a plateau/no resolved diffusion, not a
    # negative physical diffusion constant. Retain raw slopes for inspection.
    diffusion = {key: max(0.0, slope / 6 * 1e-4) for key, slope in slopes.items()}
    result.update({
        "status": "ok", "diffusion_coefficients_cm2_s": diffusion,
        "msd_slope_A2_per_ps": slopes,
        "fit_time_range_ps": [float(times[start]), float(times[-1])],
        "liquid_test": {
            "status": "ok", "is_liquid": bool(diffusion["all"] > DIFFUSION_THRESHOLD_CM2_S),
            "threshold_cm2_s": DIFFUSION_THRESHOLD_CM2_S,
            "diffusion_cm2_s": diffusion["all"],
            "msd_final_A2": result["msd_final_A2"]["all"],
            "criterion": "late-half MSD slope / 6 exceeds 1e-6 cm^2/s",
        },
    })
    return result


def _diffusion_stationarity(frames, times):
    """Compare separately re-originated early/late diffusion estimates."""
    if len(frames) < 2 * MIN_DIFFUSION_FRAMES:
        return {"status": "insufficient_data", "passed": None}
    half = len(frames) // 2
    early = _diffusion_window(frames[:half], times[:half])
    late = _diffusion_window(frames[half:], times[half:])
    if early["status"] != "ok" or late["status"] != "ok":
        return {"status": "insufficient_data", "passed": None}
    de = early["diffusion_coefficients_cm2_s"]
    dl = late["diffusion_coefficients_cm2_s"]
    # A relative tolerance alone is undefined for a frozen trajectory; the
    # same diffusion resolution used in the liquid test supplies the floor.
    difference = abs(dl["all"] - de["all"])
    scale = max(de["all"], dl["all"], DIFFUSION_THRESHOLD_CM2_S)
    threshold = max(0.5 * scale, DIFFUSION_THRESHOLD_CM2_S)
    return {
        "status": "ok", "early_cm2_s": de, "late_cm2_s": dl,
        "relative_change": difference / scale,
        "absolute_difference_cm2_s": difference,
        "threshold_cm2_s": threshold,
        "passed": bool(difference <= threshold),
        "criterion": "difference <= max(50% of larger D, 1e-6 cm^2/s); heuristic",
    }


def _temperature_windows(temperatures):
    """Keep contiguous target-temperature holds separate, even if undersampled."""
    change = np.flatnonzero(~np.isclose(np.diff(temperatures), 0, atol=1e-6, rtol=0)) + 1
    return np.split(np.arange(len(temperatures)), change), "constant_target_holds"


def convergence_report(source, timestep_fs: float = DEFAULT_TIMESTEP_FS,
                       T_target: float | None = None,
                       n_atoms: int | None = None,
                       pairs_rdf: list[tuple[str, str]] | None = None,
                       pairs_cn: list[tuple[str, str, float]] | None = None,
                       cn_cutoffs: list[float] | None = None,
                       rmax: float = 4.0, n_blocks: int = 4,
                       output_dir: str | None = None,
                       prefix: str = "convergence",
                       frame_stride: int = TRAJ_LOG_INTERVAL,
                       make_plots: bool = True, time_ps=None,
                       temperatures=None):
    """Return numerical convergence diagnostics, optionally with plots.

    ``time_ps`` supplies actual saved-frame times (including resumed runs and
    irregular final samples). Otherwise times come from a log or the MD
    timestep multiplied by ``frame_stride``. ``temperatures`` supplies the
    aligned *target* temperatures for a cooling ramp, not kinetic fluctuations.
    A ramp receives independent diffusion fits within temperature windows;
    there is deliberately no diffusion constant fitted across the entire ramp.

    MSD is corrected for centre-of-mass motion and affine cell deformation.
    Late-half fits need at least eight frames per temperature window; comparing
    early and late diffusion requires sixteen. The liquid threshold and
    diffusion-stationarity tolerance are heuristic diagnostics, not a phase
    transition or equilibration proof. Freezing is the first observed downward
    threshold crossing, with its enclosing sampled temperature bracket.

    ``make_plots=False`` performs no plotting or printing and returns strict
    JSON-compatible values; unavailable results use None, never NaN. Existing
    plotting calls retain figure keys (or PNGs and a report in ``output_dir``).
    """
    import os

    if isinstance(source, PathLike):
        source = os.fspath(source)
    is_log = isinstance(source, str) and source.endswith('.log')
    frames = [] if is_log else _load_trajectory(source)
    log_data = parse_md_log(source) if is_log else None
    if is_log:
        if n_atoms is None or n_atoms <= 0:
            raise ValueError("positive n_atoms required when using .log file")
        n_frames = len(log_data["step"])
        if time_ps is None:
            time_ps = log_data["time_ps"]
    else:
        n_frames = len(frames)
        if frames:
            n_atoms = len(frames[0])
    times = _sample_times(n_frames, timestep_fs, frame_stride, time_ps)
    targets = None if temperatures is None else np.asarray(temperatures, dtype=float)
    if targets is not None and (targets.shape != (n_frames,)
                                or not np.all(np.isfinite(targets))):
        raise ValueError("temperatures must contain one finite target temperature per frame")
    is_ramp = targets is not None and len(targets) > 1 and np.ptp(targets) > 1e-6
    report = {
        "status": "ok" if n_frames >= MIN_DIFFUSION_FRAMES else "insufficient_data",
        "n_frames": n_frames, "n_atoms": n_atoms,
        "total_time_ps": float(times[-1] - times[0]) if n_frames else 0.0,
        "elements": sorted(set(frames[0].get_chemical_symbols())) if frames else [],
        "timestep_fs": float(timestep_fs), "frame_stride": frame_stride,
        "time_range_ps": [float(times[0]), float(times[-1])] if n_frames else [],
        "target_temperature_K": T_target, "is_temperature_ramp": bool(is_ramp),
        "energy_drift_eV_per_atom_per_ps": None,
        "block_test_passed": None, "block_data": {"status": "insufficient_data"},
        "diffusion_coefficients_cm2_s": {}, "msd_final_A2": {},
        "liquid_test": {"status": "insufficient_data", "is_liquid": None,
                        "threshold_cm2_s": DIFFUSION_THRESHOLD_CM2_S},
        "stationarity": {}, "temperature_windows": [],
        "freezing_temperature_K": None,
        "diffusion_freezing": {"status": "not_applicable", "temperature_K": None,
                               "bracket_K": None},
        "warnings": [],
    }
    energy_source = source if is_log else frames
    energies = np.array([])
    try:
        if n_atoms:
            _, energies = extract_energies(energy_source, n_atoms=n_atoms)
            report["block_test_passed"], report["block_data"] = block_average_test(
                energy_source, n_blocks=n_blocks, n_atoms=n_atoms)
    except RuntimeError as exc:
        report["warnings"].append(str(exc))
    if len(energies) >= 2 and np.all(np.isfinite(energies)):
        report["energy_drift_eV_per_atom_per_ps"] = float(
            np.polyfit(times - times[0], energies, 1)[0])
    report["stationarity"]["energy"] = _energy_stationarity(energies)
    report["stationarity"]["diffusion"] = {"status": "insufficient_data", "passed": None}

    if is_ramp:
        report["liquid_test"]["status"] = "not_applicable_temperature_ramp"
        report["stationarity"] = {
            key: {"status": "not_applicable_temperature_ramp", "passed": None}
            for key in ("energy", "diffusion")}
        report["block_data"]["interpretation"] = "temperature ramp; not an equilibrium test"
        report["block_test_passed"] = None
        windows, method = _temperature_windows(targets)
        report["temperature_window_method"] = method
        for indices in windows:
            window_frames = [frames[int(i)] for i in indices] if frames else []
            window = _diffusion_window(window_frames, times[indices])
            window.update({
                "temperature_K": float(np.mean(targets[indices])),
                "temperature_range_K": [float(min(targets[indices])), float(max(targets[indices]))],
                "time_range_ps": [float(times[indices[0]]), float(times[indices[-1]])],
                "stationarity": {
                    "energy": _energy_stationarity(energies[indices] if len(energies) else []),
                    "diffusion": _diffusion_stationarity(window_frames, times[indices]),
                },
            })
            window["block_test_passed"] = None
            window["block_data"] = {"status": "insufficient_data"}
            if window_frames and len(energies):
                window["block_test_passed"], window["block_data"] = block_average_test(
                    window_frames, n_blocks=n_blocks)
            report["temperature_windows"].append(window)
        if not any(window["status"] == "ok" for window in report["temperature_windows"]):
            report["status"] = "insufficient_data"
        cooling = np.all(np.diff(targets) <= 1e-6)
        freezing = report["diffusion_freezing"]
        freezing["status"] = "insufficient_data" if cooling else "not_a_cooling_ramp"
        previous_liquid = None
        valid_windows = 0
        if cooling:
            for window in report["temperature_windows"]:
                if window["status"] != "ok":
                    continue
                valid_windows += 1
                if window["liquid_test"]["is_liquid"]:
                    previous_liquid = window["temperature_K"]
                elif previous_liquid is not None:
                    temperature = window["temperature_K"]
                    freezing.update({"status": "observed", "temperature_K": temperature,
                                     "bracket_K": [temperature, previous_liquid]})
                    report["freezing_temperature_K"] = temperature
                    break
            if freezing["status"] == "insufficient_data" and valid_windows:
                freezing["status"] = "not_observed" if previous_liquid is not None else "no_liquid_reference"
    elif frames:
        diffusion = _diffusion_window(frames, times)
        for key in ("diffusion_coefficients_cm2_s", "msd_final_A2", "liquid_test"):
            report[key] = diffusion[key]
        if "fit_time_range_ps" in diffusion:
            report["diffusion_fit_time_range_ps"] = diffusion["fit_time_range_ps"]
            report["msd_slope_A2_per_ps"] = diffusion["msd_slope_A2_per_ps"]
        report["stationarity"]["diffusion"] = _diffusion_stationarity(frames, times)
    if is_log:
        report["warnings"].append("MSD and diffusion require trajectory positions; log only supplied")
    elif frames:
        report["warnings"].append(
            "MSD assumes motion between saved frames is below half a periodic cell; "
            "liquid/freezing and diffusion-stationarity thresholds are heuristic")

    report = _json_values(report)
    lines = ["EQUILIBRATION CONVERGENCE REPORT",
             f"System: {n_atoms} atoms; frames: {n_frames}; duration: {report['total_time_ps']:.6g} ps",
             f"Status: {report['status']}",
             f"Energy drift (eV/atom/ps): {report['energy_drift_eV_per_atom_per_ps']}",
             f"Block average passed: {report['block_test_passed']}"]
    for kind, result in report["stationarity"].items():
        lines.append(f"Early-vs-late {kind} stationarity: {result['status']}; passed={result['passed']}")
    if is_ramp:
        lines.append("Diffusion evaluated independently within temperature windows:")
        for window in report["temperature_windows"]:
            liquid = window["liquid_test"]
            lines.append(f"  {window['temperature_K']:.6g} K: {window['status']}; "
                         f"D={window['diffusion_coefficients_cm2_s']} cm^2/s; "
                         f"MSD={window['msd_final_A2']} A^2; liquid={liquid['is_liquid']}")
            lines.append(f"    Block average passed: {window['block_test_passed']}")
            for kind, result in window["stationarity"].items():
                lines.append(f"    Early-vs-late {kind}: {result['status']}; passed={result['passed']}")
    else:
        lines.extend([f"Diffusion coefficients (cm^2/s): {report['diffusion_coefficients_cm2_s']}",
                      f"Final MSD (A^2): {report['msd_final_A2']}",
                      f"Liquid test: {report['liquid_test']['status']}; "
                      f"liquid={report['liquid_test']['is_liquid']}"])
    freezing = report["diffusion_freezing"]
    lines.append(f"Diffusion freezing: {freezing['status']}; temperature={freezing['temperature_K']} K; "
                 f"bracket={freezing['bracket_K']} K")
    lines.extend(f"Note: {warning}" for warning in report["warnings"])
    report["summary_text"] = "\n".join(lines)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if make_plots and n_frames >= 2:
        import matplotlib.pyplot as plt

        def store_figure(fig, key, filename):
            fig.tight_layout()
            if output_dir:
                fig.savefig(os.path.join(output_dir, f"{prefix}_{filename}.png"), dpi=150,
                            bbox_inches="tight")
                plt.close(fig)
            else:
                report[key] = fig

        if len(energies):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(times, energies, label="Potential energy")
            ax.set(xlabel="Time (ps)", ylabel="Energy (eV/atom)", title="Energy convergence")
            store_figure(fig, "fig_energy", "energy")
        if report["block_data"]["status"] == "ok":
            fig, ax = plt.subplots(figsize=(8, 4))
            means = report["block_data"]["block_means"]
            ax.plot(np.arange(1, len(means) + 1), means, "o-")
            ax.set(xlabel="Block", ylabel="Energy (eV/atom)", title="Block average test")
            store_figure(fig, "fig_blocks", "blocks")
        fig, ax = plt.subplots(figsize=(10, 3))
        kinetic_temperatures = (log_data["T_K"] if is_log else
                                np.array([atoms.get_temperature() for atoms in frames]))
        ax.plot(times, kinetic_temperatures, label="Instantaneous")
        if targets is not None:
            ax.plot(times, targets, "--", label="Target")
        elif T_target is not None:
            ax.axhline(T_target, linestyle="--", label="Target")
        ax.set(xlabel="Time (ps)", ylabel="Temperature (K)", title="Temperature")
        ax.legend()
        store_figure(fig, "fig_temperature", "temperature")
        if frames:
            # Show displacement for the ramp without fitting a global D.
            _, msd = compute_msd(frames, time_ps=times)
            fig, ax = plt.subplots(figsize=(8, 5))
            for key, values in msd.items():
                ax.plot(times, values, label=key)
            ax.set(xlabel="Time (ps)", ylabel="MSD (A^2)", title="Mean Square Displacement")
            ax.legend()
            store_figure(fig, "fig_msd", "msd")
            if n_frames >= 4:
                fig = plot_rdf_time_windows(frames, pairs=pairs_rdf, rmax=rmax,
                                            timestep_fs=timestep_fs, frame_stride=frame_stride)
                store_figure(fig, "fig_rdf_windows", "rdf_windows")
            if pairs_cn:
                fig = plot_cn_vs_time(frames, pairs_cn, cutoffs=cn_cutoffs,
                                      timestep_fs=timestep_fs, frame_stride=frame_stride)
                store_figure(fig, "fig_cn", "cn")
    if output_dir:
        with open(os.path.join(output_dir, f"{prefix}_report.txt"), "w") as handle:
            handle.write(report["summary_text"])
    if make_plots:
        print(report["summary_text"])
    return report
