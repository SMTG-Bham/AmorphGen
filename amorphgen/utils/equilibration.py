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
- Default timestep is 1.0 fs, matching AmorphGen's default_config.py.
"""

from __future__ import annotations

import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list


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
    if isinstance(source, str):
        return read(source, index=":")
    # ASE Trajectory object or other iterable
    return list(source)


def parse_md_log(logfile: str) -> dict:
    """
    Parse an AmorphGen MD stage log file.

    Expects whitespace-separated columns:
        Step  Time(ps)  T(K)  Epot(eV)  Ekin(eV)  Etot(eV)  Vol(A^3)

    Lines starting with 'Step', '-', or '->' are skipped as headers/markers.

    Returns
    -------
    dict with keys: 'step', 'time_ps', 'T_K', 'Epot_eV', 'Ekin_eV',
    'Etot_eV', 'Vol_A3' — each a numpy array.
    """
    data = {k: [] for k in ['step', 'time_ps', 'T_K', 'Epot_eV',
                             'Ekin_eV', 'Etot_eV', 'Vol_A3']}
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
                    data['step'].append(int(parts[0]))
                    data['time_ps'].append(float(parts[1]))
                    data['T_K'].append(float(parts[2]))
                    data['Epot_eV'].append(float(parts[3]))
                    data['Ekin_eV'].append(float(parts[4]))
                    data['Etot_eV'].append(float(parts[5]))
                    data['Vol_A3'].append(float(parts[6]))
                except (ValueError, IndexError):
                    continue
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


def plot_energy_convergence(source, timestep_fs: float = 1.0,
                            window_ps: float = 0.5,
                            per_atom: bool = True,
                            n_atoms: int | None = None,
                            ax=None):
    """
    Plot potential energy vs time with running average and linear drift.

    Parameters
    ----------
    source : str or list
        Trajectory file, log file (.log), or list of Atoms.
    timestep_fs : float
        MD timestep in femtoseconds (default 1.0, matching AmorphGen).
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
    import matplotlib
    matplotlib.use("Agg")
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
        time_ps = np.arange(n_steps) * timestep_fs / 1000.0

    window_steps = max(1, int(window_ps * 1000 / timestep_fs))
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
                       ) -> tuple[bool, dict]:
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
    if isinstance(source, str) and source.endswith('.log'):
        log_data = parse_md_log(source)
        if n_atoms is None:
            raise ValueError("n_atoms required for .log file")
        e_per_atom = log_data['Epot_eV'] / n_atoms
    else:
        _, e_per_atom = extract_energies(source, n_atoms)

    n_discard = int(len(e_per_atom) * discard_fraction)
    e_prod = e_per_atom[n_discard:]

    block_size = len(e_prod) // n_blocks
    block_means = np.array([
        np.mean(e_prod[i * block_size:(i + 1) * block_size])
        for i in range(n_blocks)
    ])

    overall_mean = np.mean(e_prod)
    overall_std = np.std(e_prod)
    sem = overall_std / np.sqrt(len(e_prod))

    max_deviation = np.max(np.abs(block_means - overall_mean))
    threshold = 2 * sem

    is_equilibrated = bool(max_deviation < threshold)

    return is_equilibrated, {
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
                        timestep_fs: float = 1.0,
                        n_atoms: int | None = None, ax=None):
    """Visualise block averaging: block means vs overall mean +/- 2*SEM."""
    import matplotlib
    matplotlib.use("Agg")
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

    for i in range(n_blocks):
        t_start = (n_discard + i * block_size) * timestep_fs / 1000
        t_end = (n_discard + (i + 1) * block_size) * timestep_fs / 1000
        ax.hlines(bd["block_means"][i], t_start, t_end,
                  colors="steelblue", linewidths=2.5,
                  label="Block mean" if i == 0 else None)

    t_prod_start = n_discard * timestep_fs / 1000
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

def compute_msd(traj, timestep_fs: float = 1.0,
                by_element: bool = True) -> tuple[np.ndarray, dict]:
    """
    Compute MSD from trajectory using unwrapped positions.

    Handles both orthorhombic and non-orthorhombic cells via
    fractional coordinate unwrapping.

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
    n_atoms = len(frames[0])
    symbols = frames[0].get_chemical_symbols()

    # Unwrap positions using fractional coordinate jumps
    positions = np.zeros((n_frames, n_atoms, 3))
    positions[0] = frames[0].get_positions()

    for i in range(1, n_frames):
        cell = np.array(frames[i].get_cell())
        delta = frames[i].get_positions() - frames[i - 1].get_positions()

        # Minimum image correction via fractional coordinates
        try:
            frac_delta = np.linalg.solve(cell.T, delta.T).T
            frac_delta -= np.round(frac_delta)
            delta = frac_delta @ cell
        except np.linalg.LinAlgError:
            # Fallback for degenerate cells
            L = np.array([cell[0, 0], cell[1, 1], cell[2, 2]])
            delta -= L * np.round(delta / L)

        positions[i] = positions[i - 1] + delta

    r0 = positions[0]
    time_ps = np.arange(n_frames) * timestep_fs / 1000.0
    msd_dict = {}

    if by_element:
        for elem in sorted(set(symbols)):
            mask = np.array([s == elem for s in symbols])
            disp = positions[:, mask, :] - r0[mask, :]
            msd_dict[elem] = np.mean(np.sum(disp ** 2, axis=2), axis=1)

    disp_all = positions - r0
    msd_dict["all"] = np.mean(np.sum(disp_all ** 2, axis=2), axis=1)

    return time_ps, msd_dict


def plot_msd(traj, timestep_fs: float = 1.0, ax=None):
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_ps, msd_dict = compute_msd(traj, timestep_fs=timestep_fs)

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

def plot_temperature(source, timestep_fs: float = 1.0,
                     T_target: float | None = None, ax=None):
    """
    Plot instantaneous temperature vs time.

    Parameters
    ----------
    source : str or list
        Log file (.log) or trajectory / list of Atoms.
    T_target : float, optional
        Target temperature (K). If given, shows expected fluctuation band.
    """
    import matplotlib
    matplotlib.use("Agg")
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
        time_ps = np.arange(len(temps)) * timestep_fs / 1000.0
        n_atoms = len(frames[0])

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig = ax.get_figure()

    ax.plot(time_ps, temps, alpha=0.4, lw=0.5, color="orangered")

    window = max(1, int(0.5 * 1000 / timestep_fs))
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

    Correctly handles same-species pairs (avoids double-counting).
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
        mask &= (idx_i < idx_j)
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
                          nbins: int = 100, timestep_fs: float = 1.0):
    """
    Overlay partial RDFs from different time windows.

    If the RDFs from all windows overlap, the structure is equilibrated.
    """
    import matplotlib
    matplotlib.use("Agg")
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

            t_start = start * timestep_fs / 1000
            t_end = end * timestep_fs / 1000

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
                       timestep_fs: float = 1.0
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

    time_centres_ps = np.array(time_centres) * timestep_fs / 1000.0

    return time_centres_ps, np.array(cn_avg), np.array(cn_std)


def plot_cn_vs_time(traj, pairs: list[tuple[str, str, float]],
                    cutoffs: list[float] | None = None,
                    window_size: int = 50,
                    timestep_fs: float = 1.0, ax=None):
    """
    Plot CN vs time for multiple centre-neighbour pairs.

    Parameters
    ----------
    pairs : list of (centre, neighbour, expected_cn)
        e.g. [("Si", "O", 4.0), ("O", "Si", 2.0)]
    cutoffs : list of float, optional
        Bond cutoffs per pair. None = auto from covalent radii.
    """
    import matplotlib
    matplotlib.use("Agg")
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

def convergence_report(source, timestep_fs: float = 1.0,
                       T_target: float | None = None,
                       n_atoms: int | None = None,
                       pairs_rdf: list[tuple[str, str]] | None = None,
                       pairs_cn: list[tuple[str, str, float]] | None = None,
                       cn_cutoffs: list[float] | None = None,
                       rmax: float = 4.0, n_blocks: int = 4,
                       output_dir: str | None = None,
                       prefix: str = "convergence"):
    """
    Generate a comprehensive convergence report.

    Parameters
    ----------
    source : str, list of Atoms, or Trajectory
        Trajectory file (.xyz/.traj), log file (.log), or list of Atoms.
        Log files are faster but only provide energy/temperature
        (no MSD, RDF, or CN analysis).
    timestep_fs : float
        MD timestep in femtoseconds (default 1.0).
    T_target : float, optional
        Target temperature (K) for temperature check.
    n_atoms : int, optional
        Required if source is a .log file.
    pairs_rdf : list of (str, str), optional
        Element pairs for RDF windows. Auto-detected if None.
    pairs_cn : list of (str, str, float), optional
        Element pairs + expected CN for CN tracking.
        e.g. [("Si", "O", 4.0), ("O", "Si", 2.0)]
    cn_cutoffs : list of float, optional
        Bond cutoffs for CN pairs. None = auto from covalent radii.
    rmax : float
        Max distance for RDF (must not exceed half cell length).
    n_blocks : int
        Number of blocks for block average test.
    output_dir : str, optional
        If provided, save all plots as PNG and a text report here.
    prefix : str
        Filename prefix for saved plots.

    Returns
    -------
    report : dict
        Keys include: n_frames, n_atoms, total_time_ps, elements,
        energy_drift_eV_per_atom_per_ps, block_test_passed, block_data,
        diffusion_coefficients_cm2_s, summary_text.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    is_log = isinstance(source, str) and source.endswith('.log')

    if is_log:
        log_data = parse_md_log(source)
        n_frames = len(log_data['step'])
        total_time_ps = log_data['time_ps'][-1] if n_frames > 0 else 0
        if n_atoms is None:
            raise ValueError("n_atoms required when using .log file")
        elements = []
        # Note: log time column is already in ps, no timestep inference needed
    else:
        frames = _load_trajectory(source)
        n_frames = len(frames)
        n_atoms = len(frames[0])
        total_time_ps = n_frames * timestep_fs / 1000.0
        elements = sorted(set(frames[0].get_chemical_symbols()))

    report = {
        "n_frames": n_frames,
        "n_atoms": n_atoms,
        "total_time_ps": total_time_ps,
        "elements": elements,
        "timestep_fs": timestep_fs,
    }

    lines = []
    bar = "=" * 60
    lines.append(bar)
    lines.append("EQUILIBRATION CONVERGENCE REPORT")
    lines.append(bar)
    lines.append(f"System: {n_atoms} atoms"
                 + (f", {elements}" if elements else ""))
    lines.append(f"Source: {source if isinstance(source, str) else 'trajectory'}")
    lines.append(f"Frames: {n_frames}, Total time: {total_time_ps:.1f} ps, "
                 f"dt: {timestep_fs:.1f} fs")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    def _save_or_store(fig, key, filename):
        if output_dir:
            path = os.path.join(output_dir, f"{prefix}_{filename}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            lines.append(f"  -> Saved: {path}")
        else:
            report[key] = fig

    # --- Energy convergence ---
    fig_e, drift = plot_energy_convergence(
        source, timestep_fs=timestep_fs, n_atoms=n_atoms)
    report["energy_drift_eV_per_atom_per_ps"] = drift
    ok_drift = abs(drift) < 0.001
    lines.append(f"\nEnergy drift: {drift:.6f} eV/atom/ps "
                 f"{'  [OK]' if ok_drift else '  [WARNING: > 0.001]'}")
    _save_or_store(fig_e, "fig_energy", "energy")

    # --- Block average ---
    is_eq, bd = block_average_test(source, n_blocks=n_blocks,
                                   n_atoms=n_atoms)
    report["block_test_passed"] = is_eq
    report["block_data"] = bd
    lines.append(f"Block average test: "
                 f"{'PASSED  [OK]' if is_eq else 'FAILED  [WARNING]'}")
    lines.append(f"  Block means: "
                 f"{np.array2string(bd['block_means'], precision=4)}")
    lines.append(f"  Overall: {bd['overall_mean']:.4f} +/- "
                 f"{bd['sem']:.4f} eV/atom")

    fig_b, _, _ = plot_block_averages(source, n_blocks=n_blocks,
                                      timestep_fs=timestep_fs,
                                      n_atoms=n_atoms)
    _save_or_store(fig_b, "fig_blocks", "blocks")

    # --- Temperature ---
    if is_log:
        fig_t, ax_t = plt.subplots(figsize=(10, 3))
        ax_t.plot(log_data['time_ps'], log_data['T_K'], alpha=0.6, lw=0.8,
                  color="orangered")
        window = max(1, int(0.5 * 1000 / timestep_fs))
        t_avg = running_average(log_data['T_K'], window)
        ax_t.plot(log_data['time_ps'], t_avg, color="darkred", lw=1.5,
                  label="Running avg (0.5 ps)")
        if T_target:
            ax_t.axhline(T_target, ls="--", color="black", lw=1,
                         label=f"Target: {T_target} K")
        ax_t.set_xlabel("Time (ps)")
        ax_t.set_ylabel("Temperature (K)")
        ax_t.set_title("Temperature stability")
        ax_t.legend(fontsize=9)
        fig_t.tight_layout()
    else:
        fig_t = plot_temperature(frames, timestep_fs=timestep_fs,
                                 T_target=T_target)
    _save_or_store(fig_t, "fig_temperature", "temperature")

    # --- MSD and RDF — only from trajectory, not log ---
    if not is_log:
        fig_m, D_dict = plot_msd(frames, timestep_fs=timestep_fs)
        report["diffusion_coefficients_cm2_s"] = D_dict
        lines.append("\nDiffusion coefficients:")
        for elem, D in D_dict.items():
            if elem == "all":
                continue
            status = "liquid/diffusive" if D > 1e-6 else "frozen/glass"
            lines.append(f"  D({elem}) = {D:.2e} cm^2/s  ({status})")
        _save_or_store(fig_m, "fig_msd", "msd")

        fig_r = plot_rdf_time_windows(frames, pairs=pairs_rdf,
                                      rmax=rmax, timestep_fs=timestep_fs)
        _save_or_store(fig_r, "fig_rdf_windows", "rdf_windows")

        # --- CN vs time (optional) ---
        if pairs_cn:
            fig_cn = plot_cn_vs_time(frames, pairs_cn,
                                     cutoffs=cn_cutoffs,
                                     timestep_fs=timestep_fs)
            _save_or_store(fig_cn, "fig_cn", "cn")
    else:
        lines.append("\n(MSD, RDF, and CN require trajectory file, "
                     "not .log)")

    lines.append(bar)

    text = "\n".join(lines)
    print(text)

    if output_dir:
        report_path = os.path.join(output_dir, f"{prefix}_report.txt")
        with open(report_path, "w") as f:
            f.write(text)
        print(f"  -> Report saved: {report_path}")

    report["summary_text"] = text
    return report
