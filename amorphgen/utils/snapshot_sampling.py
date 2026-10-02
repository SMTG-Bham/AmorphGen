"""Conservative, trajectory-based sampling of a high-temperature plateau.

Energy/volume autocorrelation and the slowest species' displacement both
limit the spacing. The displacement correlation is a *proxy*, exp(-MSD/a²),
not a measured structural relaxation function. Consequently the reported
effective count is a diagnostic estimate, not proof of independent glasses.
No calculator is evaluated and no extra MD is run here.
"""

from __future__ import annotations

import math

import numpy as np
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list


_CORRELATION_THRESHOLD = 0.1
_MIN_FRAMES = 8


def _autocorrelation(values):
    """Unbiased FFT autocorrelation, including lag zero (nonconstant data)."""
    x = np.asarray(values, dtype=float)
    x = x - x.mean()
    scale = np.max(np.abs(x))
    if scale == 0:
        return np.ones(len(x))
    x /= scale
    size = 1 << (2 * len(x) - 1).bit_length()
    spectrum = np.fft.rfft(x, n=size)
    covariance = np.fft.irfft(spectrum * spectrum.conj(), n=size)[:len(x)]
    covariance /= np.arange(len(x), 0, -1)
    return np.clip(covariance / covariance[0], -1.0, 1.0)


def _variable(values):
    # Treat numerical noise in a fixed NVT volume as constant.
    return np.ptp(values) > 1e-12 * max(1.0, float(np.max(np.abs(values))))


def _positive_acf(acf):
    """Initial positive sequence, truncated at the first nonpositive lag."""
    result = np.maximum(acf, 0).copy()
    stops = np.flatnonzero(acf[1:] <= 0)
    if len(stops):
        result[int(stops[0]) + 1:] = 0
    return result


def _inefficiency(acf, n):
    lags = np.arange(1, len(acf))
    return float(max(1.0, 1 + 2 * np.sum(
        _positive_acf(acf)[1:] * (1 - lags / n))))


def _decay_lag(correlation, window=3):
    """First decay below the threshold sustained over a lag window."""
    for lag in range(1, len(correlation) - window + 1):
        if np.all(correlation[lag:lag + window] <= _CORRELATION_THRESHOLD):
            return lag
    return None


def _relative_positions(frames):
    """Unwrap motion, excluding affine cell strain and global COM drift.

    Saved-frame displacements must be smaller than a minimum-image cell
    crossing; a wrapped trajectory cannot recover undersampled crossings.
    Fractional differences mapped through the mean of adjacent cells remove
    barostat dilation/shear. ``find_mic`` also handles skew cells and mixed
    periodicity, unlike independently rounding each fractional component.
    """
    positions = np.zeros((len(frames), len(frames[0]), 3))
    periodic = frames[0].pbc
    previous = frames[0]
    for i, frame in enumerate(frames[1:], 1):
        if np.any(periodic):
            cell = (previous.cell.complete().array + frame.cell.complete().array) / 2
            fractional = (frame.get_scaled_positions(wrap=False)
                          - previous.get_scaled_positions(wrap=False))
            step, _ = find_mic(fractional @ cell, cell, pbc=periodic)
        else:
            step = frame.positions - previous.positions
        positions[i] = positions[i - 1] + step
        previous = frame
    masses = frames[0].get_masses()
    positions -= np.average(positions, axis=1, weights=masses)[:, None, :]
    return positions


def _lag_msd(positions, max_lag):
    """MSD averaged over all time origins and atoms, in O(N log N)."""
    n, n_atoms, _ = positions.shape
    x = positions - positions[0]
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, n=size, axis=0)
    correlation = np.fft.irfft(
        np.sum(spectrum * spectrum.conj(), axis=(1, 2)), n=size)[:max_lag + 1]
    squared = np.sum(x * x, axis=(1, 2))
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    lag = np.arange(max_lag + 1)
    msd = (cumulative[n - lag] + cumulative[n] - cumulative[lag]
           - 2 * correlation) / ((n - lag) * n_atoms)
    msd[0] = 0
    return np.maximum(msd, 0)


def _nearest_neighbor_distance(frame):
    """Median nearest-neighbor distance, including periodic image neighbors."""
    from ase.data import covalent_radii

    cutoff = 2 * float(np.max(covalent_radii[frame.numbers]))
    indices, distances = neighbor_list("id", frame, max(cutoff, 1.0))
    nearest = np.full(len(frame), np.inf)
    np.minimum.at(nearest, indices, distances)
    # Dilute/nonperiodic inputs may have neighbors beyond the initial radius.
    for i in np.flatnonzero(~np.isfinite(nearest)):
        other = np.arange(len(frame)) != i
        if np.any(other):
            nearest[i] = np.min(frame.get_distances(i, np.flatnonzero(other), mic=True))
    if not np.all(np.isfinite(nearest)) or np.any(nearest <= 0):
        return None
    return float(np.median(nearest))


def _observables(frames):
    """Read cached scalar results only; never invoke a live calculator."""
    energies = []
    for frame in frames:
        calc = frame.calc
        energy = None if calc is None else calc.results.get("energy")
        # A calculator attached to modified atoms may hold stale results.
        if energy is None or calc.check_state(frame):
            break
        if not np.isfinite(energy):
            break
        energies.append(float(energy) / len(frame))
    result = {"volume": np.array([abs(np.linalg.det(f.cell)) for f in frames])}
    if len(energies) == len(frames):
        result["energy_per_atom"] = np.array(energies)
    return result


def _automatic_burn(observables, n):
    """Discard >=10%, then maximize the worst scalar tail effective count.

    Searching at most 64 starts through the first half avoids optimizing
    noisy autocorrelations of tiny tails. This is an equilibration heuristic,
    not a stationarity test or a guarantee that the liquid has equilibrated.
    """
    minimum = min(n - 1, max(1, math.ceil(0.1 * n)))
    if n - minimum < _MIN_FRAMES:
        return minimum
    candidates = np.unique(np.linspace(minimum, n // 2, min(64, n), dtype=int))
    best_start, best_score = minimum, -1.0
    for start in candidates:
        count = n - start
        inefficiency = 1.0
        for values in observables.values():
            tail = values[start:]
            if _variable(tail):
                acf = _autocorrelation(tail)[:count // 2 + 1]
                inefficiency = max(inefficiency, _inefficiency(acf, count))
        score = count / inefficiency
        if score > best_score:
            best_start, best_score = int(start), score
    return best_start


def _effective_count(indices, correlation):
    """N² / sum_ij rho(|t_i-t_j|), with a conservative unresolved tail."""
    count = len(indices)
    # Count pair separations without allocating a potentially huge N x N array.
    occupied = np.zeros(indices[-1] - indices[0] + 1)
    occupied[np.array(indices) - indices[0]] = 1
    size = 1 << (2 * len(occupied) - 1).bit_length()
    spectrum = np.fft.rfft(occupied, n=size)
    pairs = np.rint(np.fft.irfft(spectrum * spectrum.conj(), n=size)[:len(occupied)])
    rho = np.interp(np.arange(len(occupied)), np.arange(len(correlation)), correlation)
    denominator = count + 2 * np.dot(pairs[1:], rho[1:])
    return float(np.clip(count * count / denominator, 1, count))


def analyze_snapshot_sampling(frames, n_snapshots=20, select="decorrelated",
                              burn_in_frames=None, timestep_fs=0.5,
                              frame_stride=100, decorrelation_distance=None):
    """Select frames and return JSON-safe sampling diagnostics.

    ``decorrelated`` is count-capped: it never reduces the inferred spacing
    to meet a requested count. Unresolved relaxation yields the last frame
    only. Explicit ``uniform``/``last`` retain their old selection semantics
    while still receiving an effective-count estimate. Explicit burn-in,
    including zero, is honored; otherwise only decorrelated mode uses an
    adaptive burn-in. Time units are ps and distances are Angstrom.
    """
    frames = list(frames)
    n = len(frames)
    if not n:
        raise ValueError("Cannot sample an empty trajectory.")
    if (isinstance(n_snapshots, bool) or not np.isfinite(n_snapshots)
            or int(n_snapshots) != n_snapshots or n_snapshots < 1):
        raise ValueError("n_snapshots must be a positive integer.")
    n_snapshots = int(n_snapshots)
    if select not in {"decorrelated", "uniform", "last"}:
        raise ValueError(f"Unknown selection strategy '{select}'.")
    if not np.isfinite(timestep_fs) or timestep_fs <= 0:
        raise ValueError("timestep_fs must be finite and positive.")
    if not np.isfinite(frame_stride) or int(frame_stride) != frame_stride or frame_stride < 1:
        raise ValueError("frame_stride must be a positive integer.")
    if decorrelation_distance is not None and (
            not np.isfinite(decorrelation_distance) or decorrelation_distance <= 0):
        raise ValueError("decorrelation_distance must be finite and positive.")
    if burn_in_frames is not None:
        if (not np.isfinite(burn_in_frames)
                or int(burn_in_frames) != burn_in_frames or burn_in_frames < 0):
            raise ValueError("burn_in_frames must be a nonnegative integer.")
        if burn_in_frames >= n:
            raise ValueError(f"burn_in_frames ({burn_in_frames}) must be smaller than "
                             f"the trajectory length ({n}).")
    first = frames[0]
    if not len(first):
        raise ValueError("Cannot sample frames containing no atoms.")
    for frame in frames:
        if (not np.array_equal(frame.numbers, first.numbers)
                or not np.array_equal(frame.pbc, first.pbc)
                or not np.array_equal(frame.get_masses(), first.get_masses())):
            raise ValueError("Trajectory atom identities, masses and periodicity must be constant.")
        if (not np.isfinite(frame.positions).all() or not np.isfinite(frame.cell).all()
                or not np.isfinite(frame.get_masses()).all()
                or np.any(frame.get_masses() <= 0)):
            raise ValueError("Trajectory positions, cells and positive masses must be finite.")
        if np.any(frame.pbc) and np.any(np.linalg.norm(frame.cell, axis=1)[frame.pbc] == 0):
            raise ValueError("Periodic cell vectors must have nonzero length.")

    observables = _observables(frames)
    burn = (int(burn_in_frames) if burn_in_frames is not None else
            _automatic_burn(observables, n) if select == "decorrelated" else 0)
    tail = frames[burn:]
    count = len(tail)
    dt = float(timestep_fs * frame_stride / 1000)
    max_lag = count // 2
    warnings = []
    autocorrelation = {}
    correlation_curves = []
    scalar_spacing = 1
    scalar_resolved = True
    active_scalars = 0
    for name in ("energy_per_atom", "volume"):
        values = observables.get(name)
        if values is None:
            autocorrelation[name] = {"status": "unavailable"}
            warnings.append("Cached potential energies are unavailable; energy autocorrelation was not measured.")
            continue
        values = values[burn:]
        if not _variable(values):
            autocorrelation[name] = {"status": "constant"}
            continue
        acf = _autocorrelation(values)[:max_lag + 1]
        g = _inefficiency(acf, count)
        # A zero crossing is not relaxation: thermostats and collective modes
        # can oscillate. Require small *magnitude* for an integration-time
        # window, instead of accepting the entire negative lobe as decorrelated.
        lag = _decay_lag(np.abs(acf), window=max(3, int(math.ceil(g / 2))))
        active_scalars += 1
        scalar_resolved = scalar_resolved and lag is not None
        if lag is not None:
            scalar_spacing = max(scalar_spacing, int(math.ceil(g)), lag)
        autocorrelation[name] = {
            "status": "resolved" if lag is not None else "unresolved",
            "statistical_inefficiency": g,
            "integrated_autocorrelation_time_ps": (g - 1) * dt / 2,
            "decay_lag_frames": lag,
            "acf": acf.tolist(),
        }
        # Retain positive revivals beyond the first zero crossing, especially
        # when explicit uniform sampling lands on repeated oscillation phases.
        # Ignoring negative correlations keeps the estimate capped at N.
        correlation_curves.append(np.maximum(acf, 0))

    distance = (float(decorrelation_distance) if decorrelation_distance is not None
                else _nearest_neighbor_distance(tail[-1]))
    diffusion = {}
    diffusion_spacing = 1
    diffusion_resolved = distance is not None
    positions = _relative_positions(tail)
    symbols = np.array(first.get_chemical_symbols())
    for symbol in sorted(set(symbols)):
        msd = _lag_msd(positions[:, symbols == symbol], max_lag)
        # Flat extrapolation of the measured correlation beyond half the
        # trajectory avoids assuming unobserved relaxation or extrapolating D.
        rho = np.exp(-msd / distance ** 2) if distance is not None else np.ones(len(msd))
        lag = _decay_lag(rho)
        diffusion_resolved = diffusion_resolved and lag is not None
        if lag is not None:
            diffusion_spacing = max(diffusion_spacing, lag)
        fit_start = max(1, max_lag // 2)
        coefficient = None
        if max_lag - fit_start >= 2:
            slope = np.polyfit(np.arange(fit_start, max_lag + 1) * dt, msd[fit_start:], 1)[0]
            # 1 A²/ps = 1e-4 cm²/s. A negative finite-sample slope does not
            # establish diffusion; do not turn it into a positive coefficient.
            coefficient = float(max(0.0, slope) / 6 * 1e-4)
        diffusion[str(symbol)] = {
            "status": "resolved" if lag is not None else "unresolved",
            "decorrelation_lag_frames": lag,
            "diffusion_coefficient_cm2_s": coefficient,
            "msd_angstrom2": msd.tolist(),
        }
        correlation_curves.append(rho)

    resolved = count >= _MIN_FRAMES and scalar_resolved and diffusion_resolved
    spacing = max(scalar_spacing, diffusion_spacing) if resolved else None
    if count < _MIN_FRAMES:
        warnings.append(f"Only {count} post-burn-in frames; at least {_MIN_FRAMES} are needed to estimate relaxation.")
    if not active_scalars:
        warnings.append("No fluctuating energy/volume observable is available; the estimate uses diffusion only.")
    if not scalar_resolved:
        warnings.append("Scalar autocorrelation did not decay within the measured lag window.")
    if not diffusion_resolved:
        warnings.append("The slowest species did not diffuse far enough to resolve decorrelation within the measured lag window.")
    if distance is None:
        warnings.append("A nearest-neighbor distance could not be determined; set decorrelation_distance explicitly.")

    if select == "decorrelated":
        if spacing is None:
            indices = [n - 1]
        else:
            available = 1 + (count - 1) // spacing
            number = min(n_snapshots, available)
            # Spread across the available tail only after enforcing the
            # measured minimum spacing; a singleton uses the latest frame.
            indices = (np.linspace(burn, n - 1, number, dtype=int).tolist()
                       if number > 1 else [n - 1])
    elif select == "uniform":
        indices = np.linspace(burn, n - 1, min(n_snapshots, count), dtype=int).tolist()
    else:
        indices = list(range(max(burn, n - n_snapshots), n))
    if len(indices) < n_snapshots:
        warnings.append(f"Requested {n_snapshots} snapshots, selected {len(indices)}. "
                        "Extend stage 4 (--eq-high-steps) to obtain more decorrelated starting frames.")
    effective = min((_effective_count(indices, rho) for rho in correlation_curves),
                    default=1.0)
    if count < _MIN_FRAMES:
        effective = 1.0
    return {
        "schema_version": 1,
        "method": select,
        "status": ("resolved" if active_scalars else "diffusion_only") if resolved else "unresolved",
        "trajectory_frames": n,
        "requested_snapshots": n_snapshots,
        "selected_snapshots": len(indices),
        "selected_frame_indices": indices,
        "burn_in_frames": burn,
        "burn_in_ps": burn * dt,
        "burn_in_method": "automatic" if burn_in_frames is None and select == "decorrelated" else "explicit",
        "frame_interval_ps": dt,
        "spacing_frames": spacing,
        "spacing_ps": spacing * dt if spacing is not None else None,
        "selected_gaps_ps": (np.diff(indices) * dt).tolist(),
        "correlation_threshold": _CORRELATION_THRESHOLD,
        "decorrelation_distance_angstrom": distance,
        "distance_method": "explicit" if decorrelation_distance is not None else "median_nearest_neighbor",
        "effective_independent_snapshots": effective,
        "effective_count_method": "minimum pairwise ESS over scalar ACFs and exp(-species_MSD/distance^2)",
        "autocorrelation": autocorrelation,
        "diffusion": diffusion,
        "warnings": warnings,
        "limitations": [
            "Burn-in and effective count are heuristic diagnostics, not proof of equilibrium or independent final glasses.",
            "Wrapped trajectories assume minimum-image motion between saved frames; unresolved multiple crossings cannot be recovered.",
            "Diffusion coefficients are late-lag linear-fit diagnostics; spacing uses measured MSD, with no diffusion extrapolation.",
        ],
    }
