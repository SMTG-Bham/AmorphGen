"""Monte Carlo distribution of the empty-sphere radius at random cell points.

This measures point clearance, not connected pores, pore throats, or the
distribution of maximal cavities. Atomic spheres use ASE covalent radii by
default; that geometric convention is not a van der Waals porosity model.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import find_mic


# ASE's general MIC routine checks neighbouring reduced-cell images. Keep
# its intermediate pair/image arrays bounded even for very large structures.
_MAX_PAIRS = 16384


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_real(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        condition = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {condition}")
    return value


def _point_clearances(points, positions, cell, atom_radii):
    """Exact periodic minimum of distance-to-centre minus atomic radius."""
    clearances = np.full(len(points), np.inf)
    atom_chunk = min(len(positions), 128)
    point_chunk = max(1, _MAX_PAIRS // atom_chunk)
    for start in range(0, len(points), point_chunk):
        point_block = points[start:start + point_chunk]
        closest = np.full(len(point_block), np.inf)
        for atom_start in range(0, len(positions), atom_chunk):
            atom_stop = atom_start + atom_chunk
            vectors = (point_block[:, None, :]
                       - positions[None, atom_start:atom_stop, :])
            # Fractional-coordinate rounding alone is incorrect in skew cells.
            # ASE uses Minkowski reduction for its general minimum-image path:
            # https://docs.ase-lib.org/_modules/ase/geometry/geometry.html
            _, distances = find_mic(vectors.reshape(-1, 3), cell, pbc=True)
            distances = distances.reshape(len(point_block), -1)
            distances -= atom_radii[None, atom_start:atom_stop]
            closest = np.minimum(closest, distances.min(axis=1))
        clearances[start:start + len(point_block)] = closest
    return clearances


def _wilson_interval(successes, trials):
    """95% binomial Wilson interval, including nonzero width at 0 or 1."""
    z = 1.959963984540054
    fraction = successes / trials
    denominator = 1 + z * z / trials
    center = (fraction + z * z / (2 * trials)) / denominator
    half_width = z * np.sqrt(
        fraction * (1 - fraction) / trials + z * z / (4 * trials**2)
    ) / denominator
    return [max(0.0, float(center - half_width)),
            min(1.0, float(center + half_width))]


def compute_void_distribution(atoms_list, n_samples=10000, probe_radius=0.0,
                              radii=None, nbins=50, seed=0):
    """Sample the volume-weighted point-clearance distribution in periodic cells.

    At each uniformly sampled cell point ``x``, the clearance is
    ``min_i(min_image_distance(x, atom_i) - radius_i)`` in angstrom. Points
    with clearance >= ``probe_radius`` admit the centre of a spherical probe.
    This geometric accessibility does not imply a connected path to a pore.

    ``radius`` and ``bin_edges`` describe *clearance*, not diameter or
    clearance minus probe radius. ``probability_density`` is normalized over
    the accessible points (integral one when any are found), whereas
    ``bin_volume_fraction`` sums to ``accessible_fraction``. Frames are
    weighted by cell volume. ``accessible_volume`` is the arithmetic mean
    accessible volume per frame in angstrom cubed, not their sum.

    ``n_samples`` independent points are drawn per frame with a local NumPy
    random generator. ``seed=None`` requests nondeterministic sampling.
    Standard errors quantify Monte Carlo sampling only, not differences
    between structures. The per-frame Wilson intervals remain informative
    when no accessible/occupied points were sampled. Sample maxima are lower
    bounds on the true maximum clearance; no maximal-cavity search is done.

    Radii are in angstrom. A mapping overrides ASE covalent radii for the
    specified element symbols; unspecified elements keep the ASE defaults.
    All frames must contain atoms and have a finite full-rank 3D periodic
    cell. Returns only JSON-compatible objects; inputs are not modified.
    """
    n_samples = _positive_integer(n_samples, "n_samples")
    nbins = _positive_integer(nbins, "nbins")
    probe_radius = _finite_real(probe_radius, "probe_radius")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a non-negative integer or None")
        seed = int(seed)
    if radii is None:
        overrides = {}
    elif not isinstance(radii, Mapping):
        raise ValueError("radii must be a mapping of element symbols to radii")
    else:
        overrides = {}
        for symbol, radius in radii.items():
            if not isinstance(symbol, str) or symbol not in atomic_numbers:
                raise ValueError(f"Unknown element symbol in radii: {symbol!r}")
            overrides[symbol] = _finite_real(
                radius, f"radius for {symbol}", positive=True
            )

    atoms_list = list(atoms_list)
    if not atoms_list:
        raise ValueError("atoms_list must contain at least one structure")
    prepared = []
    used_radii = {}
    for index, atoms in enumerate(atoms_list):
        if not len(atoms):
            raise ValueError(f"Structure {index} contains no atoms")
        cell = np.asarray(atoms.cell, dtype=float)
        if (not np.all(atoms.pbc) or not np.all(np.isfinite(cell))
                or np.linalg.matrix_rank(cell) != 3):
            raise ValueError(
                f"Structure {index} requires a finite full-rank 3D periodic cell"
            )
        volume = float(abs(np.linalg.det(cell)))
        if not np.isfinite(volume) or volume <= 0:
            raise ValueError(f"Structure {index} has invalid cell volume")
        if not np.all(np.isfinite(atoms.positions)):
            raise ValueError(f"Structure {index} has non-finite atom positions")
        symbols = atoms.get_chemical_symbols()
        for symbol in set(symbols):
            used_radii[symbol] = overrides.get(
                symbol, float(covalent_radii[atomic_numbers[symbol]])
            )
            _finite_real(used_radii[symbol], f"radius for {symbol}", positive=True)
        atom_radii = np.array([used_radii[symbol] for symbol in symbols])
        positions = atoms.get_scaled_positions(wrap=True) @ cell
        prepared.append((positions, cell, volume, atom_radii))

    rng = np.random.default_rng(seed)
    samples = []
    per_structure = []
    for index, (positions, cell, volume, atom_radii) in enumerate(prepared):
        points = rng.random((n_samples, 3)) @ cell
        clearance = _point_clearances(points, positions, cell, atom_radii)
        accessible = clearance[clearance >= probe_radius]
        samples.append(accessible)
        count = len(accessible)
        fraction = count / n_samples
        stderr = float(np.sqrt(fraction * (1 - fraction) / n_samples))
        per_structure.append({
            "index": index,
            "cell_volume": volume,
            "n_samples": n_samples,
            "n_accessible": count,
            "accessible_fraction": fraction,
            "accessible_fraction_stderr": stderr,
            "accessible_fraction_interval_95": _wilson_interval(count, n_samples),
            "accessible_volume": volume * fraction,
            "accessible_volume_stderr": volume * stderr,
            "mean_clearance": float(accessible.mean()) if count else None,
            "max_clearance": float(accessible.max()) if count else None,
        })

    volumes = np.array([entry[2] for entry in prepared])
    weights = volumes / volumes.sum()
    fractions = np.array([entry["accessible_fraction"] for entry in per_structure])
    fraction = float(weights @ fractions)
    stderr = float(np.sqrt(np.sum(
        weights**2 * fractions * (1 - fractions) / n_samples
    )))
    maximum = max((float(sample.max()) for sample in samples if len(sample)),
                  default=None)
    # Keep a finite, non-degenerate axis for an empty observed distribution.
    upper = maximum if maximum is not None else probe_radius + 1.0
    if upper <= probe_radius:
        upper = probe_radius + max(1.0, abs(probe_radius) * 1e-12)
    edges = np.linspace(probe_radius, upper, nbins + 1)
    per_frame_fractions = np.array([
        np.histogram(sample, bins=edges)[0] / n_samples for sample in samples
    ])
    bin_fractions = weights @ per_frame_fractions
    bin_stderr = np.sqrt(np.sum(
        weights[:, None]**2 * per_frame_fractions * (1 - per_frame_fractions)
        / n_samples, axis=0
    ))
    density = (bin_fractions / (fraction * np.diff(edges)) if fraction > 0
               else np.zeros(nbins))
    mean = (float(sum(weight * sample.sum() / n_samples
                      for weight, sample in zip(weights, samples)) / fraction)
            if fraction > 0 else None)
    return {
        "radius": ((edges[:-1] + edges[1:]) / 2).tolist(),
        "bin_edges": edges.tolist(),
        "probability_density": density.tolist(),
        "bin_volume_fraction": bin_fractions.tolist(),
        "bin_volume_fraction_stderr": bin_stderr.tolist(),
        "accessible_fraction": fraction,
        "accessible_fraction_stderr": stderr,
        "accessible_volume": float(np.mean(volumes) * fraction),
        "accessible_volume_stderr": float(np.mean(volumes) * stderr),
        "mean_clearance": mean,
        "max_clearance": maximum,
        "per_structure": per_structure,
        "n_structures": len(atoms_list),
        "n_samples": n_samples,
        "probe_radius": probe_radius,
        "seed": seed,
        "radii": dict(sorted(used_radii.items())),
        "radius_source": "ASE covalent radii with overrides" if overrides
                         else "ASE covalent radii",
        "ensemble_weighting": "cell volume",
        "definition": "periodic point clearance; not connected-pore sizes",
        "sampling_uncertainty": "independent Monte Carlo standard errors; "
                                "excludes variation between structures",
        "length_unit": "angstrom",
        "volume_unit": "angstrom^3",
    }
