"""Harmonic vibrational density of states from finite-difference forces.

The supplied cell is treated as the full vibrational system. For a periodic
structure these are its Gamma-point modes, not a Brillouin-zone phonon DOS.
"""

from __future__ import annotations

import operator

import numpy as np
from ase import units
from ase.calculators.singlepoint import SinglePointCalculator
from scipy.integrate import trapezoid

from .uncertainty import summarize_structures

# sqrt(eV / (angstrom**2 * atomic_mass_unit)) is an angular frequency.
_FREQUENCY_TO_THZ = np.sqrt(units._e / (1e-20 * units._amu)) / (2 * np.pi * 1e12)


def _positive_finite(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _get_forces(atoms, structure_index):
    try:
        forces = np.asarray(atoms.get_forces(), dtype=float)
    except Exception as exc:
        raise ValueError(
            f"Could not evaluate forces for structure {structure_index}: {exc}"
        ) from exc
    if forces.shape != (len(atoms), 3) or not np.all(np.isfinite(forces)):
        raise ValueError(
            f"Forces for structure {structure_index} must be finite with shape "
            f"({len(atoms)}, 3)"
        )
    return forces


def compute_vibrational_dos(
    atoms_list, calculator=None, displacement=0.01, npoints=400, sigma=0.1
):
    """Compute a Gaussian-broadened harmonic vibrational density of states.

    Parameters
    ----------
    atoms_list : sequence of ase.Atoms
        Preferably relaxed structures. Each unconstrained structure contributes
        all 3N modes, including rigid translations/rotations and instabilities.
        Constraints are rejected; remove them explicitly to analyse all atoms.
    calculator : ASE calculator, optional
        Force calculator shared by the calculations. If omitted, use each
        structure's attached calculator. Stored single-point forces cannot
        describe displaced configurations and are rejected. This supports an
        MLIP or any other ASE calculator that can recalculate forces.
    displacement : float
        Central finite-difference displacement in angstrom. Requires 6N force
        evaluations per structure and dense 3N-by-3N matrix diagonalisation;
        this calculation is deliberately opt-in.
    npoints : int
        Number of frequency-grid points (at least two). Use enough points to
        resolve the Gaussian width over the returned frequency range.
    sigma : float
        Gaussian standard deviation in THz.

    Returns
    -------
    dict
        ``frequencies_thz`` and ``dos`` give a signed-frequency DOS with unit
        integral. ``projected_dos`` partitions this density by element using
        squared mass-weighted eigenvector components; the projections sum to
        ``dos``. ``mode_frequencies_thz`` holds all individual frequencies:
        negative values represent imaginary modes, not negative real
        frequencies. Structures are pooled with equal weight per mode.
        ``per_structure`` provides frequencies, instability counts and Hessian
        asymmetry diagnostics, plus individually normalized DOS curves on the
        shared frequency grid. ``uncertainty`` gives equal-weight structure
        means, standard errors, t intervals and pointwise bootstrap bands for
        DOS and its element projections. Numerically zero eigenvalues are clipped only
        within roundoff (relative to the largest eigenvalue and matrix size).

    Notes
    -----
    This is a Gamma-point spectrum of each supplied cell, not a sampled
    Brillouin-zone phonon spectrum. No relaxation or acoustic sum rule is
    applied. A stationary reference structure is the caller's responsibility.
    Original structures and their calculator attachments are preserved;
    calculators may update their ordinary internal result caches.
    """
    displacement = _positive_finite(displacement, "displacement")
    sigma = _positive_finite(sigma, "sigma")
    try:
        if isinstance(npoints, (bool, np.bool_)):
            raise TypeError
        npoints = operator.index(npoints)
    except TypeError as exc:
        raise ValueError("npoints must be an integer of at least two") from exc
    if npoints < 2:
        raise ValueError("npoints must be an integer of at least two")

    structures = list(atoms_list)
    if not structures:
        raise ValueError("At least one structure is required for vibrational DOS")

    # Validate all input structures before beginning expensive force evaluations.
    work = []
    elements = set()
    for index, atoms in enumerate(structures):
        if not len(atoms):
            raise ValueError(f"Structure {index} has no atoms")
        if atoms.constraints:
            raise ValueError(
                f"Structure {index} has constraints; remove them explicitly "
                "before computing the full vibrational DOS"
            )
        masses = np.asarray(atoms.get_masses(), dtype=float)
        if not np.all(np.isfinite(masses)) or np.any(masses <= 0):
            raise ValueError(f"Structure {index} must have finite positive masses")
        if not np.all(np.isfinite(atoms.positions)):
            raise ValueError(f"Structure {index} must have finite positions")
        if not np.all(np.isfinite(atoms.cell)):
            raise ValueError(f"Structure {index} must have a finite cell")
        periodic_vectors = np.asarray(atoms.cell)[atoms.pbc]
        if len(periodic_vectors) and np.linalg.matrix_rank(periodic_vectors) < len(periodic_vectors):
            raise ValueError(f"Periodic structure {index} requires independent periodic cell vectors")
        calc = calculator if calculator is not None else atoms.calc
        if calc is None:
            raise ValueError(f"Structure {index} requires a force calculator")
        if isinstance(calc, SinglePointCalculator):
            raise ValueError(
                f"Structure {index} uses a SinglePointCalculator; provide a "
                "calculator that can recalculate forces after displacements"
            )
        properties = getattr(calc, "implemented_properties", None)
        if properties is not None and "forces" not in properties:
            raise ValueError(f"Calculator for structure {index} does not support forces")
        copy = atoms.copy()
        copy.calc = calc
        work.append((copy, masses))
        elements.update(atoms.get_chemical_symbols())

    per_structure = []
    all_frequencies = []
    all_weights = {element: [] for element in sorted(elements)}
    for index, (atoms, masses) in enumerate(work):
        n_modes = 3 * len(atoms)
        reference_positions = atoms.positions.copy()
        hessian = np.empty((n_modes, n_modes))
        for coordinate in range(n_modes):
            atom_index, axis = divmod(coordinate, 3)
            atoms.positions[:] = reference_positions
            atoms.positions[atom_index, axis] += displacement
            force_plus = _get_forces(atoms, index).copy()
            atoms.positions[:] = reference_positions
            atoms.positions[atom_index, axis] -= displacement
            force_minus = _get_forces(atoms, index)
            hessian[:, coordinate] = -(
                force_plus - force_minus
            ).ravel() / (2 * displacement)
        if not np.all(np.isfinite(hessian)):
            raise ValueError(f"Finite-difference Hessian for structure {index} is not finite")
        asymmetry = float(np.max(np.abs(hessian - hessian.T)))
        hessian = (hessian + hessian.T) / 2
        inverse_sqrt_mass = np.repeat(masses ** -0.5, 3)
        dynamical_matrix = (
            hessian * inverse_sqrt_mass[:, None] * inverse_sqrt_mass[None, :]
        )
        if not np.all(np.isfinite(dynamical_matrix)):
            raise ValueError(f"Mass-weighted Hessian for structure {index} is not finite")
        eigenvalues, eigenvectors = np.linalg.eigh(dynamical_matrix)
        roundoff = np.finfo(float).eps * n_modes * np.max(np.abs(eigenvalues))
        eigenvalues[np.abs(eigenvalues) <= roundoff] = 0.0
        frequencies = (
            np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * _FREQUENCY_TO_THZ
        )
        all_frequencies.append(frequencies)
        symbols = np.repeat(atoms.get_chemical_symbols(), 3)
        for element in all_weights:
            weights = np.sum(eigenvectors[symbols == element] ** 2, axis=0)
            all_weights[element].append(weights)
        per_structure.append({
            "index": index,
            "n_atoms": len(atoms),
            "n_modes": n_modes,
            "frequencies_thz": frequencies,
            "imaginary_modes": int(np.count_nonzero(eigenvalues < 0)),
            "imaginary_fraction": float(np.count_nonzero(eigenvalues < 0) / n_modes),
            "mean_frequency_thz": float(frequencies.mean()),
            "hessian_asymmetry": asymmetry,
            "force_evaluations": 2 * n_modes,
        })

    frequencies = np.concatenate(all_frequencies)
    weights = {element: np.concatenate(parts) for element, parts in all_weights.items()}
    lower = min(0.0, float(frequencies.min())) - 5 * sigma
    upper = max(0.0, float(frequencies.max())) + 5 * sigma
    if not np.isfinite(lower) or not np.isfinite(upper) or not np.isfinite(upper - lower):
        raise ValueError("sigma or the vibrational frequencies exceed the supported grid range")
    grid = np.linspace(lower, upper, npoints)
    dos = np.zeros(npoints)
    projected_dos = {element: np.zeros(npoints) for element in weights}
    per_frame_dos = np.zeros((len(structures), npoints))
    per_frame_projected = {element: np.zeros_like(per_frame_dos) for element in weights}
    mode_structures = np.repeat(np.arange(len(structures)),
                                [frame["n_modes"] for frame in per_structure])
    # Normalize on the returned grid, including truncated Gaussian tails. Avoid
    # constructing an npoints-by-3N array for large amorphous structures.
    for mode_index, frequency in enumerate(frequencies):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            exponent = -0.5 * ((grid - frequency) / sigma) ** 2
        if not np.isfinite(exponent.max()):
            raise ValueError("sigma is too small to resolve on the frequency grid")
        kernel = np.exp(exponent - exponent.max())
        area = trapezoid(kernel, grid)
        if not np.isfinite(area) or area <= 0:
            raise ValueError("sigma is too small to resolve on the frequency grid")
        kernel /= area * len(frequencies)
        dos += kernel
        frame_index = mode_structures[mode_index]
        frame_kernel = kernel * len(frequencies) / per_structure[frame_index]["n_modes"]
        per_frame_dos[frame_index] += frame_kernel
        for element, element_weights in weights.items():
            projected_dos[element] += element_weights[mode_index] * kernel
            per_frame_projected[element][frame_index] += element_weights[mode_index] * frame_kernel

    for index, frame in enumerate(per_structure):
        frame["dos"] = per_frame_dos[index]
        frame["projected_dos"] = {element: curves[index]
                                  for element, curves in per_frame_projected.items()}
    uncertainty = {key: summarize_structures([frame[key] for frame in per_structure])
                   for key in ("dos", "imaginary_modes", "imaginary_fraction",
                               "mean_frequency_thz", "hessian_asymmetry")}
    uncertainty.update({f"projected_dos.{element}": summarize_structures(curves)
                        for element, curves in per_frame_projected.items()})

    return {
        "method": "harmonic_finite_difference",
        "sampling": "Gamma-point modes of each supplied structure",
        "frequency_units": "THz",
        "normalization": "per_mode",
        "frequencies_thz": grid,
        "dos": dos,
        "projected_dos": projected_dos,
        "mode_frequencies_thz": frequencies,
        "total_modes": len(frequencies),
        "imaginary_modes": sum(item["imaginary_modes"] for item in per_structure),
        "per_structure": per_structure,
        "uncertainty": uncertainty,
        "displacement_angstrom": displacement,
        "sigma_thz": sigma,
        "force_evaluations": sum(item["force_evaluations"] for item in per_structure),
    }
