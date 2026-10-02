"""Elastic stiffness and isotropic moduli from finite differences of ASE stress.

The Voigt/Reuss/Hill expressions follow the NIST atomman documentation:
https://www.ctcms.nist.gov/potentials/atomman/tutorial/3.1._ElasticConstants_class.html
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import BFGS


_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
_MODULUS_KEYS = (
    "bulk_modulus_gpa", "shear_modulus_gpa", "young_modulus_gpa", "poisson_ratio",
)


def _positive_real(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _stress(atoms, index, label):
    try:
        # Constraint transformations and kinetic stress are not part of the
        # static material stiffness. ASE uses tensile-positive stress.
        stress = np.asarray(atoms.get_stress(
            voigt=True, apply_constraint=False, include_ideal_gas=False,
        ), dtype=float)
    except Exception as exc:
        raise RuntimeError(
            f"Structure {index}: could not calculate stress at {label}: {exc}"
        ) from exc
    if stress.shape != (6,) or not np.all(np.isfinite(stress)):
        raise ValueError(f"Structure {index}: {label} stress must contain six finite values")
    return stress


def _relax(atoms, index, label, fmax, steps):
    def validate_forces():
        forces = np.asarray(atoms.get_forces(), dtype=float)
        if forces.shape != (len(atoms), 3) or not np.all(np.isfinite(forces)):
            raise ValueError(f"Structure {index}: non-finite or invalid forces at {label}")

    validate_forces()
    optimizer = BFGS(atoms, logfile=None)
    optimizer.attach(validate_forces)
    if not optimizer.run(fmax=fmax, steps=steps):
        raise RuntimeError(
            f"Structure {index}: fixed-cell atomic relaxation did not converge "
            f"at {label} within {steps} steps (fmax={fmax} eV/Angstrom)"
        )


def _isotropic_moduli(bulk, shear):
    # Preserve formal Voigt K/G for diagnosing unstable structures, but avoid
    # reporting derived E/nu as if they described a stable isotropic medium.
    valid = bulk > 0 and shear > 0
    return {
        "bulk_modulus_gpa": float(bulk),
        "shear_modulus_gpa": float(shear),
        "young_modulus_gpa": float(9 * bulk * shear / (3 * bulk + shear)) if valid else None,
        "poisson_ratio": float((3 * bulk - 2 * shear) / (2 * (3 * bulk + shear))) if valid else None,
    }


def _tensor_properties(stiffness):
    diagonal = float(np.trace(stiffness[:3, :3]))
    off_diagonal = float(stiffness[0, 1] + stiffness[0, 2] + stiffness[1, 2])
    shear_diagonal = float(np.trace(stiffness[3:, 3:]))
    bulk_voigt = (diagonal + 2 * off_diagonal) / 9
    shear_voigt = (diagonal - off_diagonal + 3 * shear_diagonal) / 15
    eigenvalues = np.linalg.eigvalsh(stiffness)
    stable = bool(np.all(eigenvalues > 0))
    condition = float(np.linalg.cond(stiffness))
    compliance_valid = stable and np.isfinite(condition) and condition <= 1e12
    warnings = []
    moduli = {"voigt": _isotropic_moduli(bulk_voigt, shear_voigt), "reuss": None, "hill": None}
    if not stable:
        warnings.append(
            "Stiffness is not positive definite. Voigt values are formal estimates; "
            "Reuss and Hill moduli are unavailable."
        )
    if not np.isfinite(condition) or condition > 1e12:
        warnings.append(
            "Stiffness is singular or ill-conditioned (condition number > 1e12); "
            "Reuss and Hill moduli are unavailable."
        )
    if compliance_valid:
        compliance = np.linalg.inv(stiffness)
        diagonal = float(np.trace(compliance[:3, :3]))
        off_diagonal = float(compliance[0, 1] + compliance[0, 2] + compliance[1, 2])
        shear_diagonal = float(np.trace(compliance[3:, 3:]))
        bulk_reuss = 1 / (diagonal + 2 * off_diagonal)
        shear_reuss = 15 / (4 * (diagonal - off_diagonal) + 3 * shear_diagonal)
        moduli["reuss"] = _isotropic_moduli(bulk_reuss, shear_reuss)
        moduli["hill"] = _isotropic_moduli(
            (bulk_voigt + bulk_reuss) / 2, (shear_voigt + shear_reuss) / 2,
        )
    return {
        "eigenvalues_gpa": eigenvalues.tolist(),
        "mechanically_stable": stable,
        "condition_number": condition if np.isfinite(condition) else None,
        "compliance_valid": bool(compliance_valid),
        "moduli": moduli,
        "warnings": warnings,
    }


def compute_elastic_moduli(atoms_list, calculator=None, strain=0.005,
                           relax=False, fmax=0.01, steps=200):
    """Compute static elastic descriptors using a stress-capable ASE calculator.

    Parameters
    ----------
    atoms_list : iterable of ase.Atoms
        Nonempty, fully periodic 3D structures. Input cells, positions,
        constraints, and calculator attachments are left unchanged.
    calculator : ASE calculator, optional
        A live calculator (for example an MLIP). If omitted, use each input's
        attached calculator. Stored single-point stresses cannot supply the
        response to strain and are rejected. Calculator result caches may be
        updated during evaluation.
    strain : float
        Positive engineering strain amplitude below one. Small amplitudes
        (typically 0.001--0.01) approximate linear response; check convergence
        against this setting for quantitative use.
    relax : bool
        If true, relax atoms with BFGS at fixed cell before evaluating the
        reference and each strained configuration. Existing atomic constraints
        are respected. Failure to converge raises an error.
    fmax : float
        Maximum force tolerance in eV/Angstrom for atomic relaxation.
    steps : int
        Maximum optimization steps per configuration.

    Returns
    -------
    dict
        JSON-compatible per-structure tensors, residual stresses, stability
        diagnostics, and Voigt/Reuss/Hill moduli in GPa (Poisson ratio is
        dimensionless). Ensemble means and population standard deviations use
        equal structure weights; unavailable values are excluded with counts.

    Notes
    -----
    Columns differentiate stress against engineering strains in ASE Voigt
    order ``xx, yy, zz, yz, xz, xy``. Shear deformation matrix entries are half
    the engineering strain. Each column uses independent positive and negative
    deformations of the reference. The raw tensor is symmetrized before
    computing moduli. There are 13 stress evaluations per structure.

    These are zero-temperature, static, tangent stress-strain coefficients.
    No finite-pressure correction is applied. Relax the cell separately to
    near-zero stress for conventional equilibrium elastic moduli. Positive
    definiteness is a diagnostic of the symmetrized tensor, not a general
    finite-pressure stability criterion. Reuss/Hill estimates are unavailable
    for nonpositive or ill-conditioned tensors. Ensemble component averages
    assume structures are expressed in comparable Cartesian orientations.
    """
    strain = _positive_real(strain, "strain")
    if strain >= 1:
        raise ValueError("strain must be less than one")
    fmax = _positive_real(fmax, "fmax")
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, Integral) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if not isinstance(relax, (bool, np.bool_)):
        raise ValueError("relax must be a boolean")
    if isinstance(atoms_list, Atoms):
        raise TypeError("atoms_list must be an iterable of Atoms, not a single Atoms object")
    structures = list(atoms_list)
    if not structures:
        raise ValueError("atoms_list must contain at least one structure")

    # Validate every structure before potentially expensive MLIP evaluations.
    calculators = []
    for index, atoms in enumerate(structures):
        if not isinstance(atoms, Atoms):
            raise TypeError(f"Structure {index} must be an ASE Atoms object")
        if not len(atoms):
            raise ValueError(f"Structure {index} must contain atoms")
        if not np.all(atoms.pbc):
            raise ValueError(f"Structure {index} must be periodic in all three directions")
        cell = np.asarray(atoms.cell)
        if (not np.all(np.isfinite(cell)) or not np.isfinite(np.linalg.det(cell))
                or abs(np.linalg.det(cell)) <= 0):
            raise ValueError(f"Structure {index} must have a finite nonsingular 3D cell")
        if not np.all(np.isfinite(atoms.positions)):
            raise ValueError(f"Structure {index} positions must be finite")
        calc = calculator if calculator is not None else atoms.calc
        if calc is None or isinstance(calc, SinglePointCalculator):
            raise ValueError(
                f"Structure {index} requires a live stress-capable calculator; "
                "stored single-point stresses cannot evaluate strained structures"
            )
        calculators.append(calc)

    results = []
    for index, (atoms, calc) in enumerate(zip(structures, calculators)):
        reference = atoms.copy()
        reference.calc = calc
        if relax:
            _relax(reference, index, "reference", fmax, steps)
        residual = _stress(reference, index, "reference") / units.GPa
        raw = np.empty((6, 6))
        for column, (i, j) in enumerate(_VOIGT_PAIRS):
            stresses = []
            for sign in (1, -1):
                deformation = np.eye(3)
                if i == j:
                    deformation[i, j] += sign * strain
                else:
                    deformation[i, j] += sign * strain / 2
                    deformation[j, i] += sign * strain / 2
                sample = reference.copy()
                sample.set_cell(np.asarray(reference.cell) @ deformation.T,
                                scale_atoms=True, apply_constraint=False)
                sample.calc = calc
                label = f"strain component {column}, sign {sign:+d}"
                if relax:
                    _relax(sample, index, label, fmax, steps)
                stresses.append(_stress(sample, index, label))
            raw[:, column] = (stresses[0] - stresses[1]) / (2 * strain * units.GPa)
        if not np.all(np.isfinite(raw)):
            raise ValueError(f"Structure {index}: strain derivatives must be finite")
        stiffness = (raw + raw.T) / 2
        properties = _tensor_properties(stiffness)
        max_residual = float(np.max(np.abs(residual)))
        symmetry_error = float(np.max(np.abs(raw - raw.T)))
        if max_residual > 0.1:
            properties["warnings"].append(
                "Residual stress exceeds 0.1 GPa. These are tangent stress-strain "
                "coefficients without finite-pressure correction."
            )
        if symmetry_error > 0.05 * max(float(np.max(np.abs(raw))), 1e-12):
            properties["warnings"].append(
                "Raw stiffness asymmetry exceeds 5% of the largest component; "
                "check strain amplitude and force/stress convergence."
            )
        results.append({
            "index": index,
            "n_atoms": len(atoms),
            "volume_angstrom3": float(reference.get_volume()),
            "stiffness_tensor_gpa": stiffness.tolist(),
            "raw_stiffness_tensor_gpa": raw.tolist(),
            "residual_stress_gpa": residual.tolist(),
            "max_residual_stress_gpa": max_residual,
            "symmetry_error_gpa": symmetry_error,
            **properties,
        })

    tensors = np.asarray([result["stiffness_tensor_gpa"] for result in results])
    ensemble = {
        "stiffness_tensor_mean_gpa": tensors.mean(axis=0).tolist(),
        "stiffness_tensor_std_gpa": tensors.std(axis=0).tolist(),
        "moduli": {},
    }
    for average in ("voigt", "reuss", "hill"):
        ensemble["moduli"][average] = {}
        for key in _MODULUS_KEYS:
            values = [result["moduli"][average][key] for result in results
                      if result["moduli"][average] is not None
                      and result["moduli"][average][key] is not None]
            ensemble["moduli"][average][key] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values)) if values else None,
                "count": len(values),
            }
    return {
        "n_structures": len(results),
        "strain": strain,
        "relaxed_ions": bool(relax),
        "voigt_order": ["xx", "yy", "zz", "yz", "xz", "xy"],
        "units": {"stiffness": "GPa", "stress": "GPa", "moduli": "GPa",
                  "poisson_ratio": "dimensionless"},
        "per_structure": results,
        "ensemble": ensemble,
    }
