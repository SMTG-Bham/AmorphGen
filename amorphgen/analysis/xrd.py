"""Coherent X-ray scattering profiles on a physically accessible 2-theta axis."""

from __future__ import annotations

from numbers import Real

import numpy as np
from ase import Atoms

from .rdf import (
    _shared_rmax,
    _smooth_sq_weighted,
    compute_structure_factor,
    compute_structure_factor_direct,
    xray_form_factor,
)
from .uncertainty import summarize_structures


def _finite_number(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or (value <= 0 if positive else value < 0)):
        sign = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a finite {sign} number")
    return float(value)


def _integer(value, name, minimum):
    if (not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_)) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def compute_xrd_pattern(atoms_list, wavelength=1.5406, qmax=None, nq=300,
                        method="direct", rmax=None, sigma_q=0.0, *,
                        q_batch=4096, confidence=0.95, n_bootstrap=1000, seed=0):
    """Calculate a coherent X-ray intensity profile per atom with ensemble bands.

    ``wavelength`` is in Angstrom (default Cu K-alpha, 1.5406 A). The returned
    ``q`` grid is in inverse Angstrom, ``two_theta`` is in degrees, and
    ``intensity`` is in electron squared per atom. This is an equal-structure
    mean, without peak-height normalization. For each structure independently,

    ``I_coh(q)/N = <f(q)>**2 * (S(q) - 1) + <f(q)**2>``

    uses that structure's composition and neutral-atom Waasmaier-Kirfel form
    factors. ``S(q)`` follows the Faber-Ziman convention. ``method='direct'``
    uses reciprocal-lattice shell means; ``'ft'`` transforms the raw RDF on a
    shared ``rmax`` grid. Direct conversion evaluates form factors at shell
    centres, an approximation that improves as q bins narrow. FT termination
    ripples, including any negative intensities, are retained.

    ``qmax=None`` selects min(15, 4*pi/wavelength). An explicit qmax must be
    physically accessible, and at most 24*pi inverse Angstrom (the form-factor
    fit's validity limit). ``nq`` is the number of q bins/points, not uniformly
    spaced angular points. The angular mapping is ``2*asin(q*wavelength/4/pi)``;
    intensities are sampled values, so no integration Jacobian is applied.
    ``rmax`` applies only to ``method='ft'``. All input structures must be
    nonempty, fully periodic, and have a finite nonsingular three-dimensional
    cell. Compositions, volumes and atom counts may differ between structures.

    ``sigma_q`` optionally Gaussian-rebins each coherent intensity curve in q
    before ensemble averaging, with reciprocal-vector weights for the direct
    method and equal point weights for FT. This is presentation smoothing,
    not an instrument-resolution model. Missing direct shells remain missing;
    ``intensity_raw`` and ``per_structure_raw`` retain the unsmoothed values.

    ``per_structure`` and ``uncertainty`` retain the independent structure
    curves and pointwise uncertainty of their mean. Whole structures, not
    reciprocal vectors, are bootstrapped. With fewer than two contributing
    structures, interval bounds are unavailable. Missing values use JSON null.
    Correlated trajectory frames require blocking before use. Ensemble bands
    exclude finite-cell, form-factor and experimental systematic errors.

    The output includes coherent self-scattering, but no anomalous/incoherent
    scattering, polarization, Lorentz, absorption, background or instrument
    response. Apply only corrections appropriate to the actual measurement.
    """
    wavelength = _finite_number(wavelength, "wavelength", positive=True)
    accessible_qmax = 4.0 * np.pi / wavelength
    requested_qmax = qmax
    qmax = min(15.0, accessible_qmax) if qmax is None else _finite_number(
        qmax, "qmax", positive=True)
    if qmax > accessible_qmax * (1.0 + 1e-12):
        raise ValueError("qmax exceeds the physically accessible 4*pi/wavelength")
    qmax = min(qmax, accessible_qmax)
    if qmax > 24.0 * np.pi:
        raise ValueError("qmax exceeds the Waasmaier-Kirfel validity limit (24*pi)")
    nq = _integer(nq, "nq", 2)
    q_batch = _integer(q_batch, "q_batch", 1)
    sigma_q = _finite_number(sigma_q, "sigma_q")
    if method not in ("direct", "ft"):
        raise ValueError("method must be 'direct' or 'ft'")
    if method == "ft" and qmax <= 0.1:
        raise ValueError("method='ft' requires qmax > 0.1 inverse Angstrom")
    if rmax is not None:
        rmax = _finite_number(rmax, "rmax", positive=True)
        if method != "ft":
            raise ValueError("rmax applies only to method='ft'")
    # Validate the statistical controls before the potentially expensive sum.
    summarize_structures([], confidence=confidence,
                         n_bootstrap=n_bootstrap, seed=seed)

    atoms_list = list(atoms_list)
    if not atoms_list:
        raise ValueError("xrd_pattern requires at least one structure")
    for index, atoms in enumerate(atoms_list):
        if not isinstance(atoms, Atoms):
            raise TypeError("atoms_list must contain ASE Atoms structures")
        if not len(atoms):
            raise ValueError(f"structure {index} contains no atoms")
        cell = np.asarray(atoms.cell, dtype=float)
        if (not np.isfinite(cell).all() or np.linalg.matrix_rank(cell) != 3
                or not np.isfinite(atoms.get_volume()) or atoms.get_volume() <= 0):
            raise ValueError(f"structure {index} needs a finite nonsingular 3D cell")
        if not np.asarray(atoms.pbc).all():
            raise ValueError(f"structure {index} must be periodic in all directions")
        if not np.isfinite(atoms.positions).all():
            raise ValueError(f"structure {index} has non-finite atomic positions")

    if method == "direct":
        sq = compute_structure_factor_direct(
            atoms_list, qmax=qmax, nq=nq, weighting="xray", q_batch=q_batch,
            sigma_q=0.0, confidence=confidence, n_bootstrap=0, seed=seed)
    else:
        rmax = _shared_rmax(atoms_list, rmax)
        sq = compute_structure_factor(
            atoms_list, qmax=qmax, nq=nq, weighting="xray", rmax=rmax,
            confidence=confidence, n_bootstrap=0, seed=seed)
    q = np.asarray(sq["q"], dtype=float)
    sq_rows = np.asarray(sq["per_structure"], dtype=float)
    form_factors = {
        symbol: xray_form_factor(symbol, q)
        for symbol in {s for a in atoms_list for s in a.get_chemical_symbols()}
    }
    raw_curves = []
    for atoms, sq_row in zip(atoms_list, sq_rows):
        symbols, counts = np.unique(atoms.get_chemical_symbols(), return_counts=True)
        fractions = counts / len(atoms)
        factors = np.asarray([form_factors[s] for s in symbols])
        f_mean = fractions @ factors
        f2_mean = fractions @ factors**2
        raw_curves.append(f_mean**2 * (sq_row - 1.0) + f2_mean)
    raw_curves = np.asarray(raw_curves)
    curves = raw_curves.copy()
    if sigma_q:
        counts = (sq["n_per_bin_per_structure"] if method == "direct"
                  else np.ones_like(curves))
        curves = np.asarray([
            _smooth_sq_weighted(q, row, count, sigma_q)
            for row, count in zip(raw_curves, counts)
        ])
    summary = summarize_structures(curves, confidence=confidence,
                                   n_bootstrap=n_bootstrap, seed=seed)
    result = {
        "q": q.tolist(),
        "two_theta": np.rad2deg(2.0 * np.arcsin(
            np.clip(q * wavelength / (4.0 * np.pi), 0.0, 1.0))).tolist(),
        "intensity": summary["mean"],
        "per_structure": summary["per_structure"],
        "uncertainty": summary,
        "method": method,
        "weighting": "xray",
        "wavelength": wavelength,
        "intensity_convention": "coherent intensity per atom",
        "intensity_units": "electron^2/atom",
        "metadata": {
            "wavelength": wavelength, "qmax": float(qmax),
            "requested_qmax": (None if requested_qmax is None
                               else float(requested_qmax)),
            "nq": nq, "method": method, "rmax": rmax, "sigma_q": sigma_q,
            "q_batch": q_batch, "weighting": "xray",
            "confidence": summary["confidence"],
            "n_bootstrap": summary["n_bootstrap"], "seed": summary["seed"],
            "form_factors": "Waasmaier-Kirfel neutral atoms",
            "conversion": "<f>^2 * (S_FZ - 1) + <f^2>, per structure",
            "form_factor_evaluation": "q bin centres" if method == "direct"
                                      else "q points",
            "angle_units": "degree", "q_units": "1/Angstrom",
            "smoothing": "coherent intensity Gaussian rebinning in q",
            "instrument_corrections": None,
        },
    }
    if method == "direct":
        result["n_per_bin"] = sq["n_per_bin"]
        result["n_per_bin_per_structure"] = sq["n_per_bin_per_structure"]
    if sigma_q:
        raw_summary = summarize_structures(raw_curves, n_bootstrap=0,
                                           confidence=confidence, seed=seed)
        result["intensity_raw"] = raw_summary["mean"]
        result["per_structure_raw"] = raw_summary["per_structure"]
    return result
