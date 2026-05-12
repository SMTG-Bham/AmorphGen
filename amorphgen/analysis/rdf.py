"""RDF, structure factor S(q), and multi-structure averaged RDF."""

from __future__ import annotations

import numpy as np
from ase.neighborlist import neighbor_list

from .cutoff import check_rmax


def _gaussian_smear(r, g_r, sigma):
    """Apply Gaussian broadening to g(r)."""
    if sigma <= 0:
        return g_r
    dr = r[1] - r[0] if len(r) > 1 else 1.0
    window_size = int(4 * sigma / dr)  # 4-sigma window
    if window_size < 1:
        return g_r
    x = np.arange(-window_size, window_size + 1) * dr
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel /= kernel.sum()
    return np.convolve(g_r, kernel, mode='same')


def compute_rdf(atoms_list, pair=None, rmax=None, nbins=200, sigma=0.0):
    """Compute the radial distribution function g(r).

    Parameters
    ----------
    atoms_list : list of Atoms
    pair : str, optional
        Pair to analyse, e.g. "Si-O". If None, total RDF.
    rmax : float, optional
        Maximum radius in A. Auto-detected from cell if None.
    nbins : int
        Number of histogram bins (default 200).
    sigma : float
        Gaussian smearing width in A (default 0.0 = no smearing).
        Typical values: 0.02-0.05 for comparison with experiment.
    """
    if not atoms_list:
        return {"r": [], "g_r": []}
    if rmax is None:
        half_cells = [min(a.cell.lengths()) / 2 for a in atoms_list]
        rmax = float(np.floor(min(half_cells) * 10) / 10)
    check_rmax(atoms_list, rmax)

    dr = rmax / nbins
    r_centres = np.linspace(dr / 2, rmax - dr / 2, nbins)
    shell_vols = 4 * np.pi * r_centres**2 * dr

    g_r = np.zeros(nbins)
    n_frames = len(atoms_list)

    for atoms in atoms_list:
        idx_i, idx_j, dists = neighbor_list('ijd', atoms, cutoff=rmax)
        syms = np.array(atoms.get_chemical_symbols())
        vol = atoms.get_volume()
        n = len(atoms)

        if pair is not None:
            p1, p2 = pair.split("-")
            n_source = int(np.sum(syms == p1))
            n_target = int(np.sum(syms == p2))
            if p1 == p2:
                rho_target = (n_target - 1) / vol
            else:
                rho_target = n_target / vol
            mask = (syms[idx_i] == p1) & (syms[idx_j] == p2)
            pair_dists = dists[mask]
        else:
            n_source = n
            rho_target = (n - 1) / vol
            pair_dists = dists

        if len(pair_dists) == 0 or n_source == 0 or rho_target <= 0:
            continue

        # Use proper histogram — skip distances outside [0, rmax]
        in_range = (pair_dists > 0) & (pair_dists < rmax)
        if not np.any(in_range):
            continue
        hist, _ = np.histogram(pair_dists[in_range], bins=nbins,
                               range=(0, rmax))

        valid = shell_vols > 0
        g_r[valid] += hist[valid] / (n_source * rho_target * shell_vols[valid])

    g_r /= n_frames

    if sigma > 0:
        g_r = _gaussian_smear(r_centres, g_r, sigma)

    return {"r": r_centres.tolist(), "g_r": g_r.tolist()}


def compute_structure_factor(atoms_list, pair=None, qmax=15.0, nq=300,
                             rmax=None):
    """Compute S(q) from g(r) via Fourier transform."""
    rdf_data = compute_rdf(atoms_list, pair=pair, rmax=rmax, nbins=500)
    r = np.array(rdf_data["r"])
    g_r = np.array(rdf_data["g_r"])
    dr = r[1] - r[0] if len(r) > 1 else 0.04

    atoms = atoms_list[0]
    n = len(atoms)
    vol = atoms.get_volume()

    if pair is not None:
        p1, p2 = pair.split("-")
        syms = atoms.get_chemical_symbols()
        n_target = sum(1 for s in syms if s == p2)
        rho = (n_target - 1) / vol if p1 == p2 else n_target / vol
    else:
        rho = (n - 1) / vol

    if rho <= 0 or len(r) == 0:
        return {"q": [], "s_q": []}

    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    q_values = np.linspace(0.1, qmax, nq)
    s_q = np.ones(nq)

    for iq, q in enumerate(q_values):
        qr = q * r
        sinc_qr = np.where(qr > 1e-12, np.sin(qr) / qr, 1.0)
        integrand = r * (g_r - 1.0) * sinc_qr
        s_q[iq] = 1.0 + 4.0 * np.pi * rho * _trapz(integrand, dx=dr)

    return {"q": q_values.tolist(), "s_q": s_q.tolist()}


def compute_averaged_rdf(atoms_list, pair=None, rmax=None, nbins=200):
    """Compute RDF per structure with mean and std."""
    if rmax is None:
        half_cells = [min(a.cell.lengths()) / 2 for a in atoms_list]
        rmax = float(np.floor(min(half_cells) * 10) / 10)

    dr = rmax / nbins
    r_centres = np.linspace(dr / 2, rmax - dr / 2, nbins)
    shell_vols = 4 * np.pi * r_centres**2 * dr

    all_g_r = []

    for atoms in atoms_list:
        idx_i, idx_j, dists = neighbor_list('ijd', atoms, cutoff=rmax)
        syms = np.array(atoms.get_chemical_symbols())
        vol = atoms.get_volume()
        n = len(atoms)

        if pair is not None:
            p1, p2 = pair.split("-")
            n_source = int(np.sum(syms == p1))
            n_target = int(np.sum(syms == p2))
            if p1 == p2:
                rho_target = (n_target - 1) / vol
            else:
                rho_target = n_target / vol
            mask = (syms[idx_i] == p1) & (syms[idx_j] == p2)
            pair_dists = dists[mask]
        else:
            n_source = n
            rho_target = (n - 1) / vol
            pair_dists = dists

        if len(pair_dists) == 0 or n_source == 0 or rho_target <= 0:
            all_g_r.append(np.zeros(nbins))
            continue

        in_range = (pair_dists > 0) & (pair_dists < rmax)
        if not np.any(in_range):
            all_g_r.append(np.zeros(nbins))
            continue
        hist, _ = np.histogram(pair_dists[in_range], bins=nbins,
                               range=(0, rmax))

        g_r = np.zeros(nbins)
        valid = shell_vols > 0
        g_r[valid] = hist[valid] / (n_source * rho_target * shell_vols[valid])
        all_g_r.append(g_r)

    all_g_r = np.array(all_g_r)

    return {
        "r": r_centres.tolist(),
        "g_r_mean": np.mean(all_g_r, axis=0).tolist(),
        "g_r_std": np.std(all_g_r, axis=0).tolist(),
        "n_structures": len(atoms_list),
    }
