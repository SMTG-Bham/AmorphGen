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


# Neutron coherent scattering lengths (fm) — common amorphous-system elements.
# Values from NIST / Sears 1992. Used when weighting="neutron".
_NEUTRON_B = {
    "H": -3.7390, "D":  6.671,  "Li": -1.90,  "Be":  7.79,  "B":   5.30,
    "C":  6.6460, "N":  9.36,   "O":  5.803,  "F":   5.654, "Na":  3.63,
    "Mg": 5.375,  "Al": 3.449,  "Si": 4.1491, "P":   5.13,  "S":   2.847,
    "Cl": 9.5770, "K":  3.67,   "Ca": 4.70,   "Ti":  -3.438,"V":  -0.3824,
    "Cr": 3.635,  "Mn": -3.73,  "Fe": 9.45,   "Co":  2.49,  "Ni": 10.3,
    "Cu": 7.718,  "Zn": 5.680,  "Ga": 7.288,  "Ge":  8.185, "As":  6.58,
    "Se": 7.970,  "Br": 6.795,  "Y":  7.75,   "Zr":  7.16,  "Nb":  7.054,
    "Mo": 6.715,  "Pd": 5.91,   "Ag": 5.922,  "Cd":  5.1,   "In":  4.065,
    "Sn": 6.225,  "Sb": 5.57,   "Te": 5.80,   "I":   5.28,  "Cs":  5.42,
    "Ba": 5.07,   "La": 8.24,   "Ce": 4.84,   "Hf":  7.77,  "Ta":  6.91,
    "W":  4.755,  "Pt": 9.6,    "Au": 7.90,   "Pb":  9.405, "Bi":  8.532,
}


def _scattering_factors(unique_symbols, weighting):
    """Return {symbol: f} dict for the chosen weighting."""
    if weighting == "xray":
        from ase.data import atomic_numbers
        return {s: float(atomic_numbers[s]) for s in unique_symbols}
    if weighting == "neutron":
        try:
            return {s: _NEUTRON_B[s] for s in unique_symbols}
        except KeyError as exc:
            raise KeyError(
                f"No neutron scattering length tabulated for element {exc}. "
                f"Add it to amorphgen.analysis.rdf._NEUTRON_B or use "
                f"weighting='xray' instead."
            ) from None
    if weighting == "unweighted":
        return {s: 1.0 for s in unique_symbols}
    raise ValueError(
        f"weighting must be 'unweighted', 'xray', or 'neutron'; "
        f"got {weighting!r}"
    )


def compute_structure_factor_direct(atoms_list, qmax=15.0, nq=300,
                                    weighting="xray", q_batch=4096):
    """Compute S(q) directly from atomic positions via the Debye formula
    evaluated at reciprocal-lattice q-vectors.

    Avoids the rmax truncation that damps the first sharp diffraction
    peak (FSDP) in the FFT-of-g(r) approach. Q-vector enumeration uses
    the reciprocal lattice of each structure, so the q-resolution is
    limited only by the simulation cell size (q_min ~ 2*pi/L).

    Parameters
    ----------
    atoms_list : list[ase.Atoms]
        Ensemble of structures (all assumed to have the same composition).
    qmax : float
        Maximum q in inverse-Angstrom.
    nq : int
        Number of q-bins between 0 and qmax for spherical averaging.
    weighting : {"xray", "neutron", "unweighted"}
        Per-element scattering factors used in the sum.
    q_batch : int
        Number of q-vectors processed per matmul. Tune for memory.

    Returns
    -------
    dict
        ``{"q": list[float], "s_q": list[float], "n_per_bin": list[int]}``.
        ``n_per_bin`` is the number of reciprocal-lattice vectors in each
        spherical shell — small values (1-3) indicate noisy estimates at
        low q.
    """
    atoms0 = atoms_list[0]
    syms0 = atoms0.get_chemical_symbols()
    unique = sorted(set(syms0))
    scat = _scattering_factors(unique, weighting)
    n_atoms = len(atoms0)

    # Composition fractions and <f>
    c = np.array([syms0.count(s) for s in unique], dtype=float) / n_atoms
    f_per_elem = np.array([scat[s] for s in unique])
    f_mean = float((c * f_per_elem).sum())
    if f_mean == 0:
        return {"q": [], "s_q": [], "n_per_bin": []}

    # Spherical-shell accumulators
    q_edges = np.linspace(0.0, qmax, nq + 1)
    q_centres = 0.5 * (q_edges[:-1] + q_edges[1:])
    sq_sum = np.zeros(nq)
    sq_count = np.zeros(nq, dtype=int)

    for atoms in atoms_list:
        positions = atoms.get_positions()
        syms = atoms.get_chemical_symbols()
        f_atoms = np.array([scat[s] for s in syms])
        N = len(atoms)
        denom = N * f_mean * f_mean

        # Reciprocal lattice: rows are b_i with a_i . b_j = 2pi delta_ij
        recip = 2.0 * np.pi * np.linalg.inv(atoms.cell.array).T  # (3,3)
        recip_min = float(np.linalg.norm(recip, axis=1).min())
        if recip_min == 0:
            continue
        n_max = int(np.ceil(qmax / recip_min)) + 1

        # Build all integer triplets (n1,n2,n3), exclude origin, build q
        ns = np.arange(-n_max, n_max + 1)
        grid = np.array(np.meshgrid(ns, ns, ns, indexing="ij")).reshape(3, -1).T
        # drop origin
        grid = grid[~np.all(grid == 0, axis=1)]
        q_vecs = grid @ recip                          # (M, 3)
        q_mags = np.linalg.norm(q_vecs, axis=1)
        keep = q_mags <= qmax
        q_vecs = q_vecs[keep]
        q_mags = q_mags[keep]

        # Process in batches to control memory
        M = q_vecs.shape[0]
        for start in range(0, M, q_batch):
            stop = min(start + q_batch, M)
            qb = q_vecs[start:stop]                    # (m, 3)
            phases = qb @ positions.T                  # (m, N)
            # |Σ f_i e^{i q·r_i}|² = (Σ f cos)² + (Σ f sin)²
            cos_sum = (f_atoms * np.cos(phases)).sum(axis=1)
            sin_sum = (f_atoms * np.sin(phases)).sum(axis=1)
            sq_vals = (cos_sum * cos_sum + sin_sum * sin_sum) / denom

            # Bin by |q|
            bin_idx = np.clip(((q_mags[start:stop] / qmax) * nq).astype(int),
                              0, nq - 1)
            np.add.at(sq_sum, bin_idx, sq_vals)
            np.add.at(sq_count, bin_idx, 1)

    sq = np.where(sq_count > 0, sq_sum / np.maximum(sq_count, 1), np.nan)
    return {
        "q": q_centres.tolist(),
        "s_q": sq.tolist(),
        "n_per_bin": sq_count.tolist(),
    }


def compute_structure_factor(atoms_list, pair=None, qmax=15.0, nq=300,
                             rmax=None, weighting="unweighted"):
    """Compute the structure factor S(q) from g(r) via Fourier transform.

    Parameters
    ----------
    atoms_list : list[ase.Atoms]
        Ensemble of structures.
    pair : str, optional
        Specific A-B partial (e.g. ``"Ga-O"``). If ``None``, computes the
        total S(q). The ``weighting`` argument only affects the total
        case; explicit partials are always returned as their own
        Faber-Ziman S_AB(q).
    qmax : float
        Maximum q in inverse-Angstrom (default 15.0).
    nq : int
        Number of q points (default 300).
    rmax : float, optional
        Max radius for the underlying g(r). ``None`` = auto (half cell).
    weighting : {"unweighted", "xray", "neutron"}, default ``"unweighted"``
        How to combine partials into the total S(q):

        * ``"unweighted"`` — single FT of the all-atom g(r). Fast,
          useful for ensemble-vs-ensemble comparisons. Does NOT match
          experimental X-ray/neutron S(Q) in general because it omits
          per-element scattering weights.
        * ``"xray"`` — Faber-Ziman partials weighted by atomic numbers
          squared (Z_A·Z_B). The Z² approximation is exact only at
          q = 0; quantitatively good below q ~ 5 inverse-Angstrom
          (covers the FSDP and main-peak region).
        * ``"neutron"`` — same Faber-Ziman combination but weighted by
          tabulated neutron coherent scattering lengths (b_A·b_B). Uses
          a built-in table of ~50 common elements; raises KeyError for
          unsupported species.

    Returns
    -------
    dict
        ``{"q": list[float], "s_q": list[float]}``.

    Notes
    -----
    For X-ray S(Q) of amorphous oxides the FSDP at ~1.5-2.5
    inverse-Angstrom is dominated by heavy-atom cation-cation
    correlations and cancels in the unweighted sum because
    the cation-anion partial dips at the same q. The ``"xray"``
    weighting recovers it. See ``examples/test_structure_factor.py``
    for a worked example on a-Ga2O3 (Kaewmeechai et al., Phys. Rev. B
    111, 035203, 2025).
    """
    if weighting not in ("unweighted", "xray", "neutron"):
        raise ValueError(
            f"weighting must be 'unweighted', 'xray', or 'neutron'; "
            f"got {weighting!r}"
        )

    # ── Weighted-total path (Faber-Ziman combination of partials) ────────
    if pair is None and weighting in ("xray", "neutron"):
        from ase.data import atomic_numbers
        atoms = atoms_list[0]
        syms = atoms.get_chemical_symbols()
        unique = sorted(set(syms))
        if not unique:
            return {"q": [], "s_q": []}
        total_n = len(syms)
        fractions = {s: syms.count(s) / total_n for s in unique}

        if weighting == "xray":
            scat = {s: float(atomic_numbers[s]) for s in unique}
        else:  # neutron
            try:
                scat = {s: _NEUTRON_B[s] for s in unique}
            except KeyError as exc:
                raise KeyError(
                    f"No neutron scattering length tabulated for element "
                    f"{exc}. Add it to amorphgen.analysis.rdf._NEUTRON_B "
                    f"or use weighting='xray' instead."
                )

        # Compute every unique partial once
        partials = {}
        q_arr = None
        for i, s1 in enumerate(unique):
            for s2 in unique[i:]:
                pstr = f"{s1}-{s2}"
                sq = compute_structure_factor(
                    atoms_list, pair=pstr, qmax=qmax, nq=nq, rmax=rmax,
                    weighting="unweighted",   # avoid recursion
                )
                if q_arr is None:
                    q_arr = np.array(sq["q"])
                partials[(s1, s2)] = np.array(sq["s_q"])

        if q_arr is None:
            return {"q": [], "s_q": []}

        # Faber-Ziman: w_AB = (2 - delta_AB) c_A c_B b_A b_B / <b>^2
        mean_scat = sum(fractions[s] * scat[s] for s in unique)
        norm = mean_scat * mean_scat if mean_scat != 0 else 1.0

        s_total = np.zeros_like(q_arr)
        weight_sum = 0.0
        for (s1, s2), s_part in partials.items():
            mult = 1.0 if s1 == s2 else 2.0
            w = mult * fractions[s1] * fractions[s2] * scat[s1] * scat[s2] / norm
            s_total += w * s_part
            weight_sum += w
        if weight_sum > 0:
            s_total /= weight_sum

        return {"q": q_arr.tolist(), "s_q": s_total.tolist()}

    # ── Unweighted / partial path (original behaviour) ───────────────────
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
