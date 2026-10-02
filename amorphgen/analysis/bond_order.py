"""Steinhardt q6 and Lechner--Dellago neighbour-averaged qbar6.

References: Steinhardt et al., Phys. Rev. B 28, 784 (1983),
doi:10.1103/PhysRevB.28.784; Lechner and Dellago, J. Chem. Phys. 129,
114707 (2008), doi:10.1063/1.2977970.
"""

from __future__ import annotations

from math import factorial
from numbers import Integral

import numpy as np
from ase.neighborlist import neighbor_list
from scipy.special import lpmv

from .cutoff import parse_cutoff_spec, resolve_cutoffs


def _positive_cutoff(value):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("bond-order cutoff must be finite and positive") from exc
    if not np.isfinite(number) or number <= 0:
        raise ValueError("bond-order cutoff must be finite and positive")


def _resolve_cutoff(atoms_list, cutoff):
    if isinstance(cutoff, (bool, np.bool_)):
        raise ValueError("bond-order cutoff must be finite and positive")
    spec = parse_cutoff_spec(cutoff)
    values = spec.values() if isinstance(spec, dict) else [spec]
    for value in values:
        if isinstance(value, str) and value in ("auto", "auto-rdf"):
            continue
        _positive_cutoff(value)
    frames = [atoms for atoms in atoms_list if len(atoms)]
    if not frames:
        return {} if isinstance(spec, (str, dict)) else float(spec)

    # A complete pair table is already a concrete cutoff specification.
    # Supplying its otherwise-unused base avoids computing an RDF again
    # when an analyser or a melt-survival comparison reuses this table.
    if isinstance(spec, dict) and "default" not in spec:
        symbols = sorted(set(frames[0].get_chemical_symbols()))
        pairs = [(a, b) for i, a in enumerate(symbols) for b in symbols[i:]]
        if all(f"{a}-{b}" in spec or f"{b}-{a}" in spec for a, b in pairs):
            spec = {"default": max(spec.values()), **spec}
    resolved, _ = resolve_cutoffs(frames, spec)
    for value in resolved.values() if isinstance(resolved, dict) else [resolved]:
        _positive_cutoff(value)
    if isinstance(resolved, dict):
        for atoms in frames:
            symbols = sorted(set(atoms.get_chemical_symbols()))
            if any(f"{a}-{b}" not in resolved and f"{b}-{a}" not in resolved
                   for i, a in enumerate(symbols) for b in symbols[i:]):
                raise ValueError("pair-specific bond-order cutoffs require "
                                 "the same element types in all structures")
    return resolved


def _validate_geometry(atoms, index):
    if not np.isfinite(atoms.positions).all():
        raise ValueError(f"structure {index}: atom positions must be finite")
    if not np.isfinite(atoms.cell.array).all():
        raise ValueError(f"structure {index}: cell vectors must be finite")
    periodic = atoms.cell.array[np.asarray(atoms.pbc, dtype=bool)]
    if len(periodic) and np.linalg.matrix_rank(periodic) < len(periodic):
        raise ValueError(f"structure {index}: periodic cell vectors must be "
                         "nonzero and linearly independent")


def _q6_vectors(n_atoms, source, vectors, counts):
    """Average normalized complex Y_6m over each atom's bond directions."""
    qlm = np.zeros((n_atoms, 13), dtype=complex)
    if not len(source):
        return qlm
    lengths = np.linalg.norm(vectors, axis=1)
    cos_theta = np.clip(vectors[:, 2] / lengths, -1.0, 1.0)
    phi = np.arctan2(vectors[:, 1], vectors[:, 0])
    # lpmv includes the Condon--Shortley phase. This explicit spherical
    # harmonic form supports scipy>=1.10, including releases after removal
    # of sph_harm, without its incompatible sph_harm_y angle convention.
    for m in range(7):
        normalization = np.sqrt(13 / (4 * np.pi)
                                * factorial(6 - m) / factorial(6 + m))
        harmonic = normalization * lpmv(m, 6, cos_theta) * np.exp(1j * m * phi)
        np.add.at(qlm[:, 6 + m], source, harmonic)
        if m:
            np.add.at(qlm[:, 6 - m], source, (-1)**m * harmonic.conjugate())
    qlm /= np.maximum(counts, 1)[:, None]
    return qlm


def _clusters(ordered, source, target):
    """Label connected ordered atoms, counting cell atoms, not images."""
    parent = np.arange(len(ordered))

    def find(atom):
        while parent[atom] != atom:
            parent[atom] = parent[parent[atom]]
            atom = parent[atom]
        return atom

    edges = ordered[source] & ordered[target] & (source < target)
    for i, j in zip(source[edges], target[edges]):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    ids = np.full(len(ordered), -1, dtype=int)
    labels = {}
    for atom in np.flatnonzero(ordered):
        root = find(atom)
        ids[atom] = labels.setdefault(root, len(labels))
    sizes = np.bincount(ids[ordered]) if np.any(ordered) else np.array([], dtype=int)
    return ids, int(sizes.max()) if len(sizes) else 0


def compute_bond_order(atoms_list, cutoff="auto-rdf", qbar6_threshold=0.3,
                       min_neighbors=4):
    """Compute q6, qbar6 and connected crystal-like clusters for each frame.

    The Steinhardt vector is the mean of ``Y_6m`` over all neighbours
    inside the cutoff. Lechner--Dellago averaging then averages those
    *complex vectors* over the central atom and its neighbour images,
    before taking the rotational invariant ``sqrt(4*pi/13*sum(|q6m|^2))``.
    Both definitions include periodic self-images and repeated images in
    small cells, using ASE's full periodic neighbour list, including for
    triclinic cells. An isolated atom has q6 = qbar6 = 0.

    An atom is labelled ordered when ``qbar6 >= qbar6_threshold`` and its
    neighbour count is at least ``min_neighbors``. The threshold is a
    material- and cutoff-dependent diagnostic, not a universal crystal
    classifier: calibrate it against the relevant crystal and liquid.
    Ordered atoms sharing a neighbour-list edge form a cluster. Cluster
    sizes count unique atoms in the simulation cell, even when a periodic
    cluster connects to itself across the boundary.

    Parameters
    ----------
    atoms_list : iterable of ase.Atoms
        Structures in input order. Empty frames and an empty list produce
        zero counts, fractions and means.
    cutoff : float, dict, or str
        Neighbour cutoff in Angstrom; accepts the same scalar, pair-table
        and automatic forms as :class:`StructureAnalyser`. All numeric
        cutoffs must be positive and finite. Resolved cutoffs are shared
        across frames so their results can be compared.
    qbar6_threshold : float
        Inclusive ordered-atom threshold, between 0 and 1 (default 0.3).
    min_neighbors : int
        Minimum neighbour-image count to classify an atom (default 4).

    Returns
    -------
    dict
        ``parameters`` stores the resolved ``cutoff``, ``qbar6_threshold``
        and ``min_neighbors``. ``per_structure`` contains atom arrays
        ``q6``, ``qbar6``, ``neighbor_counts``, ``ordered`` and ``cluster_ids``
        (disordered atoms have cluster ID -1), plus ``index``, ``n_atoms``,
        ``ordered_count``, ``ordered_fraction``, ``largest_cluster_size``,
        ``largest_cluster_fraction``, ``q6_mean`` and ``qbar6_mean``.
        Fractions use all atoms in that frame as the denominator.
        Top-level scalar summaries are arithmetic means over frames,
        including ``largest_cluster_size``; ``n_structures`` is the count.
    """
    try:
        threshold = float(qbar6_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("qbar6_threshold must be finite and between 0 and 1") from exc
    if (isinstance(qbar6_threshold, (bool, np.bool_))
            or not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("qbar6_threshold must be finite and between 0 and 1")
    if (isinstance(min_neighbors, (bool, np.bool_))
            or not isinstance(min_neighbors, Integral) or min_neighbors < 1):
        raise ValueError("min_neighbors must be a positive integer")
    frames = list(atoms_list)
    for index, atoms in enumerate(frames):
        _validate_geometry(atoms, index)
    resolved = _resolve_cutoff(frames, cutoff)
    ase_cutoff = ({tuple(pair.split("-")): value for pair, value in resolved.items()}
                  if isinstance(resolved, dict) else resolved)
    per_structure = []
    for index, atoms in enumerate(frames):
        n_atoms = len(atoms)
        if n_atoms:
            source, target, vectors = neighbor_list("ijD", atoms, ase_cutoff)
            distances = np.linalg.norm(vectors, axis=1)
            if not np.isfinite(distances).all() or np.any(distances <= 1e-12):
                raise ValueError(f"structure {index}: overlapping atoms or "
                                 "invalid neighbour vectors prevent bond-order analysis")
        else:
            source = target = np.array([], dtype=int)
            vectors = np.empty((0, 3))
        counts = np.bincount(source, minlength=n_atoms)
        qlm = _q6_vectors(n_atoms, source, vectors, counts)
        averaged = qlm.copy()
        np.add.at(averaged, source, qlm[target])
        averaged /= (counts + 1)[:, None]
        q6 = np.sqrt(4 * np.pi / 13 * np.sum(np.abs(qlm)**2, axis=1))
        qbar6 = np.sqrt(4 * np.pi / 13 * np.sum(np.abs(averaged)**2, axis=1))
        ordered = (counts >= min_neighbors) & (qbar6 >= threshold)
        cluster_ids, largest = _clusters(ordered, source, target)
        ordered_count = int(ordered.sum())
        per_structure.append({
            "index": index, "n_atoms": n_atoms,
            "q6": q6, "qbar6": qbar6, "neighbor_counts": counts,
            "ordered": ordered, "cluster_ids": cluster_ids,
            "ordered_count": ordered_count,
            "ordered_fraction": ordered_count / n_atoms if n_atoms else 0.0,
            "largest_cluster_size": largest,
            "largest_cluster_fraction": largest / n_atoms if n_atoms else 0.0,
            "q6_mean": float(q6.mean()) if n_atoms else 0.0,
            "qbar6_mean": float(qbar6.mean()) if n_atoms else 0.0,
        })
    result = {
        "parameters": {"cutoff": resolved, "qbar6_threshold": threshold,
                       "min_neighbors": int(min_neighbors)},
        "n_structures": len(frames), "per_structure": per_structure,
    }
    for key in ("q6_mean", "qbar6_mean", "ordered_fraction",
                "largest_cluster_size", "largest_cluster_fraction"):
        result[key] = (float(np.mean([frame[key] for frame in per_structure]))
                       if per_structure else 0.0)
    return result
