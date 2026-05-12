"""Structural analysis: density, coordination, bond distances, bond angles."""

from __future__ import annotations

import numpy as np
from collections import Counter, defaultdict
from ase.neighborlist import neighbor_list


def compute_density(atoms_list: list) -> dict:
    """Compute density for each structure."""
    if not atoms_list:
        return {"values": [], "mean": 0.0, "std": 0.0}
    densities = []
    for atoms in atoms_list:
        mass_g = sum(atoms.get_masses()) / 6.022e23
        vol_cm3 = atoms.get_volume() * 1e-24
        densities.append(mass_g / vol_cm3)
    return {
        "values": densities,
        "mean": float(np.mean(densities)),
        "std": float(np.std(densities)),
    }


def build_neighbour_dict(atoms, cutoff, get_cutoff_fn):
    """Build a neighbour dictionary for one frame."""
    idx_i, idx_j, dists, vecs = neighbor_list(
        'ijdD', atoms, cutoff=cutoff
    )
    syms = atoms.get_chemical_symbols()
    nbr_dict = defaultdict(list)

    for k in range(len(idx_i)):
        si = syms[idx_i[k]]
        sj = syms[idx_j[k]]
        cut = get_cutoff_fn(si, sj)
        if dists[k] <= cut:
            nbr_dict[idx_i[k]].append(
                (idx_j[k], sj, dists[k], vecs[k])
            )

    return nbr_dict, syms


def compute_coordination(atoms_list, max_cutoff, get_cutoff_fn,
                         pair=None) -> dict:
    """Compute coordination numbers with percentage distribution."""
    cn_data = {}

    for atoms in atoms_list:
        nbr_dict, syms = build_neighbour_dict(atoms, max_cutoff,
                                               get_cutoff_fn)
        unique = sorted(set(syms))
        n = len(atoms)

        for s1 in unique:
            for s2 in unique:
                if pair is not None:
                    p1 = pair.split("-")
                    if not ((s1 == p1[0] and s2 == p1[1]) or
                            (s1 == p1[1] and s2 == p1[0])):
                        continue
                key = f"{s1}-{s2}"
                for a in range(n):
                    if syms[a] != s1:
                        continue
                    cn = sum(1 for _, sj, _, _ in nbr_dict[a] if sj == s2)
                    cn_data.setdefault(key, []).append(cn)

    result = {}
    for key, cns in sorted(cn_data.items()):
        cns = np.array(cns)
        counts = Counter(cns)
        total = len(cns)
        distribution = {
            int(cn): round(100.0 * count / total, 1)
            for cn, count in sorted(counts.items())
        }
        result[key] = {
            "mean": float(np.mean(cns)),
            "std": float(np.std(cns)),
            "min": int(np.min(cns)),
            "max": int(np.max(cns)),
            "distribution": distribution,
            "total_atoms": total,
        }
    return result


def compute_bond_distances(atoms_list, max_cutoff, get_cutoff_fn,
                           pair=None) -> dict:
    """Compute pair distance statistics."""
    dist_data = {}

    for atoms in atoms_list:
        nbr_dict, syms = build_neighbour_dict(atoms, max_cutoff,
                                               get_cutoff_fn)
        for a in range(len(atoms)):
            for _, sj, d, _ in nbr_dict[a]:
                s1 = syms[a]
                key = f"{s1}-{sj}" if s1 <= sj else f"{sj}-{s1}"
                if pair is not None and key != pair:
                    p_rev = "-".join(pair.split("-")[::-1])
                    if key != p_rev:
                        continue
                dist_data.setdefault(key, []).append(d)

    result = {}
    for key, ds in sorted(dist_data.items()):
        ds = np.array(ds)
        result[key] = {
            "mean": float(np.mean(ds)),
            "std": float(np.std(ds)),
            "min": float(np.min(ds)),
            "max": float(np.max(ds)),
            "count": len(ds),
        }
    return result


def compute_all_angles(atoms_list, max_cutoff, get_cutoff_fn,
                       triplet=None, bonding_only=True) -> dict:
    """Compute all bond angles."""
    bonding_pairs = None
    if bonding_only:
        try:
            from ..pipeline.random_gen import _classify_bond
        except ImportError:
            from amorphgen.pipeline.random_gen import _classify_bond

        bonding_pairs = set()
        unique = sorted(set(atoms_list[0].get_chemical_symbols()))
        # In multi-element systems we exclude same-element covalent
        # pairs because those are anion-anion contacts (e.g.\ O-O in
        # SiO2), not real bonds. In single-element systems (a-Si,
        # a-C, a-Ge, ...) the same-element covalent pair *is* the
        # bond, so we must keep it.
        single_element = len(unique) == 1
        for s1 in unique:
            for s2 in unique:
                bond_type = _classify_bond(s1, s2)
                if bond_type == "ionic" or bond_type == "metallic" or \
                   (bond_type == "covalent" and (s1 != s2 or single_element)):
                    bonding_pairs.add((s1, s2))
                    bonding_pairs.add((s2, s1))

    angle_data = {}

    for atoms in atoms_list:
        nbr_dict, syms = build_neighbour_dict(atoms, max_cutoff,
                                               get_cutoff_fn)
        for a in range(len(atoms)):
            sym_a = syms[a]
            nbrs = nbr_dict[a]

            if bonding_pairs is not None:
                nbrs_filtered = [
                    (j, sj, d, v) for j, sj, d, v in nbrs
                    if (sym_a, sj) in bonding_pairs
                ]
            else:
                nbrs_filtered = nbrs

            for p in range(len(nbrs_filtered)):
                for q in range(p + 1, len(nbrs_filtered)):
                    _, s1, _, v1 = nbrs_filtered[p]
                    _, s2, _, v2 = nbrs_filtered[q]
                    sa, sb = sorted([s1, s2])
                    key = f"{sa}-{sym_a}-{sb}"

                    if triplet is not None and key != triplet:
                        continue

                    norm_v1 = np.linalg.norm(v1)
                    norm_v2 = np.linalg.norm(v2)
                    if norm_v1 < 1e-12 or norm_v2 < 1e-12:
                        continue

                    cos_a = np.dot(v1, v2) / (norm_v1 * norm_v2)
                    cos_a = np.clip(cos_a, -1, 1)
                    angle_data.setdefault(key, []).append(
                        np.degrees(np.arccos(cos_a)))

    return angle_data


def compute_bond_angle_stats(angle_data: dict) -> dict:
    """Convert raw angle data to statistics."""
    result = {}
    for key, angles in sorted(angle_data.items()):
        angles = np.array(angles)
        result[key] = {
            "mean": float(np.mean(angles)),
            "std": float(np.std(angles)),
            "min": float(np.min(angles)),
            "max": float(np.max(angles)),
            "count": len(angles),
        }
    return result
