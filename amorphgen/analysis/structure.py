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
        # Rules for which same-element pairs count as "bonded":
        # - Single-element system (a-Si, a-C, a-Ge, Cu, ...): X-X IS
        #   the bond (covalent or metallic), so keep it.
        # - Multi-element system containing an anion (oxides, halides,
        #   chalcogenides, ...): same-element pairs are second-shell
        #   contacts mediated by the anion, NOT real first-shell bonds.
        #   This excludes O-O in SiO2 (covalent same-element) AND
        #   Hf-Hf in HfO2 (metallic same-element).
        # - Pure-metal alloy (NiTi, CuZr, all elements metallic): X-X
        #   IS a real bond (alloy chemistry), so keep it.
        single_element = len(unique) == 1
        has_anion_bond = any(
            _classify_bond(s1, s2) == "ionic"
            for s1 in unique for s2 in unique if s1 != s2
        )
        for s1 in unique:
            for s2 in unique:
                bond_type = _classify_bond(s1, s2)
                # Same-element pair in a multi-element system: keep
                # only if pure-metal alloy (no anion to mediate
                # second-shell contacts).
                if s1 == s2 and not single_element:
                    if bond_type == "metallic" and not has_anion_bond:
                        bonding_pairs.add((s1, s2))
                    # otherwise: skip (same-element non-bond in
                    # an anion-containing compound)
                    continue
                # Different-element or single-element case:
                if bond_type == "ionic" or bond_type == "metallic" or \
                   bond_type == "covalent":
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


def compute_dimers(atoms_list, threshold_frac: float = 0.85) -> dict:
    """Detect unphysically close same-element / anion-anion pairs ("dimers").

    A dimer is a HOMONUCLEAR (same-element) pair whose distance falls below
    ``threshold_frac`` times the radii-derived minimum separation for that
    pair (``utils.radii.default_minsep``) — the classic "wrong bond" defect
    of amorphous networks: O-O peroxide, S-S disulfide, N-N (~N2), Cl-Cl,
    P-P, and metal-metal dimers. In anion-bearing systems, same-element
    METAL pairs are additionally skipped (cations pack closer than the
    metallic threshold around shared anions — relaxed a-Li3OCl has physical
    Li-Li at 2.27-2.38 A that would otherwise flag 13 false dimers).

    Only same-element pairs are checked. Cross-element contacts are NOT
    flagged, because they conflate real bonds with non-defects: a
    polyanion former bonds covalently to its anions (phosphate P-O ~1.5 A,
    thiophosphate P-S ~2.0 A, LiPON P-N ~1.5 A), and those short contacts
    are the STRUCTURE, not a defect — flagging them would drown the report
    (KTiOPO4 relaxed shows 23 real P-O "bonds" that are not dimers). Genuine
    defects in those same structures are homonuclear (S-S, P-P, N-N) and are
    still caught.

    With the default 0.85, O-O flags below ~1.9 A (0.85 x 2.24 A), the
    peroxide signature that cold-relaxed MLIP structures develop from loose
    seeds; such structures sit higher in energy and should usually be
    discarded from an ensemble (rank by energy, keep the dimer-free
    members).

    Returns
    -------
    dict
        ``{"pairs": {pair: {"count", "min_distance", "threshold"}},
        "per_structure": [int, ...], "total": int, "n_structures": int,
        "threshold_frac": float}`` — ``pairs`` contains only pairs with at
        least one dimer; counts are summed over all structures.
    """
    from ..utils.radii import default_minsep, NONMETALS, METALLOIDS

    symbols = sorted({s for atoms in atoms_list
                      for s in atoms.get_chemical_symbols()})
    minsep = default_minsep(symbols)
    has_anions = any(s in NONMETALS for s in symbols)

    # Homonuclear pairs only (see docstring). Same-element metal pairs are
    # skipped when anions are present (metallic threshold is the wrong
    # yardstick for cations packed around anions).
    thresholds = {}
    for pair, d in minsep.items():
        a, b = pair.split("-")
        if a != b:
            continue
        a_is_metal = a not in NONMETALS and a not in METALLOIDS
        if not a_is_metal or not has_anions:
            thresholds[pair] = threshold_frac * d
    if not thresholds:
        return {"pairs": {}, "per_structure": [0] * len(atoms_list),
                "total": 0, "n_structures": len(atoms_list),
                "threshold_frac": threshold_frac}

    max_cut = max(thresholds.values())
    pair_stats = {}
    per_structure = []
    for atoms in atoms_list:
        syms = np.array(atoms.get_chemical_symbols())
        i, j, d = neighbor_list("ijd", atoms, cutoff=max_cut)
        mask = i < j                       # each direction once
        # Dedup periodic images: in cells smaller than ~2x the cutoff the
        # same (i, j) pair can appear once per image — keep only the
        # minimum-image distance so a single close contact is not counted
        # twice.
        pair_min: dict = {}
        for ii, jj, dd in zip(i[mask], j[mask], d[mask]):
            idx = (int(ii), int(jj))
            if idx not in pair_min or dd < pair_min[idx]:
                pair_min[idx] = float(dd)
        n_here = 0
        for (ii, jj), dd in pair_min.items():
            key = "-".join(sorted((syms[ii], syms[jj])))
            thr = thresholds.get(key)
            if thr is None or dd >= thr:
                continue
            n_here += 1
            st = pair_stats.setdefault(
                key, {"count": 0, "min_distance": float("inf"),
                      "threshold": thr})
            st["count"] += 1
            st["min_distance"] = min(st["min_distance"], dd)
        per_structure.append(n_here)

    return {"pairs": pair_stats, "per_structure": per_structure,
            "total": sum(per_structure), "n_structures": len(atoms_list),
            "threshold_frac": threshold_frac}


def format_dimer_report(result: dict) -> str:
    """Human-readable table for :func:`compute_dimers` output."""
    lines = ["", "=" * 65,
             f"  Dimer check  (threshold = {result['threshold_frac']:.2f} x minsep, "
             f"{result['n_structures']} structure(s))",
             "=" * 65]
    if result["total"] == 0:
        lines.append("  DIMER-FREE: no unphysical close contacts found.")
    else:
        lines.append(f"  {'pair':<8s} {'count':>6s} {'min dist (A)':>13s} "
                     f"{'threshold (A)':>14s}")
        lines.append("  " + "-" * 45)
        for pair, st in sorted(result["pairs"].items()):
            lines.append(f"  {pair:<8s} {st['count']:>6d} "
                         f"{st['min_distance']:>13.2f} {st['threshold']:>14.2f}")
        bad = sum(1 for n in result["per_structure"] if n)
        lines.append(f"\n  {result['total']} dimer(s) in {bad}/"
                     f"{result['n_structures']} structure(s). Dimer-bearing "
                     f"structures usually rank higher in energy — consider "
                     f"discarding them from the ensemble.")
    lines.append("=" * 65)
    return "\n".join(lines)
