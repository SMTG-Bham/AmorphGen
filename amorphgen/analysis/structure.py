"""Structural analysis: density, coordination, bond distances, bond angles."""

from __future__ import annotations

import numpy as np
from collections import Counter, defaultdict
from ase.neighborlist import neighbor_list

from .uncertainty import (summarize_observations, summarize_site_groups,
                          summarize_structures)
from ..utils.common import compute_density_gcm3


def compute_density(atoms_list: list) -> dict:
    """Compute density for each structure."""
    densities = [compute_density_gcm3(atoms) for atoms in atoms_list]
    uncertainty = summarize_structures(densities)
    return {
        "values": densities,
        "mean": float(np.mean(densities)) if densities else None,
        "std": float(np.std(densities)) if densities else None,
        "per_structure": densities,
        "uncertainty": uncertainty,
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


# a same-element metal pair is a first-shell bond above this metal fraction
_METAL_RICH_BOND_FRACTION = 0.70


from ..utils.radii import anion_elements      # noqa: F401  (re-exported)


def is_bonding_pair(s1: str, s2: str, composition) -> bool:
    """Whether an s1-s2 contact counts as a first-shell BOND in this compound
    (the rule shared by the coordination report, the total coordination, the
    bond angles and the CN plot).

    * Hydrogenated group-IV network (C/Si/Ge plus H, at most one H per host
      atom): host-host pairs follow the H-free host's rules below, X-H is a
      bond and H-H is not. The composition gate is shared with generation
      through :func:`~amorphgen.utils.radii._hydrogenated_host`.
    * Compound with an anion: a pair is a bond only when exactly one member is
      an anion, and which elements those are comes from :func:`anion_elements`
      (charge balance), not from a fixed list. Cation-cation contacts (Ga-In,
      and hetero pairs the radii table calls covalent such as Al-Si or Na-Si in
      aluminosilicate glasses) and anion-anion contacts are second-shell
      neighbours mediated by the anion.
    * No anion (a-Si, SiC, GaAs, alloys): the radii classification decides. A
      same-element pair is a bond in a single-element system, and in a
      metal-rich composition (at least 70 % metal atoms, which covers the
      metallic glasses Ni80P20, Fe80B20, Pd80Si20 as well as CuZr); Ga-Ga in
      GaAs or Ti-Ti in TiC is a second-shell contact, since those compounds are
      not metals.

    ``composition`` may be a mapping of counts (preferred: the hydrogen and
    metal fractions need them) or a bare set of symbols (one of each assumed).
    """
    try:
        from ..pipeline.random_gen import _classify_bond
        from ..utils.radii import NONMETALS, METALLOIDS, _hydrogenated_host
    except ImportError:
        from amorphgen.pipeline.random_gen import _classify_bond
        from amorphgen.utils.radii import (
            NONMETALS, METALLOIDS, _hydrogenated_host,
        )
    counts = (dict(composition) if hasattr(composition, "items")
              else {e: 1 for e in composition})
    host = _hydrogenated_host(counts)
    if host is not None:
        if "H" in (s1, s2):
            return s1 != s2
        counts = host
    elements = set(counts)
    anions = anion_elements(counts)
    if anions and anions != elements:
        return (s1 in anions) != (s2 in anions)
    bt = _classify_bond(s1, s2)
    if s1 == s2:
        if len(elements) == 1:
            return True
        n_metal = sum(n for e, n in counts.items()
                      if e not in NONMETALS and e not in METALLOIDS)
        metal_fraction = n_metal / sum(counts.values())
        return bt == "metallic" and metal_fraction >= _METAL_RICH_BOND_FRACTION
    return bt in ("ionic", "covalent", "metallic")


def compute_coordination(atoms_list, max_cutoff, get_cutoff_fn,
                         pair=None) -> dict:
    """Compute pooled site coordination and uncertainty of structure means."""
    cn_data = {}
    neighbour_species = sorted({s for atoms in atoms_list
                                for s in atoms.get_chemical_symbols()})

    for structure_index, atoms in enumerate(atoms_list):
        nbr_dict, syms = build_neighbour_dict(atoms, max_cutoff,
                                               get_cutoff_fn)
        unique = sorted(set(syms))
        n = len(atoms)

        for s1 in unique:
            for s2 in neighbour_species:
                if pair is not None:
                    p1 = pair.split("-")
                    if not ((s1 == p1[0] and s2 == p1[1]) or
                            (s1 == p1[1] and s2 == p1[0])):
                        continue
                key = f"{s1}-{s2}"
                if key not in cn_data:
                    cn_data[key] = [[] for _ in atoms_list]
                for a in range(n):
                    if syms[a] != s1:
                        continue
                    cn = sum(1 for _, sj, _, _ in nbr_dict[a] if sj == s2)
                    cn_data[key][structure_index].append(cn)

    result = {}
    for key, groups in sorted(cn_data.items()):
        result[key] = summarize_site_groups(groups)
    return result


def compute_bond_distances(atoms_list, max_cutoff, get_cutoff_fn,
                           pair=None) -> dict:
    """Keep pooled bond spread and uncertainty of equal-weight structure means."""
    dist_data = {}

    for structure_index, atoms in enumerate(atoms_list):
        nbr_dict, syms = build_neighbour_dict(atoms, max_cutoff,
                                               get_cutoff_fn)
        for a in range(len(atoms)):
            for j, sj, d, _ in nbr_dict[a]:
                if j < a:            # each bond once (the list holds both directions)
                    continue
                s1 = syms[a]
                key = f"{s1}-{sj}" if s1 <= sj else f"{sj}-{s1}"
                if pair is not None and key != pair:
                    p_rev = "-".join(pair.split("-")[::-1])
                    if key != p_rev:
                        continue
                if key not in dist_data:
                    dist_data[key] = [[] for _ in atoms_list]
                dist_data[key][structure_index].append(d)

    result = {}
    for key, groups in sorted(dist_data.items()):
        ds = np.array([d for group in groups for d in group])
        per_structure = [float(np.mean(group)) if group else None for group in groups]
        result[key] = summarize_observations(ds, per_structure)
    return result


class BondAngleData(dict):
    """Legacy pooled angle dictionary retaining aligned per-structure samples."""

    def __init__(self, n_structures=0):
        super().__init__()
        self.per_structure = [{} for _ in range(n_structures)]


def compute_all_angles(atoms_list, max_cutoff, get_cutoff_fn,
                       triplet=None, bonding_only=True) -> dict:
    """Compute all bond angles."""
    angle_data = BondAngleData(len(atoms_list))

    for structure_index, atoms in enumerate(atoms_list):
        nbr_dict, syms = build_neighbour_dict(atoms, max_cutoff,
                                               get_cutoff_fn)
        bonding_pairs = None
        if bonding_only:
            unique = sorted(set(syms))
            comp = Counter(syms)
            bonding_pairs = {(s1, s2) for s1 in unique for s2 in unique
                             if is_bonding_pair(s1, s2, comp)}
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
                    angle = float(np.degrees(np.arccos(cos_a)))
                    angle_data.setdefault(key, []).append(angle)
                    angle_data.per_structure[structure_index].setdefault(
                        key, []).append(angle)

    return angle_data


def compute_bond_angle_stats(angle_data: dict) -> dict:
    """Keep pooled angle spread and uncertainty of equal-weight structure means."""
    result = {}
    for key, angles in sorted(angle_data.items()):
        angles = np.array(angles)
        if angles.size == 0:
            continue
        groups = getattr(angle_data, "per_structure", None)
        # A plain dictionary has no structure identities.  Treat it as one
        # pooled sample and leave uncertainty unavailable, never infer n from
        # the number of angles.
        per_structure = ([float(np.mean(group[key])) if group.get(key) else None
                          for group in groups] if groups is not None
                         else [float(np.mean(angles))])
        result[key] = summarize_observations(angles, per_structure)
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

    # The counts, not the bare element set: which nonmetals are cations (the
    # P of a phosphate) and which anions bond to themselves (S2 2- in FeS2)
    # follow the stoichiometry, and one of each reads Li2O as a peroxide.
    counts = Counter(s for atoms in atoms_list
                     for s in atoms.get_chemical_symbols())
    minsep = default_minsep(dict(counts))
    has_anions = any(s in NONMETALS for s in counts)

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
    max_cut = max(thresholds.values(), default=0.0)
    pair_stats = {}
    per_structure = []
    per_structure_sites = []
    per_structure_pairs = []
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
        dimer_sites = set()
        local_pairs = {}
        for (ii, jj), dd in pair_min.items():
            key = "-".join(sorted((syms[ii], syms[jj])))
            thr = thresholds.get(key)
            if thr is None or dd >= thr:
                continue
            n_here += 1
            dimer_sites.update((ii, jj))
            local_pairs.setdefault(key, []).append(dd)
            st = pair_stats.setdefault(
                key, {"count": 0, "min_distance": float("inf"),
                      "threshold": thr})
            st["count"] += 1
            st["min_distance"] = min(st["min_distance"], dd)
        per_structure.append(n_here)
        per_structure_sites.append(len(dimer_sites))
        per_structure_pairs.append(local_pairs)

    site_fractions = [count / len(atoms) if len(atoms) else None
                      for count, atoms in zip(per_structure_sites, atoms_list)]
    for key, stats in pair_stats.items():
        counts = [len(group.get(key, [])) for group in per_structure_pairs]
        stats["per_structure"] = counts
        stats["uncertainty"] = summarize_structures(counts)
        stats["min_distance_uncertainty"] = summarize_structures([
            min(group[key]) if key in group else None for group in per_structure_pairs])
    total_sites = sum(len(atoms) for atoms in atoms_list)
    presence = [float(count > 0) for count in per_structure]
    return {"pairs": pair_stats, "per_structure": per_structure,
            "total": sum(per_structure), "n_structures": len(atoms_list),
            "threshold_frac": threshold_frac,
            "uncertainty": summarize_structures(per_structure),
            "fraction_of_sites": (sum(per_structure_sites) / total_sites
                                  if total_sites else None),
            "fraction_of_structures": float(np.mean(presence)) if presence else None,
            "site_fraction_uncertainty": summarize_structures(site_fractions),
            "structure_fraction_uncertainty": summarize_structures(presence)}


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
    site_fraction = result.get("fraction_of_sites")
    structure_fraction = result.get("fraction_of_structures")
    if site_fraction is not None and structure_fraction is not None:
        lines.append(f"  Fraction of sites in dimers: {100 * site_fraction:.1f}%; "
                     f"fraction of structures with dimers: {100 * structure_fraction:.1f}%.")
    lines.append("=" * 65)
    return "\n".join(lines)


def compute_polyhedral_connectivity(atoms_list, max_cutoff, get_cutoff_fn,
                                    cation=None, anion=None) -> dict:
    """Corner-, edge- and face-sharing between cation-centred polyhedra.

    Two cations are *linked* when they share at least one anion neighbour
    (within the pair cutoff).  One shared anion = corner-sharing, two =
    edge-sharing, three or more = face-sharing.  Reports the linkage
    distribution and, per cation, how many are involved in at least one
    edge- or face-sharing link (the quantity that separates a corner-
    sharing network glass such as a-SiO2 / a-GeO2 from a random packing).

    Parameters
    ----------
    cation, anion : str, optional
        Element symbols.  Default: anion = most electronegative element,
        cations = every other element (pooled, and also reported per species).
    """
    try:
        from ..utils.radii import PAULING_EN
    except ImportError:  # pragma: no cover
        from amorphgen.utils.radii import PAULING_EN
    if not atoms_list:
        return {"error": "no structures"}
    syms0 = sorted({s for atoms in atoms_list for s in atoms.get_chemical_symbols()})
    if anion is None:
        anion = max(syms0, key=lambda s: PAULING_EN.get(s, 2.0))
    cations = [cation] if cation else [s for s in syms0 if s != anion]
    if not cations:
        return {"error": "single-element system: no cation/anion split"}

    link = Counter()                      # corner / edge / face counts
    per_species = {c: {"n": 0, "edge_or_face": 0, "face": 0,
                       "corner_links": 0, "edge_links": 0, "face_links": 0}
                   for c in cations}
    per_structure_edge = []
    per_structure_face = []
    per_structure_links = []
    per_structure_species = {c: [] for c in cations}
    n_cat_total = 0; n_ef_total = 0

    for atoms in atoms_list:
        nbrs, syms = build_neighbour_dict(atoms, max_cutoff, get_cutoff_fn)
        an_of = {}
        for i, s in enumerate(syms):
            if s in cations:
                an_of[i] = {j for (j, sj, d, v) in nbrs[i] if sj == anion}
        # cation pairs through a shared anion
        shared = defaultdict(int)
        for i, aset in an_of.items():
            for a in aset:
                for (j, sj, d, v) in nbrs[a]:
                    if sj in cations and j > i:
                        shared[(i, j)] += 1
        ef = set(); face = set(); links_of = defaultdict(lambda: Counter())
        local_links = Counter()
        for (i, j), n in shared.items():
            kind = "corner" if n == 1 else ("edge" if n == 2 else "face")
            link[kind] += 1
            local_links[kind] += 1
            links_of[i][kind] += 1; links_of[j][kind] += 1
            if n >= 2:
                ef.update((i, j))
            if n >= 3:
                face.update((i, j))
        n_cat = len(an_of); n_cat_total += n_cat; n_ef_total += len(ef)
        per_structure_edge.append(100.0 * len(ef) / n_cat if n_cat else None)
        per_structure_face.append(100.0 * len(face) / n_cat if n_cat else None)
        per_structure_links.append(local_links)
        for i in an_of:
            ps = per_species[syms[i]]
            ps["n"] += 1; ps["edge_or_face"] += (i in ef); ps["face"] += (i in face)
            for kind in ("corner", "edge", "face"):
                ps[f"{kind}_links"] += links_of[i][kind]
        for c in cations:
            sites = [i for i in an_of if syms[i] == c]
            count = len(sites)
            per_structure_species[c].append({
                "n_atoms": count,
                "edge_or_face_percent": (100.0 * sum(i in ef for i in sites) / count
                                         if count else None),
                "face_percent": (100.0 * sum(i in face for i in sites) / count
                                 if count else None),
                **{f"mean_{kind}_links": (sum(links_of[i][kind] for i in sites) / count
                                          if count else None)
                   for kind in ("corner", "edge", "face")},
            })

    total_links = sum(link.values())
    out = {
        "cations": cations, "anion": anion,
        "n_links": total_links,
        "link_percent": {k: (100.0 * link[k] / total_links if total_links else 0.0)
                         for k in ("corner", "edge", "face")},
        "cation_edge_or_face_percent": (100.0 * n_ef_total / n_cat_total
                                        if n_cat_total else 0.0),
        "per_structure_edge_percent": per_structure_edge,
        "uncertainty": summarize_structures(per_structure_edge),
        "link_percent_uncertainty": {
            kind: summarize_structures([
                100.0 * links[kind] / sum(links.values()) if links else None
                for links in per_structure_links])
            for kind in ("corner", "edge", "face")},
        "n_links_uncertainty": summarize_structures([
            sum(links.values()) for links in per_structure_links]),
        "fraction_of_sites": (n_ef_total / n_cat_total if n_cat_total else None),
        "fraction_of_structures": float(np.mean([
            value is not None and value > 0 for value in per_structure_edge])),
        "site_fraction_uncertainty": summarize_structures([
            value / 100 if value is not None else None for value in per_structure_edge]),
        "structure_fraction_uncertainty": summarize_structures([
            float(value is not None and value > 0) for value in per_structure_edge]),
        "fraction_definition": "cation sites / structures with any edge- or face-sharing cation",
        "face_percent_uncertainty": summarize_structures(per_structure_face),
        "per_species": {},
    }
    for c, ps in per_species.items():
        n = ps["n"] or 1
        out["per_species"][c] = {
            "n_atoms": ps["n"],
            "edge_or_face_percent": 100.0 * ps["edge_or_face"] / n,
            "face_percent": 100.0 * ps["face"] / n,
            "mean_corner_links": ps["corner_links"] / n,
            "mean_edge_links": ps["edge_links"] / n,
            "mean_face_links": ps["face_links"] / n,
            "per_structure": per_structure_species[c],
            "uncertainty": {
                field: summarize_structures([row[field] for row in per_structure_species[c]])
                for field in per_structure_species[c][0]},
        }
    return out


def format_connectivity_report(r: dict) -> str:
    if "error" in r:
        return f"\n  Polyhedral connectivity: {r['error']}"
    lp = r["link_percent"]
    lines = [f"\n  Polyhedral connectivity ({'/'.join(r['cations'])}-centred, "
             f"shared {r['anion']} neighbours):",
             f"    cation-cation links: {r['n_links']} "
             f"(corner {lp['corner']:.1f}%, edge {lp['edge']:.1f}%, face {lp['face']:.1f}%)",
             f"    cations in >=1 edge/face-sharing pair: "
             f"{r['cation_edge_or_face_percent']:.1f}%"]
    for c, ps in r["per_species"].items():
        lines.append(f"    {c}: {ps['n_atoms']} atoms, edge/face-sharing {ps['edge_or_face_percent']:.1f}%, "
                     f"links per atom corner {ps['mean_corner_links']:.2f} / edge "
                     f"{ps['mean_edge_links']:.2f} / face {ps['mean_face_links']:.2f}")
    if r.get("fraction_of_sites") is not None:
        lines.append(f"    Fraction of cation sites in edge/face links: "
                     f"{100 * r['fraction_of_sites']:.1f}%; fraction of structures "
                     f"with such sites: {100 * r['fraction_of_structures']:.1f}%")
    if len(r["per_structure_edge_percent"]) > 1:
        v = np.array([value for value in r["per_structure_edge_percent"] if value is not None])
        if not len(v):
            return "\n".join(lines)
        lines.append(f"    per-structure edge/face-sharing cations: "
                     f"{v.mean():.1f}%, structure SD {v.std():.1f}% "
                     f"(min {v.min():.1f}, max {v.max():.1f})")
    uncertainty = r.get("uncertainty")
    if uncertainty and uncertainty["sem"] is not None:
        lines.append(f"    Edge/face site-percent mean: SEM={uncertainty['sem']:.2f}%; "
                     f"{100 * uncertainty['confidence']:g}% t CI "
                     f"[{uncertainty['ci_low']:.2f}, {uncertainty['ci_high']:.2f}]%")
    elif uncertainty:
        lines.append("    SEM/t interval unavailable: fewer than two structures with cation sites.")
    return "\n".join(lines)
