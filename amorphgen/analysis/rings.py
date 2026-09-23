"""Ring statistics for network glasses."""

from __future__ import annotations

import numpy as np
from collections import Counter, defaultdict, deque
from ase.neighborlist import neighbor_list


def compute_ring_statistics(atoms_list, bond_pair=None, cutoff=None,
                            max_ring=12, get_cutoff_fn=None):
    """
    Compute ring size distribution for network glasses.

    For each edge in the network graph (A-B-A path through bridging B),
    find the shortest ring by BFS excluding that edge.
    """
    # Auto-detect bond pair: ring NODES are the least electronegative
    # element (network former / cation, e.g. Si in SiO2) and the BRIDGES
    # are the most electronegative one (anion).  Sorting symbols
    # alphabetically (old behaviour) picked O as the node for SiO2 and
    # reported 3-rings for cristobalite instead of 6-rings.
    if bond_pair is None:
        unique = sorted(set(atoms_list[0].get_chemical_symbols()))
        if len(unique) == 1:
            bond_pair = (unique[0], unique[0])
        else:
            try:
                from ..utils.radii import PAULING_EN
            except ImportError:
                from amorphgen.utils.radii import PAULING_EN
            en = lambda sym: PAULING_EN.get(sym, 2.0)
            bond_pair = (min(unique, key=en), max(unique, key=en))

    ring_counts = Counter()

    for atoms in atoms_list:
        syms = atoms.get_chemical_symbols()
        n = len(atoms)
        p1, p2 = bond_pair

        if cutoff is None and get_cutoff_fn is not None:
            cut = get_cutoff_fn(p1, p2)
        elif cutoff is not None:
            cut = cutoff
        else:
            cut = 2.5  # fallback

        # Build adjacency for p1-p2 bonds
        idx_i, idx_j, dists = neighbor_list('ijd', atoms, cutoff=cut)
        adj = defaultdict(set)
        for k in range(len(idx_i)):
            si, sj = syms[idx_i[k]], syms[idx_j[k]]
            if (si == p1 and sj == p2) or (si == p2 and sj == p1):
                adj[idx_i[k]].add(idx_j[k])

        # Build network graph: p1-p1 edges through bridging p2, or the
        # direct p1-p1 bonds for a single-element network (a-Si, a-Ge, C).
        p1_indices = [i for i in range(n) if syms[i] == p1]
        net_adj = defaultdict(set)
        if p1 == p2:
            for a in p1_indices:
                net_adj[a] = {b for b in adj[a] if b != a}
        else:
            for a in p1_indices:
                for bridge in adj[a]:
                    if syms[bridge] == p2:
                        for b in adj[bridge]:
                            if b != a and syms[b] == p1:
                                net_adj[a].add(b)

        # For each edge, find shortest ring
        counted_edges = set()
        for a in p1_indices:
            for b in net_adj[a]:
                edge = (min(a, b), max(a, b))
                if edge in counted_edges:
                    continue
                counted_edges.add(edge)

                dist = {a: 0}
                queue = deque([a])
                found = None

                while queue:
                    cur = queue.popleft()
                    if dist[cur] >= max_ring:
                        break
                    for nb in net_adj[cur]:
                        if cur == a and nb == b:
                            continue
                        if nb == b:
                            found = dist[cur] + 2
                            break
                        if nb not in dist:
                            dist[nb] = dist[cur] + 1
                            queue.append(nb)
                    if found:
                        break

                if found and found <= max_ring:
                    ring_counts[found] += 1

    total = sum(ring_counts.values())
    sizes = sorted(ring_counts.keys())
    counts = [ring_counts[s] for s in sizes]
    fractions = [c / total * 100 if total > 0 else 0 for c in counts]

    return {
        "ring_sizes": sizes,
        "counts": counts,
        "fractions": fractions,
        "bond_pair": bond_pair,
        "total_rings": total,
    }
