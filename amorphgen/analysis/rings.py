"""Ring statistics for network glasses."""

from __future__ import annotations

import numpy as np
from collections import Counter, defaultdict, deque
from ase.neighborlist import neighbor_list
from .uncertainty import summarize_structures


def compute_ring_statistics(atoms_list, bond_pair=None, cutoff=None,
                            max_ring=12, get_cutoff_fn=None):
    """
    Compute ring size distribution for network glasses.

    For each edge in the network graph (A-B-A path through bridging B),
    find the shortest ring by BFS excluding that edge.

    The graph is periodic: every bond carries the cell offset of its partner,
    and a path closes a ring only when it comes back to the image it started
    from. A path that comes back to another periodic image of its start has
    crossed the cell; counting it as a ring (old behaviour) gave an 8-atom
    diamond cell 4-rings instead of 6-rings.
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
    per_structure = []

    for atoms in atoms_list:
        frame_counts = Counter()
        syms = atoms.get_chemical_symbols()
        n = len(atoms)
        p1, p2 = bond_pair

        if cutoff is None and get_cutoff_fn is not None:
            cut = get_cutoff_fn(p1, p2)
        elif cutoff is not None:
            cut = cutoff
        else:
            cut = 2.5  # fallback

        # Build adjacency for p1-p2 bonds: (partner, cell offset of partner)
        idx_i, idx_j, shifts = neighbor_list('ijS', atoms, cutoff=cut)
        adj = defaultdict(set)
        for i, j, s in zip(idx_i.tolist(), idx_j.tolist(), shifts.tolist()):
            si, sj = syms[i], syms[j]
            if (si == p1 and sj == p2) or (si == p2 and sj == p1):
                adj[i].add((j, *s))

        # Build network graph: p1-p1 edges through bridging p2, or the
        # direct p1-p1 bonds for a single-element network (a-Si, a-Ge, C).
        # Each edge is (b, sx, sy, sz): atom b in the cell at that offset.
        p1_indices = [i for i in range(n) if syms[i] == p1]
        net_adj = defaultdict(set)
        if p1 == p2:
            for a in p1_indices:
                net_adj[a] = set(adj[a])
        else:
            for a in p1_indices:
                for bridge, *s1 in adj[a]:
                    if syms[bridge] == p2:
                        for b, *s2 in adj[bridge]:
                            s = (s1[0] + s2[0], s1[1] + s2[1], s1[2] + s2[2])
                            # a -> bridge -> the same a is no edge; a -> bridge
                            # -> another image of a is one, in a small cell
                            if syms[b] == p1 and (b != a or s != (0, 0, 0)):
                                net_adj[a].add((b, *s))

        # For each edge, find shortest ring
        counted_edges = set()
        for a in p1_indices:
            for edge in net_adj[a]:
                b, sx, sy, sz = edge
                key = min((a, b, sx, sy, sz), (b, a, -sx, -sy, -sz))
                if key in counted_edges:
                    continue
                counted_edges.add(key)

                # BFS over (atom, cell offset) from a in the home cell to the
                # image of b this edge bonds to, without the edge itself
                start = (a, 0, 0, 0)
                dist = {start: 0}
                queue = deque([start])
                found = None

                while queue:
                    cur = queue.popleft()
                    d = dist[cur]
                    c, cx, cy, cz = cur
                    for nb, nx, ny, nz in net_adj[c]:
                        state = (nb, cx + nx, cy + ny, cz + nz)
                        if state == edge and cur == start:
                            continue
                        if state == edge:
                            found = d + 2
                            break
                        # a state at depth d + 1 closes rings of d + 3 atoms
                        if state not in dist and d + 3 <= max_ring:
                            dist[state] = d + 1
                            queue.append(state)
                    if found:
                        break

                if found and found <= max_ring:
                    ring_counts[found] += 1
                    frame_counts[found] += 1
        per_structure.append({"counts": dict(frame_counts),
                              "total_rings": sum(frame_counts.values())})

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
        "per_structure": per_structure,
        "uncertainty": {
            "total_rings": summarize_structures([r["total_rings"] for r in per_structure]),
            "counts": {size: summarize_structures([
                r["counts"].get(size, 0) for r in per_structure]) for size in sizes},
            "fractions": {size: summarize_structures([
                r["counts"].get(size, 0) / r["total_rings"] if r["total_rings"] else None
                for r in per_structure]) for size in sizes},
            "fraction_of_structures": {size: summarize_structures([
                float(r["counts"].get(size, 0) > 0) for r in per_structure]) for size in sizes},
        },
        "fraction_of_structures": {
            size: sum(r["counts"].get(size, 0) > 0 for r in per_structure) / len(per_structure)
            for size in sizes},
        "fraction_of_structures_definition": "structures with at least one ring of this size",
    }
