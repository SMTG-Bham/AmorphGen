"""Ring statistics for network glasses."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols
from ase.neighborlist import neighbor_list

from .uncertainty import summarize_structures


def _ring_size_summary(counts):
    """Population statistics of shortest-ring observations, weighted by edges."""
    total = sum(counts.values())
    if not total:
        return dict.fromkeys(("mean_ring_size", "std_ring_size",
                              "min_ring_size", "max_ring_size"))
    mean = sum(size * count for size, count in counts.items()) / total
    variance = sum(count * (size - mean)**2 for size, count in counts.items()) / total
    return {
        "mean_ring_size": float(mean), "std_ring_size": float(np.sqrt(variance)),
        "min_ring_size": min(counts), "max_ring_size": max(counts),
    }


def _validate_cutoff(cutoff):
    """Keep ASE's scalar and pair-mapping cutoff conventions, with metadata."""
    def positive(value):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value <= 0):
            raise ValueError("cutoff must be a finite positive number")
        return float(value)

    if not isinstance(cutoff, Mapping):
        value = positive(cutoff)
        return value, value
    if not cutoff:
        raise ValueError("cutoff mapping must contain at least one pair")
    validated, metadata = {}, {}
    for pair, value in cutoff.items():
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("cutoff mapping keys must be pairs of element symbols or atomic numbers")
        symbols = []
        for item in pair:
            if isinstance(item, str) and item in atomic_numbers:
                symbols.append(item)
            elif (isinstance(item, Integral) and not isinstance(item, (bool, np.bool_))
                  and 0 <= item < len(chemical_symbols)):
                symbols.append(chemical_symbols[item])
            else:
                raise ValueError("cutoff mapping keys must be valid element pairs")
        value = positive(value)
        validated[pair] = value
        # JSON cannot encode ASE's tuple mapping keys. Symmetric entries are
        # resolved in input order, just as ASE applies its pair cutoffs.
        metadata["-".join(sorted(symbols))] = value
    return validated, metadata


def compute_ring_statistics(atoms_list, bond_pair=None, cutoff=None,
                            max_ring=12, get_cutoff_fn=None):
    """
    Compute the shortest-ring distribution observed by network edges.

    For each edge in the network graph (A-B-A path through bridging B),
    find the shortest ring by BFS excluding that edge. Counts are edge
    observations, not unique cycles: an isolated six-member ring contributes
    six observations. Ring sizes count A nodes, not the intervening B atoms.
    For a same-element pair, the network uses direct bonds instead.

    The graph is periodic: every bond carries the cell offset of its partner,
    and a path closes a ring only when it comes back to the image it started
    from. A path that comes back to another periodic image of its start has
    crossed the cell; counting it as a ring (old behaviour) gave an 8-atom
    diamond cell 4-rings instead of 6-rings. Nonperiodic structures and partial
    periodicity are also supported; a periodic axis requires a cell vector.

    ``max_ring`` truncates the search. An unresolved edge has no closure up
    to that size and may belong to a larger ring; it is not necessarily
    acyclic. ``total_rings`` retains its legacy meaning and equals
    ``n_ring_edges``. Top-level graph counts are summed over structures, and
    size means/spreads are pooled over resolved edges. ``ring_edge_fraction``
    is on the 0..1 scale, whereas legacy ``fractions`` are percentages.
    No-edge fractions and no-ring size summaries are None. ``uncertainty``
    summarizes equal-weight structure values, with structures as sampling
    units rather than their correlated edges. The size standard deviation
    is descriptive (population standard deviation), not a standard error.
    ``cutoff`` accepts a positive scalar or ASE pair-cutoff mapping in
    angstrom. Mapping metadata uses canonical ``"A-B"`` string keys for JSON.
    """
    if (isinstance(max_ring, (bool, np.bool_))
            or not isinstance(max_ring, Integral) or max_ring < 3):
        raise ValueError("max_ring must be an integer of at least 3")
    max_ring = int(max_ring)
    try:
        atoms_list = list(atoms_list)
    except TypeError as exc:
        raise ValueError("atoms_list must be an iterable of structures") from exc
    if not atoms_list:
        raise ValueError("atoms_list must contain at least one structure")
    all_symbols = set()
    for index, atoms in enumerate(atoms_list):
        if not isinstance(atoms, Atoms):
            raise ValueError(f"Structure {index} must be an ASE Atoms object")
        if not len(atoms):
            raise ValueError(f"Structure {index} contains no atoms")
        if not np.all(np.isfinite(atoms.positions)):
            raise ValueError(f"Structure {index} has non-finite atom positions")
        cell = np.asarray(atoms.cell, dtype=float)
        if not np.all(np.isfinite(cell)):
            raise ValueError(f"Structure {index} has a non-finite cell")
        nonzero = np.any(cell != 0, axis=1)
        if np.any(atoms.pbc & ~nonzero):
            raise ValueError(f"Structure {index} requires a cell vector for each periodic axis")
        if np.any(nonzero) and np.linalg.matrix_rank(cell[nonzero]) < nonzero.sum():
            raise ValueError(f"Structure {index} requires linearly independent cell vectors")
        all_symbols.update(atoms.get_chemical_symbols())

    # Auto-detect bond pair: ring NODES are the least electronegative
    # element (network former / cation, e.g. Si in SiO2) and the BRIDGES
    # are the most electronegative one (anion).  Sorting symbols
    # alphabetically (old behaviour) picked O as the node for SiO2 and
    # reported 3-rings for cristobalite instead of 6-rings.
    if bond_pair is None:
        unique = sorted(all_symbols)
        if len(unique) == 1:
            bond_pair = (unique[0], unique[0])
        else:
            try:
                from ..utils.radii import PAULING_EN
            except ImportError:
                from amorphgen.utils.radii import PAULING_EN
            # A symbol tie-break makes both the result and equal-EN cases
            # independent of frame order, while keeping distinct endpoints.
            en = lambda sym: (PAULING_EN.get(sym, 2.0), sym)
            bond_pair = (min(unique, key=en), max(unique, key=en))
    else:
        if isinstance(bond_pair, (str, bytes)):
            raise ValueError("bond_pair must contain two element symbols")
        try:
            bond_pair = tuple(bond_pair)
        except TypeError as exc:
            raise ValueError("bond_pair must contain two element symbols") from exc
        if len(bond_pair) != 2 or any(
                not isinstance(symbol, str) or symbol not in atomic_numbers
                for symbol in bond_pair):
            raise ValueError("bond_pair must contain two valid element symbols")

    p1, p2 = bond_pair
    cut = (cutoff if cutoff is not None else
           get_cutoff_fn(p1, p2) if get_cutoff_fn is not None else 2.5)
    cut, cutoff_metadata = _validate_cutoff(cut)

    ring_counts = Counter()
    per_structure = []

    for index, atoms in enumerate(atoms_list):
        frame_counts = Counter()
        syms = atoms.get_chemical_symbols()
        n = len(atoms)
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
        n_edges = len(counted_edges)
        n_ring_edges = sum(frame_counts.values())
        per_structure.append({
            "index": index, "counts": dict(sorted(frame_counts.items())),
            "total_rings": n_ring_edges,
            "n_network_nodes": len(p1_indices), "n_network_edges": n_edges,
            "n_ring_edges": n_ring_edges,
            "n_unresolved_edges": n_edges - n_ring_edges,
            "ring_edge_fraction": n_ring_edges / n_edges if n_edges else None,
            **_ring_size_summary(frame_counts),
        })

    total = sum(ring_counts.values())
    sizes = sorted(ring_counts.keys())
    counts = [ring_counts[s] for s in sizes]
    fractions = [c / total * 100 if total > 0 else 0 for c in counts]
    n_edges = sum(frame["n_network_edges"] for frame in per_structure)
    scalar_metrics = ("total_rings", "mean_ring_size", "std_ring_size",
                      "min_ring_size", "max_ring_size", "n_network_nodes",
                      "n_network_edges", "n_ring_edges", "n_unresolved_edges",
                      "ring_edge_fraction")

    return {
        "method": "shortest_cycle_per_network_edge",
        "count_definition": (
            "one shortest-cycle observation per undirected periodic network edge; "
            "counts and total_rings count edge observations, not unique cycles"),
        "ring_size_definition": (
            "number of network-former nodes in the shortest cycle; bridging atoms "
            "are omitted for unlike-element bond pairs"),
        "unresolved_edge_definition": (
            "network edge without a closed cycle of size <= max_ring; "
            "may belong to a larger ring and is not necessarily acyclic"),
        "cutoff": cutoff_metadata, "max_ring": max_ring, "n_structures": len(atoms_list),
        "ring_sizes": sizes,
        "counts": counts,
        "fractions": fractions,
        "bond_pair": bond_pair,
        "total_rings": total,
        "n_network_nodes": sum(frame["n_network_nodes"] for frame in per_structure),
        "n_network_edges": n_edges, "n_ring_edges": total,
        "n_unresolved_edges": n_edges - total,
        "ring_edge_fraction": total / n_edges if n_edges else None,
        **_ring_size_summary(ring_counts),
        "per_structure": per_structure,
        "uncertainty": {
            **{key: summarize_structures([r[key] for r in per_structure])
               for key in scalar_metrics},
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
        "fraction_of_structures_definition": (
            "structures with at least one network edge whose shortest ring has this size"),
    }
