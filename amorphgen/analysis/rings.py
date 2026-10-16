"""Ring statistics for network glasses."""

from __future__ import annotations

import warnings
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


RING_CRITERIA = ("shortest", "primitive")
_CRITERION_ALIASES = {"guttman": "shortest"}   # the literature name of the shortest-ring-per-edge rule


def _edge_key(a, oa, b, ob):
    """Canonical key of the periodic network edge between the unfolded nodes (a, oa) and (b, ob)."""
    s = (ob[0] - oa[0], ob[1] - oa[1], ob[2] - oa[2])
    return min((a, b, *s), (b, a, -s[0], -s[1], -s[2]))


def _unfolded_bfs(net_adj, src, depth):
    """BFS from (src, home cell) over (atom, offset) states: distances and shortest-path predecessors."""
    start = (src, (0, 0, 0))
    dist = {start: 0}
    preds = defaultdict(list)
    queue = deque([start])
    while queue:
        u = queue.popleft()
        d = dist[u]
        if d >= depth:
            continue
        a, off = u
        for b, sx, sy, sz in net_adj[a]:
            v = (b, (off[0] + sx, off[1] + sy, off[2] + sz))
            if v not in dist:
                dist[v] = d + 1
                queue.append(v)
                preds[v].append(u)
            elif dist[v] == d + 1:
                preds[v].append(u)
    return dist, preds


def _shortest_paths(preds, target, start):
    """Every shortest path start -> target through the predecessor lists."""
    out = []
    stack = [(target, [target])]
    while stack:
        u, path = stack.pop()
        if u == start:
            out.append(path[::-1])
            continue
        for p in preds[u]:
            stack.append((p, path + [p]))
    return out


def _closer_than(net_adj, u, v, bound):
    """True if the unfolded network distance u -> v is below ``bound``."""
    dist = {u: 0}
    queue = deque([u])
    while queue:
        x = queue.popleft()
        d = dist[x]
        if d + 1 >= bound:
            continue
        a, off = x
        for b, sx, sy, sz in net_adj[a]:
            y = (b, (off[0] + sx, off[1] + sy, off[2] + sz))
            if y not in dist:
                if y == v:
                    return True
                dist[y] = d + 1
                queue.append(y)
    return False


def _is_primitive(net_adj, ring):
    """A ring is primitive when no two of its nodes are closer through the network than round it."""
    n = len(ring)
    for i in range(n):
        for j in range(i + 2, n):
            ring_distance = min(j - i, n - (j - i))
            if ring_distance >= 2 and _closer_than(net_adj, ring[i], ring[j], ring_distance):
                return False
    return True


def _canonical_ring(ring):
    """Rotation- and direction-independent key of a ring of unfolded nodes, offsets relative to its first node."""
    n = len(ring)
    best = None
    for seq in (ring, ring[::-1]):
        for k in range(n):
            o0 = seq[k][1]
            key = tuple((seq[(k + t) % n][0],
                         (seq[(k + t) % n][1][0] - o0[0], seq[(k + t) % n][1][1] - o0[1],
                          seq[(k + t) % n][1][2] - o0[2])) for t in range(n))
            if best is None or key < best:
                best = key
    return best


def _primitive_rings(net_adj, p1_indices, max_ring):
    """Unique primitive rings of the periodic network graph, as {canonical key: size}.

    Rings through each node are built from pairs of node-disjoint shortest paths
    (the construction of Yuan and Cormack, Comput. Mater. Sci. 24, 343 (2002)): a
    ring of 2d nodes joins two shortest paths to a node at distance d, a ring of
    2d + 1 nodes two shortest paths to the ends of an edge whose ends are both at
    distance d. A candidate is kept when it has no shortcut (Goetzke and Klein,
    J. Non-Cryst. Solids 127, 215 (1991)). Offsets make a loop that winds round
    the cell no ring.
    """
    rings = {}
    depth = max_ring // 2 + 1
    for s in p1_indices:
        start = (s, (0, 0, 0))
        dist, preds = _unfolded_bfs(net_adj, s, depth)
        by_distance = defaultdict(list)
        for v, d in dist.items():
            by_distance[d].append(v)
        for d in range(1, depth + 1):
            if d >= 2 and 2 * d <= max_ring:          # even rings; d = 1 would be a double edge
                for x in by_distance[d]:
                    paths = _shortest_paths(preds, x, start)
                    for i in range(len(paths)):
                        inner_i = set(paths[i][1:-1])
                        for j in range(i + 1, len(paths)):
                            if inner_i.isdisjoint(paths[j][1:-1]):
                                ring = paths[i][:-1] + paths[j][::-1][:-1]
                                key = _canonical_ring(ring)
                                if key not in rings and _is_primitive(net_adj, ring):
                                    rings[key] = len(ring)
            if 2 * d + 1 <= max_ring:                  # odd rings through an edge at distance d
                for x in by_distance[d]:
                    a, off = x
                    for b, sx, sy, sz in net_adj[a]:
                        y = (b, (off[0] + sx, off[1] + sy, off[2] + sz))
                        if dist.get(y) != d or y <= x:
                            continue
                        paths_x = _shortest_paths(preds, x, start)
                        paths_y = _shortest_paths(preds, y, start)
                        for p1 in paths_x:
                            tail = set(p1[1:])
                            for p2 in paths_y:
                                if tail.isdisjoint(p2[1:]):
                                    ring = p1 + p2[::-1][:-1]
                                    key = _canonical_ring(ring)
                                    if key not in rings and _is_primitive(net_adj, ring):
                                        rings[key] = len(ring)
    return rings


def compute_ring_statistics(atoms_list, bond_pair=None, cutoff=None,
                            max_ring=12, get_cutoff_fn=None, criterion="shortest"):
    """
    Compute ring statistics of the network graph by one of two criteria.

    ``criterion="shortest"`` (default; ``"guttman"`` is accepted as its literature
    name): the shortest-ring distribution observed by
    network edges. For each edge in the network graph (A-B-A path through
    bridging B), find the shortest ring by BFS excluding that edge (Guttman,
    J. Non-Cryst. Solids 116, 145 (1990)). Counts are edge observations, not
    unique cycles: an isolated six-member ring contributes six observations.

    ``criterion="primitive"``: the census of unique primitive rings, rings that
    cannot be shortened by a path through the network (Goetzke and Klein,
    J. Non-Cryst. Solids 127, 215 (1991); Yuan and Cormack, Comput. Mater. Sci.
    24, 343 (2002)), each counted once. ``counts`` are unique rings, ``fractions``
    per cent of rings, ``rings_per_node`` the rings of each size per network
    node (the R_C(n) of the RINGS code, Le Roux and Jund, Comput. Mater. Sci.
    49, 70 (2010)) and ``nodes_in_ring_fraction`` the fraction of nodes on at
    least one ring of that size. The search enumerates shortest-path pairs to
    depth ``max_ring // 2 + 1``, which grows quickly with the node degree: a
    400-atom close-packed oxide takes seconds at ``max_ring=8`` and much longer
    at 12, a 648-atom silica network seconds at 12. In a close-packed oxide
    the census is dominated by 3-rings whatever the preparation route (every
    anion bridging three cations is a triangle), so it discriminates for
    network glasses only.

    Ring sizes count A nodes, not the intervening B atoms. For a same-element
    pair, the network uses direct bonds instead.

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
    ``bond_pair`` is ``(nodes, bridges)``; each may be one element or a
    ``/``-joined group, ``("Ga/In/Zn", "O")``. Auto-detection (``None``):
    with three or more elements the nodes are every element that is not an
    anion and the bridges the anion(s), by the charge-balance rule of
    :func:`amorphgen.utils.radii.anion_elements` (the rule the coordination
    and connectivity reports use for a bond); with two elements the less
    electronegative one is the node. Taking one cation as the node (old
    behaviour, the least electronegative) traced a Zn-O-Zn sub-network in
    a-InGaZnO4 and reported its triangles as the ring statistics of the
    oxide; Ga and In nodes were not in the graph. Ring sizes count nodes of
    any species. Every non-anion element is a node, modifiers included, so
    for a modified glass (Na-Al-Si-O) name the formers: ``("Si/Al", "O")``. ``cutoff`` accepts a positive scalar or ASE pair-cutoff
    mapping in angstrom; with ``get_cutoff_fn`` and no ``cutoff``, every
    node-bridge pair gets its own cutoff. Mapping metadata uses canonical
    ``"A-B"`` string keys for JSON.
    """
    if (isinstance(max_ring, (bool, np.bool_))
            or not isinstance(max_ring, Integral) or max_ring < 3):
        raise ValueError("max_ring must be an integer of at least 3")
    max_ring = int(max_ring)
    criterion = _CRITERION_ALIASES.get(criterion, criterion)
    if criterion not in RING_CRITERIA:
        raise ValueError(f"criterion must be one of {', '.join(RING_CRITERIA)}, not {criterion!r}")
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
                from ..utils.radii import PAULING_EN, anion_elements
            except ImportError:
                from amorphgen.utils.radii import PAULING_EN, anion_elements
            counts = Counter()
            for atoms in atoms_list:
                counts.update(atoms.get_chemical_symbols())
            anions = anion_elements(counts) if len(unique) > 2 else set()
            if anions and anions != set(unique):
                bond_pair = ("/".join(s for s in unique if s not in anions),
                             "/".join(s for s in unique if s in anions))
            else:
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
                not isinstance(group, str) or not group
                or any(symbol not in atomic_numbers for symbol in group.split("/"))
                for group in bond_pair):
            raise ValueError("bond_pair must contain two valid element symbols "
                             "(or /-joined groups such as Ga/In/Zn)")

    p1, p2 = bond_pair
    nodes, bridges = frozenset(p1.split("/")), frozenset(p2.split("/"))
    if p1 != p2 and nodes & bridges:
        raise ValueError("bond_pair nodes and bridges must not share an element")
    if cutoff is not None:
        cut = cutoff
    elif get_cutoff_fn is None:
        cut = 2.5
    elif len(nodes) == 1 and len(bridges) == 1:
        cut = get_cutoff_fn(p1, p2)
    else:                                   # one cutoff per node-bridge pair
        cut = {(a, b): get_cutoff_fn(a, b) for a in sorted(nodes) for b in sorted(bridges)}
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
            if (si in nodes and sj in bridges) or (si in bridges and sj in nodes):
                adj[i].add((j, *s))

        # Build network graph: node-node edges through a bridging atom, or
        # the direct bonds for a single-element network (a-Si, a-Ge, C).
        # Each edge is (b, sx, sy, sz): atom b in the cell at that offset.
        p1_indices = [i for i in range(n) if syms[i] in nodes]
        net_adj = defaultdict(set)
        if p1 == p2:
            for a in p1_indices:
                net_adj[a] = set(adj[a])
        else:
            for a in p1_indices:
                for bridge, *s1 in adj[a]:
                    if syms[bridge] in bridges:
                        for b, *s2 in adj[bridge]:
                            s = (s1[0] + s2[0], s1[1] + s2[1], s1[2] + s2[2])
                            # a -> bridge -> the same a is no edge; a -> bridge
                            # -> another image of a is one, in a small cell
                            if syms[b] in nodes and (b != a or s != (0, 0, 0)):
                                net_adj[a].add((b, *s))

        counted_edges = set()
        for a in p1_indices:
            for b, sx, sy, sz in net_adj[a]:
                counted_edges.add(min((a, b, sx, sy, sz), (b, a, -sx, -sy, -sz)))
        n_edges = len(counted_edges)
        if criterion == "primitive":
            # Unique primitive rings; an edge is resolved when it lies on one
            degree = 2 * n_edges / len(p1_indices) if p1_indices else 0.0
            if index == 0 and max_ring > 8 and degree > 5:
                warnings.warn(
                    f"primitive-ring search to max_ring={max_ring} on a network with "
                    f"{degree:.1f} links per node (a close-packed oxide) enumerates shortest-path "
                    "pairs exponentially in the depth and can take minutes per structure; "
                    "max_ring=8 finds the same rings in seconds there", stacklevel=2)
            found_rings = _primitive_rings(net_adj, p1_indices, max_ring)
            ring_edges = set()
            members = defaultdict(set)
            for key, size in found_rings.items():
                frame_counts[size] += 1
                ring_counts[size] += 1
                for t in range(size):
                    (a, oa), (b, ob) = key[t], key[(t + 1) % size]
                    ring_edges.add(_edge_key(a, oa, b, ob))
                    members[size].add(a)
            n_ring_edges = len(ring_edges)
            n_nodes = len(p1_indices)
            per_structure.append({
                "index": index, "counts": dict(sorted(frame_counts.items())),
                "total_rings": sum(frame_counts.values()),
                "n_network_nodes": n_nodes, "n_network_edges": n_edges,
                "n_ring_edges": n_ring_edges,
                "n_unresolved_edges": n_edges - n_ring_edges,
                "ring_edge_fraction": n_ring_edges / n_edges if n_edges else None,
                "rings_per_node": {size: count / n_nodes for size, count in sorted(frame_counts.items())},
                "nodes_in_ring_fraction": {size: len(members[size]) / n_nodes
                                           for size in sorted(frame_counts)},
                **_ring_size_summary(frame_counts),
            })
            continue

        # shortest ring per edge (Guttman): for each edge, find its shortest ring
        for a in p1_indices:
            for edge in net_adj[a]:
                b, sx, sy, sz = edge
                key = min((a, b, sx, sy, sz), (b, a, -sx, -sy, -sz))
                if key not in counted_edges:
                    continue
                counted_edges.discard(key)

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

    n_nodes = sum(frame["n_network_nodes"] for frame in per_structure)
    n_ring_edges = sum(frame["n_ring_edges"] for frame in per_structure)
    if criterion == "primitive":
        definitions = {
            "method": "primitive_rings",
            "count_definition": (
                "unique primitive rings (no two ring nodes closer through the network than "
                "round the ring), each counted once; counts and total_rings are rings, "
                "rings_per_node is rings of that size per network node"),
            "ring_size_definition": (
                "number of network-former nodes (any node species) in the ring; "
                "bridging atoms are omitted for unlike-element bond pairs"),
            "unresolved_edge_definition": (
                "network edge on no primitive ring of size <= max_ring; "
                "may belong to a larger ring and is not necessarily acyclic"),
            "fraction_of_structures_definition": (
                "structures with at least one primitive ring of this size"),
        }
        extra = {
            "rings_per_node": [ring_counts[s] / n_nodes if n_nodes else None for s in sizes],
            "nodes_in_ring_fraction": [
                sum(r["nodes_in_ring_fraction"].get(s, 0.0) * r["n_network_nodes"] for r in per_structure)
                / n_nodes if n_nodes else None for s in sizes],
        }
        extra_uncertainty = {
            "rings_per_node": {size: summarize_structures([
                r["rings_per_node"].get(size, 0.0) for r in per_structure]) for size in sizes},
            "nodes_in_ring_fraction": {size: summarize_structures([
                r["nodes_in_ring_fraction"].get(size, 0.0) for r in per_structure]) for size in sizes},
        }
    else:
        definitions = {
            "method": "shortest_cycle_per_network_edge",
            "count_definition": (
                "one shortest-cycle observation per undirected periodic network edge; "
                "counts and total_rings count edge observations, not unique cycles"),
            "ring_size_definition": (
                "number of network-former nodes (any node species) in the shortest cycle; "
                "bridging atoms are omitted for unlike-element bond pairs"),
            "unresolved_edge_definition": (
                "network edge without a closed cycle of size <= max_ring; "
                "may belong to a larger ring and is not necessarily acyclic"),
            "fraction_of_structures_definition": (
                "structures with at least one network edge whose shortest ring has this size"),
        }
        extra, extra_uncertainty = {}, {}

    return {
        **definitions,
        "criterion": criterion,
        "cutoff": cutoff_metadata, "max_ring": max_ring, "n_structures": len(atoms_list),
        "ring_sizes": sizes,
        "counts": counts,
        "fractions": fractions,
        **extra,
        "bond_pair": bond_pair,
        "nodes": sorted(nodes), "bridges": sorted(bridges),
        "total_rings": total,
        "n_network_nodes": n_nodes,
        "n_network_edges": n_edges, "n_ring_edges": n_ring_edges,
        "n_unresolved_edges": n_edges - n_ring_edges,
        "ring_edge_fraction": n_ring_edges / n_edges if n_edges else None,
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
            **extra_uncertainty,
        },
        "fraction_of_structures": {
            size: sum(r["counts"].get(size, 0) > 0 for r in per_structure) / len(per_structure)
            for size in sizes},
    }
