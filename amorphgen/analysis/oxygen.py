"""Oxygen connectivity relative to explicitly selected network formers."""

from __future__ import annotations

from collections import Counter

import numpy as np
from ase.data import atomic_numbers
from ase.neighborlist import neighbor_list

from .cutoff import parse_cutoff_spec, resolve_cutoffs


DEFAULT_NETWORK_FORMERS = ("Al", "B", "Ge", "P", "Si")
OXYGEN_SPECIES = ("free", "non_bridging", "bridging", "tricluster",
                  "higher_coordinated")


def compute_oxygen_speciation(atoms_list, network_formers=None, cutoff="auto-rdf"):
    """Classify O by the number of bonded network-former neighbours.

    Zero, one, two, three and four-or-more neighbours are labelled ``free``,
    ``non_bridging``, ``bridging``, ``tricluster`` and ``higher_coordinated``.
    Fractions are in [0, 1] and pooled by oxygen count across frames.
    Periodic images are distinct neighbours, including in small unit cells.

    ``network_formers`` is an iterable of element symbols (or a comma/plus
    separated string). By default, the Al, B, Ge, P and Si present in the
    ensemble are selected. Other oxides require an explicit selection;
    alkali/alkaline-earth modifiers must not be counted as network formers.
    ``cutoff`` accepts the same forms as :class:`StructureAnalyser`.
    All frames must have the same element set; atom counts may differ.

    This is a geometric connectivity descriptor, not an assignment of
    charge, bond order, or hydroxyl speciation. "Free" means no selected
    network-former neighbour, even if other atoms are nearby.
    """
    atoms_list = list(atoms_list)
    if not atoms_list or any(len(a) == 0 for a in atoms_list):
        raise ValueError("Oxygen speciation requires nonempty structures.")
    element_sets = [set(a.get_chemical_symbols()) for a in atoms_list]
    if any(elements != element_sets[0] for elements in element_sets[1:]):
        raise ValueError("Oxygen speciation requires the same element set in each structure; "
                         "analyse different chemistries separately.")
    elements = element_sets[0]
    inferred = network_formers is None
    if inferred:
        formers = sorted(elements.intersection(DEFAULT_NETWORK_FORMERS))
    else:
        if isinstance(network_formers, str):
            network_formers = [s.strip() for s in
                               network_formers.replace("+", ",").split(",")]
        formers = sorted(set(network_formers))
        if not formers or any(s not in atomic_numbers or s in ("O", "H", "X")
                              for s in formers):
            raise ValueError("network_formers must contain element symbols other than O or H.")
        missing = set(formers) - elements
        if missing:
            raise ValueError(f"Network formers absent from structures: {sorted(missing)}")
    if "O" in elements and not formers:
        raise ValueError("No conventional network formers found; specify network_formers explicitly.")
    if any(not np.isfinite(a.positions).all() or not np.isfinite(a.cell).all()
           for a in atoms_list):
        raise ValueError("Structure positions and cells must be finite.")
    for atoms in atoms_list:
        periodic_vectors = np.asarray(atoms.cell)[atoms.pbc]
        if len(periodic_vectors) and np.linalg.matrix_rank(periodic_vectors) < len(periodic_vectors):
            raise ValueError("Periodic directions require nonzero, independent cell vectors.")

    parsed = parse_cutoff_spec(cutoff)
    # The analyser already supplies resolved pair cutoffs. Do not recompute
    # an unused RDF baseline when every relevant pair is explicitly known.
    if isinstance(parsed, dict) and all(
            f"O-{s}" in parsed or f"{s}-O" in parsed for s in formers):
        resolved = parsed
    else:
        resolved, _ = resolve_cutoffs(atoms_list, parsed)
    pair_cutoffs = {}
    for former in formers:
        value = (resolved if np.isscalar(resolved) else
                 resolved.get(f"O-{former}", resolved.get(f"{former}-O", 0.0)))
        if not np.isfinite(value) or value < 0:
            raise ValueError("Oxygen bond cutoffs must be finite and nonnegative.")
        pair_cutoffs[("O", former)] = float(value)

    def summarise(counts):
        values = {name: 0 for name in OXYGEN_SPECIES}
        for cn in counts:
            values[OXYGEN_SPECIES[min(int(cn), 4)]] += 1
        total = len(counts)
        return {"total_oxygen": total, "counts": values,
                "fractions": {k: v / total if total else 0.0 for k, v in values.items()},
                "coordination_distribution": dict(sorted(Counter(counts).items()))}

    per_structure, all_counts = [], []
    for frame, atoms in enumerate(atoms_list):
        symbols = np.asarray(atoms.get_chemical_symbols())
        oxygen_indices = np.flatnonzero(symbols == "O")
        cn = np.zeros(len(atoms), dtype=int)
        if pair_cutoffs and len(oxygen_indices):
            i = neighbor_list("i", atoms, pair_cutoffs)
            np.add.at(cn, i[symbols[i] == "O"], 1)
        counts = cn[oxygen_indices].tolist()
        per_structure.append({"index": frame, **summarise(counts),
                              "oxygen_indices": oxygen_indices.tolist(),
                              "network_former_coordination": counts})
        all_counts.extend(counts)
    return {**summarise(all_counts), "n_structures": len(atoms_list),
            "network_formers": formers, "network_formers_inferred": inferred,
            "cutoffs": {f"O-{s}": c for (_, s), c in pair_cutoffs.items()},
            "per_structure": per_structure}
