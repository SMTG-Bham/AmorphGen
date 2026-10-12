"""Raw, directed-pair RDF histograms shared by scattering and shell detection."""

import numpy as np
from ase.neighborlist import neighbor_list


def raw_frame_rdfs(atoms, pairs, rmax, shell_volumes):
    """Normalize requested RDFs using one neighbour search for a structure.

    ``pairs`` contains A-B labels, or None for the all-atom RDF. Shell volumes
    are 4*pi*r**2*dr on the caller's common grid. Only distances strictly inside
    (0, rmax) count; periodic images and both homonuclear directions are kept.
    Missing species and same-species singletons return None, leaving callers
    to distinguish missing observations from zero contributions to a cutoff
    estimate. An observable pair without neighbours returns a zero curve.
    """
    source, target, distances = neighbor_list("ijd", atoms, cutoff=rmax)
    symbols = np.asarray(atoms.get_chemical_symbols())
    volume = atoms.get_volume()
    nbins = len(shell_volumes)
    curves = {}
    for pair in pairs:
        if pair is None:
            n_source = len(atoms)
            n_target = n_source - 1
            selected = distances
        else:
            first, second = pair.split("-")
            n_source = int(np.sum(symbols == first))
            n_target = int(np.sum(symbols == second)) - int(first == second)
            selected = distances[(symbols[source] == first) & (symbols[target] == second)]
        if n_source == 0 or n_target <= 0:
            curves[pair] = None
            continue
        inside = (selected > 0) & (selected < rmax)
        counts, _ = np.histogram(selected[inside], bins=nbins, range=(0, rmax))
        curves[pair] = counts / (n_source * (n_target / volume) * shell_volumes)
    return curves
