"""Shared numerical diagnostics for the executable tutorial notebooks.

Run each notebook from its own tutorial directory and import this module from
the parent ``Tutorials`` directory. These teaching diagnostics deliberately
retain the historical minimum-image RDF and geometric-angle conventions;
the production analyser includes periodic images in RDFs and skips
zero-length bond vectors when calculating angles.
"""

import numpy as np
from ase.neighborlist import neighbor_list

from amorphgen.analysis.structure import build_neighbour_dict
from amorphgen.utils.common import compute_density_gcm3


def get_density(atoms):
    """Density in g/cm³ using standard elemental masses, as in the tutorials."""
    standard_masses = atoms.copy()
    standard_masses.set_masses(None)
    return compute_density_gcm3(standard_masses)


def minimum_image_rdf(atoms, rmax=6.0, nbins=150):
    """Unbroadened total RDF using each distinct atom pair's minimum image.

    Distances are in Å and g(r) is dimensionless. Normalization uses
    N(N−1)/2 pairs and midpoint shell volumes, preserving the original
    tutorials even when rmax exceeds half a cell length. Such a cutoff does
    not include additional periodic images as the production RDF does.
    """
    distances = atoms.get_all_distances(mic=True)
    pairs = distances[np.triu_indices_from(distances, k=1)]
    pairs = pairs[pairs < rmax]
    hist, edges = np.histogram(pairs, bins=nbins, range=(0, rmax))
    r = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]
    n = len(atoms)
    n_pairs = n * (n - 1) / 2
    shell_volume = 4 * np.pi * r**2 * dr
    return r, hist * atoms.get_volume() / (n_pairs * shell_volume)


def coordination_numbers(atoms, central, neighbour, cutoff):
    """Per-site integer CN, in central-atom order, including periodic images."""
    neighbours, symbols = build_neighbour_dict(atoms, cutoff, lambda *_: cutoff)
    return np.array([
        sum(partner == neighbour for _, partner, _, _ in neighbours[index])
        for index, symbol in enumerate(symbols) if symbol == central
    ], dtype=int)


def bond_angles(atoms, central, neighbour, cutoff):
    """Geometric neighbour-central-neighbour angles in degrees.

    Unlike the production bonded-angle analysis, this does not classify
    chemical bonds or discard zero-length vectors; coincident neighbours
    retain the historical NaN result.
    """
    indices, partners, vectors = neighbor_list("ijD", atoms, cutoff)
    symbols = np.array(atoms.get_chemical_symbols())
    angles = []
    for index in np.flatnonzero(symbols == central):
        selected = vectors[(indices == index) & (symbols[partners] == neighbour)]
        for a in range(len(selected)):
            for b in range(a + 1, len(selected)):
                cosine = np.dot(selected[a], selected[b]) / (
                    np.linalg.norm(selected[a]) * np.linalg.norm(selected[b]))
                angles.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    return np.array(angles)
