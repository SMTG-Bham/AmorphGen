"""Voronoi tessellation analysis for coordination polyhedra."""

from __future__ import annotations

import numpy as np
from collections import Counter


def compute_voronoi(atoms_list, element=None):
    """
    Voronoi tessellation analysis.

    Computes Voronoi indices (n3, n4, n5, n6) for each atom.
    """
    from scipy.spatial import Voronoi as SciVoronoi

    all_indices = []

    for atoms in atoms_list:
        syms = atoms.get_chemical_symbols()
        n = len(atoms)
        pos = atoms.get_positions()
        cell = np.array(atoms.get_cell())

        # Create supercell images for PBC
        images = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    shift = dx * cell[0] + dy * cell[1] + dz * cell[2]
                    images.append(pos + shift)
        all_pos = np.vstack(images)

        try:
            vor = SciVoronoi(all_pos)
        except Exception as exc:
            import warnings
            warnings.warn(f"Voronoi tessellation failed for structure "
                          f"({len(atoms)} atoms): {exc}")
            continue

        central_start = 13 * n
        central_end = 14 * n

        for atom_idx in range(central_start, central_end):
            real_idx = atom_idx - central_start
            if element is not None and syms[real_idx] != element:
                continue

            region_idx = vor.point_region[atom_idx]
            region = vor.regions[region_idx]

            if -1 in region or len(region) == 0:
                continue

            n_faces = {3: 0, 4: 0, 5: 0, 6: 0}
            for ridge_idx, (p1, p2) in enumerate(vor.ridge_points):
                if p1 == atom_idx or p2 == atom_idx:
                    face_verts = vor.ridge_vertices[ridge_idx]
                    if -1 not in face_verts:
                        nv = len(face_verts)
                        if nv in n_faces:
                            n_faces[nv] += 1

            idx_tuple = (n_faces[3], n_faces[4], n_faces[5], n_faces[6])
            all_indices.append(idx_tuple)

    distribution = Counter(all_indices)
    total = len(all_indices)
    mean_faces = np.mean([sum(idx) for idx in all_indices]) if all_indices else 0

    top = distribution.most_common(10)
    top_formatted = [(idx, count, count / total * 100) for idx, count in top]

    return {
        "indices": all_indices,
        "distribution": dict(distribution),
        "top_10": top_formatted,
        "mean_faces": float(mean_faces),
        "total_atoms": total,
    }
