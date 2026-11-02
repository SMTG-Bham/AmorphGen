"""Voronoi tessellation analysis for coordination polyhedra."""

from __future__ import annotations

import numpy as np
from collections import Counter
from .uncertainty import summarize_structures


def compute_voronoi(atoms_list, element=None):
    """
    Voronoi tessellation analysis.

    Computes Voronoi indices (n3, n4, n5, n6) for each atom.
    """
    from scipy.spatial import Voronoi as SciVoronoi

    all_indices = []
    per_structure = []

    for atoms in atoms_list:
        frame_indices = []
        per_structure.append({"indices": frame_indices, "mean_faces": None,
                              "tessellation_succeeded": False})
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
        per_structure[-1]["tessellation_succeeded"] = True

        # Each ridge is a face shared by its two endpoints. Accumulate the
        # central cells' faces in one pass instead of scanning every ridge
        # again for each atom.
        face_counts = [[0, 0, 0, 0] for _ in range(n)]
        for (p1, p2), face_verts in zip(vor.ridge_points, vor.ridge_vertices):
            first_central = central_start <= p1 < central_end
            second_central = central_start <= p2 < central_end
            if not (first_central or second_central):
                continue
            nv = len(face_verts)
            if nv < 3 or nv > 6 or -1 in face_verts:
                continue
            if first_central:
                face_counts[p1 - central_start][nv - 3] += 1
            if second_central and p2 != p1:
                face_counts[p2 - central_start][nv - 3] += 1

        for atom_idx in range(central_start, central_end):
            real_idx = atom_idx - central_start
            if element is not None and syms[real_idx] != element:
                continue

            region_idx = vor.point_region[atom_idx]
            region = vor.regions[region_idx]

            if -1 in region or len(region) == 0:
                continue

            idx_tuple = tuple(face_counts[real_idx])
            all_indices.append(idx_tuple)
            frame_indices.append(idx_tuple)
        if frame_indices:
            per_structure[-1]["mean_faces"] = float(np.mean([sum(i) for i in frame_indices]))

    distribution = Counter(all_indices)
    total = len(all_indices)
    mean_faces = np.mean([sum(idx) for idx in all_indices]) if all_indices else 0

    top = distribution.most_common(10)
    top_formatted = [(idx, count, count / total * 100) for idx, count in top]

    prevalence = {idx: summarize_structures([
        float(idx in frame["indices"]) if frame["tessellation_succeeded"] else None
        for frame in per_structure]) for idx in distribution}
    return {
        "indices": all_indices,
        "distribution": dict(distribution),
        "top_10": top_formatted,
        "mean_faces": float(mean_faces),
        "total_atoms": total,
        "per_structure": per_structure,
        "fraction_of_sites": {idx: count / total for idx, count in distribution.items()},
        "fraction_of_structures": {idx: summary["mean"] for idx, summary in prevalence.items()},
        "fraction_of_structures_definition": "successfully tessellated structures with at least one such Voronoi site",
        "uncertainty": {
            "mean_faces": summarize_structures([frame["mean_faces"] for frame in per_structure]),
            "fraction_of_sites": {idx: summarize_structures([
                frame["indices"].count(idx) / len(frame["indices"]) if frame["indices"] else None
                for frame in per_structure]) for idx in distribution},
            "fraction_of_structures": prevalence,
        },
    }
