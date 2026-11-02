"""Voronoi face counts and central-cell selection."""

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from amorphgen.analysis.voronoi import compute_voronoi


@pytest.mark.parametrize("element,expected", [
    (None, [(1, 1, 1, 0), (0, 0, 1, 1), (0, 0, 0, 1)]),
    ("Si", [(1, 1, 1, 0), (0, 0, 0, 1)]),
    ("Cu", []),
])
def test_central_faces_count_both_endpoints_and_exclude_invalid_geometry(
        monkeypatch, element, expected):
    atoms = Atoms("SiOSiOSi", positions=np.zeros((5, 3)), cell=[5, 5, 5], pbc=True)
    # The home cell occupies points 65..69 in the 27-image tessellation.
    point_region = np.zeros(27 * len(atoms), dtype=int)
    point_region[65:70] = np.arange(5)
    ridges = [
        ((65, 0), [0, 1, 2]),            # central first endpoint, triangle
        ((0, 65), [0, 1, 2, 3]),         # central second endpoint, quadrilateral
        ((65, 66), [0, 1, 2, 3, 4]),     # shared by two central cells
        ((66, 100), [0, 1, 2, 3, 4, 5]),
        ((65, 66), [-1, 0, 1]),          # unbounded face
        ((65, 66), list(range(7))),       # outside the n3..n6 index
        ((65, 1), [0, 1]),
        ((67, 2), [0, 1, 2]),            # unbounded region
        ((68, 3), [0, 1, 2, 3]),         # empty region
        ((69, 4), [0, 1, 2, 3, 4, 5]),
        ((3, 4), [0, 1, 2, 3]),         # neither endpoint in the home cell
    ]
    tessellation = SimpleNamespace(
        point_region=point_region,
        regions=[[0, 1, 2], [0, 1, 2], [-1, 0, 1], [], [0, 1, 2]],
        ridge_points=np.array([points for points, _ in ridges]),
        ridge_vertices=[vertices for _, vertices in ridges],
    )
    monkeypatch.setattr("scipy.spatial.Voronoi", lambda positions: tessellation)

    result = compute_voronoi([atoms], element=element)

    assert result["indices"] == expected
    assert result["total_atoms"] == len(expected)
    assert list(result["distribution"]) == expected
    assert result["mean_faces"] == (np.mean([sum(index) for index in expected])
                                    if expected else 0)
    assert result["per_structure"] == [{
        "indices": expected,
        "mean_faces": result["mean_faces"] if expected else None,
        "tessellation_succeeded": True,
    }]


@pytest.mark.parametrize("symbol,crystal,a,index", [
    ("Cu", "fcc", 3.6, (0, 12, 0, 0)),
    ("Fe", "bcc", 2.87, (0, 6, 0, 8)),
])
def test_crystal_voronoi_polyhedra(symbol, crystal, a, index):
    atoms = bulk(symbol, crystal, a=a, cubic=True)

    result = compute_voronoi([atoms])

    assert result["indices"] == [index] * len(atoms)
    assert result["distribution"] == {index: len(atoms)}
    assert result["mean_faces"] == sum(index)
