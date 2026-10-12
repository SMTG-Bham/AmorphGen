"""Characterize tutorial diagnostics without executing generation or MD cells."""

import ast
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.neighborlist import neighbor_list


TUTORIALS = Path(__file__).resolve().parents[1] / "Tutorials"
spec = importlib.util.spec_from_file_location(
    "tutorial_analysis_helpers", TUTORIALS / "analysis_helpers.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


@pytest.mark.parametrize("cell", [
    [4, 4, 4], [[4, 0, 0], [1.1, 3.5, 0], [0.2, 0.4, 4.2]],
])
@pytest.mark.parametrize("rmax,nbins", [(1.8, 9), (6.0, 17)])
def test_minimum_image_rdf_retains_pair_normalization_and_grid(cell, rmax, nbins):
    atoms = Atoms("Si2O2", scaled_positions=[
        [.1, .1, .1], [.1, .1, .1], [.8, .1, .1], [.2, .4, .5],
    ], cell=cell, pbc=True)
    distances = [atoms.get_distance(i, j, mic=True)
                 for i, j in itertools.combinations(range(len(atoms)), 2)]
    edges = np.linspace(0, rmax, nbins + 1)
    counts = np.histogram([d for d in distances if d < rmax], edges)[0]
    expected_r = (edges[1:] + edges[:-1]) / 2
    pairs = len(atoms) * (len(atoms) - 1) / 2
    expected_g = counts * atoms.get_volume() / (
        pairs * 4 * np.pi * expected_r**2 * (edges[1] - edges[0]))

    r, g = helpers.minimum_image_rdf(atoms, rmax=rmax, nbins=nbins)

    np.testing.assert_array_equal(r, expected_r)
    np.testing.assert_allclose(g, expected_g)
    assert r.shape == g.shape == (nbins,)


def test_minimum_image_rdf_remains_distinct_from_periodic_image_rdf():
    from amorphgen.analysis.rdf import compute_rdf

    atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 0, 0]], cell=[4] * 3, pbc=True)
    _, small = helpers.minimum_image_rdf(atoms, rmax=1.8, nbins=9)
    canonical = compute_rdf([atoms], rmax=1.8, nbins=9, sigma=0, n_bootstrap=0)
    np.testing.assert_allclose(small, canonical["g_r"])
    _, large = helpers.minimum_image_rdf(atoms, rmax=6, nbins=30)
    with pytest.warns(UserWarning, match="half the shortest cell"):
        canonical = compute_rdf([atoms], rmax=6, nbins=30, sigma=0, n_bootstrap=0)
    assert not np.allclose(large, canonical["g_r"])


@pytest.mark.parametrize("override_masses", [False, True])
def test_density_retains_standard_elemental_masses_and_gcm3(override_masses):
    atoms = Atoms("SiO2", cell=[[4, 0, 0], [1, 5, 0], [0, .2, 6]], pbc=True)
    if override_masses:
        atoms.set_masses([100, 200, 300])
    original_masses = atoms.get_masses().copy()
    expected = (sum(atomic_masses[atomic_numbers[s]] for s in atoms.symbols)
                / 6.022e23 / (atoms.get_volume() * 1e-24))
    assert helpers.get_density(atoms) == pytest.approx(expected, rel=1e-14)
    np.testing.assert_array_equal(atoms.get_masses(), original_masses)


@pytest.mark.parametrize("central,partner", [("Si", "O"), ("Si", "Si"), ("Xe", "O")])
@pytest.mark.parametrize("periodic", [False, True])
def test_coordination_preserves_site_order_cutoff_and_periodic_images(central, partner, periodic):
    atoms = Atoms("OSiOSi", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0]],
                  cell=[[3, 0, 0], [.3, 3, 0], [0, 0, 3]], pbc=periodic)
    cutoff = 3.1
    shifts = itertools.product(range(-2, 3) if periodic else [0], repeat=3)
    shifts = list(shifts)
    expected = []
    for i, symbol in enumerate(atoms.symbols):
        if symbol != central:
            continue
        count = 0
        for j, other in enumerate(atoms.symbols):
            if other != partner:
                continue
            for shift in shifts:
                if i == j and shift == (0, 0, 0):
                    continue
                distance = np.linalg.norm(atoms.positions[j] + np.array(shift) @ atoms.cell
                                          - atoms.positions[i])
                count += distance < cutoff
        expected.append(count)
    actual = helpers.coordination_numbers(atoms, central, partner, cutoff)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype.kind == "i"


@pytest.mark.parametrize("coincident", [False, True])
def test_angles_remain_geometric_degrees_including_coincident_nan(coincident):
    positions = [[0, 0, 0], [0 if coincident else 1, 0, 0], [0, 1, 0], [-1, 0, 0]]
    atoms = Atoms("SiO3", positions=positions, cell=[8] * 3, pbc=True)
    i, j, vectors = neighbor_list("ijD", atoms, 1.5)
    selected = vectors[(i == 0) & (np.array(atoms.get_chemical_symbols())[j] == "O")]
    expected = []
    for a, b in itertools.combinations(selected, 2):
        expected.append(np.nan if not np.linalg.norm(a) * np.linalg.norm(b) else
                        np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))
    with np.errstate(invalid="ignore"):
        actual = helpers.bond_angles(atoms, "Si", "O", 1.5)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert helpers.bond_angles(atoms, "Xe", "O", 1.5).shape == (0,)


@pytest.mark.parametrize("prefix,rmax,nbins", [
    ("T2", None, None), ("T3", 6.0, 100), ("T4", 8.0, 200),
    ("T5", 6.0, 150), ("T6", None, None),
])
def test_notebooks_bind_shared_diagnostics_without_running_simulations(
        prefix, rmax, nbins, monkeypatch):
    notebook = next(TUTORIALS.glob(f"{prefix}_*/*.ipynb"))
    monkeypatch.chdir(notebook.parent)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    nodes = []
    for cell in json.loads(notebook.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.FunctionDef):
                assert node.name not in {"get_density", "coordination_numbers", "bond_angles"}
                if node.name == "compute_rdf" and rmax is not None:
                    nodes.append(node)
            if isinstance(node, ast.Import) and any(a.name == "sys" for a in node.names):
                nodes.append(node)
            elif isinstance(node, ast.ImportFrom) and node.module in {
                    "pathlib", "functools", "analysis_helpers"}:
                nodes.append(node)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                if ast.unparse(node.value.func) == "sys.path.insert":
                    nodes.append(node)
            elif isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "compute_rdf" for t in node.targets):
                nodes.append(node)
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(notebook), "exec"), namespace)
    if rmax is not None:
        atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 0, 0]], cell=[4] * 3, pbc=True)
        r, g = namespace["compute_rdf"](atoms)
        expected = helpers.minimum_image_rdf(atoms, rmax=rmax, nbins=nbins)
        np.testing.assert_array_equal(r, expected[0])
        np.testing.assert_array_equal(g, expected[1])
        override = namespace["compute_rdf"](atoms, 2.0, 10)
        expected_override = helpers.minimum_image_rdf(atoms, rmax=2.0, nbins=10)
        np.testing.assert_array_equal(override, expected_override)
