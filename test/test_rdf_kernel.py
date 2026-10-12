"""Independent count/normalization checks for the shared raw RDF kernel."""

from contextlib import nullcontext

import numpy as np
import pytest
from ase import Atoms

from amorphgen.analysis import cutoff
from amorphgen.analysis.rdf import compute_rdf


def test_partial_histogram_excludes_overlap_and_rmax_but_keeps_bin_boundaries():
    # Four O targets: overlap (excluded), r=1 and r=2 (right-hand bins),
    # and r=rmax (excluded). All four still enter the target number density.
    atoms = Atoms("SiO4", positions=[[0, 0, 0], [0, 0, 0], [1, 0, 0],
                                    [2, 0, 0], [3, 0, 0]], cell=[10] * 3)
    result = compute_rdf([atoms], pair="Si-O", rmax=3, nbins=3,
                         sigma=0, n_bootstrap=0)
    radii = np.array([0.5, 1.5, 2.5])
    expected = np.array([0, 1, 1]) / (4 / atoms.get_volume() * 4 * np.pi * radii**2)
    np.testing.assert_array_equal(result["r"], radii)
    np.testing.assert_allclose(result["g_r"], expected)


def test_homonuclear_histogram_counts_both_directions_with_n_minus_one_density():
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 0, 0]], cell=[10] * 3)
    result = compute_rdf([atoms], pair="Si-Si", rmax=3, nbins=3,
                         sigma=0, n_bootstrap=0)
    # Two directed contacts / (two sources * one available target / volume).
    expected = [0, atoms.get_volume() / (4 * np.pi * 1.5**2), 0]
    np.testing.assert_allclose(result["g_r"], expected)
    total = compute_rdf([atoms], rmax=3, nbins=3, sigma=0, n_bootstrap=0)
    assert total["g_r"] == result["g_r"]


def test_raw_rdf_normalizes_each_frame_by_its_own_volume():
    small = Atoms("SiO", positions=[[0, 0, 0], [1, 0, 0]], cell=[10] * 3)
    large = small.copy()
    large.set_cell([20] * 3)
    result = compute_rdf([small, large], pair="Si-O", rmax=3, nbins=3,
                         sigma=0, n_bootstrap=0)
    curves = np.asarray(result["per_structure"])
    np.testing.assert_allclose(curves[1], 8 * curves[0])
    np.testing.assert_allclose(result["g_r"], (curves[0] + curves[1]) / 2)


@pytest.mark.parametrize("symbols,pair", [("Si", "Si-Si"), ("O2", "Si-O")])
def test_unobservable_pair_is_missing_for_rdf(symbols, pair):
    # The singleton's periodic self-images lie inside rmax, but N-1 is zero.
    if symbols == "Si":
        atoms = Atoms("Si", cell=[2] * 3, pbc=True)
        warning = pytest.warns(UserWarning, match="exceeds half")
    else:
        atoms = Atoms("O2", positions=[[0, 0, 0], [0.7, 0, 0]], cell=[8] * 3)
        warning = nullcontext()
    with warning:
        result = compute_rdf([atoms], pair=pair, rmax=3, nbins=3,
                             sigma=0, n_bootstrap=0)
    assert np.isnan(result["g_r"]).all()
    assert result["per_structure"] == [[None, None, None]]


def test_cutoff_keeps_missing_frame_zero_policy_and_one_search_per_frame(monkeypatch):
    from amorphgen.analysis import _rdf_kernel

    first = Atoms("SiO", positions=[[0, 0, 0], [1, 0, 0]], cell=[10] * 3)
    missing = Atoms("Si", cell=[10] * 3)
    rdf = compute_rdf([first, missing], pair="O-Si", rmax=3, nbins=3,
                      sigma=0, n_bootstrap=0)
    captured = []

    def minimum(radius, curve):
        captured.append(curve.copy())
        return 2.0

    monkeypatch.setattr(cutoff, "_first_minimum", minimum)
    original = _rdf_kernel.neighbor_list
    searches = []

    def neighbors(*args, **kwargs):
        searches.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(_rdf_kernel, "neighbor_list", neighbors)
    result = cutoff.auto_cutoff_rdf([first, missing], rmax=3, nbins=3)
    assert result == {"O-O": 2.0, "O-Si": 2.0, "Si-Si": 2.0}
    assert len(searches) == 2
    # Cutoff fitting retains both frames in its denominator. Public RDF
    # averages only the one available curve. Same-species singletons give no
    # normalized observations and remain zero in the cutoff estimate.
    np.testing.assert_array_equal(captured[0], np.zeros(3))
    np.testing.assert_allclose(captured[1], np.asarray(rdf["g_r"]) / 2)
    np.testing.assert_array_equal(captured[2], np.zeros(3))
