"""Coherent intensity conventions, angle conversion and ensemble sampling."""

import json

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from amorphgen.analysis import xrd
from amorphgen.analysis.rdf import _smooth_sq_weighted, xray_form_factor


def _structure(seed=0, symbols="Si4O8", length=8.0):
    atoms = Atoms(symbols, cell=[length] * 3, pbc=True)
    atoms.positions = np.random.default_rng(seed).uniform(0, length, (len(atoms), 3))
    return atoms


def _rows(result, key="per_structure"):
    return np.asarray(result[key], dtype=float)


def test_self_scattering_and_per_structure_composition(monkeypatch):
    q = np.array([1.0, 2.0, 3.0])
    sq_rows = np.array([[1., 1., 1.], [.5, 1., 2.]])

    def fake_sq(atoms_list, **kwargs):
        assert kwargs["weighting"] == "xray"
        assert kwargs["sigma_q"] == 0
        return {"q": q.tolist(), "per_structure": sq_rows.tolist(),
                "n_per_bin": [11, 11, 11],
                "n_per_bin_per_structure": [[1, 1, 1], [10, 10, 10]]}

    monkeypatch.setattr(xrd, "compute_structure_factor_direct", fake_sq)
    result = xrd.compute_xrd_pattern([_structure(symbols="O2"),
                                      _structure(symbols="Si2O2")],
                                     qmax=4, nq=3, n_bootstrap=100, seed=7)
    oxygen, silicon = (xray_form_factor(s, q) for s in ("O", "Si"))
    expected = np.array([
        oxygen**2,
        (.5 * oxygen + .5 * silicon)**2 * (sq_rows[1] - 1)
        + .5 * oxygen**2 + .5 * silicon**2,
    ])
    np.testing.assert_allclose(_rows(result), expected)
    np.testing.assert_allclose(result["intensity"], expected.mean(axis=0))
    np.testing.assert_allclose(result["uncertainty"]["sem"],
                               np.abs(expected[0] - expected[1]) / 2)
    assert result["uncertainty"]["n_per_point"] == [2, 2, 2]
    assert result["intensity_units"] == "electron^2/atom"
    np.testing.assert_allclose(result["two_theta"],
                               np.degrees(2 * np.arcsin(q * 1.5406 / (4 * np.pi))))
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("method", ["direct", "ft"])
def test_real_profiles_equal_weight_independent_compositions_and_cells(method):
    atoms = [_structure(1, "Si4O8", 8), _structure(2, "Si8O4", 10)]
    kwargs = dict(qmax=5, nq=20, method=method, n_bootstrap=100, seed=17)
    if method == "ft":
        kwargs["rmax"] = 3
    result = xrd.compute_xrd_pattern(atoms, **kwargs)
    singles = [xrd.compute_xrd_pattern([a], **kwargs) for a in atoms]
    rows = np.asarray([s["intensity"] for s in singles], dtype=float)
    np.testing.assert_allclose(_rows(result), rows, equal_nan=True)
    present = np.isfinite(rows).all(axis=0)
    assert present.sum() > 5
    np.testing.assert_allclose(np.asarray(result["intensity"])[present].astype(float),
                               rows[:, present].mean(axis=0))
    repeated = xrd.compute_xrd_pattern(atoms, **kwargs)
    assert repeated["uncertainty"]["bootstrap_low"] == result["uncertainty"]["bootstrap_low"]
    json.dumps(result, allow_nan=False)


def test_copper_first_reflection_has_bragg_angle():
    copper = bulk("Cu", "fcc", a=3.6, cubic=True)
    result = xrd.compute_xrd_pattern([copper], wavelength=1.5406,
                                     qmax=5, nq=200, n_bootstrap=0)
    intensity = np.asarray(result["intensity"], dtype=float)
    peak = np.nanargmax(intensity)
    d111 = 3.6 / np.sqrt(3)
    expected_two_theta = np.degrees(2 * np.arcsin(1.5406 / (2 * d111)))
    assert result["two_theta"][peak] == pytest.approx(expected_two_theta, abs=.25)
    assert all(x is None for x in result["uncertainty"]["sem"])
    assert all(x is None for x in result["uncertainty"]["bootstrap_low"])


def test_smoothing_applies_after_coherent_conversion_and_preserves_missing():
    result = xrd.compute_xrd_pattern([_structure()], qmax=5, nq=100,
                                     sigma_q=.2, n_bootstrap=0)
    raw = _rows(result, "per_structure_raw")[0]
    counts = np.asarray(result["n_per_bin_per_structure"])[0]
    expected = _smooth_sq_weighted(result["q"], raw, counts, .2)
    np.testing.assert_allclose(_rows(result)[0], expected, equal_nan=True)
    missing = counts == 0
    assert missing.any()
    assert np.isnan(_rows(result)[0, missing]).all()
    assert np.isnan(raw[missing]).all()
    assert np.max(np.abs(raw[~missing] - expected[~missing])) > .1
    json.dumps(result, allow_nan=False)


def test_default_qmax_tracks_wavelength():
    result = xrd.compute_xrd_pattern([_structure()], wavelength=4,
                                     nq=20, n_bootstrap=0)
    assert result["metadata"]["qmax"] == pytest.approx(np.pi)
    assert result["metadata"]["requested_qmax"] is None
    assert all(0 < angle < 180 for angle in result["two_theta"])


@pytest.mark.parametrize("kwargs,match", [
    ({"wavelength": 0}, "wavelength"),
    ({"wavelength": np.nan}, "wavelength"),
    ({"qmax": 10}, "physically accessible"),
    ({"qmax": 80, "wavelength": .1}, "validity limit"),
    ({"qmax": -1}, "qmax"),
    ({"nq": 1}, "nq"),
    ({"nq": True}, "nq"),
    ({"q_batch": 0}, "q_batch"),
    ({"sigma_q": -1}, "sigma_q"),
    ({"sigma_q": np.inf}, "sigma_q"),
    ({"method": "bad"}, "method"),
    ({"method": "ft", "qmax": .05}, "qmax > 0.1"),
    ({"rmax": 3}, "only to method='ft'"),
    ({"method": "ft", "rmax": 0}, "rmax"),
    ({"confidence": 1}, "confidence"),
    ({"n_bootstrap": -1}, "n_bootstrap"),
])
def test_invalid_controls_raise_before_scattering(kwargs, match):
    with pytest.raises(ValueError, match=match):
        xrd.compute_xrd_pattern([_structure()], **kwargs)


def test_empty_and_invalid_structures_rejected():
    with pytest.raises(ValueError, match="at least one"):
        xrd.compute_xrd_pattern([])
    with pytest.raises(ValueError, match="no atoms"):
        xrd.compute_xrd_pattern([Atoms(cell=[8] * 3, pbc=True)])
    with pytest.raises(TypeError, match="ASE Atoms"):
        xrd.compute_xrd_pattern([object()])
    nonperiodic = _structure()
    nonperiodic.pbc[2] = False
    with pytest.raises(ValueError, match="periodic"):
        xrd.compute_xrd_pattern([nonperiodic])
    bad_cell = _structure()
    bad_cell.cell[2] = 0
    with pytest.raises(ValueError, match="nonsingular"):
        xrd.compute_xrd_pattern([bad_cell])
    bad_cell.cell[2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        xrd.compute_xrd_pattern([bad_cell])
    bad_position = _structure()
    bad_position.positions[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite atomic positions"):
        xrd.compute_xrd_pattern([bad_position])
