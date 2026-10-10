"""Empty and degenerate inputs, option branches and crystal oracles for RDF, S(q), T(r) and structure descriptors."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from scipy.stats import t

from amorphgen.analysis.rdf import (
    compute_averaged_rdf,
    compute_rdf,
    compute_structure_factor,
    compute_structure_factor_direct,
    compute_total_correlation,
    coordination_from_Tr,
    first_Tr_peak,
    format_Tr_scan,
    xray_form_factor,
)
from amorphgen.analysis.structure import (
    compute_all_angles,
    compute_bond_angle_stats,
    compute_bond_distances,
    compute_coordination,
    compute_polyhedral_connectivity,
    format_connectivity_report,
)


def _random(symbols, length=8.0, seed=0):
    rng = np.random.default_rng(seed)
    atoms = Atoms(symbols, cell=[length] * 3, pbc=True)
    atoms.positions = rng.uniform(0, length, (len(atoms), 3))
    return atoms


def _shell(r, r0, amplitude, width):
    return amplitude * np.exp(-((r - r0) / width) ** 2)


def _tr_result(r, g, rho):
    return {"r": r.tolist(), "g_r": g.tolist(), "rho": rho,
            "T_r": (4 * np.pi * r * rho * g).tolist()}


# ─── g(r), S(q) and T(r) ──────────────────────────────────────────────────────

@pytest.mark.parametrize("function,axis,value,extra", [
    (compute_rdf, "r", "g_r", {}),
    (compute_structure_factor, "q", "s_q", {}),
    (compute_structure_factor_direct, "q", "s_q", {"n_per_bin": []}),
    (compute_averaged_rdf, "r", "g_r_mean", {"g_r_std": []}),
])
def test_empty_ensemble_gives_empty_curves(function, axis, value, extra):
    result = function(iter([]))
    assert result[axis] == [] and result[value] == []
    assert result["per_structure"] == []
    assert result["uncertainty"]["n_structures"] == 0
    for key, expected in extra.items():
        assert result[key] == expected


def test_total_correlation_needs_a_structure():
    with pytest.raises(ValueError, match="no structures given"):
        compute_total_correlation([])


def test_smearing_narrower_than_a_quarter_bin_leaves_the_histogram_unchanged():
    atoms = _random("Si8O16")
    kwargs = dict(rmax=4.0, nbins=20, n_bootstrap=0)       # dr = 0.2 A
    raw = compute_rdf([atoms], sigma=0, **kwargs)["g_r"]
    # The 4-sigma window is 0.8 bins, so there is no kernel to apply.
    assert compute_rdf([atoms], sigma=0.04, **kwargs)["g_r"] == raw
    assert compute_rdf([atoms], sigma=0.06, **kwargs)["g_r"] != raw


@pytest.mark.parametrize("call,match", [
    (lambda: xray_form_factor("Es", 1.0), r"'Es'.*H\.\.Cf"),
    (lambda: compute_structure_factor_direct([_random("Es2Cu2")], qmax=4, nq=10),
     r"X-ray.*'Es'"),
    (lambda: compute_structure_factor_direct([_random("Ar4")], qmax=4, nq=10,
                                             weighting="neutron"),
     r"neutron scattering length.*'Ar'.*_NEUTRON_B"),
    (lambda: compute_structure_factor([_random("Ar4")], qmax=4, nq=10,
                                      weighting="neutron"),
     r"neutron scattering length.*'Ar'"),
])
def test_untabulated_elements_name_the_missing_scattering_factor(call, match):
    with pytest.raises(KeyError, match=match):
        call()


@pytest.mark.parametrize("function", [compute_structure_factor_direct,
                                      compute_structure_factor,
                                      compute_total_correlation])
def test_every_scattering_path_rejects_an_unknown_weighting(function):
    with pytest.raises(ValueError, match="weighting must be .*got 'bogus'"):
        function([_random("Si4O8")], weighting="bogus", qmax=4, nq=10)


@pytest.mark.parametrize("function,kwargs,axis,value", [
    (compute_structure_factor_direct, dict(qmax=6.0, nq=30, weighting="xray"), "q", "s_q"),
    (compute_rdf, dict(nbins=30, sigma=0), "r", "g_r"),
])
def test_equivalent_sheared_cell_gives_the_same_curve(function, kwargs, axis, value):
    # (a1, a1 + a2, a3) spans the same lattice as the cube, so every
    # reciprocal vector and every pair distance is unchanged. A single
    # integer bound taken from the shortest reciprocal vector misses part of
    # the |q| <= qmax sphere in the sheared basis.
    length = 6.0
    cubic = _random("Si4O8", length, seed=1)
    sheared = cubic.copy()
    sheared.set_cell([[length, 0, 0], [length, length, 0], [0, 0, length]])
    expected = function([cubic], n_bootstrap=0, **kwargs)
    result = function([sheared], n_bootstrap=0, **kwargs)
    assert result[axis] == expected[axis]
    np.testing.assert_allclose(np.asarray(result[value], dtype=float),
                               np.asarray(expected[value], dtype=float),
                               rtol=1e-9, atol=1e-12, equal_nan=True)
    if "n_per_bin" in expected:
        assert result["n_per_bin"] == expected["n_per_bin"]


@pytest.mark.parametrize("pair,weighting,unobservable", [
    ("Si-O", "unweighted", _random("Si8", seed=2)),           # no O partner
    (None, "unweighted", Atoms("Si", cell=[8] * 3, pbc=True)),  # N - 1 = 0
    (None, "xray", Atoms(cell=[8] * 3, pbc=True)),            # no atoms
])
def test_ft_sq_of_an_unobservable_structure_is_missing_not_zero(pair, weighting, unobservable):
    observed = _random("Si4O8")
    kwargs = dict(pair=pair, weighting=weighting, qmax=6, nq=20, rmax=4,
                  n_bootstrap=0)
    alone = compute_structure_factor([observed], **kwargs)
    result = compute_structure_factor([observed, unobservable], **kwargs)
    assert result["per_structure"][1] == [None] * 20
    assert result["s_q"] == alone["s_q"]
    assert result["uncertainty"]["n_structures"] == 1


@pytest.mark.parametrize("r,T", [
    (np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.0, 1.0, 0.0, 1.0])),  # too short
    (np.linspace(1, 6, 100), 4 * np.pi * 0.07 * np.linspace(1, 6, 100)),  # no maximum
])
def test_first_tr_peak_reports_none_without_a_local_maximum(r, T):
    assert first_Tr_peak({"r": r.tolist(), "T_r": T.tolist(), "rho": 0.07}) == (None, None, None)


def test_first_tr_peak_ignores_a_shell_inside_rmin():
    r = np.linspace(0.05, 6, 600)
    result = _tr_result(r, 1 + _shell(r, 0.6, 3.0, 0.08), 0.07)
    assert first_Tr_peak(result) == (None, None, None)
    assert first_Tr_peak(result, rmin=0.5)[0] == pytest.approx(0.6, abs=0.02)


def test_first_tr_peak_derives_g_from_density_when_g_is_absent():
    # The 1.5 A shell is prominent in T(r) but its weighted g is 0.45, below
    # min_g, so the first resolved shell is the one at 3.0 A. Without g_r the
    # same decision must come from T / (4 pi rho r).
    r = np.linspace(0.05, 6.0, 600)
    rho = 0.07
    g = _shell(r, 1.5, 0.45, 0.1) + _shell(r, 3.0, 0.9, 0.15)
    with_g = _tr_result(r, g, rho)
    assert first_Tr_peak(with_g, min_g=0.0)[0] == pytest.approx(1.5, abs=0.02)
    pk, lo, hi = first_Tr_peak(with_g)
    assert pk == pytest.approx(3.0, abs=0.02) and lo < pk < hi
    without_g = {key: value for key, value in with_g.items() if key != "g_r"}
    assert first_Tr_peak(without_g) == (pk, lo, hi)


def test_coordination_window_outside_the_grid_is_an_error():
    result = {"r": [1.0, 2.0, 3.0], "T_r": [1.0, 2.0, 3.0]}
    with pytest.raises(ValueError, match="no T\\(r\\) points between 5 and 6 A"):
        coordination_from_Tr(result, 5, 6)


def test_tr_scan_table_shows_failed_and_unresolved_transforms():
    error = ("compute_total_correlation: only 0 usable S(Q) points for "
             "structure 0 above qmin=30.0")
    rows = [{"qmax": 5.0, "window": "lorch", "error": error},
            {"qmax": 12.0, "window": None, "r_peak": None, "r_lo": None,
             "r_hi": None, "count": None, "result": {}}]
    lines = format_Tr_scan(rows).splitlines()
    assert f"  lorch       5.0   {error[:36]}" in lines
    assert error[:37] not in "\n".join(lines)
    assert "  none       12.0      no resolved first peak" in lines
    assert not any("spread" in line for line in lines)


# ─── Coordination, distances and angles in rock salt ──────────────────────────

@pytest.fixture
def nacl():
    """64-atom rock salt: 6 Cl at a/2 and 12 Na at a/sqrt(2) around each Na."""
    return bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat(2)


def _nacl_cutoffs(a, b):
    return 3.2 if a != b else 4.2      # first shell, and like pairs to the second


def test_coordination_pair_filter_keeps_both_directions(nacl):
    full = compute_coordination([nacl], 4.2, _nacl_cutoffs)
    assert full["Na-Cl"]["distribution"] == {6: 100.0}
    assert full["Na-Na"]["mean"] == 12
    for pair in ("Na-Cl", "Cl-Na"):
        only = compute_coordination([nacl], 4.2, _nacl_cutoffs, pair=pair)
        assert set(only) == {"Na-Cl", "Cl-Na"}
        assert only["Na-Cl"] == full["Na-Cl"] and only["Cl-Na"] == full["Cl-Na"]


@pytest.mark.parametrize("pair", ["Na-Cl", "Cl-Na"])
def test_bond_distance_pair_filter_accepts_either_order(nacl, pair):
    assert set(compute_bond_distances([nacl], 4.2, _nacl_cutoffs)) == {
        "Cl-Cl", "Cl-Na", "Na-Na"}
    distances = compute_bond_distances([nacl], 4.2, _nacl_cutoffs, pair=pair)
    assert list(distances) == ["Cl-Na"]
    assert distances["Cl-Na"]["count"] == 32 * 6          # each bond once
    assert distances["Cl-Na"]["mean"] == pytest.approx(5.64 / 2)


def test_angle_triplet_filter_and_non_bonding_neighbours(nacl):
    assert set(compute_all_angles([nacl], 4.2, _nacl_cutoffs)) == {
        "Cl-Na-Cl", "Na-Cl-Na"}
    octahedra = compute_all_angles([nacl], 4.2, _nacl_cutoffs, triplet="Cl-Na-Cl")
    assert list(octahedra) == ["Cl-Na-Cl"]
    angles = np.round(octahedra["Cl-Na-Cl"], 6)
    # Of the 15 Cl-Na-Cl pairs around an octahedron, 12 are cis and 3 trans.
    assert np.count_nonzero(angles == 90) == 32 * 12
    assert np.count_nonzero(angles == 180) == 32 * 3
    everything = compute_all_angles([nacl], 4.2, _nacl_cutoffs, bonding_only=False)
    assert len(everything["Na-Na-Na"]) == 32 * 66        # C(12, 2)
    assert len(everything["Cl-Na-Na"]) == 32 * 6 * 12


def test_overlapping_atoms_add_no_angle_for_the_zero_length_contact():
    atoms = Atoms("Si2O2", positions=[[5, 5, 5], [5, 5, 5], [6.6, 5, 5], [5, 6.6, 5]],
                  cell=[10] * 3, pbc=True)
    angles = compute_all_angles([atoms], 2.0, lambda a, b: 2.0, bonding_only=False)
    assert "O-Si-Si" not in angles                 # Si-Si contact has length zero
    assert angles["O-Si-O"] == pytest.approx([90.0, 90.0])
    assert all(np.isfinite(values).all() for values in angles.values())


def test_angle_statistics_skip_triplets_without_angles():
    stats = compute_bond_angle_stats({"O-Si-O": [], "Si-O-Si": [130.0, 150.0]})
    assert list(stats) == ["Si-O-Si"]
    assert stats["Si-O-Si"]["mean"] == 140 and stats["Si-O-Si"]["count"] == 2


# ─── Polyhedral connectivity ──────────────────────────────────────────────────

def _cscl_type(n=3):
    """Cs in a cube of 8 Cl: per Cs, 6 face-, 12 edge- and 8 corner-sharing links."""
    return Atoms("CsCl", scaled_positions=[[.5, .5, .5], [0, 0, 0]],
                 cell=[4.12] * 3, pbc=True).repeat(n)


def _reo3_type(n=3):
    """ReO3 geometry with Cs on the Re site: octahedra sharing only corners."""
    return Atoms("CsCl3", scaled_positions=[[0, 0, 0], [.5, 0, 0], [0, .5, 0], [0, 0, .5]],
                 cell=[3.75] * 3, pbc=True).repeat(n)


def _cation_anion(a, b):
    return 3.8 if a != b else 0.0


def test_cubic_polyhedra_share_faces_edges_and_corners():
    result = compute_polyhedral_connectivity([_cscl_type()], 3.8, _cation_anion)
    assert result["anion"] == "Cl" and result["cations"] == ["Cs"]
    assert result["n_links"] == 27 * 26 // 2
    assert result["link_percent"] == pytest.approx(
        {"corner": 100 * 108 / 351, "edge": 100 * 162 / 351, "face": 100 * 81 / 351})
    cs = result["per_species"]["Cs"]
    assert (cs["mean_face_links"], cs["mean_edge_links"], cs["mean_corner_links"]) == (6, 12, 8)
    assert cs["face_percent"] == 100


def test_connectivity_report_summarises_structures_and_their_uncertainty():
    result = compute_polyhedral_connectivity([_cscl_type(), _reo3_type()], 3.8,
                                             _cation_anion)
    assert result["per_structure_edge_percent"] == [100.0, 0.0]
    assert result["cation_edge_or_face_percent"] == 50
    text = format_connectivity_report(result)
    assert ("per-structure edge/face-sharing cations: 50.0%, structure SD 50.0% "
            "(min 0.0, max 100.0)") in text
    half_width = t.ppf(0.975, 1) * 50                    # SEM = 100 / sqrt(2) / sqrt(2)
    assert (f"SEM=50.00%; 95% t CI [{50 - half_width:.2f}, "
            f"{50 + half_width:.2f}]%") in text


@pytest.mark.parametrize("structures,message", [
    ([], "no structures"),
    ([bulk("Cu", "fcc", a=3.6)], "single-element system"),
])
def test_connectivity_without_a_cation_anion_split_is_explained(structures, message):
    result = compute_polyhedral_connectivity(structures, 3.0, lambda a, b: 3.0)
    assert message in result["error"]
    assert format_connectivity_report(result) == (
        f"\n  Polyhedral connectivity: {result['error']}")


def test_absent_requested_cation_reports_no_per_structure_statistics():
    result = compute_polyhedral_connectivity([_cscl_type(2), _cscl_type(2)], 3.8,
                                             _cation_anion, cation="Na")
    assert result["per_structure_edge_percent"] == [None, None]
    assert result["n_links"] == 0 and result["per_species"]["Na"]["n_atoms"] == 0
    text = format_connectivity_report(result)
    assert "per-structure" not in text and "nan" not in text
