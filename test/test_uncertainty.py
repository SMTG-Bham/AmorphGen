"""Uncertainty must use structures rather than their correlated sites."""

import json

import numpy as np
import pytest
from ase import Atoms
from scipy.stats import t

from amorphgen.analysis.uncertainty import summarize_site_groups, summarize_structures
from amorphgen.analysis.structure import (
    BondAngleData, compute_all_angles, compute_bond_angle_stats,
    compute_bond_distances, compute_coordination, compute_density, compute_dimers,
)


def test_structure_sem_and_student_t_interval():
    result = summarize_structures([1, 3, 5])
    assert result["mean"] == 3
    assert result["std"] == 2
    assert result["sem"] == pytest.approx(2 / np.sqrt(3))
    half_width = t.ppf(0.975, 2) * 2 / np.sqrt(3)
    assert result["ci_low"] == pytest.approx(3 - half_width)
    assert result["ci_high"] == pytest.approx(3 + half_width)
    assert result["sampling_unit"] == "structure"
    assert result["n_structures"] == 3


@pytest.mark.parametrize("values,mean,n", [([], None, 0), ([3], 3, 1),
                                           ([None, np.nan], None, 0),
                                           ([None, 3, np.inf], 3, 1)])
def test_insufficient_structures_have_no_invented_certainty(values, mean, n):
    result = summarize_structures(values)
    assert result["mean"] == mean
    assert result["n_structures"] == n
    assert result["n_total_structures"] == len(values)
    for key in ("std", "sem", "ci_low", "ci_high", "bootstrap_low", "bootstrap_high"):
        assert result[key] is None
    json.dumps(result, allow_nan=False)


def test_missing_structures_are_not_zero_observations():
    result = summarize_structures([2, None, 4, np.nan])
    assert result["per_structure"] == [2, None, 4, None]
    assert result["n_structures"] == 2
    assert result["mean"] == 3
    assert result["sem"] == 1


def test_bootstrap_resamples_whole_curves_and_is_reproducible():
    curves = np.array([[1, 3], [2, 5], [4, 9], [8, 17]])
    result = summarize_structures(curves, n_bootstrap=400, seed=82)
    assert result == summarize_structures(curves, n_bootstrap=400, seed=82)
    # A row resample preserves this affine relation in every bootstrap draw.
    for key in ("mean", "ci_low", "ci_high", "bootstrap_low", "bootstrap_high"):
        assert result[key][1] == pytest.approx(2 * result[key][0] + 1)
    assert result["sem"][1] == pytest.approx(2 * result["sem"][0])
    assert result["band_type"] == "pointwise"


def test_curve_missing_bins_and_rows_preserve_alignment():
    result = summarize_structures([[1, None, 4], None, [3, 7, None]])
    assert result["n_structures"] == 2
    assert result["n_total_structures"] == 3
    assert result["n_per_point"] == [2, 1, 1]
    assert result["mean"] == [2, 7, 4]
    assert result["sem"] == [1, None, None]
    assert result["bootstrap_low"][1:] == [None, None]
    assert result["per_structure"][1] == [None, None, None]
    json.dumps(result, allow_nan=False)


def test_repeated_sites_do_not_create_additional_independent_samples():
    original = summarize_site_groups([[1, 3], [3, 5]])
    repeated = summarize_site_groups([[1, 3] * 100, [3, 5] * 100])
    assert original["uncertainty"] == repeated["uncertainty"]
    assert original["uncertainty"]["sem"] == 1


def test_unequal_structure_sizes_separate_site_fraction_and_prevalence():
    bonded = Atoms("Si2", positions=[[0, 0, 0], [1, 0, 0]], cell=[30] * 3)
    isolated = Atoms("Si9", positions=[[3 * i, 0, 0] for i in range(9)],
                     cell=[30] * 3)
    result = compute_coordination([bonded, isolated], 1.1,
                                  lambda a, b: 1.1)["Si-Si"]
    assert result["mean"] == pytest.approx(2 / 11)
    assert result["pooled_mean"] == result["mean"]
    assert result["uncertainty"]["mean"] == 0.5
    assert result["uncertainty"]["sem"] == 0.5
    assert result["per_structure"] == [1, 0]
    assert result["fraction_of_sites"][1] == pytest.approx(2 / 11)
    assert result["fraction_of_structures"][1] == 0.5
    assert result["site_fraction_uncertainty"][1]["per_structure"] == [1, 0]


def test_absent_centre_is_missing_but_absent_neighbour_is_zero_coordination():
    sio = Atoms("SiO", positions=[[0, 0, 0], [1, 0, 0]], cell=[10] * 3)
    si = Atoms("Si", cell=[10] * 3)
    o = Atoms("O", cell=[10] * 3)
    result = compute_coordination([sio, si, o], 1.1, lambda a, b: 1.1)["Si-O"]
    assert result["per_structure"] == [1, 0, None]
    assert result["uncertainty"]["n_structures"] == 2
    assert result["fraction_of_structures"][1] == pytest.approx(1 / 3)


def test_bond_distances_keep_missing_structure_and_equal_weight():
    first = Atoms("Si2", positions=[[0, 0, 0], [1, 0, 0]], cell=[30] * 3)
    second = Atoms("Si8", positions=[[x, 0, 0] for x in (0, 2, 6, 8, 12, 14, 18, 20)],
                   cell=[30] * 3)
    missing = Atoms("Si", cell=[30] * 3)
    result = compute_bond_distances([first, missing, second], 2.1,
                                    lambda a, b: 2.1)["Si-Si"]
    assert result["mean"] == pytest.approx(1.8)
    assert result["per_structure"] == [1, None, 2]
    assert result["uncertainty"]["mean"] == 1.5
    assert result["uncertainty"]["sem"] == 0.5


def test_angle_statistics_preserve_structure_identity_and_legacy_dict():
    raw = BondAngleData(3)
    raw["O-Si-O"] = [90] * 10 + [150]
    raw.per_structure = [{"O-Si-O": [90] * 10}, {}, {"O-Si-O": [150]}]
    result = compute_bond_angle_stats(raw)["O-Si-O"]
    assert result["mean"] == pytest.approx(1050 / 11)
    assert result["uncertainty"]["mean"] == 120
    assert result["uncertainty"]["sem"] == 30
    assert result["per_structure"] == [90, None, 150]
    legacy = compute_bond_angle_stats(dict(raw))["O-Si-O"]
    assert legacy["uncertainty"]["sem"] is None


def test_raw_angles_include_empty_structures_without_adding_zero_angles():
    bent = Atoms("OSiO", positions=[[1, 0, 0], [0, 0, 0], [0, 1, 0]],
                 cell=[10] * 3)
    empty = Atoms("Si", cell=[10] * 3)
    raw = compute_all_angles([bent, empty], 1.1, lambda a, b: 1.1)
    assert raw["O-Si-O"] == [90]
    assert raw.per_structure == [{"O-Si-O": [90]}, {}]
    assert compute_all_angles([], 1.1, lambda a, b: 1.1) == {}


def test_density_and_dimer_uncertainty_are_per_structure():
    a = Atoms("SiO2", positions=[[5, 5, 5], [0, 0, 0], [1, 0, 0]], cell=[10] * 3)
    b = Atoms("SiO2", positions=[[5, 5, 5], [0, 0, 0], [4, 0, 0]], cell=[20] * 3)
    density = compute_density([a, b])
    assert density["uncertainty"]["per_structure"] == density["values"]
    assert compute_density([])["uncertainty"]["mean"] is None
    dimers = compute_dimers([a, b])
    assert dimers["per_structure"] == [1, 0]
    assert dimers["uncertainty"]["sem"] == 0.5
    assert dimers["fraction_of_sites"] == pytest.approx(1 / 3)
    assert dimers["fraction_of_structures"] == 0.5


@pytest.mark.parametrize("kwargs", [{"confidence": 1}, {"confidence": 0},
                                   {"n_bootstrap": -1}, {"n_bootstrap": 1.5}])
def test_invalid_uncertainty_settings(kwargs):
    with pytest.raises(ValueError):
        summarize_structures([1, 2], **kwargs)
