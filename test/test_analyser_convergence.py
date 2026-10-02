"""Convergence across the actual analyser and stochastic descriptor boundary."""

import json
import warnings

import numpy as np
import pytest
from ase import Atoms

from amorphgen.analysis import StructureAnalyser, convergence_report
from amorphgen.analysis.voids import compute_void_distribution


def silica_ensemble():
    return [Atoms("SiO2", positions=[[0, 0, 0], [d, 0, 0], [0, d, 0]],
                  cell=[length] * 3, pbc=True)
            for d, length in [(1.5, 8), (1.6, 9), (1.7, 10)]]


def test_analyser_report_uses_structure_means_and_is_order_independent():
    structures = silica_ensemble()
    sa = StructureAnalyser(structures, cutoff=1.8)
    tolerances = {"density": 0.01, "bond_distance.O-Si": 0.02}
    report = sa.convergence_report(tolerances)
    reverse = StructureAnalyser(structures[::-1], cutoff=1.8)
    assert report == reverse.convergence_report(tolerances)
    rows = report["descriptors"]
    assert rows["density"]["units"] == "g/cm^3"
    assert rows["bond_angle.O-Si-O"]["units"] == "degrees"
    assert rows["total_coordination.Si"]["mean"] == 2
    assert rows["bond_distance.O-Si"]["mean"] == pytest.approx(1.6)
    assert rows["bond_distance.O-Si"]["estimated_additional_structures"] > 0
    assert report["analysis_settings"]["cutoff"] == 1.8
    json.dumps(report, allow_nan=False)


def test_default_cutoffs_and_report_are_order_independent_with_mixed_species():
    structures = silica_ensemble()
    structures[0] += Atoms("H", positions=[[3, 3, 3]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forward = StructureAnalyser(structures)
        reverse = StructureAnalyser(structures[::-1])
        first = forward.convergence_report({"density": 0.01})
        second = reverse.convergence_report({"density": 0.01})
    assert any("different compositions" in str(item.message) for item in caught)
    assert forward.cutoff == reverse.cutoff
    assert first == second
    assert forward.atoms_list == structures
    assert reverse.atoms_list == structures[::-1]


def test_analyser_accepts_extra_curve_summary_on_the_same_ensemble():
    sa = StructureAnalyser(silica_ensemble(), cutoff=1.8)
    rdf = sa.rdf(rmax=3, nbins=20, n_bootstrap=0)
    report = sa.convergence_report({"rdf.total": 0.5},
                                   descriptors={"rdf.total": rdf["uncertainty"]})
    row = report["descriptors"]["rdf.total"]
    assert row["n_per_point"] == [3] * 20
    assert len(row["half_width_per_point"]) == 20


def test_extra_descriptor_units_are_not_inferred_from_a_core_name_prefix():
    sa = StructureAnalyser(silica_ensemble(), cutoff=1.8)
    report = sa.convergence_report(descriptors={
        "density.custom": {"per_structure": [1, 2, 3], "units": "custom units"}})
    assert report["descriptors"]["density.custom"]["units"] == "custom units"
    assert report["descriptors"]["density"]["units"] == "g/cm^3"


def test_analyser_rejects_overwriting_core_and_misaligned_additions():
    sa = StructureAnalyser(silica_ensemble(), cutoff=1.8)
    with pytest.raises(ValueError, match="replace core"):
        sa.convergence_report({"density": 1}, descriptors={"density": [0, 0, 0]})
    with pytest.raises(ValueError):
        sa.convergence_report({"custom": 1}, descriptors={"custom": [0, 1]})


def test_seeded_void_observations_and_convergence_survive_structure_reordering():
    structures = silica_ensemble()
    forward = compute_void_distribution(structures, n_samples=50, nbins=5, seed=19)
    reverse = compute_void_distribution(structures[::-1], n_samples=50, nbins=5, seed=19)
    for key, summary in forward["uncertainty"].items():
        assert summary["per_structure"] == reverse["uncertainty"][key]["per_structure"][::-1]
    descriptors = {key: value for key, value in forward["uncertainty"].items()}
    assert convergence_report(descriptors, {"accessible_fraction": 0.01}) == (
        convergence_report(reverse["uncertainty"], {"accessible_fraction": 0.01}))
    np.testing.assert_array_equal(structures[0].positions, silica_ensemble()[0].positions)
