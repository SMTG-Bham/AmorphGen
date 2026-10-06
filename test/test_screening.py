"""Screen labels, selection policy and transparent population accounting."""

import csv
import json

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator

from amorphgen.analysis.screening import (
    format_screening_report,
    mark_screening_analysed,
    screen_structures,
    validate_screening_config,
    write_screening_outputs,
)


def test_labels_and_exclusion_are_independent_and_analysis_is_explicit():
    atoms = bulk("Si", "diamond", a=5.43)
    config = {"coordination": {"allowed": {"Si": [3]}}}
    labelled = screen_structures([atoms], config, cutoff=2.6)
    assert labelled["summary"] == {"generated": 1, "passed": 0,
                                    "labelled": 1, "analysed": 0, "excluded": 0}
    assert labelled["retained_indices"] == [0]
    assert labelled["per_structure"][0]["labels"] == ["coordination"]
    mark_screening_analysed(labelled, [0])
    assert labelled["summary"]["analysed"] == 1
    config["coordination"]["exclude"] = True
    excluded = screen_structures([atoms], config, cutoff=2.6)
    assert excluded["per_structure"][0]["labels"] == ["coordination"]
    assert excluded["retained_indices"] == []
    with pytest.raises(ValueError, match="excluded"):
        mark_screening_analysed(excluded, [0])


def test_default_coordination_infers_targets_and_tolerance():
    si = bulk("Si", "diamond", a=5.43)
    frame = screen_structures([si], {"coordination": True}, cutoff=2.6)["per_structure"][0]
    assert frame["passed"]
    assert frame["screens"]["coordination"]["metrics"]["allowed"] == {"Si": [4]}
    # auto_target_cn infers flexible four-to-six coordinate oxide cations.
    oxide = Atoms("Al2O3", positions=[[0, 0, 0], [5, 0, 0], [1, 0, 0],
                                    [0, 1, 0], [0, 0, 1]], cell=[15] * 3)
    metrics = screen_structures([oxide], {"coordination": True}, cutoff=2)["per_structure"][0]["screens"]["coordination"]["metrics"]
    assert metrics["allowed"] == {"Al": [4, 5, 6]}
    assert metrics["checked_sites"] == 2
    assert metrics["unassessed_elements"] == ["O"]


def test_coordination_counts_bonds_not_cation_contacts():
    atoms = Atoms("Al2O3", positions=[[0, 0, 0], [0.5, 0, 0],
                                    [0, 1, 0], [0, -1, 0], [0, 0, 1]])
    frame = screen_structures([atoms], {"coordination": {"allowed": {"Al": [3]}}}, cutoff=2)["per_structure"][0]
    assert frame["passed"]
    assert frame["screens"]["coordination"]["metrics"]["coordination_numbers"][:2] == [3, 3]


def test_coordination_fraction_uses_only_assessed_sites_and_strict_limit():
    atoms = Atoms("Si2H", positions=[[0, 0, 0], [1, 0, 0], [4, 0, 0]])
    config = {"coordination": {"allowed": {"Si": [0]}, "max_fraction": 1}}
    frame = screen_structures([atoms], config, cutoff=1.1)["per_structure"][0]
    assert frame["passed"]
    assert frame["screens"]["coordination"]["metrics"]["unexpected_fraction"] == 1


def test_unknown_coordination_does_not_silently_pass():
    atom = Atoms("He", cell=[10] * 3)
    result = screen_structures([atom], {"coordination": {"exclude": True}}, cutoff=2)
    assert result["per_structure"][0]["labels"] == ["coordination_unavailable"]
    assert result["summary"]["excluded"] == 1


def test_crystal_order_uses_existing_periodic_bond_order():
    atom = bulk("Cu", "fcc", a=4)
    result = screen_structures([atom], {"crystal_like": {"max_fraction": 0.8}}, cutoff=3)
    record = result["per_structure"][0]["screens"]["crystal_like"]
    assert record["status"] == "labelled"
    assert record["metrics"]["ordered_fraction"] == 1
    assert record["metrics"]["largest_cluster_fraction"] == 1
    assert record["metrics"]["qbar6_mean"] == pytest.approx(0.5745242597)
    result = screen_structures([atom], {"crystal_like": {"qbar6_threshold": 0.7}}, cutoff=3)
    assert result["summary"]["passed"] == 1


def test_crystal_cluster_threshold_can_trigger_independently():
    atom = bulk("Cu", "fcc", a=4)
    result = screen_structures([atom], {"crystal_like": {
        "max_fraction": 1, "max_cluster_fraction": 0.5}}, cutoff=3)
    assert result["per_structure"][0]["labels"] == ["crystal_like"]


def test_crystal_cutoff_override_is_independent_of_coordination():
    atom = bulk("Cu", "fcc", a=4)
    result = screen_structures([atom], {
        "coordination": {"allowed": {"Cu": [0]}},
        "crystal_like": {"cutoff": 3}}, cutoff=1)
    assert result["per_structure"][0]["screens"]["coordination"]["status"] == "passed"
    assert result["per_structure"][0]["labels"] == ["crystal_like"]
    assert result["parameters"]["crystal_like_cutoff"] == 3


def test_crystal_cutoff_override_does_not_resolve_unused_global_cutoff(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("unused default cutoff should not require an RDF")

    monkeypatch.setattr("amorphgen.analysis.cutoff.auto_cutoff_rdf", fail)
    atom = bulk("Cu", "fcc", a=4)
    result = screen_structures([atom], {"crystal_like": {"cutoff": {"default": 3}}})
    assert result["per_structure"][0]["labels"] == ["crystal_like"]


def test_close_contacts_include_heteronuclear_pairs_and_periodic_self_images():
    atoms = Atoms("SiO", positions=[[0, 0, 0], [0.2, 0, 0]], cell=[10] * 3)
    frame = screen_structures([atoms], {"close_contacts": True})["per_structure"][0]
    assert frame["labels"] == ["close_contacts"]
    metrics = frame["screens"]["close_contacts"]["metrics"]
    assert metrics["count"] == 1
    assert metrics["contacts"][0]["pair"] == "O-Si"
    tiny = Atoms("Si", cell=[0.5, 10, 10], pbc=True)
    frame = screen_structures([tiny], {"close_contacts": {"min_distance": 0.8}})["per_structure"][0]
    metrics = frame["screens"]["close_contacts"]["metrics"]
    assert metrics["count"] == 1
    assert metrics["contacts"][0]["atoms"] == [0, 0]
    assert metrics["contacts"][0]["shift"] == [1, 0, 0]


def test_close_contact_lower_bound_is_strict_and_pair_table_complete():
    atoms = Atoms("SiO", positions=[[0, 0, 0], [1, 0, 0]])
    result = screen_structures([atoms], {"close_contacts": {"min_distance": 1}})
    assert result["summary"]["passed"] == 1
    result = screen_structures([atoms], {"close_contacts": {"min_distance": {"Si-O": 2}}})
    assert result["per_structure"][0]["labels"] == ["close_contacts_unavailable"]
    result = screen_structures([atoms], {"close_contacts": {"min_distance": {
        "Si-O": 2, "Si-Si": 1, "O-O": 1}}})
    assert result["per_structure"][0]["labels"] == ["close_contacts"]


def test_density_and_energy_use_inclusive_bounds_and_per_atom_energy():
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.calc = SinglePointCalculator(atoms, energy=-10)
    result = screen_structures([atoms], {"density": {"min": 2, "max": 3},
                                        "energy": {"min": -5, "max": -5}})
    assert result["summary"]["passed"] == 1
    energy = result["per_structure"][0]["screens"]["energy"]["metrics"]
    assert energy == {"value": -5, "units": "eV/atom"}
    atoms.info["energy"] = -8
    result = screen_structures([atoms], {"energy": {"max": -5, "exclude": True}})
    assert result["summary"]["excluded"] == 1


def test_energy_does_not_evaluate_calculators():
    class NeverCalculate(Calculator):
        implemented_properties = ["energy"]

        def calculate(self, *args, **kwargs):
            pytest.fail("screening evaluated a calculator")

    atoms = Atoms("Si")
    atoms.calc = NeverCalculate()
    result = screen_structures([atoms], {"energy": {"max": 0}})
    assert result["per_structure"][0]["labels"] == ["energy_unavailable"]


def test_stale_calculator_energy_does_not_pass():
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.calc = SinglePointCalculator(atoms, energy=-10)
    atoms.positions[0, 0] += 0.1
    result = screen_structures([atoms], {"energy": {"max": 0}})
    record = result["per_structure"][0]["screens"]["energy"]
    assert record["status"] == "unavailable"
    assert "stale" in record["metrics"]["reason"]


def test_density_exact_analysis_bound_passes():
    from amorphgen.analysis.structure import compute_density

    atoms = bulk("Si", "diamond", a=5.43)
    density = compute_density([atoms])["mean"]
    result = screen_structures([atoms], {"density": {"min": density, "max": density}})
    assert result["summary"]["passed"] == 1


@pytest.mark.parametrize("value,expected", [(True, []), (False, ["unconverged"]),
                                           ("False", ["unconverged_unavailable"])])
def test_unconverged_requires_explicit_boolean(value, expected):
    atoms = Atoms("Si")
    atoms.info["relaxation_converged"] = value
    result = screen_structures([atoms], {"unconverged": True})
    assert result["per_structure"][0]["labels"] == expected


def test_generic_convergence_does_not_stand_in_for_relaxation():
    atoms = Atoms("Si")
    atoms.info["converged"] = True
    result = screen_structures([atoms], {"unconverged": {"exclude": True}})
    assert result["per_structure"][0]["labels"] == ["unconverged_unavailable"]
    assert result["summary"]["excluded"] == 1


def test_missing_and_nonfinite_measurements_are_json_safe_unavailable():
    atoms = Atoms("Si", info={"energy": np.nan})
    atoms.positions[:] = np.nan
    result = screen_structures([atoms, Atoms()], {
        "coordination": True, "crystal_like": True,
        "close_contacts": True, "density": {"min": 1},
        "energy": {"max": 0}, "unconverged": True}, cutoff=2)
    json.dumps(result, allow_nan=False)
    for frame in result["per_structure"]:
        assert not frame["passed"]
        assert all(s["status"] == "unavailable" for s in frame["screens"].values())
    assert result["summary"]["labelled"] == 2


def test_exports_preserve_sources_and_overlapping_population_counts(tmp_path):
    frames = [Atoms("Si", info={"relaxation_converged": flag}) for flag in [True, False]]
    result = screen_structures(frames, {"unconverged": True}, source_names=["a.xyz", "b.xyz"])
    mark_screening_analysed(result, result["retained_indices"])
    files = write_screening_outputs(result, tmp_path / "screen")
    with open(files["json"]) as handle:
        assert json.load(handle) == result
    with open(files["structures"], newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["source"] == "b.xyz"
    assert rows[1]["labelled"] == rows[1]["analysed"] == "True"
    assert rows[1]["excluded"] == "False"
    with open(files["summary"], newline="") as handle:
        summary = next(csv.DictReader(handle))
    assert summary == {"generated": "2", "passed": "1", "labelled": "1",
                       "analysed": "2", "excluded": "0"}
    report = format_screening_report(result)
    assert all(word in report for word in ("generated", "passed", "labelled", "analysed"))


@pytest.mark.parametrize("config", [
    "all", {"typo": True}, {"energy": True}, {"density": {}},
    {"unconverged": {"exclude": "false"}},
    {"coordination": {"allowed": {"Si": [True]}}},
    {"coordination": {"allowed": {"Si": []}}},
    {"coordination": {"allowed": {"Invalid": [4]}}},
    {"coordination": {"max_fraction": 2}},
    {"crystal_like": {"qbar6_threshold": np.nan}},
    {"crystal_like": {"min_neighbors": 2.5}},
    {"crystal_like": {"cutoff": -1}},
    {"crystal_like": {"cutoff": True}},
    {"crystal_like": {"cutoff": {"default": True}}},
    {"crystal_like": {"cutoff": {"Si-Si": True}}},
    {"close_contacts": {"min_distance": 0}},
    {"close_contacts": {"min_distance": {"Si-O": 1, "O-Si": 2}}},
    {"close_contacts": {"min_distance": {"Si-Zz": 1}}},
    {"energy": {"min": 2, "max": 1}},
    {"density": {"min": -1}},
])
def test_invalid_config_fails_before_evaluation(config):
    with pytest.raises(ValueError):
        screen_structures([], config)


def test_config_defaults_disabled_screens_empty_ensembles_and_validation():
    config = validate_screening_config(True)
    assert set(config) == {"coordination", "crystal_like", "close_contacts", "unconverged"}
    assert all(not screen["exclude"] for screen in config.values())
    assert validate_screening_config({"density": {"enabled": False}}) == {}
    assert screen_structures([], True)["summary"]["generated"] == 0
    atoms = Atoms("Si")
    with pytest.raises(ValueError, match="source_names"):
        screen_structures([atoms], False, source_names=[])
    result = screen_structures([atoms], False)
    assert result["summary"]["passed"] == 1
    for indices in ([True], [-1], [1]):
        with pytest.raises(ValueError, match="indices"):
            mark_screening_analysed(result, indices)


def test_complete_cutoff_table_does_not_recompute_rdf(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("resolved cutoffs should not recompute RDF")

    monkeypatch.setattr("amorphgen.analysis.cutoff.auto_cutoff_rdf", fail)
    si = bulk("Si", "diamond", a=5.43)
    result = screen_structures([si], {"coordination": True, "crystal_like": True}, cutoff={"Si-Si": 2.6})
    assert result["per_structure"][0]["screens"]["coordination"]["status"] == "passed"


def test_direct_api_resolves_cutoffs_in_deterministic_order(monkeypatch):
    import amorphgen.analysis.screening as screening

    first = bulk("Si", "diamond", a=5.43)
    second = first.copy()
    second.positions[0, 0] = 0.1
    orders = []

    def resolve(frames, cutoff):
        orders.append([id(atoms) for atoms in frames])
        return 2.6

    monkeypatch.setattr(screening, "_resolve_cutoff", resolve)
    screening.screen_structures([first, second], {"coordination": True})
    screening.screen_structures([second, first], {"coordination": True})
    assert orders[0] == orders[1]
