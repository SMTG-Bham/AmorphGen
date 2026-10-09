"""Screen decisions remain auditable while only retained candidates are analysed."""

import csv
import json

import pytest
import yaml
from ase import Atoms
from ase.io import write

from amorphgen_test_helpers import run_cli

from amorphgen.analysis import StructureAnalyser
from amorphgen.configs import load_yaml_config


@pytest.fixture
def candidates(tmp_path):
    directory = tmp_path / "candidates"
    directory.mkdir()
    for index, energy in enumerate([-10.0, 10.0]):
        atoms = Atoms("SiO2", positions=[[0, 0, 0], [1.6, 0, 0], [0, 1.6, 0]],
                      cell=[8, 8, 8], pbc=True)
        atoms.info["potential_energy"] = energy
        write(directory / f"candidate_{index}.xyz", atoms)
    return directory


@pytest.mark.parametrize("exclude,analysed", [(False, 2), (True, 1)])
def test_labels_and_exclusions_reach_analysis_and_tables(
        tmp_path, candidates, monkeypatch, exclude, analysed):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"analysis": {
        "screening": {"energy": {"max": 0, "exclude": exclude}},
    }}))
    report_path = tmp_path / "report.txt"
    prefix = tmp_path / "audit" / "screening"
    received = []
    original = StructureAnalyser.summary

    def summary(self, **kwargs):
        received.append((len(self.atoms_list), list(self._file_list)))
        return original(self, **kwargs)

    monkeypatch.setattr(StructureAnalyser, "summary", summary)
    run_cli(monkeypatch, [
        "--analyse", "--input-dir", candidates, "--config", config, "--cutoff", "1.8",
        "--screening-output", prefix, "--save-report", report_path,
    ])
    audit = json.loads(prefix.with_suffix(".json").read_text())
    assert audit["summary"]["generated"] == 2
    assert audit["summary"]["passed"] == 1
    assert audit["summary"]["labelled"] == 1
    assert audit["summary"]["analysed"] == analysed
    assert received == [(analysed, [str(candidates / f"candidate_{i}.xyz")
                                   for i in range(analysed)])]
    assert audit["per_structure"][1]["labels"] == ["energy"]
    assert audit["per_structure"][1]["excluded"] is exclude
    assert audit["per_structure"][1]["analysed"] is (not exclude)
    with open(f"{prefix}_structures.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(key in rows[0] for key in ("source", "passed", "labels", "analysed"))
    assert "generated" in report_path.read_text().lower()
    assert "analysed" in report_path.read_text().lower()
    assert (prefix.parent / "screening_summary.csv").exists()


def test_all_excluded_writes_audit_without_attempting_empty_analysis(
        tmp_path, candidates, monkeypatch, capsys):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"analysis": {
        "screening": {"energy": {"min": 100, "exclude": True}},
    }}))

    def forbidden(*args, **kwargs):
        raise AssertionError("Empty ensemble must not be analysed")

    monkeypatch.setattr(StructureAnalyser, "summary", forbidden)
    run_cli(monkeypatch, [
        "--analyse", "--input-dir", candidates, "--config", config, "--cutoff", "1.8",
        "--work-dir", tmp_path / "out",
    ])
    audit = json.loads((tmp_path / "out" / "screening.json").read_text())
    assert audit["summary"]["generated"] == audit["summary"]["labelled"] == 2
    assert audit["summary"]["analysed"] == audit["summary"]["passed"] == 0
    assert "No structures retained" in capsys.readouterr().out


def test_failed_analysis_does_not_claim_candidates_were_analysed(
        tmp_path, candidates, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"analysis": {
        "screening": {"energy": {"max": 0}},
    }}))

    def failure(*args, **kwargs):
        raise RuntimeError("analysis failed")

    monkeypatch.setattr(StructureAnalyser, "summary", failure)
    with pytest.raises(RuntimeError, match="analysis failed"):
        run_cli(monkeypatch, [
            "--analyse", "--input-dir", candidates, "--config", config, "--cutoff", "1.8",
            "--work-dir", tmp_path / "out",
        ])
    audit = json.loads((tmp_path / "out" / "screening.json").read_text())
    assert audit["retained_indices"] == [0, 1]
    assert audit["analysed_indices"] == []
    assert audit["summary"]["analysed"] == 0


def test_api_selection_keeps_cutoffs_file_alignment_and_original_candidates(candidates):
    analyser = StructureAnalyser(str(candidates), cutoff=1.8)
    subset, report = analyser.screened({"energy": {"max": 0, "exclude": True}})
    assert subset.cutoff == analyser.cutoff
    assert subset._cutoff_mode == analyser._cutoff_mode
    assert subset._file_list == analyser._file_list[:1]
    assert len(subset.atoms_list) == 1
    assert len(analyser.atoms_list) == 2
    assert report["summary"]["analysed"] == 0
    assert "labels" not in analyser.atoms_list[1].info


def test_collected_vasp_retains_verified_relaxation_metadata(tmp_path):
    from amorphgen.cli import _collect_ensemble_final

    source = tmp_path / "quench" / "run_0000"
    source.mkdir(parents=True)
    atoms = Atoms("SiO2", positions=[[0, 0, 0], [1.6, 0, 0], [0, 1.6, 0]],
                  cell=[8, 8, 8], pbc=True)
    atoms.info["relaxation_converged"] = False
    write(source / "final_amorphous.xyz", atoms)
    output = tmp_path / "final"
    _collect_ensemble_final(str(source.parent), str(output), "vasp", "mq",
                            {"vasp": ("vasp", ".vasp")})
    result = StructureAnalyser(str(output), cutoff=1.8).screen({"unconverged": {}})
    assert result["per_structure"][0]["labels"] == ["unconverged"]


def test_screen_defaults_and_output_override(tmp_path, candidates, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"analysis": {
        "screening_output": str(tmp_path / "unused"),
    }}))
    run_cli(monkeypatch, [
        "--analyse", "--input-dir", candidates, "--config", config, "--cutoff", "1.8", "--screen",
        "--screening-output", tmp_path / "audit",
    ])
    report = json.loads((tmp_path / "audit.json").read_text())
    assert set(report["per_structure"][0]["screens"]) == {
        "coordination", "crystal_like", "close_contacts", "unconverged"}
    assert report["summary"]["analysed"] == 2
    assert not (tmp_path / "unused.json").exists()


@pytest.mark.parametrize("settings", [
    {"energy": {"max": float("nan")}},
    {"density": {"min": 3, "max": 2}},
    {"density": {"max": 2, "exclude": "yes"}},
    {"coordination": {"allowed": {"Si": [True]}}},
    {"enrgy": {"max": 0}},
])
def test_invalid_yaml_screening_rejected(tmp_path, settings):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"analysis": {"screening": settings}}))
    with pytest.raises(ValueError, match="screening"):
        load_yaml_config(config)


@pytest.mark.parametrize("arguments", [
    ["--random-gen", "--screen"],
    ["--analyse", "missing.xyz", "--screening-output", "audit"],
])
def test_screening_mode_errors_precede_expensive_work(monkeypatch, arguments):
    with pytest.raises(SystemExit) as error:
        run_cli(monkeypatch, [*arguments])
    assert error.value.code == 1
