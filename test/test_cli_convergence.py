"""CLI declarations, YAML precedence, and convergence report exports."""

import json

import pytest
import yaml
from ase import Atoms
from ase.io import write

from amorphgen_test_helpers import run_cli

from amorphgen.cli import _convergence_options, _get_parser, _requires_calculator
from amorphgen.configs import load_yaml_config


@pytest.fixture
def ensemble_file(tmp_path):
    source = tmp_path / "silica.xyz"
    structures = [Atoms("SiO2", positions=[[0, 0, 0], [1.5 + i * .05, 0, 0],
                                         [0, 1.5 + i * .05, 0]],
                        cell=[7 + i * .2] * 3, pbc=True)
                  for i in range(3)]
    write(source, structures)
    return source


def test_cli_declarations_override_yaml_per_descriptor():
    args = _get_parser().parse_args([
        "--analyse", "input.xyz", "--tolerance", "density=0.02",
        "--tolerance", "rdf.total=0.1", "--convergence-confidence", "0.9",
        "--convergence-max-structures", "2000"])
    settings = _convergence_options(args, {
        "convergence": False, "tolerances": {"density": .5, "coordination.Si-O": .2},
        "convergence_confidence": .8, "convergence_max_structures": 50})
    assert settings == (True, {"density": .02, "coordination.Si-O": .2,
                              "rdf.total": .1}, .9, 2000)
    assert not _requires_calculator(args)


@pytest.mark.parametrize("value", ["density", "=1", "density=oops", "density=0",
                                   "density=-1", "density=nan", "density=inf"])
def test_invalid_cli_tolerances_are_parser_errors(value, capsys):
    with pytest.raises(SystemExit) as exc:
        _get_parser().parse_args(["--analyse", "input.xyz", "--tolerance", value])
    assert exc.value.code == 2
    assert "tolerance" in capsys.readouterr().err


@pytest.mark.parametrize("analysis", [
    {"tolerances": {"density": 0}},
    {"tolerances": {"density": -1}},
    {"tolerances": {"density": float("nan")}},
    {"tolerances": {"density": float("inf")}},
    {"tolerances": {"density": True}},
    {"tolerances": {"density": "small"}},
    {"tolerances": {1: .1}},
    {"tolerances": {"": .1}},
    {"tolerances": []},
    {"convergence_confidence": 0},
    {"convergence_confidence": 1},
    {"convergence_confidence": float("nan")},
    {"convergence_confidence": True},
    {"convergence_max_structures": 1},
    {"convergence_max_structures": True},
    {"convergence_max_structures": 10.5},
])
def test_invalid_yaml_convergence_settings(tmp_path, analysis):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"analysis": analysis}))
    with pytest.raises(ValueError, match="analysis"):
        load_yaml_config(path)


@pytest.mark.parametrize("flags", [
    ["--convergence-confidence", "nan"],
    ["--convergence-confidence", "1"],
    ["--convergence-max-structures", "0"],
])
def test_invalid_cli_settings_fail_before_loading_structures(monkeypatch, flags, capsys):
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, ["--analyse", "missing.xyz", "--convergence", *flags])
    assert exc.value.code == 1
    assert "Error: convergence analysis:" in capsys.readouterr().out


def test_convergence_exports_and_selected_optional_descriptors(
        tmp_path, monkeypatch, ensemble_file, capsys):
    import amorphgen.utils
    from amorphgen.analysis import StructureAnalyser
    from amorphgen.analysis import descriptors

    def forbidden(**kwargs):
        raise AssertionError("Convergence analysis must not construct an MLIP")

    monkeypatch.setattr(amorphgen.utils, "get_calculator", forbidden)
    # Existing descriptor plotting has its own coverage; exercise the new
    # convergence exporters here without writing unrelated plot collections.
    monkeypatch.setattr(StructureAnalyser, "plot", lambda self, **kwargs: None)
    monkeypatch.setattr(descriptors, "save_descriptor", lambda *args, **kwargs: None)
    output = tmp_path / "plots"
    report = tmp_path / "report.txt"
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"analysis": {
        "tolerances": {"density": .5, "coordination.Si-O": .2,
                       "voids.accessible_fraction": .1},
        "convergence_confidence": .8, "convergence_max_structures": 500,
        "voids": True, "void_samples": 20,
        "save_plot": str(output), "save_report": str(report),
    }}))
    run_cli(monkeypatch, ["--analyse", ensemble_file, "--config", config,
                          "--cutoff", "1.8", "--tolerance", "density=0.001",
                          "--tolerance", "rdf.total=0.1",
                          "--tolerance", "angle_distribution.O-Si-O=0.01",
                          "--convergence-confidence", ".9",
                          "--convergence-max-structures", "2000",
                          "--save-pdf", "--dpi", "50"])
    for filename in ("analysis_convergence.json", "analysis_convergence.txt",
                     "analysis_convergence_summary.csv", "analysis_convergence_curves.csv",
                     "analysis_convergence_density.png", "analysis_convergence_density.pdf"):
        assert (output / filename).stat().st_size > 0
    result = json.loads((output / "analysis_convergence.json").read_text())
    assert result["confidence"] == .9
    assert result["max_structures"] == 2000
    assert result["descriptors"]["density"]["tolerance"] == .001
    assert result["descriptors"]["coordination.Si-O"]["tolerance"] == .2
    assert result["descriptors"]["voids.accessible_fraction"]["tolerance"] == .1
    assert result["descriptors"]["rdf.total"]["tolerance"] == .1
    assert result["descriptors"]["angle_distribution.O-Si-O"]["tolerance"] == .01
    assert "convergence" in report.read_text().lower()
    assert "convergence" in capsys.readouterr().out.lower()


@pytest.mark.parametrize("name", ["densitty", "rdf.foo", "rdf.Bogus-O",
                                  "angle_distribution.Si-O", "angle_distribution.Foo-O-Si"])
def test_unknown_descriptor_is_an_explicit_cli_error(monkeypatch, ensemble_file, capsys, name):
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, ["--analyse", ensemble_file, "--cutoff", "1.8",
                              "--tolerance", f"{name}=.01"])
    assert exc.value.code == 1
    text = capsys.readouterr().out
    assert "Error: convergence analysis:" in text
    assert name in text


def test_selected_dimers_and_missing_energy_are_reported(
        tmp_path, monkeypatch, ensemble_file, capsys):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"analysis": {
        "check_dimers": True, "energy_ranking": True,
        "tolerances": {"dimers.count": .1, "energy.per_atom": .01},
    }}))
    run_cli(monkeypatch, ["--analyse", ensemble_file, "--cutoff", "1.8",
                          "--config", config])
    output = capsys.readouterr().out
    assert "dimers.count" in output
    assert "energy.per_atom" in output
    assert "No energy data" in output


@pytest.mark.parametrize("use_yaml", [False, True])
def test_convergence_without_tolerances_reports_uncertainty(
        tmp_path, monkeypatch, ensemble_file, capsys, use_yaml):
    args = ["--analyse", ensemble_file, "--cutoff", "1.8"]
    if use_yaml:
        config = tmp_path / "config.yaml"
        config.write_text("analysis:\n  convergence: true\n")
        args += ["--config", config]
    else:
        args += ["--convergence"]
    run_cli(monkeypatch, args)
    output = capsys.readouterr().out
    assert "convergence" in output.lower()
    assert "density" in output.lower()
