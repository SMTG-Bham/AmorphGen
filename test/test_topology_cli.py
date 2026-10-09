"""Public void/ring controls, YAML precedence and complete geometry-only exports."""

import csv
import json

import pytest
import yaml
from ase.build import bulk
from ase.io import write

from amorphgen_test_helpers import run_cli

from amorphgen.analysis import StructureAnalyser
from amorphgen.configs.yaml_config import _validate_config


@pytest.mark.parametrize("use_cli", [False, True])
def test_topology_controls_and_exports(tmp_path, monkeypatch, use_cli):
    import amorphgen.utils

    def forbidden(**kwargs):
        raise AssertionError("Topology analysis must not construct a calculator")

    monkeypatch.setattr(amorphgen.utils, "get_calculator", forbidden)
    source = tmp_path / "diamond.xyz"
    write(source, bulk("Si", "diamond", a=5.43, cubic=True))
    output = tmp_path / "plots"
    report = tmp_path / "report.txt"
    config = tmp_path / "analysis.yaml"
    config.write_text(yaml.safe_dump({"analysis": {
        "rings": "Si-Si", "ring_max_size": 5, "ring_cutoff": 2.6,
        "voids": True, "void_samples": 32, "void_probe_radius": 0.25,
        "void_probe_radii": [0, 0.25, 0.5],
        "save_plot": str(output), "save_report": str(report),
    }}))
    arguments = ["--analyse", source, "--config", config, "--cutoff", "2.7", "--dpi", "40"]
    if use_cli:
        arguments += ["--ring-max-size", "6", "--ring-cutoff", "2.5",
                      "--void-probe-radii", "1", "0.25", "0", "1"]
    run_cli(monkeypatch, arguments)
    rings = json.loads((output / "analysis_rings.json").read_text())
    voids = json.loads((output / "analysis_voids.json").read_text())
    assert rings["max_ring"] == (6 if use_cli else 5)
    assert rings["cutoff"] == (2.5 if use_cli else 2.6)
    assert rings["n_network_edges"] == 16
    assert rings["n_ring_edges"] == (16 if use_cli else 0)
    assert rings["n_unresolved_edges"] == (0 if use_cli else 16)
    assert rings["mean_ring_size"] == (6 if use_cli else None)
    assert rings["structure_files"] == [str(source)]
    assert voids["probe_curve"]["radii"] == ([0, 0.25, 1] if use_cli else [0, 0.25, 0.5])
    assert voids["probe_curve"]["accessible_fraction"][1] == pytest.approx(
        voids["accessible_fraction"])
    for stem in ("rings_structures", "rings_per_structure", "voids_structures", "voids_probe"):
        assert (output / f"analysis_{stem}.csv").stat().st_size > 0
    assert (output / "analysis_voids_probe.png").stat().st_size > 0
    with (output / "analysis_rings.csv").open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["ring_size", "count", "fraction_percent"]
    assert len(rows) == (2 if use_cli else 1)
    assert "Ring statistics" in report.read_text()
    assert "Void distribution" in report.read_text()


@pytest.mark.parametrize("flag,value,match", [
    ("--ring-max-size", "2", "max_ring"),
    ("--ring-cutoff", "0", "cutoff"),
    ("--rings", "Si-O-X", "bond_pair"),
    ("--void-probe-radii", "-1", "probe_radii"),
])
def test_topology_bad_cli_parameters_are_friendly_errors(tmp_path, monkeypatch, capsys,
                                                       flag, value, match):
    source = tmp_path / "diamond.xyz"
    write(source, bulk("Si", "diamond", a=5.43, cubic=True))
    with pytest.raises(SystemExit) as error:
        run_cli(monkeypatch, ["--analyse", source, "--cutoff", "2.6", "--rings",
                              "--voids", "--void-samples", "8", flag, value])
    assert error.value.code == 1
    assert match in capsys.readouterr().out


@pytest.mark.parametrize("key,value", [
    ("ring_max_size", 2), ("ring_max_size", True), ("ring_max_size", 4.5),
    ("ring_cutoff", 0), ("ring_cutoff", float("nan")), ("ring_cutoff", True),
    ("void_probe_radii", []), ("void_probe_radii", [True]),
    ("void_probe_radii", [-1]), ("void_probe_radii", [float("inf")]),
    ("void_probe_radii", [[0, 1]]), ("void_probe_radii", "0 1"),
])
def test_invalid_topology_yaml_is_rejected(key, value):
    warnings, errors = _validate_config({"analysis": {key: value}}, "test.yaml")
    assert warnings == []
    assert any(f"analysis.{key}" in error for error in errors)


def test_topology_yaml_defaults_and_supported_values():
    assert _validate_config({"analysis": {
        "ring_max_size": 16, "ring_cutoff": None, "void_probe_radii": None,
    }}, "test.yaml") == ([], [])


def test_void_api_forwards_probe_grid():
    sa = StructureAnalyser([bulk("Si", "diamond", a=5.43, cubic=True)], cutoff=2.6)
    result = sa.void_distribution(n_samples=32, probe_radii=[1, 0, 1])
    assert result["probe_curve"]["radii"] == [0, 1]
