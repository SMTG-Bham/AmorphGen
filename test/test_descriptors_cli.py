"""Optional descriptor dispatch, calculator gating and exported artifacts."""

import json

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.build import bulk
from ase.io import write
from scipy.integrate import trapezoid

from amorphgen_test_helpers import run_cli

from amorphgen.cli import _get_parser, _requires_calculator


def test_structural_descriptors_need_no_calculator_and_cli_overrides_yaml(tmp_path, monkeypatch):
    import amorphgen.utils

    def forbidden(**kwargs):
        raise AssertionError("Geometry analysis must not construct an MLIP")

    monkeypatch.setattr(amorphgen.utils, "get_calculator", forbidden)
    atoms = Atoms("SiO2", positions=[[0, 0, 0], [1.6, 0, 0], [0, 1.6, 0]],
                  cell=[8, 8, 8], pbc=True)
    source = tmp_path / "silica.xyz"
    write(source, atoms)
    out = tmp_path / "plots"
    report = tmp_path / "report.txt"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"analysis": {
        "voids": True, "void_samples": 20, "void_seed": 12,
        "void_radii": {"O": 1.2}, "oxygen_speciation": True,
        "network_formers": ["Si"], "save_report": str(report), "save_plot": str(out),
    }}))
    run_cli(monkeypatch, ["--analyse", source, "--config", cfg, "--cutoff", "1.8",
                          "--void-samples", "32", "--void-seed", "0", "--save-pdf", "--dpi", "50"])
    void = json.loads((out / "analysis_voids.json").read_text())
    oxygen = json.loads((out / "analysis_oxygen_speciation.json").read_text())
    assert void["n_samples"] == 32
    assert void["seed"] == 0
    assert void["radii"]["O"] == 1.2
    assert oxygen["counts"]["non_bridging"] == 2
    assert oxygen["structure_files"] == [str(source)]
    for descriptor in ("voids", "oxygen_speciation"):
        for suffix in ("json", "csv", "png", "pdf"):
            assert (out / f"analysis_{descriptor}.{suffix}").stat().st_size > 0
    assert "Void distribution" in report.read_text()
    assert "Oxygen speciation" in report.read_text()


def test_elastic_and_vdos_end_to_end_with_live_ase_calculator(tmp_path, monkeypatch):
    import amorphgen.utils
    from ase.calculators.lj import LennardJones

    # Exercise all force/stress calculations without downloading an MLIP.
    # AmorphGen's own classical factory currently provides no stress, so use
    # ASE's stress-capable LJ through the shared calculator factory interface.
    requests = []

    def calculator_factory(**kwargs):
        requests.append(kwargs)
        return LennardJones(epsilon=0.0104, sigma=3.4, rc=8.0)

    monkeypatch.setattr(amorphgen.utils, "get_calculator", calculator_factory)
    source = tmp_path / "argon.xyz"
    write(source, bulk("Ar", "fcc", a=5.4, cubic=True))
    out = tmp_path / "plots"
    report = tmp_path / "report.txt"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({
        "model": "lj", "device": "cpu",
        "classical_params": {"params": {"Ar-Ar": {"epsilon": 0.0104, "sigma": 3.4}},
                             "cutoff": 8.0},
        "analysis": {"elastic": True, "vdos": True, "vdos_npoints": 100,
                     "vdos_sigma": 0.2, "save_plot": str(out), "save_report": str(report)},
    }))
    run_cli(monkeypatch, ["--analyse", source, "--config", cfg, "--cutoff", "4.0",
                          "--vdos-npoints", "150", "--elastic-strain", "0.002", "--dpi", "50"])
    elastic = json.loads((out / "analysis_elastic.json").read_text())
    vdos = json.loads((out / "analysis_vdos.json").read_text())
    assert elastic["strain"] == 0.002
    assert np.asarray(elastic["per_structure"][0]["stiffness_tensor_gpa"]).shape == (6, 6)
    assert elastic["ensemble"]["moduli"]["hill"]["bulk_modulus_gpa"]["mean"] > 0
    assert vdos["total_modes"] == 12
    assert len(requests) == 1
    assert requests[0]["model"] == "lj"
    assert requests[0]["device"] == "cpu"
    assert requests[0]["classical_params"]["params"]["Ar-Ar"]["sigma"] == 3.4
    assert requests[0]["classical_params"]["cutoff"] == 8.0
    assert len(vdos["frequencies_thz"]) == 150
    assert trapezoid(vdos["dos"], vdos["frequencies_thz"]) == pytest.approx(1)
    for descriptor in ("elastic", "vdos"):
        for suffix in ("json", "csv", "png"):
            assert (out / f"analysis_{descriptor}.{suffix}").stat().st_size > 0
    assert (out / "analysis_elastic_tensor.csv").exists()
    assert "Elastic moduli" in report.read_text()
    assert "Vibrational density" in report.read_text()


@pytest.mark.parametrize("flag", ["--elastic", "--vdos"])
def test_calculator_gate_includes_cli_and_yaml(flag):
    parser = _get_parser()
    args = parser.parse_args(["--analyse", "input.xyz", flag])
    assert _requires_calculator(args)
    args = parser.parse_args(["--analyse", "input.xyz"])
    assert _requires_calculator(args, {flag[2:]: True})
    assert not _requires_calculator(args, {"voids": True, "oxygen_speciation": True})


def test_descriptor_validation_is_a_cli_error(tmp_path, monkeypatch, capsys):
    source = tmp_path / "argon.xyz"
    write(source, bulk("Ar", "fcc", a=5.4, cubic=True))
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, ["--analyse", source, "--cutoff", "4", "--voids", "--void-samples", "0"])
    assert exc.value.code == 1
    assert "n_samples" in capsys.readouterr().out


@pytest.mark.parametrize("enable_in_yaml", [False, True])
def test_bond_order_exports_and_cli_precedence(tmp_path, monkeypatch, enable_in_yaml):
    import csv
    import amorphgen.utils

    def forbidden(**kwargs):
        raise AssertionError("Bond order must not construct a calculator")

    monkeypatch.setattr(amorphgen.utils, "get_calculator", forbidden)
    source = tmp_path / "copper.xyz"
    write(source, bulk("Cu", cubic=True))
    out = tmp_path / "plots"
    report = tmp_path / "report.txt"
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"analysis": {
        "bond_order": enable_in_yaml, "qbar6_threshold": 0.9,
        "order_min_neighbors": 20, "cutoff": 2.8,
        "save_plot": str(out), "save_report": str(report),
    }}))
    flags = [] if enable_in_yaml else ["--bond-order"]
    run_cli(monkeypatch, ["--analyse", source, "--config", config, *flags,
                          "--qbar6-threshold", "0.4", "--order-min-neighbors", "6",
                          "--save-pdf", "--dpi", "50"])
    result = json.loads((out / "analysis_bond_order.json").read_text())
    assert result["parameters"] == {
        "cutoff": 2.8, "qbar6_threshold": 0.4, "min_neighbors": 6}
    row = result["per_structure"][0]
    assert row["q6"] == pytest.approx([np.sqrt(169 / 512)] * 4)
    assert row["qbar6"] == pytest.approx(row["q6"])
    assert row["ordered_fraction"] == 1
    assert row["largest_cluster_size"] == 4
    assert result["structure_files"] == [str(source)]
    for suffix in ("json", "csv", "png", "pdf"):
        assert (out / f"analysis_bond_order.{suffix}").stat().st_size > 0
    with (out / "analysis_bond_order_atoms.csv").open() as handle:
        atoms = list(csv.DictReader(handle))
    assert len(atoms) == 4
    assert [int(atom["neighbor_count"]) for atom in atoms] == [12] * 4
    assert all(atom["ordered"] == "1" for atom in atoms)
    assert "largest cluster" in report.read_text()
    args = _get_parser().parse_args(["--analyse", str(source), "--bond-order"])
    assert not _requires_calculator(args)


@pytest.mark.parametrize("option,value", [("--qbar6-threshold", "nan"),
                                         ("--order-min-neighbors", "0")])
def test_bond_order_invalid_settings_are_cli_errors(tmp_path, monkeypatch, capsys,
                                                  option, value):
    source = tmp_path / "copper.xyz"
    write(source, bulk("Cu", cubic=True))
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, ["--analyse", source, "--cutoff", "2.8",
                              "--bond-order", option, value])
    assert exc.value.code == 1
    assert "Error: descriptor analysis:" in capsys.readouterr().out


@pytest.mark.parametrize("use_cli", [False, True])
def test_gete_order_shell_is_independent_of_chemical_shell(tmp_path, monkeypatch, use_cli):
    source = tmp_path / "gete.xyz"
    write(source, bulk("GeTe", "rocksalt", a=6.0, cubic=True))
    out = tmp_path / "plots"
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"analysis": {
        "bond_order": True, "cutoff": 4.5,
        "order_cutoff": 4.5 if use_cli else 3.5,
        "save_plot": str(out),
    }}))
    flags = ["--order-cutoff", "3.5"] if use_cli else []
    run_cli(monkeypatch, ["--analyse", source, "--config", config, *flags, "--dpi", "50"])
    result = json.loads((out / "analysis_bond_order.json").read_text())
    row = result["per_structure"][0]
    assert result["parameters"]["cutoff"] == 3.5
    assert row["neighbor_counts"] == [6] * 8
    assert row["qbar6"] == pytest.approx([np.sqrt(1 / 8)] * 8)
    assert row["largest_cluster_size"] == 8
