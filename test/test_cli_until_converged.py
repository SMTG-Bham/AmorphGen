"""Native sequential generation dispatch and predeclared CLI/YAML settings."""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import yaml

from amorphgen.cli import _get_parser, _until_convergence_options, main
from amorphgen.configs import load_yaml_config


def _arguments():
    return ["--random-gen", "--relax", "--engine", "torchsim", "--until-converged",
            "--composition", "Si=4", "--seed", "42", "--tolerance", "density=.1",
            "--descriptor-bounds", "density=0,10"]


def _parse(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["amorphgen", *map(str, arguments)])
    return _get_parser().parse_args()


def _run(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["amorphgen", *map(str, arguments)])
    return main()


@pytest.fixture
def controller(monkeypatch):
    runner = Mock(return_value={"status": "converged", "n_structures": 32})
    module = ModuleType("amorphgen.pipeline.until_converged")
    module.run_until_converged = runner
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr("amorphgen.utils.calculators.require_backend", lambda *a, **kw: None)
    monkeypatch.setattr("amorphgen.utils.calculators.require_dtype", lambda *a, **kw: None)
    return runner


def test_resolves_bounded_defaults_without_changing_analysis_projection(monkeypatch):
    args = _parse(monkeypatch, _arguments())
    result = _until_convergence_options(args, {"engine": "torchsim"})
    assert result == {
        "targets": {"density": {"bounds": [0.0, 10.0], "tolerance": .1, "components": 1}},
        "batch_size": 8, "min_structures": 2, "max_structures": 1000,
        "confidence": .95, "cutoff": None,
    }


def test_cli_overrides_yaml_per_target_and_sampling_control(monkeypatch):
    arguments = _arguments() + ["--convergence-batch-size", "12",
        "--convergence-min-structures", "24", "--convergence-max-structures", "120",
        "--convergence-confidence", ".9", "--cutoff", "2.5"]
    args = _parse(monkeypatch, arguments)
    result = _until_convergence_options(args, {
        "engine": "torchsim", "random_gen": {"convergence_batch_size": 3,
                                               "convergence_min_structures": 9},
        "analysis": {"tolerances": {"density": .3, "coordination.Si-Si": .2},
                     "descriptor_bounds": {"density": [1, 20], "coordination.Si-Si": [0, 20]},
                     "convergence_max_structures": 999, "convergence_confidence": .8,
                     "cutoff": 3.0}})
    assert result["targets"] == {
        "density": {"bounds": [0., 10.], "tolerance": .1, "components": 1},
        "coordination.Si-Si": {"bounds": [0, 20], "tolerance": .2, "components": 1},
    }
    assert [result[key] for key in ("batch_size", "min_structures", "max_structures")] == [12, 24, 120]
    assert result["confidence"] == .9
    assert result["cutoff"] == 2.5


@pytest.mark.parametrize("value", ["density", "=1,2", "density=1", "density=1,2,3",
                                     "density=2,1", "density=1,1", "density=0,nan",
                                     "density=-inf,1", "density=a,b"])
def test_invalid_descriptor_bounds_are_parser_errors(value):
    with pytest.raises(SystemExit) as exc:
        _get_parser().parse_args(["--descriptor-bounds", value])
    assert exc.value.code == 2


@pytest.mark.parametrize("extra,remove,message", [
    ([], ["--random-gen"], "--random-gen"),
    ([], ["--relax"], "--relax"),
    (["--engine", "ase"], [], "--engine torchsim"),
    ([], ["--seed", "42"], "--seed"),
    (["--seed", "-1"], [], "--seed"),
    (["--indices", "1-2"], [], "--indices"),
    (["-n", "5"], [], "--n-structures"),
    (["--format", "vasp"], [], "xyz"),
    (["--convergence-batch-size", "0"], [], "batch size"),
    (["--convergence-min-structures", "1"], [], "min structures"),
    (["--convergence-min-structures", "10", "--convergence-max-structures", "4"], [], "max structures"),
    (["--convergence-confidence", "nan"], [], "confidence"),
    (["--tolerance", "densitty=.1", "--descriptor-bounds", "densitty=0,1"], [], "densitty"),
    (["--tolerance", "coordination.Si-Si=.1", "--descriptor-bounds", "coordination.Si-Si=0,30"], [], "numeric --cutoff"),
    (["--tolerance", "coordination.Si-Si=.1", "--descriptor-bounds", "coordination.Si-Si=0,30", "--cutoff", "auto"], [], "numeric --cutoff"),
    ([], ["--descriptor-bounds", "density=0,10"], "matching descriptor bounds"),
])
def test_invalid_mode_and_contract_fail_before_loading_backend(
        monkeypatch, tmp_path, extra, remove, message, capsys):
    arguments = [arg for arg in _arguments() if arg not in remove] + extra
    backend = Mock(side_effect=AssertionError("must fail before backend check"))
    monkeypatch.setattr("amorphgen.utils.calculators.require_backend", backend)
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, arguments + ["-o", str(tmp_path / "output")])
    assert exc.value.code == 1
    assert message in capsys.readouterr().out
    assert not (tmp_path / "output").exists()
    backend.assert_not_called()


def test_dispatch_preserves_generation_and_relax_settings(tmp_path, monkeypatch, controller):
    output = tmp_path / "output"
    arguments = _arguments() + ["-o", str(output), "--target-density", "2.3",
        "--density-scale", "1.2", "--no-sc", "--minsep", "Si-Si=1.8",
        "--fmax", ".04", "--opt-steps", "77", "--optimizer", "FIRE",
        "--cell-filter", "none", "--batch-size", "2", "--resume"]
    assert _run(monkeypatch, arguments) is None
    args, kwargs = controller.call_args
    assert args == ({"Si": 4}, str(output))
    assert kwargs["targets"]["density"] == {"bounds": [0., 10.], "tolerance": .1, "components": 1}
    assert kwargs["generation"]["seed"] == 42
    assert kwargs["generation"]["target_cn"] == {}
    assert kwargs["generation"]["minsep"] == {"Si-Si": 1.8}
    assert kwargs["generation"]["target_density"] == 2.3
    assert kwargs["generation"]["density_scale"] == 1.2
    assert kwargs["cfg_override"]["opt"] == {
        "fmax": .04, "max_steps": 77, "optimizer": "FIRE", "cell_filter": "none", "batch_size": 2}
    assert kwargs["resume"] is True
    # The controller owns output locking. Dispatch must not acquire a nested lock.
    assert not output.exists()


@pytest.mark.parametrize("status,expected", [("max_structures_reached", 2), ("failed", 1)])
def test_unconverged_or_failed_controller_cannot_report_cli_success(
        tmp_path, monkeypatch, controller, status, expected, capsys):
    controller.return_value = {"status": status, "n_structures": 20}
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, _arguments() + ["-o", str(tmp_path / "output")])
    assert exc.value.code == expected
    assert status in capsys.readouterr().out


def test_controller_error_is_reported_without_success(tmp_path, monkeypatch, controller, capsys):
    controller.side_effect = ValueError("bounds violated")
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, _arguments() + ["-o", str(tmp_path / "output")])
    assert exc.value.code == 1
    assert "bounds violated" in capsys.readouterr().out


def test_yaml_enables_native_until_mode_and_relaxation(tmp_path, monkeypatch, controller):
    config = tmp_path / "sequential.yaml"
    config.write_text(yaml.safe_dump({
        "engine": "torchsim", "seed": 10,
        "random_gen": {"until_converged": True, "relax": True, "composition": {"Si": 4},
                       "convergence_batch_size": 4, "convergence_min_structures": 8, "seed": 11},
        "opt": {"max_steps": 200, "fmax": .02, "cell_filter": "none"},
        "analysis": {"tolerances": {"coordination.Si-Si": .2},
                     "descriptor_bounds": {"coordination.Si-Si": [0, 30]}, "cutoff": 2.5,
                     "convergence_max_structures": 40, "convergence_confidence": .9}}))
    _run(monkeypatch, ["--random-gen", "--config", str(config), "-o", str(tmp_path / "output")])
    kwargs = controller.call_args.kwargs
    assert [kwargs[key] for key in ("batch_size", "min_structures", "max_structures")] == [4, 8, 40]
    assert kwargs["generation"]["seed"] == 11
    assert kwargs["cfg_override"]["opt"]["fmax"] == .02
    assert kwargs["cutoff"] == 2.5


@pytest.mark.parametrize("config", [
    {"random_gen": {"until_converged": 1}},
    {"random_gen": {"convergence_batch_size": 0}},
    {"random_gen": {"convergence_batch_size": True}},
    {"random_gen": {"convergence_min_structures": 1}},
    {"analysis": {"descriptor_bounds": {"density": [1, 1]}}},
    {"analysis": {"descriptor_bounds": {"density": [2, 1]}}},
    {"analysis": {"descriptor_bounds": {"density": [True, 2]}}},
    {"analysis": {"descriptor_bounds": {"density": [0, float("inf")]}}},
    {"analysis": {"descriptor_bounds": {"density": [0]}}},
    {"analysis": {"descriptor_bounds": {"density": "0,2"}}},
    {"analysis": {"descriptor_bounds": {1: [0, 2]}}},
])
def test_invalid_yaml_sequential_fields_fail_validation(tmp_path, config):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_yaml_config(path)
