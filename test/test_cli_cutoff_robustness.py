"""CLI and YAML control the same cutoff window in reports and exports."""

import sys

import pytest
import yaml
from ase import Atoms
from ase.io import write

from amorphgen.analysis import StructureAnalyser
from amorphgen.cli import _get_parser, _requires_calculator, main
from amorphgen.configs import load_yaml_config


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "silica.xyz"
    write(path, Atoms("SiO2", positions=[[0, 0, 0], [1.99, 0, 0], [0, 2.05, 0]],
                     cell=[10] * 3, pbc=True))
    return path


def test_cutoff_window_is_available_without_calculator():
    args = _get_parser().parse_args(["--analyse", "source.xyz", "--cutoff-window", ".2"])
    assert args.cutoff_window == .2
    assert not _requires_calculator(args)


@pytest.mark.parametrize("yaml_window,cli_window,expected,per_structure", [
    (None, None, .1, False),
    (.2, None, .2, False),
    (.2, .3, .3, False),
    (.2, .3, .3, True),
])
def test_width_precedence_reaches_saved_report_and_plot_exports(
        tmp_path, monkeypatch, source, yaml_window, cli_window, expected, per_structure):
    report_path = tmp_path / "report.txt"
    plot_dir = tmp_path / "plots"
    received = []
    monkeypatch.setattr(StructureAnalyser, "plot", lambda self, **kwargs: received.append(kwargs))
    args = ["amorphgen", "--analyse", str(source), "--cutoff", "2",
            "--save-report", str(report_path), "--save-plot", str(plot_dir)]
    if yaml_window is not None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({"analysis": {"cutoff_window": yaml_window}}))
        args.extend(["--config", str(config_path)])
    if cli_window is not None:
        args.extend(["--cutoff-window", str(cli_window)])
    if per_structure:
        args.append("--per-structure")
    monkeypatch.setattr(sys, "argv", args)
    main()
    assert f"Cutoff robustness (+/- {expected:.3f} A)" in report_path.read_text()
    assert received[0]["cutoff_window"] == expected


@pytest.mark.parametrize("value", ["0", "-.1", "nan", "inf"])
def test_invalid_cli_window_fails_before_structure_loading(monkeypatch, value, capsys):
    monkeypatch.setattr(sys, "argv", ["amorphgen", "--analyse", "missing.xyz",
                                      "--cutoff-window", value])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "Error: cutoff robustness:" in capsys.readouterr().out


@pytest.mark.parametrize("value", [0, -.1, float("nan"), float("inf"), True, "small"])
def test_invalid_yaml_window_is_rejected(tmp_path, value):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"analysis": {"cutoff_window": value}}))
    with pytest.raises(ValueError, match="cutoff_window"):
        load_yaml_config(path)
