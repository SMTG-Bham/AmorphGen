"""Public scattering APIs and complete calculator-free CLI workflows."""

import json

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.io import write

from amorphgen_test_helpers import run_cli as _run

from amorphgen.analysis import StructureAnalyser
from amorphgen.cli import _get_parser, _requires_calculator


@pytest.fixture
def ensemble(tmp_path):
    directory = tmp_path / "structures"
    directory.mkdir()
    first = Atoms("Si4", positions=[[1, 1, 1], [2, 2, 2], [4, 3, 5], [6, 5, 4]],
                  cell=[8, 8, 8], pbc=True)
    second = first.copy()
    second.positions[0, 0] += 0.25
    for index, atoms in enumerate((first, second)):
        write(directory / f"structure_{index}.xyz", atoms)
    return directory, StructureAnalyser(str(directory), cutoff=2.5)


def test_public_comparison_wrapper_and_defaults(ensemble, tmp_path):
    _, sa = ensemble
    sq = sa.structure_factor(qmax=4, nq=30, weighting="xray", n_bootstrap=0)
    measured = tmp_path / "measured.csv"
    np.savetxt(measured, np.column_stack([sq["q"], sq["s_q"]]),
               delimiter=",", header="q,S(q)", comments="")
    result = sa.compare_experiment(
        measured, kind="S(q)", method="ft", load_options={"skiprows": 1},
        calculation_options={"qmax": 4, "nq": 30}, n_bootstrap=0)
    assert result["metrics"]["rmse"] == pytest.approx(0, abs=1e-14)
    assert result["metrics"]["chi_square"] is None
    assert result["uncertainty"]["n_per_point"] == [2] * 30
    assert result["calculation"]["rmax"] == 4
    assert result["calculation"]["weighting"] == "xray"
    assert result["calculation"]["normalization"] == "Faber-Ziman"
    assert result["experiment_metadata"]["skiprows"] == 1
    json.dumps(result, allow_nan=False)


def test_public_tr_wrapper_has_transform_metadata(ensemble):
    _, sa = ensemble
    options = {"qmin": 0.3, "qmax": 4, "nq": 40, "nr": 30,
               "rmax": 5, "sigma_q": 0, "window": "none"}
    curve = sa.total_correlation(**options, n_bootstrap=0)
    measured = {"kind": "tr", "x": curve["r"], "observed": curve["T_r"]}
    result = sa.compare_experiment(measured, calculation_options=options, n_bootstrap=0)
    assert result["metrics"]["rmse"] == pytest.approx(0, abs=1e-14)
    for key, value in options.items():
        assert result["calculation"][key] == value


@pytest.mark.parametrize("data, options, message", [
    ([], {}, "path or a mapping"),
    ({"kind": "tr"}, {"method": "ft"}, "direct"),
    ({"kind": "sq"}, {"calculation_options": {"seed": 4}}, "pass seed directly"),
])
def test_wrapper_rejects_invalid_requests(ensemble, data, options, message):
    with pytest.raises(ValueError, match=message):
        ensemble[1].compare_experiment(data, **options)


def test_experiment_sq_cli_exports_and_overrides_yaml(ensemble, tmp_path, monkeypatch):
    directory, sa = ensemble
    curve = sa.structure_factor(qmax=4, nq=30, weighting="neutron", n_bootstrap=0)
    measured = tmp_path / "sq.csv"
    # Include extra columns to exercise selected coordinate/value/sigma indices.
    np.savetxt(measured, np.column_stack([
        np.zeros(30), curve["q"], curve["s_q"], np.full(30, 0.1)]),
        delimiter=",", header="unused,q,s,sigma", comments="")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"analysis": {
        "experiment_sq": str(tmp_path / "wrong.dat"),
        "sq_method": "direct", "sq_qmax": 8, "sq_nq": 50,
        "sq_weighting": "xray", "experiment_skiprows": 0,
    }}))
    out, report = tmp_path / "plots", tmp_path / "fit.txt"
    _run(monkeypatch, ["--analyse", "--input-dir", directory, "--config", config,
                      "--cutoff", "2.5", "--experiment-sq", measured,
                      "--experiment-skiprows", "1", "--experiment-columns", "1", "2", "3",
                      "--sq-method", "ft", "--sq-weighting", "neutron", "--sq-qmax", "4",
                      "--sq-nq", "30", "--sq-fit-range", "1", "3",
                      "--save-plot", out, "--save-report", report, "--dpi", "40"])
    result = json.loads((out / "analysis_experiment_sq.json").read_text())
    assert result["metrics"]["rmse"] == pytest.approx(0, abs=1e-14)
    assert result["metrics"]["chi_square"] == pytest.approx(0, abs=1e-14)
    assert result["overlap"]["n_excluded_x_range"] > 0
    assert result["calculation"]["method"] == "ft"
    assert result["calculation"]["weighting"] == "neutron"
    assert result["calculation"]["qmax"] == 4
    assert len(result["structure_files"]) == 2
    assert "Experimental S(q) comparison" in report.read_text()
    for suffix in ("json", "csv", "txt", "png"):
        assert (out / f"analysis_experiment_sq.{suffix}").stat().st_size > 0


def test_experiment_tr_and_xrd_yaml_workflow(ensemble, tmp_path, monkeypatch):
    directory, sa = ensemble
    curve = sa.total_correlation(qmin=0.3, qmax=4, window="none", n_bootstrap=0)
    measured = tmp_path / "tr.dat"
    np.savetxt(measured, np.column_stack([curve["r"], curve["T_r"]]))
    out = tmp_path / "out"
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"analysis": {
        "experiment_tr": str(measured), "tr_qrange": [0.3, 4], "tr_window": "none",
        "tr_fit_range": [1, 5], "xrd": True, "xrd_qmax": 4, "xrd_nq": 30,
        "xrd_wavelength": 1.0, "save_plot": str(out), "dpi": 40,
    }}))
    _run(monkeypatch, ["--analyse", "--input-dir", directory, "--config", config,
                      "--cutoff", "2.5", "--xrd-wavelength", "1.5406"])
    comparison = json.loads((out / "analysis_experiment_tr.json").read_text())
    assert comparison["metrics"]["rmse"] == pytest.approx(0, abs=1e-14)
    assert comparison["metrics"]["chi_square"] is None
    xrd = json.loads((out / "analysis_xrd.json").read_text())
    assert xrd["wavelength"] == 1.5406
    assert len(xrd["q"]) == 30
    assert xrd["uncertainty"]["n_structures"] == 2
    assert len(xrd["structure_files"]) == 2


def test_cli_rejects_malformed_experiment(ensemble, tmp_path, monkeypatch, capsys):
    path = tmp_path / "bad.dat"
    path.write_text("1 2 0\n2 3 1\n")
    with pytest.raises(SystemExit) as error:
        _run(monkeypatch, ["--analyse", "--input-dir", ensemble[0], "--cutoff", "2.5",
                          "--experiment-sq", path, "--sq-qmax", "4", "--sq-nq", "30"])
    assert error.value.code == 1
    assert "experimental sigma" in capsys.readouterr().out


def test_scattering_options_need_no_calculator():
    args = _get_parser().parse_args(["--analyse", "structures/", "--xrd",
                                    "--experiment-sq", "sq.dat", "--experiment-tr", "tr.dat"])
    assert not _requires_calculator(args)
