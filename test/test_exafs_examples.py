"""Reproducibility contracts for the retained-input EXAFS examples."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
from ase.build import bulk
from ase.io import write


def load_study_module():
    path = Path(__file__).resolve().parents[1] / "examples/exafs/ga2o3_validation.py"
    spec = importlib.util.spec_from_file_location("ga2o3_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_study_retains_inputs_matches_sites_and_rejects_changes(tmp_path):
    study = load_study_module()
    inputs = tmp_path / "structures"
    inputs.mkdir()
    for n in range(2):
        write(inputs / f"cell{n}.xyz", bulk("Ga", "fcc", a=4+n*.1, cubic=True))
    experiment = tmp_path / "experiment.dat"
    np.savetxt(experiment, np.c_[np.arange(1, 14, .05), np.ones(260)])
    args = SimpleNamespace(output=tmp_path / "study", structures=inputs, pattern="*.xyz",
                           experiment=experiment, radii=[3, 4], max_sites=3, seed=0, rpath=6)
    study.prepare(args)
    saved = json.loads((args.output / "study.json").read_text())
    assert len(saved["tasks"]) == 18
    assert saved["site_ranks"]["r3"] == saved["site_ranks"]["r4"]
    assert (args.output / "inputs/ensemble.extxyz").exists()
    assert study.sha(args.output / "inputs/cell0.xyz") == study.sha(inputs / "cell0.xyz")
    study.prepare(args)  # unchanged study is idempotent
    args.radii = [3, 5]
    with pytest.raises(ValueError, match="changed"):
        study.prepare(args)


def test_study_defaults_are_the_validated_protocol(tmp_path):
    study = load_study_module()
    defaults = study.build_parser().parse_args(["prepare", "--output", str(tmp_path / "x")])
    assert (defaults.rpath, defaults.nleg, defaults.criteria, defaults.stoichiometry) == \
        (8, 4, "1.0 0.5", "structure")
    assert defaults.radii == [8, 10, 12]
    inputs = tmp_path / "structures"
    inputs.mkdir()
    for n in range(2):
        write(inputs / f"cell{n}.xyz", bulk("Ga", "fcc", a=4+n*.1, cubic=True))
    experiment = tmp_path / "experiment.dat"
    np.savetxt(experiment, np.c_[np.arange(1, 14, .05), np.ones(260)])
    args = SimpleNamespace(output=tmp_path / "study", structures=inputs, pattern="*.xyz",
                           experiment=experiment, radii=[3, 4], max_sites=3, seed=0, rpath=3,
                           nleg=4, criteria="1.0 0.5", stoichiometry="structure")
    study.prepare(args)
    saved = json.loads((args.output / "study.json").read_text())
    assert saved["config"]["criteria"] == "1.0 0.5"
    assert saved["config"]["stoichiometry"] == "structure"
    protocol = next((args.output / "r3").glob("*/feff.inp")).read_text()
    legacy = next((args.output / "legacy_60").glob("*/feff.inp")).read_text()
    assert "CRITERIA  1.0 0.5" in protocol and "RPATH     3" in protocol
    assert "CRITERIA" not in legacy and "RPATH     6.0" in legacy
    manifest = json.loads((args.output / "r3/manifest.json").read_text())
    assert manifest["stoichiometry"] == "structure"
    assert json.loads((args.output / "legacy_60/manifest.json").read_text())["stoichiometry"] == "cluster"


def test_spectral_fit_recovers_known_adjustments_without_clamping():
    study = load_study_module()
    k = np.arange(.05, 15.01, .05)
    chi = np.sin(4*k)*np.exp(-.03*k)/(1+k*k)
    kd = np.arange(2.5, 12.01, .05)
    expected = [2.0, 4.0, .002, .3]
    scale, de0, sigma2, offset = expected
    y = scale*kd**2*np.interp(np.sqrt(kd**2-de0*.2625), k, chi)*np.exp(-2*sigma2*kd**2)+offset
    fit, values = study.fit_curve(k, chi, np.c_[kd, y])
    assert [fit["scale"], fit["delta_E0_eV"], fit["extra_sigma2_A2"], fit["offset"]] == pytest.approx(expected, abs=1e-5)
    assert fit["normalized_squared_residual"] < 1e-10
    assert values.shape[1] == 4
    with pytest.raises(ValueError, match="coverage"):
        study.fit_curve(k[k>=3], chi[k>=3], np.c_[kd, y])


def test_convergence_metric_uses_unfitted_weighted_spectra():
    study = load_study_module()
    k = np.arange(.05, 14.01, .05)
    chi = np.sin(k)
    assert study.relative_rms(k, 1.1*chi, chi) == pytest.approx(.1)
    assert study.relative_rms(k, chi, chi) == 0
