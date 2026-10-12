"""MD stage outputs contain convergence evidence from the saved samples."""

import json

import numpy as np
import pytest
from ase.calculators.emt import EMT
from ase.io import read

from amorphgen.pipeline import equilibrate, melt_cell, quench
from amorphgen.utils import equilibration, md_diagnostics


def _fixed_config(stage="high", steps=20, timestep=.5, traj_format="extxyz"):
    return {
        "seed": 4,
        "device": "cpu",
        "traj_format": traj_format,
        f"eq_{stage}": {
            "ensemble": "NVT", "T": 300, "steps": steps,
            "timestep": timestep, "friction": .01, "make_cubic": False,
        },
    }


def _load_report(directory, stem):
    """Reject Python's otherwise accepted non-JSON NaN and Infinity values."""
    def reject_constant(value):
        raise AssertionError(f"Non-finite JSON literal: {value}")

    path = directory / f"{stem}_diagnostics.json"
    report = json.loads(path.read_text(), parse_constant=reject_constant)
    assert (directory / f"{stem}_diagnostics.txt").read_text().strip() == report["summary_text"].strip()
    return report


@pytest.mark.parametrize("stage,number", [("premelt", 2), ("high", 4), ("low", 6)])
def test_short_fixed_stages_report_insufficient_evidence(cu_bulk, tmp_path, stage, number):
    equilibrate.run(cu_bulk, _fixed_config(stage), EMT(), stage=stage, work_dir=tmp_path)

    report = _load_report(tmp_path, f"stage{number}_eq")
    assert report["stage"] == number
    assert report["engine"] == "ase"
    assert report["status"] == "insufficient_data"
    assert report["n_frames"] == 1
    assert report["total_time_ps"] == 0
    assert report["liquid_test"]["is_liquid"] is None
    assert report["freezing_temperature_K"] is None
    assert report["warnings"]
    assert not list(tmp_path.glob("*.png"))


def test_fixed_stage_runs_existing_checks_with_exact_saved_times(cu_bulk, tmp_path, monkeypatch):
    called = {"compute_msd": 0, "block_average_test": 0, "convergence_report": 0}
    report_kwargs = []

    def track(name, original):
        def wrapper(*args, **kwargs):
            called[name] += 1
            if name == "convergence_report":
                report_kwargs.append(kwargs)
            return original(*args, **kwargs)
        return wrapper

    for name in ("compute_msd", "block_average_test"):
        monkeypatch.setattr(equilibration, name, track(name, getattr(equilibration, name)))
    monkeypatch.setattr(md_diagnostics, "convergence_report",
                        track("convergence_report", md_diagnostics.convergence_report))

    # This timestep cannot be recovered exactly from the rounded log time.
    timestep = .123456
    equilibrate.run(cu_bulk, _fixed_config(steps=800, timestep=timestep), EMT(),
                    work_dir=tmp_path)

    report = _load_report(tmp_path, "stage4_eq")
    expected_steps = np.arange(0, 801, 100)
    expected_times = expected_steps * timestep / 1000
    assert all(called.values()), called
    assert report_kwargs[0]["make_plots"] is False
    np.testing.assert_allclose(report_kwargs[0]["time_ps"], expected_times, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(report["sample_steps"], expected_steps)
    np.testing.assert_allclose(report["sample_time_ps"], expected_times, rtol=0, atol=1e-15)
    assert report["total_time_ps"] == pytest.approx(expected_times[-1], abs=1e-15)
    assert report["n_frames"] == 9
    assert report["frame_stride"] == 100
    assert report["source"] == {
        "log": str(tmp_path / "stage4_eq.log"),
        "trajectory": str(tmp_path / "stage4_eq_traj.xyz"),
    }
    assert "energy" in report["stationarity"]
    assert "diffusion" in report["stationarity"]
    assert report["msd_final_A2"]["all"] >= 0
    log = equilibration.parse_md_log(tmp_path / "stage4_eq.log")
    assert np.isfinite(log["P_GPa"]).all()
    assert (log["density_g_cm3"] > 0).all()


@pytest.mark.parametrize("module,section,number,targets", [
    (melt_cell, "melt", 3, [400, 500, 600]),
    (quench, "quench", 5, [500, 400, 300]),
])
def test_ramp_reports_resolved_temperature_holds(cu_bulk, tmp_path, monkeypatch,
                                                module, section, number, targets):
    calls = []
    original = md_diagnostics.convergence_report

    def capture(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(md_diagnostics, "convergence_report", capture)
    cfg = {
        "seed": 4, "device": "cpu", "traj_format": "extxyz",
        section: {
            "ensemble": "NVT", "T_start": 300 if number == 3 else 600,
            "T_end": targets[-1], "T_step": 100 if number == 3 else -100,
            "steps_per_T": 300, "timestep": .25, "friction": .01,
            "make_cubic": False,
        },
    }
    module.run(cu_bulk, cfg, EMT(), work_dir=tmp_path)

    report = _load_report(tmp_path, f"stage{number}_{section}")
    assert report["stage"] == number
    # A boundary frame belongs to the hold just completed; step zero belongs
    # to the first hold. This preserves the schedule's partial endpoints.
    np.testing.assert_array_equal(calls[0]["temperatures"],
                                  [targets[0]] * 4 + [targets[1]] * 3 + [targets[2]] * 3)
    assert len(report["temperature_windows"]) == 3
    assert [window["temperature_K"] for window in report["temperature_windows"]] == targets
    assert all(window["status"] == "insufficient_data" for window in report["temperature_windows"])
    assert report["n_frames"] == 10
    assert report["total_time_ps"] == pytest.approx(.225)
    assert report["freezing_temperature_K"] is None
    assert not list(tmp_path.glob("*.png"))


def test_resumed_stage_rewrites_report_for_full_saved_history(cu_bulk, tmp_path):
    equilibrate.run(cu_bulk, _fixed_config(steps=200), EMT(), work_dir=tmp_path)
    first = _load_report(tmp_path, "stage4_eq")
    assert first["n_frames"] == 3

    equilibrate.run(cu_bulk, _fixed_config(steps=400), EMT(), work_dir=tmp_path, resume=True)

    resumed = _load_report(tmp_path, "stage4_eq")
    assert resumed["n_frames"] == 5
    assert resumed["sample_steps"] == [0, 100, 200, 300, 400]
    assert resumed["total_time_ps"] == pytest.approx(.2)
    assert len(read(tmp_path / "stage4_eq_traj.xyz", index=":")) == 5


def test_configured_binary_trajectory_is_read_despite_xyz_suffix(cu_bulk, tmp_path):
    equilibrate.run(cu_bulk, _fixed_config(steps=200, traj_format="traj"), EMT(),
                    work_dir=tmp_path)

    report = _load_report(tmp_path, "stage4_eq")
    assert report["status"] != "unavailable"
    assert report["n_frames"] == 3
    assert report["sample_steps"] == [0, 100, 200]
    assert len(read(tmp_path / "stage4_eq_traj.xyz", index=":", format="traj")) == 3


def test_unavailable_evidence_is_saved_without_a_liquid_claim(tmp_path):
    report = md_diagnostics.write_stage_diagnostics(
        tmp_path / "missing.log", tmp_path / "missing.xyz", timestep_fs=.5, stage=4)

    saved = _load_report(tmp_path, "missing")
    assert saved == report
    assert saved["status"] == "unavailable"
    assert saved["warnings"]
    assert saved.get("freezing_temperature_K") is None


def test_log_trajectory_mismatch_cannot_use_guessed_times(cu_bulk, tmp_path):
    equilibrate.run(cu_bulk, _fixed_config(), EMT(), work_dir=tmp_path)
    with (tmp_path / "stage4_eq.log").open("a") as log:
        log.write("100 0.05 300 0 0 0 40 0 8\n")

    md_diagnostics.write_stage_diagnostics(
        tmp_path / "stage4_eq.log", tmp_path / "stage4_eq_traj.xyz",
        timestep_fs=.5, stage=4)

    report = _load_report(tmp_path, "stage4_eq")
    assert report["status"] == "unavailable"
    assert any("sample counts differ" in message for message in report["warnings"])


def test_batched_writer_diagnostics_keep_partial_final_block_times(cu_bulk, tmp_path):
    from amorphgen.utils.torchsim_md import _RunWriter

    cu_bulk.calc = EMT()
    writer = _RunWriter(tmp_path, "batch.log", "batch.xyz")
    steps = [100, 200, 300, 400, 500, 600, 700, 750]
    for step in steps:
        writer.write(cu_bulk, step, .5)
    md_diagnostics.write_stage_diagnostics(
        tmp_path / "batch.log", tmp_path / "batch.xyz",
        timestep_fs=.5, stage=6, engine="torchsim", T_target=300)

    report = _load_report(tmp_path, "batch")
    assert report["engine"] == "torchsim"
    assert report["sample_steps"] == steps
    np.testing.assert_allclose(report["sample_time_ps"], np.array(steps) * .0005)
    assert report["total_time_ps"] == pytest.approx(.325)
    assert report["diffusion_coefficients_cm2_s"]["all"] == pytest.approx(0, abs=1e-15)
    assert report["liquid_test"]["is_liquid"] is False


def test_duplicate_log_steps_cannot_create_a_valid_diffusion_report(cu_bulk, tmp_path):
    from amorphgen.utils.torchsim_md import _RunWriter

    cu_bulk.calc = EMT()
    writer = _RunWriter(tmp_path, "duplicate.log", "duplicate.xyz")
    for step in [100, 100]:
        writer.write(cu_bulk, step, .5)
    md_diagnostics.write_stage_diagnostics(
        tmp_path / "duplicate.log", tmp_path / "duplicate.xyz",
        timestep_fs=.5, stage=4, engine="torchsim")

    report = _load_report(tmp_path, "duplicate")
    assert report["status"] == "unavailable"
    assert any("strictly increasing" in warning for warning in report["warnings"])
