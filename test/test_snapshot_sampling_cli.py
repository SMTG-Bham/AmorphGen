"""Snapshot diagnostics reach CLI workflows without changing quench identities."""

import json
from pathlib import Path
import sys

import pytest
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from amorphgen import cli
from amorphgen.utils.common import extract_snapshots


def _trajectory(path, count=16):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(count):
        atoms = bulk("Cu", cubic=True)
        atoms.set_momenta([[0.0, 0.0, 0.0]] * len(atoms))
        atoms.calc = SinglePointCalculator(atoms, energy=-float(i % 3))
        frames.append(atoms)
    write(path, frames)
    return frames


@pytest.fixture
def mq_run(tmp_path, monkeypatch):
    from amorphgen.pipeline import batch_quench, run_pipeline
    import amorphgen.utils

    work = tmp_path / "mq"
    original = tmp_path / "crystal.xyz"
    write(original, bulk("Cu", cubic=True))
    trajectory = work / "shared" / "stage4_eq_traj.xyz"
    _trajectory(trajectory)
    calls = []

    class SharedPipeline:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, stages, resume):
            assert stages == [1, 2, 3, 4]

    def quench(**kwargs):
        report = json.loads((work / "snapshot_sampling.json").read_text())
        assert (work / "snapshot_sampling.txt").exists()
        assert len(kwargs["snapshot_files"]) == report["selected_snapshots"]
        calls.append(kwargs)
        directory = Path(kwargs["work_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        write(directory / "final_amorphous.xyz", read(kwargs["snapshot_files"][0]))

    monkeypatch.setattr(run_pipeline, "MeltQuenchPipeline", SharedPipeline)
    monkeypatch.setattr(amorphgen.utils, "get_calculator", lambda **kwargs: object())
    monkeypatch.setattr(batch_quench, "run", quench)
    monkeypatch.setattr(cli, "_collect_ensemble_final", lambda *args, **kwargs: None)
    args = cli._get_parser().parse_args([
        str(original), "--mq-ensemble", "--resume", "-n", "4", "-o", str(work),
    ])
    return args, work, trajectory, calls


def test_mq_automatic_sampling_reports_before_quench_and_resumes(mq_run):
    args, work, trajectory, calls = mq_run
    override = {"eq_high": {"timestep": 1.25}}
    cli._run_mq_ensemble(args, override)
    report = json.loads((work / "snapshot_sampling.json").read_text())
    assert report["method"] == "decorrelated"
    assert report["burn_in_frames"] > 0
    assert report["frame_interval_ps"] == 0.125
    assert report["selected_frame_indices"] == [15]
    assert report["effective_independent_snapshots"] == 1
    assert len(report["selected_frame_fingerprints"]) == 1
    text = (work / "snapshot_sampling.txt").read_text()
    assert "Autocorrelation energy_per_atom" in text
    assert "Diffusion Cu" in text
    assert "Effective independent snapshots" in text
    assert "Limitation:" in text
    cli._run_mq_ensemble(args, override)
    assert len(calls) == 2
    assert calls[0]["snapshot_files"] == calls[1]["snapshot_files"]


@pytest.mark.parametrize("change", ["positions", "energy", "momenta", "settings"])
def test_resume_rejects_changed_sources_before_overwriting(mq_run, change):
    args, work, trajectory, calls = mq_run
    cli._run_mq_ensemble(args, {})
    report_before = (work / "snapshot_sampling.json").read_bytes()
    snapshot = Path(calls[0]["snapshot_files"][0])
    snapshot_before = snapshot.read_bytes()
    frames = read(trajectory, index=":")
    if change == "positions":
        frames[-1].positions[0, 0] += 0.1
    elif change == "energy":
        frames[-1].calc = SinglePointCalculator(frames[-1], energy=123.0)
    elif change == "momenta":
        frames[-1].set_momenta([[1.0, 0.0, 0.0]] * len(frames[-1]))
    else:
        args.select = "last"
    write(trajectory, frames)
    with pytest.raises(ValueError, match="Snapshot selection or source frames changed"):
        cli._run_mq_ensemble(args, {})
    assert len(calls) == 1
    assert (work / "snapshot_sampling.json").read_bytes() == report_before
    assert snapshot.read_bytes() == snapshot_before


def test_resume_legacy_quenches_requires_explicit_legacy_selection(mq_run):
    args, work, trajectory, calls = mq_run
    directory = work / "quench_runs"
    directory.mkdir()
    write(directory / "final_amorphous.xyz", bulk("Cu", cubic=True))
    with pytest.raises(ValueError, match="snapshot_sampling.json is missing"):
        cli._run_mq_ensemble(args, {})
    assert not calls
    args.select = "last"
    args.n_structures = 1
    cli._run_mq_ensemble(args, {})
    assert len(calls) == 1
    report = json.loads((work / "snapshot_sampling.json").read_text())
    assert report["selected_frame_indices"] == [15]
    assert report["burn_in_frames"] == 0


def test_resume_detects_custom_named_interrupted_quench(mq_run):
    args, work, trajectory, calls = mq_run
    directory = work / "quench_runs"
    directory.mkdir()
    write(directory / "cooling.xyz", bulk("Cu", cubic=True))
    with pytest.raises(ValueError, match="snapshot_sampling.json is missing"):
        cli._run_mq_ensemble(args, {"quench": {"traj_file": "cooling.xyz"}})
    assert not calls


@pytest.mark.parametrize("mq,expected_resume", [(True, False), (False, True)])
def test_absolute_stage4_output_only_counts_when_stage4_is_downstream(
        mq_run, mq, expected_resume):
    args, work, trajectory, calls = mq_run
    args.batch_stages = [4, 5, 6, 7]
    kwargs = cli._snapshot_sampling_kwargs(
        args, {"eq_high": {"traj_file": str(trajectory.resolve())}}, mq=mq,
        report_path=str(work / "snapshot_sampling.json"),
        quench_dir=str(work / "quench_runs"),
    )
    assert kwargs["resume"] is expected_resume


def test_plain_extraction_preserves_uniform_default(tmp_path):
    trajectory = tmp_path / "trajectory.xyz"
    _trajectory(trajectory)
    output = tmp_path / "snapshots"
    paths = extract_snapshots(trajectory, n_snapshots=3, output_dir=output)
    assert [Path(path).name for path in paths] == [
        "snapshot_0000_frame00000.xyz", "snapshot_0001_frame00007.xyz",
        "snapshot_0002_frame00015.xyz",
    ]
    assert not (output / "snapshot_sampling.json").exists()


def test_extract_cli_decorrelated_uses_configured_stage_timestep(tmp_path, monkeypatch):
    trajectory = tmp_path / "trajectory.xyz"
    _trajectory(trajectory)
    config = tmp_path / "config.yaml"
    config.write_text("eq_high:\n  timestep: 1.25\n")
    output = tmp_path / "snapshots"
    monkeypatch.setattr(sys, "argv", [
        "amorphgen", "--extract-snapshots", str(trajectory), "--select", "decorrelated",
        "--config", str(config), "-n", "4", "-o", str(output),
    ])
    cli.main()
    report = json.loads((output / "snapshot_sampling.json").read_text())
    assert report["frame_interval_ps"] == 0.125
    assert report["selected_frame_indices"] == [15]


def test_batch_decorrelated_extraction_excludes_stale_snapshots(tmp_path, monkeypatch):
    from amorphgen.pipeline import batch_quench
    import amorphgen.utils

    trajectory = tmp_path / "trajectory.xyz"
    _trajectory(trajectory)
    output = tmp_path / "batch"
    stale = output / "snapshots_extracted" / "snapshot_9999_stale.xyz"
    stale.parent.mkdir(parents=True)
    write(stale, bulk("Cu", cubic=True))
    monkeypatch.setattr(cli, "_requires_calculator", lambda *args: False)
    monkeypatch.setattr(amorphgen.utils, "get_calculator", lambda **kwargs: object())
    calls = []
    monkeypatch.setattr(batch_quench, "run", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", [
        "amorphgen", "--batch-quench", "--snapshot-dir", str(trajectory),
        "--select", "decorrelated", "--n-runs", "4", "-o", str(output),
    ])
    cli.main()
    assert len(calls[0]["snapshot_files"]) == 1
    assert calls[0]["select"] == "uniform"
    assert "9999" not in calls[0]["snapshot_files"][0]
    assert (output / "snapshot_sampling.json").exists()


def test_batch_decorrelated_requires_trajectory_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_requires_calculator", lambda *args: False)
    monkeypatch.setattr(sys, "argv", [
        "amorphgen", "--batch-quench", "--snapshot-dir", str(tmp_path),
        "--select", "decorrelated", "-o", str(tmp_path / "output"),
    ])
    with pytest.raises(ValueError, match="requires a trajectory file"):
        cli.main()
