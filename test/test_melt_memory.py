"""Crystal-order retention uses original identities and melt endpoints."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.io import write

from amorphgen import cli
from amorphgen.analysis import melt_memory as memory


@pytest.fixture
def ordered_frames():
    # Two disconnected octahedral shells: only each six-coordinated central
    # atom passes min_neighbors=4. Distancing a shell removes its order.
    octahedron = np.vstack(([0, 0, 0], np.eye(3), -np.eye(3)))
    initial = Atoms("Cu14", positions=np.vstack((octahedron, octahedron + [100, 0, 0])))
    partial = initial.copy()
    partial.positions[8:] = [100, 0, 0] + 10 * octahedron[1:]
    none = partial.copy()
    none.positions[1:7] = 10 * octahedron[1:]
    return initial, partial, none


def test_full_partial_and_no_initial_order_survival(ordered_frames):
    initial, partial, none = ordered_frames
    report = memory.compute_melt_memory(
        initial, {"full": initial, "partial": partial, "none": none}, cutoff=1.1)
    assert report["initial"]["ordered_indices"] == [0, 7]
    full, partial_row, absent = report["comparisons"]
    assert [row["survival_fraction"] for row in report["comparisons"]] == [1, 0.5, 0]
    assert [row["initial_ordered_count"] for row in report["comparisons"]] == [2, 2, 2]
    assert partial_row["retained_ordered_indices"] == [0]
    assert partial_row["ordered_fraction"] == pytest.approx(1 / 14)
    assert full["largest_cluster_size"] == 1
    assert absent["retained_ordered_indices"] == []
    assert "does not establish continuous" in report["interpretation"]


def test_no_initial_order_has_undefined_survival_but_current_order(ordered_frames):
    initial, _, none = ordered_frames
    report = memory.compute_melt_memory(none, {"ordered_later": initial}, cutoff=1.1)
    row = report["comparisons"][0]
    assert row["status"] == "unavailable"
    assert row["survival_fraction"] is None
    assert row["initial_ordered_count"] == 0
    assert row["newly_ordered_count"] == 2
    assert row["ordered_count"] == 2
    text = memory.format_melt_memory(report)
    assert "survival unavailable" in text
    assert "largest ordered cluster 1/14" in text
    assert "--qbar6-threshold" in text


def test_identity_validation_never_truncates_or_sorts(ordered_frames):
    initial, _, _ = ordered_frames
    with pytest.raises(ValueError, match="Atom count differs"):
        memory.compute_melt_memory(initial, {"fewer": initial[:-1]}, cutoff=1.1)
    binary = initial.copy()
    binary.symbols[0] = "Ge"
    reordered = binary[list(range(1, len(binary))) + [0]]
    assert sorted(reordered.get_chemical_symbols()) == sorted(binary.get_chemical_symbols())
    with pytest.raises(ValueError, match="Element sequence differs"):
        memory.compute_melt_memory(binary, {"reordered": reordered}, cutoff=1.1)


def test_fixed_reference_cutoff_and_current_geometry(monkeypatch):
    initial = bulk("Cu", cubic=True)
    expanded = initial.copy()
    expanded.set_cell(initial.cell * 4, scale_atoms=True)
    nonperiodic = initial.copy()
    nonperiodic.pbc = False
    calls = []
    real_compute = memory.compute_bond_order

    def record(atoms, **kwargs):
        calls.append(kwargs)
        return real_compute(atoms, **kwargs)

    monkeypatch.setattr(memory, "compute_bond_order", record)
    result = memory.compute_melt_memory(initial, {"expanded": expanded, "isolated": nonperiodic})
    assert calls[0]["cutoff"] == "auto-rdf"
    assert all(call["cutoff"] == result["parameters"]["cutoff"] for call in calls[1:])
    assert result["initial"]["ordered_count"] == len(initial)
    assert result["comparisons"][0]["survival_fraction"] == 0
    assert result["comparisons"][1]["survival_fraction"] == 0


def test_gete_starting_crystal_survival_with_calibrated_first_shell_cutoff():
    crystal = bulk("GeTe", "rocksalt", a=6.0, cubic=True)
    expanded = crystal.copy()
    expanded.set_cell(crystal.cell * 3, scale_atoms=True)
    report = memory.compute_melt_memory(
        crystal, {"unchanged": crystal, "expanded": expanded}, cutoff=3.5)
    assert report["initial"]["ordered_count"] == len(crystal)
    assert report["initial"]["qbar6_mean"] == pytest.approx(np.sqrt(1 / 8))
    assert [row["survival_fraction"] for row in report["comparisons"]] == [1, 0]
    assert report["comparisons"][0]["largest_cluster_size"] == len(crystal)


def test_report_uses_stage_endpoints_and_legacy_last_frame(tmp_path, ordered_frames):
    initial, partial, none = ordered_frames
    input_file = tmp_path / "crystal.xyz"
    write(input_file, initial)
    shared = tmp_path / "shared"
    shared.mkdir()
    write(shared / "stage3_melted.xyz", partial)
    # Stage trajectories can have later/earlier order and must not replace
    # the checkpoint; old stage4_eq.xyz trajectories use their last frame.
    write(shared / "stage3_melt_traj.xyz", initial)
    write(shared / "stage4_eq.xyz", [initial, none])
    write(shared / "stage4_eq_traj.xyz", initial)
    snapshot = tmp_path / "snapshot_0000.xyz"
    write(snapshot, initial)
    report = memory.report_melt_memory(input_file, shared, [snapshot], tmp_path, cutoff=1.1)
    assert [row["kind"] for row in report["comparisons"]] == ["stage3", "stage4", "snapshot"]
    assert [row["survival_fraction"] for row in report["comparisons"]] == [0.5, 0, 1]
    loaded = json.loads((tmp_path / "melt_memory.json").read_text())
    assert loaded == report
    with (tmp_path / "melt_memory.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["survival_fraction"]) for row in rows] == [0.5, 0, 1]
    assert "50.0% (1/2)" in (tmp_path / "melt_memory.txt").read_text()


def test_missing_resume_checkpoint_and_bad_identity_are_unavailable(tmp_path, ordered_frames):
    initial, _, _ = ordered_frames
    input_file = tmp_path / "crystal.xyz"
    write(input_file, initial)
    # A resumed legacy run may have a trajectory but no stage3 checkpoint.
    write(tmp_path / "stage3_melt_traj.xyz", initial)
    write(tmp_path / "stage4_eq.xyz", initial[:-1])
    report = memory.report_melt_memory(input_file, tmp_path, [], tmp_path, cutoff=1.1)
    assert len(report["comparisons"]) == 2
    assert all(row["survival_fraction"] is None for row in report["comparisons"])
    assert all(row["status"] == "unavailable" for row in report["comparisons"])
    assert "Atom count differs" in report["comparisons"][1]["reason"]
    assert report["comparisons"][0]["retained_ordered_count"] is None


def test_configured_checkpoint_names_and_missing_original(tmp_path, ordered_frames):
    initial, partial, _ = ordered_frames
    input_file = tmp_path / "crystal.xyz"
    write(input_file, initial)
    write(tmp_path / "custom_melt.xyz", partial)
    write(tmp_path / "custom_eq.xyz", initial)
    config = {"melt": {"output_xyz": "custom_melt.xyz"},
              "eq_high": {"output_xyz": "custom_eq.xyz"}}
    report = memory.report_melt_memory(input_file, tmp_path, [], tmp_path,
                                       cutoff=1.1, cfg_override=config)
    assert [row["survival_fraction"] for row in report["comparisons"]] == [0.5, 1]
    report = memory.report_melt_memory(tmp_path / "missing.xyz", tmp_path, [], tmp_path)
    assert report["initial"]["status"] == "unavailable"
    assert all(row["survival_fraction"] is None for row in report["comparisons"])
    assert report["parameters"] is None


@pytest.mark.parametrize("options", [
    {"cutoff": -1}, {"qbar6_threshold": 1.1}, {"min_neighbors": 0},
])
def test_preflight_rejects_invalid_settings_even_without_input(tmp_path, options):
    with pytest.raises(ValueError):
        memory.prepare_melt_memory(tmp_path / "missing.xyz", **options)


def test_prepared_reference_is_reused_after_original_file_changes(tmp_path, ordered_frames):
    initial, partial, none = ordered_frames
    original = tmp_path / "crystal.xyz"
    write(original, initial)
    prepared = memory.prepare_melt_memory(original, cutoff=1.1)
    write(original, none)
    write(tmp_path / "stage3_melted.xyz", partial)
    write(tmp_path / "stage4_eq.xyz", none)
    report = memory.report_melt_memory(original, tmp_path, [], tmp_path, prepared=prepared)
    assert report["initial"]["ordered_indices"] == [0, 7]
    assert report["parameters"]["cutoff"] == 1.1
    assert [row["survival_fraction"] for row in report["comparisons"]] == [0.5, 0]


def test_mq_reports_before_quenching_and_excludes_stale_snapshots(tmp_path, monkeypatch, ordered_frames):
    from amorphgen.pipeline import batch_quench, run_pipeline
    import amorphgen.utils

    initial, partial, none = ordered_frames
    original = tmp_path / "crystal.xyz"
    write(original, initial)
    work = tmp_path / "ensemble"
    snapshots = work / "snapshots"
    snapshots.mkdir(parents=True)
    write(snapshots / "snapshot_9999_stale.xyz", initial)
    events = []

    class SharedPipeline:
        def __init__(self, input_file, work_dir, cfg_override):
            self.work_dir = Path(work_dir)

        def run(self, stages, resume):
            assert stages == [1, 2, 3, 4]
            assert resume
            events.append("shared")
            self.work_dir.mkdir(parents=True, exist_ok=True)
            write(self.work_dir / "stage3_melted.xyz", partial)
            write(self.work_dir / "stage4_eq.xyz", none)
            write(self.work_dir / "stage4_eq_traj.xyz", [initial, partial, none])

    def fake_quench(**kwargs):
        events.append("quench")
        report = json.loads((work / "melt_memory.json").read_text())
        assert [row["survival_fraction"] for row in report["comparisons"]] == [0.5, 0, 0]
        assert len(kwargs["snapshot_files"]) == 1
        assert "9999" not in kwargs["snapshot_files"][0]

    monkeypatch.setattr(run_pipeline, "MeltQuenchPipeline", SharedPipeline)
    monkeypatch.setattr(amorphgen.utils, "get_calculator", lambda **kwargs: object())
    monkeypatch.setattr(batch_quench, "run", fake_quench)
    monkeypatch.setattr(cli, "_collect_ensemble_final", lambda *args, **kwargs: None)
    args = cli._get_parser().parse_args([
        str(original), "--mq-ensemble", "--resume", "--select", "last",
        "--cutoff", "1.1", "-o", str(work),
    ])
    cli._run_mq_ensemble(args, {})
    assert events == ["shared", "quench"]


def test_mq_saves_melt_endpoints_when_snapshot_burn_in_is_invalid(tmp_path, monkeypatch, ordered_frames):
    from amorphgen.pipeline import run_pipeline

    initial, partial, none = ordered_frames
    original = tmp_path / "crystal.xyz"
    write(original, initial)
    work = tmp_path / "ensemble"

    class SharedPipeline:
        def __init__(self, input_file, work_dir, cfg_override):
            self.work_dir = Path(work_dir)

        def run(self, stages, resume):
            self.work_dir.mkdir(parents=True, exist_ok=True)
            write(self.work_dir / "stage3_melted.xyz", partial)
            write(self.work_dir / "stage4_eq.xyz", none)
            write(self.work_dir / "stage4_eq_traj.xyz", [initial, none])

    monkeypatch.setattr(run_pipeline, "MeltQuenchPipeline", SharedPipeline)
    args = cli._get_parser().parse_args([
        str(original), "--mq-ensemble", "--burn-in-frames", "2", "-o", str(work),
    ])
    with pytest.raises(ValueError, match=r"burn_in_frames \(2\).*trajectory length \(2\)"):
        cli._run_mq_ensemble(args, {"analysis": {"cutoff": 1.1}})
    report = json.loads((work / "melt_memory.json").read_text())
    assert report["parameters"]["cutoff"] == 1.1
    assert [row["kind"] for row in report["comparisons"]] == ["stage3", "stage4"]
    assert [row["survival_fraction"] for row in report["comparisons"]] == [0.5, 0]
    assert (work / "melt_memory.csv").is_file()
    assert (work / "melt_memory.txt").is_file()
    assert not (work / "quench_runs").exists()
