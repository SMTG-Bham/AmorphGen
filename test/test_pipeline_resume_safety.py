"""Resume must verify protocol identity before touching saved run data."""

import json

import pytest
from ase.io import write

from amorphgen.pipeline import run_pipeline
from amorphgen.pipeline.run_pipeline import MeltQuenchPipeline
from amorphgen.utils.run_lock import run_lock


@pytest.fixture
def completed_pipeline(cu_bulk, emt_calc, tmp_path, monkeypatch):
    input_file = tmp_path / "input.xyz"
    write(input_file, cu_bulk)
    pipe = MeltQuenchPipeline(str(input_file), str(tmp_path / "run"),
                             cfg_override={"seed": 12}, calc=emt_calc)

    def stage(atoms, cfg, calc):
        write("stage1_opt.xyz", atoms)
        return atoms

    monkeypatch.setattr(run_pipeline.opt_cell, "run", stage)
    pipe.run(stages=[1])
    return pipe


def _files(directory):
    from pathlib import Path
    return {str(p): p.read_bytes() for p in Path(directory).rglob("*") if p.is_file()}


@pytest.mark.parametrize("config,key", [
    ({"seed": 13}, "config.seed"),
    ({"model": "chgnet"}, "config.model"),
    ({"default_dtype": "float32"}, "config.default_dtype"),
    ({"opt": {"fmax": 0.3}}, "config.opt.fmax"),
    ({"quench": {"steps_per_T": 100}}, "config.quench.steps_per_T"),
])
def test_changed_config_refused_without_writes(completed_pipeline, config, key):
    pipe = completed_pipeline
    before = _files(pipe.work_dir)
    for name, value in config.items():
        if isinstance(value, dict):
            pipe.cfg[name].update(value)
        else:
            pipe.cfg[name] = value
    with pytest.raises(ValueError, match=key):
        pipe.run(stages=[1], resume=True)
    assert _files(pipe.work_dir) == before
    with run_lock(pipe.work_dir):
        pass  # A rejected invocation must release ownership too.


def test_changed_stage_selection_refused(completed_pipeline):
    pipe = completed_pipeline
    before = _files(pipe.work_dir)
    with pytest.raises(ValueError, match="settings changed: stages"):
        pipe.run(stages=[7, 1], resume=True)
    assert _files(pipe.work_dir) == before


def test_changed_input_contents_refused(completed_pipeline, cu_bulk):
    pipe = completed_pipeline
    cu_bulk.positions[0, 0] += 0.2
    write(pipe.input_file, cu_bulk)
    before = _files(pipe.work_dir)
    with pytest.raises(ValueError, match="input.sha256"):
        pipe.run(stages=[1], resume=True)
    assert _files(pipe.work_dir) == before


def test_changed_injected_calculator_parameters_refused(completed_pipeline):
    pipe = completed_pipeline
    pipe._injected_calc.set(asap_cutoff=True)
    before = _files(pipe.work_dir)
    with pytest.raises(ValueError, match="calculator.parameters"):
        pipe.run(stages=[1], resume=True)
    assert _files(pipe.work_dir) == before


def test_changed_model_contents_at_same_path_refused(
        cu_bulk, emt_calc, tmp_path, monkeypatch):
    input_file = tmp_path / "input.xyz"
    write(input_file, cu_bulk)
    model = tmp_path / "custom.model"
    model.write_bytes(b"original weights")
    monkeypatch.setattr(run_pipeline, "get_calculator", lambda **kwargs: emt_calc)

    def stage(atoms, cfg, calc):
        write("stage1_opt.xyz", atoms)
        return atoms

    monkeypatch.setattr(run_pipeline.opt_cell, "run", stage)
    pipe = MeltQuenchPipeline(str(input_file), str(tmp_path / "run"),
                             {"model_path": str(model), "device": "cpu"})
    pipe.run(stages=[1])
    model.write_bytes(b"different weights")
    before = _files(pipe.work_dir)
    with pytest.raises(ValueError, match="requested_model.sha256"):
        pipe.run(stages=[1], resume=True)
    assert _files(pipe.work_dir) == before

    # Fresh execution may reuse its already loaded calculator. Its manifest
    # must not allow a new calculator loaded from the replaced file to resume.
    pipe.run(stages=[1])
    new_pipe = MeltQuenchPipeline(pipe.input_file, pipe.work_dir, pipe.cfg)
    before = _files(pipe.work_dir)
    with pytest.raises(ValueError, match="calculator.sha256"):
        new_pipe.run(stages=[1], resume=True)
    assert _files(pipe.work_dir) == before


@pytest.mark.parametrize("legacy_manifest", [False, True])
def test_legacy_outputs_require_verifiable_manifest(
        cu_bulk, emt_calc, tmp_path, legacy_manifest):
    write(tmp_path / "input.xyz", cu_bulk)
    pipe = MeltQuenchPipeline(str(tmp_path / "input.xyz"), str(tmp_path / "run"), calc=emt_calc)
    write(tmp_path / "run" / "stage1_opt.xyz", cu_bulk)
    if legacy_manifest:
        (tmp_path / "run" / "run_manifest.json").write_text(json.dumps({
            "schema_version": 1, "attempts": [{"config": pipe.cfg}],
        }))
    before = _files(pipe.work_dir)
    with pytest.raises(ValueError, match="Cannot resume: .*verifiable"):
        pipe.run(stages=[1], resume=True)
    after = _files(pipe.work_dir)
    after.pop(str(tmp_path / "run" / ".amorphgen.lock"))
    assert after == before


def test_pipeline_lock_precedes_manifest_and_calculator(completed_pipeline, monkeypatch):
    pipe = completed_pipeline
    before = _files(pipe.work_dir)
    monkeypatch.setattr(pipe, "_resume_settings", lambda *args: pytest.fail("already locked"))
    with run_lock(pipe.work_dir):
        with pytest.raises(RuntimeError, match="Another run"):
            pipe.run(stages=[1], resume=True)
    assert _files(pipe.work_dir) == before


def test_corrupt_checkpoint_reruns_stage(completed_pipeline, monkeypatch):
    pipe = completed_pipeline
    from pathlib import Path
    (Path(pipe.work_dir) / "stage1_opt.xyz").write_text("")
    calls = []

    def stage(atoms, cfg, calc):
        calls.append(1)
        write("stage1_opt.xyz", atoms)
        return atoms

    monkeypatch.setattr(run_pipeline.opt_cell, "run", stage)
    pipe.run(stages=[1], resume=True)
    assert calls == [1]


def test_stale_outputs_after_fresh_failure_are_not_skipped(completed_pipeline, monkeypatch):
    pipe = completed_pipeline

    def fail(atoms, cfg, calc):
        raise RuntimeError("interrupted fresh run")

    monkeypatch.setattr(run_pipeline.opt_cell, "run", fail)
    with pytest.raises(RuntimeError, match="interrupted fresh run"):
        pipe.run(stages=[1])
    # The old stage1_opt.xyz still exists, but the new stage did not finish.
    with pytest.raises(RuntimeError, match="interrupted fresh run"):
        pipe.run(stages=[1], resume=True)


def test_custom_checkpoint_does_not_reuse_stale_default(completed_pipeline, monkeypatch):
    pipe = completed_pipeline
    pipe.cfg["opt"]["output_xyz"] = "custom.xyz"

    def stage(atoms, cfg, calc):
        atoms.positions[0, 0] += 0.75
        write("custom.xyz", atoms)
        return atoms

    monkeypatch.setattr(run_pipeline.opt_cell, "run", stage)
    expected = pipe.run(stages=[1])
    monkeypatch.setattr(pipe, "_get_calc", lambda: pytest.fail("completed stage"))
    actual = pipe.run(stages=[1], resume=True)
    assert actual.positions == pytest.approx(expected.positions)


def test_fresh_stage_failure_does_not_adopt_old_frames(cu_bulk, emt_calc, tmp_path, monkeypatch):
    write(tmp_path / "input.xyz", cu_bulk)
    pipe = MeltQuenchPipeline(str(tmp_path / "input.xyz"), str(tmp_path / "run"), calc=emt_calc)
    trajectory = tmp_path / "run" / "stage5_quench_traj.xyz"
    legacy = tmp_path / "run" / "stage5_quench.xyz"
    write(trajectory, cu_bulk)
    write(legacy, cu_bulk)

    def fail_before_outputs(atoms, cfg, calc, resume):
        assert not trajectory.exists()
        assert not legacy.exists()
        raise RuntimeError("setup failed")

    monkeypatch.setattr(run_pipeline.quench, "run", fail_before_outputs)
    with pytest.raises(RuntimeError, match="setup failed"):
        pipe.run(stages=[5])
    with pytest.raises(RuntimeError, match="setup failed"):
        pipe.run(stages=[5], resume=True)


def test_wrapped_calculator_settings_checked(completed_pipeline):
    from amorphgen.utils.repulsion import with_repulsive_core
    pipe = completed_pipeline
    pipe._injected_calc = with_repulsive_core(pipe._injected_calc,
                                            {"enabled": True, "strength": 1.0})
    pipe.run(stages=[1])
    pipe._injected_calc.base_calculator.set(asap_cutoff=True)
    with pytest.raises(ValueError, match="calculator.parameters.base_calculator"):
        pipe.run(stages=[1], resume=True)


def test_changed_factory_settings_invalidate_shared_calculator(
        cu_bulk, emt_calc, tmp_path, monkeypatch):
    write(tmp_path / "input.xyz", cu_bulk)
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return emt_calc

    monkeypatch.setattr(run_pipeline, "get_calculator", factory)
    monkeypatch.setattr(run_pipeline.opt_cell, "run", lambda atoms, cfg, calc: atoms)
    pipe = MeltQuenchPipeline(str(tmp_path / "input.xyz"), str(tmp_path / "run"),
                             {"device": "cpu", "model": "lj"})
    pipe.run(stages=[1])
    pipe.cfg["model"] = "buckingham"
    pipe.run(stages=[1])
    assert [call["model"] for call in calls] == ["lj", "buckingham"]


def test_changed_settings_rejected_before_frame_resume(cu_bulk, emt_calc, tmp_path, monkeypatch):
    write(tmp_path / "input.xyz", cu_bulk)
    pipe = MeltQuenchPipeline(str(tmp_path / "input.xyz"), str(tmp_path / "run"), calc=emt_calc)

    def interrupted(atoms, cfg, calc, resume):
        write("stage5_quench_traj.xyz", atoms)
        raise KeyboardInterrupt()

    monkeypatch.setattr(run_pipeline.quench, "run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        pipe.run(stages=[5])
    pipe.cfg["quench"]["timestep"] = 1.0
    before = _files(pipe.work_dir)
    with pytest.raises(ValueError, match="config.quench.timestep"):
        pipe.run(stages=[5], resume=True)
    assert _files(pipe.work_dir) == before


def test_resume_empty_work_dir_starts_fresh(cu_bulk, emt_calc, tmp_path, monkeypatch):
    write(tmp_path / "input.xyz", cu_bulk)
    pipe = MeltQuenchPipeline(str(tmp_path / "input.xyz"), str(tmp_path / "run"), calc=emt_calc)
    monkeypatch.setattr(run_pipeline.opt_cell, "run", lambda atoms, cfg, calc: atoms)
    assert len(pipe.run(stages=[1], resume=True)) == len(cu_bulk)
