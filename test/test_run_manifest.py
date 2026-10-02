"""Durable provenance and lifecycle records for pipeline invocations."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from ase.io import write

from amorphgen import __version__
from amorphgen.pipeline import manifest as manifest_module
from amorphgen.pipeline import run_pipeline
from amorphgen.pipeline.run_pipeline import MeltQuenchPipeline
from amorphgen.utils.common import run_index_for


def _manifest(work_dir):
    document = json.loads((work_dir / "run_manifest.json").read_text())
    assert document["schema_version"] == 1
    return document


def _assert_finished(record):
    started = datetime.fromisoformat(record["started_at"])
    finished = datetime.fromisoformat(record["finished_at"])
    assert started.tzinfo is not None
    assert finished >= started
    assert record["elapsed_seconds"] >= 0


def _unexpected_calculator(*args, **kwargs):
    pytest.fail("This invocation must not load a calculator")


def test_failure_resume_and_noop_preserve_attempt_history(
        cu_bulk, emt_calc, tmp_work_dir, monkeypatch):
    input_path = tmp_work_dir / "input.xyz"
    write(input_path, cu_bulk)
    work_dir = tmp_work_dir / "run"
    pipeline = MeltQuenchPipeline(
        str(input_path), work_dir=str(work_dir), calc=emt_calc,
        cfg_override={"seed": 17, "device": "cpu"},
    )
    failure = RuntimeError("quench failed after optimisation")
    observed_stages = []

    def optimise(atoms, cfg, calc):
        attempt = _manifest(work_dir)["attempts"][-1]
        assert attempt["status"] == "running"
        assert [stage["status"] for stage in attempt["stages"]] == [
            "running", "pending", "pending",
        ]
        assert attempt["stages"][0]["started_at"]
        observed_stages.append(1)
        write(work_dir / "stage1_opt.xyz", atoms)
        return atoms

    def fail_quench(atoms, cfg, calc, resume):
        attempt = _manifest(work_dir)["attempts"][-1]
        assert attempt["status"] == "running"
        assert [stage["status"] for stage in attempt["stages"]] == [
            "completed", "running", "pending",
        ]
        _assert_finished(attempt["stages"][0])
        observed_stages.append(5)
        raise failure

    monkeypatch.setattr(run_pipeline.opt_cell, "run", optimise)
    monkeypatch.setattr(run_pipeline.quench, "run", fail_quench)
    monkeypatch.setattr(run_pipeline.final_opt, "run", _unexpected_calculator)

    with pytest.raises(RuntimeError) as caught:
        pipeline.run(stages=[1, 5, 7])

    assert caught.value is failure
    assert Path.cwd() == tmp_work_dir
    assert observed_stages == [1, 5]
    first = _manifest(work_dir)["attempts"][0]
    assert first["status"] == "failed"
    assert first["error"]["type"].endswith("RuntimeError")
    assert first["error"]["message"] == str(failure)
    assert [stage["status"] for stage in first["stages"]] == [
        "completed", "failed", "pending",
    ]
    assert first["stages"][1]["error"] == first["error"]
    _assert_finished(first)
    _assert_finished(first["stages"][1])

    def resume_quench(atoms, cfg, calc, resume):
        document = _manifest(work_dir)
        assert document["attempts"][0] == first
        current = document["attempts"][1]
        assert current["status"] == "running"
        assert current["resume"] is True
        assert current["input_file"] == str(work_dir / "stage1_opt.xyz")
        assert [stage["status"] for stage in current["stages"]] == [
            "skipped", "running", "pending",
        ]
        assert resume is True
        observed_stages.append(5)
        write(work_dir / "stage5_quenched.xyz", atoms)
        return atoms

    def final_optimise(atoms, cfg, calc):
        current = _manifest(work_dir)["attempts"][-1]
        assert [stage["status"] for stage in current["stages"]] == [
            "skipped", "completed", "running",
        ]
        _assert_finished(current["stages"][1])
        observed_stages.append(7)
        write(work_dir / "stage7_opt.xyz", atoms)
        return atoms

    monkeypatch.setattr(run_pipeline.opt_cell, "run", _unexpected_calculator)
    monkeypatch.setattr(run_pipeline.quench, "run", resume_quench)
    monkeypatch.setattr(run_pipeline.final_opt, "run", final_optimise)
    result = pipeline.run(stages=[1, 5, 7], resume=True)

    assert len(result) == len(cu_bulk)
    assert Path.cwd() == tmp_work_dir
    assert observed_stages == [1, 5, 5, 7]
    previous_attempts = _manifest(work_dir)["attempts"]
    second = previous_attempts[1]
    assert previous_attempts[0] == first
    assert first["seed"] == first["config"]["seed"] == 17
    assert second["seed"] == second["config"]["seed"] == 17
    assert second["config"] == first["config"]
    assert second["status"] == "completed"
    assert second["requested_stages"] == [1, 5, 7]
    assert [stage["status"] for stage in second["stages"]] == [
        "skipped", "completed", "completed",
    ]
    _assert_finished(second)

    monkeypatch.setattr(pipeline, "_get_calc", _unexpected_calculator)
    result = pipeline.run(stages=[1, 5, 7], resume=True)

    attempts = _manifest(work_dir)["attempts"]
    assert len(attempts) == 3
    assert attempts[:2] == previous_attempts
    assert attempts[2]["status"] == "completed"
    assert attempts[2]["resume"] is True
    assert attempts[2]["input_file"] == str(work_dir / "stage7_opt.xyz")
    assert [stage["status"] for stage in attempts[2]["stages"]] == [
        "skipped", "skipped", "skipped",
    ]
    _assert_finished(attempts[2])
    assert len(result) == len(cu_bulk)
    assert Path.cwd() == tmp_work_dir


def test_records_effective_config_model_hash_and_seed_index(
        cu_bulk, emt_calc, tmp_work_dir, monkeypatch):
    write(tmp_work_dir / "input.xyz", cu_bulk)
    model_path = tmp_work_dir / "custom.model"
    model_data = b"local model checkpoint\x00\xff"
    model_path.write_bytes(model_data)
    work_dir = tmp_work_dir / "run_0003"
    emt_calc.dtype = "float64"
    emt_calc.device = "cpu"
    factory_calls = []
    effective_indices = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        return emt_calc

    def stage(atoms, cfg, calc):
        attempt = _manifest(work_dir)["attempts"][-1]
        assert attempt["status"] == "running"
        assert attempt["model"]["sha256"] == hashlib.sha256(model_data).hexdigest()
        effective_indices.append(run_index_for(cfg))
        return atoms

    monkeypatch.setattr(run_pipeline, "get_calculator", factory)
    monkeypatch.setattr(run_pipeline.opt_cell, "run", stage)
    pipeline = MeltQuenchPipeline(
        "unused.xyz", work_dir="run_0003", cfg_override={
            "model": "mace-mpa-0", "model_path": "custom.model",
            "seed": 83, "device": "cpu", "default_dtype": "auto",
            "opt": {"fmax": 0.125},
        },
    )

    pipeline.run(stages=[1], input_file="input.xyz")

    attempt = _manifest(work_dir)["attempts"][0]
    assert attempt["package_version"] == __version__
    assert attempt["config"] == pipeline.cfg
    assert attempt["config"]["opt"]["fmax"] == 0.125
    assert "max_steps" in attempt["config"]["opt"]
    assert attempt["seed"] == 83
    assert attempt["seed_index"] == effective_indices[0]
    assert attempt["seed_index"] != run_index_for(pipeline.cfg)
    assert attempt["engine"] == "ase"
    assert attempt["input_file"] == str(tmp_work_dir / "input.xyz")
    assert attempt["requested_stages"] == [1]
    assert attempt["resume"] is False
    assert attempt["status"] == "completed"
    assert attempt["precision"] == {"requested": "auto", "resolved": "float64"}
    assert attempt["device"] == {"requested": "cpu", "resolved": "cpu"}
    assert attempt["model"]["name"] == "mace-mpa-0"
    assert attempt["model"]["path"] == str(model_path)
    assert attempt["model"]["sha256"] == hashlib.sha256(model_data).hexdigest()
    assert attempt["model"]["hash_source"]
    assert attempt["model"]["hash_unavailable_reason"] is None
    assert attempt["model"]["calculator_class"].endswith(".EMT")
    assert [stage["stage"] for stage in attempt["stages"]] == [1]
    assert attempt["stages"][0]["name"] == pipeline.STAGE_NAMES[1]
    assert factory_calls[0]["default_dtype"] == "auto"
    _assert_finished(attempt)
    _assert_finished(attempt["stages"][0])
    assert Path.cwd() == tmp_work_dir

    model_path.write_bytes(b"replacement checkpoint not loaded by the calculator")
    pipeline.run(stages=[1], input_file="input.xyz")

    attempts = _manifest(work_dir)["attempts"]
    assert len(factory_calls) == 1
    assert attempts[0] == attempt
    assert attempts[1]["model"]["sha256"] == attempt["model"]["sha256"]


def test_injected_calculator_is_not_attributed_to_configured_model(
        cu_bulk, emt_calc, tmp_work_dir, monkeypatch):
    input_path = tmp_work_dir / "input.xyz"
    write(input_path, cu_bulk)
    unused_model_path = tmp_work_dir / "unused.model"
    unused_model_path.write_bytes(b"not the injected calculator")
    work_dir = tmp_work_dir / "run"
    monkeypatch.setattr(run_pipeline, "get_calculator", _unexpected_calculator)
    monkeypatch.setattr(run_pipeline.opt_cell, "run", lambda atoms, cfg, calc: atoms)
    pipeline = MeltQuenchPipeline(
        str(input_path), work_dir=str(work_dir), calc=emt_calc,
        cfg_override={
            "model": "mace-mpa-0", "model_path": str(unused_model_path),
            "seed": 4, "seed_index": 91,
        },
    )

    pipeline.run(stages=[1])

    attempt = _manifest(work_dir)["attempts"][0]
    assert attempt["seed_index"] == 91
    assert attempt["config"]["model_path"] == str(unused_model_path)
    assert attempt["model"]["name"] != "mace-mpa-0"
    assert attempt["model"]["path"] is None
    assert attempt["model"]["sha256"] is None
    assert attempt["model"]["hash_unavailable_reason"]
    assert attempt["model"]["calculator_class"].endswith(".EMT")


@pytest.mark.parametrize("failure_point", ["input", "calculator"])
def test_setup_failure_is_recorded_before_any_stage(
        cu_bulk, tmp_work_dir, monkeypatch, failure_point):
    input_path = tmp_work_dir / "input.xyz"
    write(input_path, cu_bulk)
    work_dir = tmp_work_dir / "run"
    failure = RuntimeError(f"{failure_point} setup failed")

    def fail(*args, **kwargs):
        current = _manifest(work_dir)["attempts"][-1]
        assert current["status"] == "running"
        assert current["stages"][0]["status"] == "pending"
        raise failure

    if failure_point == "input":
        monkeypatch.setattr(run_pipeline, "read", fail)
        monkeypatch.setattr(run_pipeline, "get_calculator", _unexpected_calculator)
    else:
        monkeypatch.setattr(run_pipeline, "get_calculator", fail)
    monkeypatch.setattr(run_pipeline.opt_cell, "run", _unexpected_calculator)
    pipeline = MeltQuenchPipeline(
        str(input_path), work_dir=str(work_dir), cfg_override={"device": "cpu"},
    )

    with pytest.raises(RuntimeError) as caught:
        pipeline.run(stages=[1])

    assert caught.value is failure
    assert Path.cwd() == tmp_work_dir
    attempt = _manifest(work_dir)["attempts"][0]
    assert attempt["status"] == "failed"
    assert attempt["error"]["message"] == str(failure)
    assert attempt["error"]["type"].endswith("RuntimeError")
    assert attempt["stages"][0]["status"] == "pending"
    _assert_finished(attempt)


def test_keyboard_interrupt_records_interrupted_stage_and_restores_cwd(
        cu_bulk, emt_calc, tmp_work_dir, monkeypatch):
    input_path = tmp_work_dir / "input.xyz"
    write(input_path, cu_bulk)
    work_dir = tmp_work_dir / "run"
    interruption = KeyboardInterrupt("cancelled during optimisation")

    def interrupt(atoms, cfg, calc):
        assert _manifest(work_dir)["attempts"][-1]["stages"][0]["status"] == "running"
        raise interruption

    monkeypatch.setattr(run_pipeline.opt_cell, "run", interrupt)
    pipeline = MeltQuenchPipeline(
        str(input_path), work_dir=str(work_dir), calc=emt_calc,
    )
    invocation_dir = tmp_work_dir / "caller"
    invocation_dir.mkdir()
    monkeypatch.chdir(invocation_dir)

    with pytest.raises(KeyboardInterrupt) as caught:
        pipeline.run(stages=[1, 7])

    assert caught.value is interruption
    assert Path.cwd() == invocation_dir
    attempt = _manifest(work_dir)["attempts"][0]
    assert attempt["status"] == "interrupted"
    assert attempt["error"]["type"].endswith("KeyboardInterrupt")
    assert [stage["status"] for stage in attempt["stages"]] == [
        "interrupted", "pending",
    ]
    _assert_finished(attempt)
    _assert_finished(attempt["stages"][0])


@pytest.mark.parametrize("existing", ["{invalid json", '{"schema_version": 1, "attempts": {}}'])
def test_corrupt_existing_manifest_is_not_overwritten(
        cu_bulk, emt_calc, tmp_work_dir, monkeypatch, existing):
    input_path = tmp_work_dir / "input.xyz"
    write(input_path, cu_bulk)
    work_dir = tmp_work_dir / "run"
    pipeline = MeltQuenchPipeline(
        str(input_path), work_dir=str(work_dir), calc=emt_calc,
    )
    manifest_path = work_dir / "run_manifest.json"
    manifest_path.write_text(existing)
    monkeypatch.setattr(run_pipeline.opt_cell, "run", _unexpected_calculator)

    with pytest.raises(ValueError):
        pipeline.run(stages=[1])

    assert manifest_path.read_text() == existing
    assert Path.cwd() == tmp_work_dir


def test_atomic_save_failure_preserves_previous_document(tmp_work_dir, monkeypatch):
    manifest = manifest_module.RunManifest(
        tmp_work_dir, "input.xyz", {"seed": 17}, [1],
        MeltQuenchPipeline.STAGE_NAMES, False,
    )
    original = manifest.path.read_bytes()
    manifest.attempt["seed"] = 19

    def fail_replace(source, destination):
        raise OSError("unable to replace manifest")

    monkeypatch.setattr(manifest_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="unable to replace manifest"):
        manifest.save()

    assert manifest.path.read_bytes() == original
    assert _manifest(tmp_work_dir)["attempts"][0]["seed"] == 17
    assert not list(tmp_work_dir.glob(".run_manifest.*.tmp"))


def test_manifest_write_failure_does_not_mask_stage_exception(
        cu_bulk, emt_calc, tmp_work_dir, monkeypatch, capsys):
    input_path = tmp_work_dir / "input.xyz"
    write(input_path, cu_bulk)
    work_dir = tmp_work_dir / "run"
    failure = RuntimeError("original simulation failure")
    durable_documents = []

    def fail_replace(source, destination):
        raise OSError("manifest filesystem unavailable")

    def fail_stage(atoms, cfg, calc):
        durable_documents.append(_manifest(work_dir))
        monkeypatch.setattr(manifest_module.os, "replace", fail_replace)
        raise failure

    monkeypatch.setattr(run_pipeline.opt_cell, "run", fail_stage)
    pipeline = MeltQuenchPipeline(
        str(input_path), work_dir=str(work_dir), calc=emt_calc,
    )

    with pytest.raises(RuntimeError) as caught:
        pipeline.run(stages=[1])

    assert caught.value is failure
    assert Path.cwd() == tmp_work_dir
    assert _manifest(work_dir) == durable_documents[0]
    assert not list(work_dir.glob(".run_manifest.*.tmp"))
    assert "manifest filesystem unavailable" in capsys.readouterr().err


def test_config_snapshot_serializes_non_json_values(tmp_work_dir, emt_calc):
    model_path = tmp_work_dir / "custom.model"
    config = {
        "seed": np.int64(23),
        "model_path": model_path,
        "classical_params": {("Cu", "Cu"): {"sigma": np.float64(2.1)}},
        "safety": {"reference": emt_calc},
    }
    manifest = manifest_module.RunManifest(
        tmp_work_dir, "input.xyz", config, [1],
        MeltQuenchPipeline.STAGE_NAMES, False,
    )

    config["classical_params"][("Cu", "Cu")]["sigma"] = 9.0
    manifest.save()

    attempt = _manifest(tmp_work_dir)["attempts"][0]
    assert attempt["seed"] == attempt["config"]["seed"] == 23
    assert attempt["config"]["model_path"] == str(model_path)
    assert attempt["config"]["classical_params"] == {
        "('Cu', 'Cu')": {"sigma": 2.1},
    }
    assert attempt["config"]["safety"]["reference"] == {
        "python_type": "ase.calculators.emt.EMT",
    }


def test_real_emt_stages_write_checkpoints_and_completed_manifest(
        cu_bulk, emt_calc, tmp_work_dir):
    input_path = tmp_work_dir / "input.xyz"
    write(input_path, cu_bulk)
    work_dir = tmp_work_dir / "run"
    pipeline = MeltQuenchPipeline(
        str(input_path), work_dir=str(work_dir), calc=emt_calc,
        cfg_override={
            "seed": 11,
            "opt": {"cell_filter": "none", "max_steps": 2},
            "eq_premelt": {
                "ensemble": "NVT", "steps": 2, "T": 300, "timestep": 0.1,
            },
        },
    )

    result = pipeline.run(stages=[1, 2])

    assert len(result) == len(cu_bulk)
    assert np.isfinite(result.get_positions()).all()
    assert (work_dir / "stage1_opt.xyz").is_file()
    assert (work_dir / "stage2_eq.xyz").is_file()
    assert (work_dir / "pipeline_summary.log").is_file()
    attempt = _manifest(work_dir)["attempts"][0]
    assert attempt["status"] == "completed"
    assert [stage["status"] for stage in attempt["stages"]] == [
        "completed", "completed",
    ]
    _assert_finished(attempt)
    assert Path.cwd() == tmp_work_dir
