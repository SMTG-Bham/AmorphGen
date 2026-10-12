"""Edge cases of format conversion, pipeline orchestration and Slurm workflow rendering."""

import json
import os
import runpy
import sys

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.io import read, write

from amorphgen.pipeline import run_pipeline
from amorphgen.pipeline.run_pipeline import (MeltQuenchPipeline, _calculator_parameters,
                                             _resume_config_value)
from amorphgen.slurm import generate, load_workflow
from amorphgen.utils.convert import _plan_outputs, convert


def _silica_chain():
    return Atoms("OSiO", positions=[[0, 0, 0], [1.6, 0, 0], [3.2, 0, 0]],
                 cell=[6, 6, 6], pbc=True)


# ── convert ─────────────────────────────────────────────────────────────────

def test_convert_rejects_unknown_format_before_writing(tmp_path):
    write(tmp_path / "s.xyz", _silica_chain())
    with pytest.raises(ValueError, match="unknown format 'pdb'"):
        convert(str(tmp_path / "s.xyz"), output_format="pdb", output_dir=str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_convert_missing_path_and_empty_directory_are_distinct_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        convert(str(tmp_path / "missing.xyz"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no structure files found"):
        convert(str(tmp_path / "empty"))
    assert not (tmp_path / "empty_vasp").exists()


@pytest.mark.parametrize("output_format,sort,symbols", [
    ("vasp", True, ["O", "O", "Si"]),
    ("vasp", False, ["O", "Si", "O"]),
    ("cif", True, ["O", "Si", "O"]),
    ("xyz", True, ["O", "Si", "O"]),
])
def test_directory_conversion_defaults_and_species_order(tmp_path, output_format, sort, symbols):
    source = tmp_path / "snaps"
    source.mkdir()
    write(source / "chain.xyz", _silica_chain())
    paths = convert(str(source) + "/", output_format=output_format, sort=sort, verbose=False)
    expected = tmp_path / f"snaps_{output_format}" / f"chain.{output_format}"
    assert paths == [str(expected)]
    converted = read(expected)
    assert converted.get_chemical_symbols() == symbols
    original = _silica_chain()
    by_symbol = {}
    for symbol, position in zip(original.get_chemical_symbols(), original.positions):
        by_symbol.setdefault(symbol, []).append(position)
    expected_positions = [by_symbol[symbol].pop(0) for symbol in symbols]
    np.testing.assert_allclose(converted.positions, expected_positions, atol=1e-6)


def test_output_names_stay_distinct_whatever_the_input_order(tmp_path):
    # A sorted directory listing never puts s_xyz.* before s.*; callers' lists can.
    files = []
    for name in ("s_xyz.cif", "s.cif", "s.xyz"):
        (tmp_path / name).write_text("")
        files.append(str(tmp_path / name))
    out = tmp_path / "out"
    destinations = _plan_outputs(files, str(out), ".vasp")
    assert [os.path.basename(path) for path in destinations] == [
        "s_xyz.vasp", "s.vasp", "s_xyz_2.vasp"]


@pytest.mark.parametrize("link", ["hard", "symbolic"])
def test_outputs_that_alias_each_other_are_refused_before_writing(tmp_path, link):
    source = tmp_path / "in"
    source.mkdir()
    for name in ("a.xyz", "b.xyz"):
        write(source / name, _silica_chain())
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.vasp").write_text("keep me")
    if link == "hard":
        os.link(out / "a.vasp", out / "b.vasp")
    else:
        os.symlink(out / "a.vasp", out / "b.vasp")
    with pytest.raises(ValueError, match="aliases another output file"):
        convert(str(source), output_format="vasp", output_dir=str(out), verbose=False)
    assert (out / "a.vasp").read_text() == "keep me"


# ── MeltQuenchPipeline ──────────────────────────────────────────────────────

def test_calculator_parameters_stop_at_wrapper_cycles():
    class Wrapper:
        def __init__(self, value):
            self.parameters = {"value": value}
            self.base_calculator = None

    outer, inner = Wrapper(1), Wrapper(2)
    outer.base_calculator, inner.base_calculator = inner, outer
    parameters = _calculator_parameters(outer)
    assert parameters["parameters"] == {"value": 1}
    assert parameters["base_calculator"]["parameters"] == {"value": 2}
    assert parameters["base_calculator"]["base_calculator"] is None


def test_resume_settings_capture_nested_reference_calculators():
    def settings(sigma):
        return _resume_config_value({"reference": [LennardJones(sigma=sigma), (1, 2)], 3: "x"})

    first = settings(2.0)
    assert first == settings(2.0)
    assert first != settings(2.5)
    calculator, pair = first["reference"]
    assert pair == [1, 2] and first["3"] == "x"
    assert calculator["model"]["calculator_class"] == "ase.calculators.lj.LennardJones"
    assert "hash_unavailable_reason" not in calculator["model"]
    assert calculator["parameters"]["parameters"]["sigma"] == 2.0
    assert json.loads(json.dumps(first)) == first


def test_legacy_mace_model_key_fills_an_empty_model(tmp_path):
    pipe = MeltQuenchPipeline("input.xyz", work_dir=str(tmp_path / "run"),
                              cfg_override={"mace_model": "chgnet", "model": None})
    assert pipe.cfg["model"] == "chgnet"


def test_resume_point_stops_at_a_stage_without_checkpoint(tmp_path):
    pipe = MeltQuenchPipeline("input.xyz", work_dir=str(tmp_path))
    write(tmp_path / "stage1_opt.xyz", bulk("Cu", cubic=True))
    assert pipe._find_resume_point([1, 9]) == ([9], str(tmp_path / "stage1_opt.xyz"))


@pytest.fixture
def recorded_stages(monkeypatch):
    calls = []

    def stage(name):
        def run(atoms, cfg, calc, **kwargs):
            calls.append((name, kwargs.get("stage")))
            return atoms
        return run

    for module, name in ((run_pipeline.opt_cell, "opt"), (run_pipeline.equilibrate, "eq"),
                         (run_pipeline.melt_cell, "melt"), (run_pipeline.quench, "quench"),
                         (run_pipeline.final_opt, "final")):
        monkeypatch.setattr(module, "run", stage(name))
    return calls


def _manifest_stages(work_dir):
    attempt = json.loads((work_dir / "run_manifest.json").read_text())["attempts"][-1]
    return attempt["status"], [(s["stage"], s["status"]) for s in attempt["stages"]]


def test_run_without_stages_executes_the_full_protocol_in_order(tmp_path, recorded_stages):
    source = tmp_path / "input.xyz"
    write(source, bulk("Cu", cubic=True))
    work = tmp_path / "run"
    MeltQuenchPipeline(str(source), work_dir=str(work), calc=EMT()).run()
    assert recorded_stages == [("opt", None), ("eq", "premelt"), ("melt", None), ("eq", "high"),
                               ("quench", None), ("eq", "low"), ("final", None)]
    assert _manifest_stages(work) == ("completed", [(s, "completed") for s in range(1, 8)])
    assert "Stages:  [1, 2, 3, 4, 5, 6, 7]" in (work / "pipeline_summary.log").read_text()


def test_unknown_stage_is_recorded_as_skipped(tmp_path, recorded_stages, capsys):
    source = tmp_path / "input.xyz"
    structure = bulk("Cu", cubic=True)
    write(source, structure)
    work = tmp_path / "run"
    result = MeltQuenchPipeline(str(source), work_dir=str(work), calc=EMT()).run(stages=[2, 9])
    np.testing.assert_allclose(result.positions, structure.positions)
    assert recorded_stages == [("eq", "premelt")]
    assert "WARNING: Unknown stage 9 - skipping." in capsys.readouterr().out
    assert _manifest_stages(work) == ("completed", [(2, "completed"), (9, "skipped")])
    summary = (work / "pipeline_summary.log").read_text()
    assert "Per-atom stage 2" in summary and "Per-atom stage 9" not in summary


# ── Slurm workflows ─────────────────────────────────────────────────────────

def _workflow_file(tmp_path, jobs=None, **fields):
    if jobs is None:
        jobs = [{"name": "one", "commands": [["true"]]}]
    data = {"version": 1, "root": str(tmp_path / "root"), "jobs": jobs, **fields}
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.mark.parametrize("fields,message", [
    ({"version": 2}, "Only workflow version 1 is supported"),
    ({"version": True}, "Only workflow version 1 is supported"),
    ({"root": ["somewhere"]}, "root must be text or a number"),
    ({"resources": {"partition": ""}}, "Empty resource directive: partition"),
    ({"environment": ["module"]}, "environment must be a mapping with string keys"),
    ({"jobs": []}, "jobs must be a nonempty list"),
    ({"jobs": ["one"]}, "job must be a mapping with string keys"),
    ({"jobs": [{"name": "one", "commands": [["true"]], "needs": "zero"}]},
     "one.needs must be a list of job names"),
    ({"jobs": [{"name": "one", "commands": [["true"]], "work_dir": ""}]},
     "Invalid work_dir for one"),
    ({"jobs": [{"name": "one", "commands": [["true"]], "work_dir": "{work_dir}/x"}]},
     "Invalid work_dir for one"),
    ({"jobs": [{"name": "one", "commands": ["true"]}]},
     "one.commands entries must be nonempty argv lists"),
    ({"jobs": [{"name": "one", "commands": [[]]}]},
     "one.commands entries must be nonempty argv lists"),
    ({"jobs": [{"name": "one", "commands": [["", "x"]]}]}, "Empty command in one"),
    ({"jobs": [{"name": "one", "commands": [["echo", True]]}]},
     "command argument must be text or a number"),
    ({"jobs": [
        {"name": "gen", "array": "0-1", "work_dir": "g/{task_id}", "commands": [["true"]]},
        {"name": "collect", "needs": ["gen"], "dependency": "aftercorr",
         "commands": [["true"]]}]},
     "aftercorr requires an array and at least one dependency"),
])
def test_invalid_workflows_name_the_problem(tmp_path, fields, message):
    with pytest.raises(ValueError, match=message):
        load_workflow(_workflow_file(tmp_path, **fields))
    assert not (tmp_path / "root").exists()


def test_workflow_must_be_a_mapping(tmp_path):
    path = tmp_path / "workflow.yaml"
    path.write_text("- one\n- two\n")
    with pytest.raises(ValueError, match="workflow must be a mapping with string keys"):
        load_workflow(path)


def test_generic_profile_activates_the_configured_venv(tmp_path):
    workflow = load_workflow(_workflow_file(tmp_path, environment={"venv": "env space"}))
    (script, _) = generate(workflow, tmp_path / "jobs")
    text = script.read_text()
    activate = str((tmp_path / "root" / "env space").resolve()) + "/bin/activate"
    assert f"source '{activate}'" in text
    assert "AMORPHGEN_VENV" not in text.split('amorphgen_slurm_main "$0" "$@"')[1]


def test_slurm_module_entry_point_writes_scripts(tmp_path, monkeypatch, capsys):
    path = _workflow_file(tmp_path)
    monkeypatch.delitem(sys.modules, "amorphgen.slurm")
    monkeypatch.setattr(sys, "argv", ["amorphgen-slurm", str(path),
                                      "--output-dir", str(tmp_path / "jobs")])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("amorphgen.slurm", run_name="__main__")
    assert exc.value.code == 0
    assert (tmp_path / "jobs" / "one.slurm").is_file()
    assert (tmp_path / "jobs" / "submit.sh").is_file()
    assert "Inspect these scripts" in capsys.readouterr().out
