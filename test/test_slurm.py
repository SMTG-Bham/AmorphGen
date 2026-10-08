"""Portable workflow generation and submission without a Slurm installation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from amorphgen.slurm import array_indices, generate, load_workflow, main


def workflow(tmp_path, jobs=None, **kwargs):
    data = {"version": 1, "root": str(tmp_path / "run space"), "jobs": jobs or [
        {"name": "one", "work_dir": "out", "commands": [["true"]]}], **kwargs}
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(data))
    return load_workflow(path)


@pytest.mark.skipif(not shutil.which("setsid"), reason="Slurm jobs require Linux setsid")
def test_portable_array_executes_literal_arguments_and_isolates_tasks(tmp_path):
    tricky = "spaces 'quotes' $(touch INJECTED) `touch ALSO_INJECTED`; $HOME"
    data = workflow(tmp_path, [{"name": "array", "array": "0-1%1", "work_dir": "out/{task_id}",
        "commands": [[sys.executable, "-c",
            "import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))",
            "{work_dir}/argv.json", tricky, "{root}", "", "{task_id}"]]}])
    paths = generate(data, tmp_path / "jobs space")
    for path in paths:
        subprocess.run(["bash", "-n", str(path)], check=True)
    for task in range(2):
        env = {**os.environ, "SLURM_ARRAY_TASK_ID": str(task), "AMORPHGEN_VENV": ""}
        subprocess.run(["bash", str(paths[0])], env=env, check=True, timeout=10)
        output = Path(data["root"]) / "out" / str(task) / "argv.json"
        assert json.loads(output.read_text()) == [tricky, data["root"], "", str(task)]
    script = paths[0].read_text()
    assert "#SBATCH --array=0-1%1" in script
    assert "%A_%a.out" in script
    assert "#SBATCH --signal=B:USR1@120" in script
    assert "module load" not in script.split('amorphgen_slurm_main "$0" "$@"')[1]
    assert not list(Path(data["root"]).rglob("*INJECTED*"))


@pytest.mark.parametrize("mktemp_style", ["native", "bsd"])
def test_submission_topological_ids_dependencies_and_failure(tmp_path, mktemp_style):
    jobs = [
        {"name": "collect", "needs": ["relax"], "commands": [["true"]]},
        {"name": "relax", "array": "0-4:2%2", "work_dir": "relax/{task_id}",
         "needs": ["gen"], "dependency": "aftercorr", "commands": [["true"]]},
        {"name": "gen", "array": "0,2,4", "work_dir": "gen/{task_id}", "commands": [["true"]]},
    ]
    data = workflow(tmp_path, jobs)
    paths = generate(data, tmp_path / "jobs")
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    if mktemp_style == "bsd":
        # Model BSD's trailing-X substitution on every platform. With a .tsv
        # suffix it instead creates the literal template, so reuse fails.
        mktemp = fakebin / "mktemp"
        mktemp.write_text(f"#!{sys.executable}\n" + '''import os,sys,tempfile
from pathlib import Path
template = sys.argv[1]
prefix = template.rstrip('X')
if prefix == template:
    try:
        fd = os.open(template, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        print(f'mktemp: {template}: File exists', file=sys.stderr)
        sys.exit(1)
    path = template
else:
    fd, path = tempfile.mkstemp(prefix=Path(prefix).name, dir=Path(prefix).parent)
os.close(fd)
print(path)
''')
        mktemp.chmod(0o755)
    sbatch = fakebin / "sbatch"
    calls = tmp_path / "calls.jsonl"
    sbatch.write_text(f"#!{sys.executable}\n" + '''import json,os,sys
from pathlib import Path
path = Path(os.environ['CALLS'])
previous = path.read_text().splitlines() if path.exists() else []
with path.open('a') as stream: stream.write(json.dumps(sys.argv[1:])+'\\n')
if os.environ.get('FAIL') == str(len(previous)): sys.exit(9)
print(str(101+len(previous))+';cluster-a')
''')
    sbatch.chmod(0o755)
    env = {**os.environ, "PATH": f"{fakebin}:{os.environ['PATH']}", "CALLS": str(calls)}
    result = subprocess.run(["bash", str(paths[-1]), "--account=my-account"], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    arguments = [json.loads(line) for line in calls.read_text().splitlines()]
    assert [Path(args[-1]).stem for args in arguments] == ["gen", "relax", "collect"]
    assert all(args[:2] == ["--parsable", "--account=my-account"] for args in arguments)
    assert "--dependency=aftercorr:101" in arguments[1]
    assert "--dependency=afterok:102" in arguments[2]
    assert "collect\t103" in result.stdout
    assert (Path(data["root"]) / "logs").is_dir()
    calls.unlink()
    env["FAIL"] = "1"
    result = subprocess.run(["bash", str(paths[-1])], env=env, capture_output=True, timeout=10)
    assert result.returncode == 9, result.stderr
    assert len(calls.read_text().splitlines()) == 2  # downstream was not submitted
    receipts = list((tmp_path / "jobs").glob("submitted.*.tsv"))
    assert len(receipts) == 2
    assert {p.read_text() for p in receipts} == {
        "gen\t101\nrelax\t102\ncollect\t103\n", "gen\t101\n"}


@pytest.mark.parametrize("expression", ["", "1-0", "0-5:0", "-1", "0%0", "0%1%2", "0;true", "1,", "0-2:foo"])
def test_bad_array_rejected(expression):
    with pytest.raises(ValueError):
        array_indices(expression)


@pytest.mark.parametrize("change", [
    {"jobs": [{"name": "a", "commands": "echo x"}]},
    {"jobs": [{"name": "a", "commands": [["true"]], "array": "0-2", "work_dir": "shared"}]},
    {"jobs": [{"name": "a", "commands": [["true"]], "array": "0-2", "work_dir": "out/{task_id}/../shared"}]},
    {"jobs": [{"name": "a", "commands": [["true"]], "needs": ["missing"]}]},
    {"jobs": [{"name": "a", "commands": [["true"]], "needs": ["a"]}]},
    {"jobs": [{"name": "../escape", "commands": [["true"]]}]},
    {"resources": {"mem": "8G\n#SBATCH --account=wrong"}},
    {"resources": {"wrap": "echo bad"}},
    {"checkpoint": {"signal_seconds": 0}},
    {"checkpoint": {"requeue": "false"}},
    {"environment": {"modules": "Python"}},
    {"profile": []},
    {"jobs": [{"name": "a", "commands": [["true"]], "dependency": []}]},
    {"typo": True},
])
def test_bad_workflow_fails_before_writes(tmp_path, change):
    with pytest.raises(ValueError):
        workflow(tmp_path, **change)
    assert not (tmp_path / "run space").exists()


def test_mismatched_corresponding_arrays_rejected(tmp_path):
    jobs = [{"name": "a", "array": "0-1", "work_dir": "a/{task_id}", "commands": [["true"]]},
            {"name": "b", "array": "1-2", "work_dir": "b/{task_id}", "commands": [["true"]],
             "needs": ["a"], "dependency": "aftercorr"}]
    with pytest.raises(ValueError, match="matching indices"):
        workflow(tmp_path, jobs)


def test_bluebear_profile_resources_and_environment_override(tmp_path):
    data = workflow(tmp_path, profile="bluebear", resources={"gres": "gpu:1", "partition": "test"})
    paths = generate(data, tmp_path / "jobs")
    script = paths[0].read_text()
    assert "#SBATCH --qos=bbgpu" in script
    assert "#SBATCH --partition=test" in script
    assert "module load bear-apps/2024a/live Python/3.12.3-GCCcore-13.3.0" in script
    assert "${AMORPHGEN_VENV:?" in script
    assert "#SBATCH --account" not in script
    data = workflow(tmp_path, profile="bluebear", environment={"modules": [], "venv": "venv space"})
    assert data["modules"] == []
    assert data["jobs"][0]["resources"]["qos"] == "bbdefault"


def test_overwrite_explicit_and_cli_generation_only(tmp_path, capsys):
    workflow(tmp_path)
    yaml_path = tmp_path / "workflow.yaml"
    args = [str(yaml_path), "--output-dir", str(tmp_path / "jobs")]
    assert main(args) == 0
    assert "Inspect these scripts" in capsys.readouterr().out
    sentinel = tmp_path / "jobs" / "one.slurm"
    sentinel.write_text("manual change")
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert sentinel.read_text() == "manual change"
    assert main([*args, "--force"]) == 0
    assert sentinel.read_text().startswith("#!/bin/bash")


def test_bundled_example_and_all_amorphgen_commands_parse(tmp_path):
    from amorphgen.cli import _get_parser
    source = Path(__file__).resolve().parents[1] / "examples" / "slurm_workflow.yaml"
    data = load_workflow(source)
    for job in data["jobs"]:
        for command in job["commands"]:
            if command[0] == "amorphgen":
                _get_parser().parse_args([token.replace("{task_id}", "0") for token in command[1:]])
    # Keep test artifacts out of the checkout.
    data["root"] = str(tmp_path)
    for path in generate(data, tmp_path / "jobs"):
        subprocess.run(["bash", "-n", str(path)], check=True)
