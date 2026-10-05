"""Render portable Slurm workflows without contacting a scheduler.

Use ``amorphgen-slurm workflow.yaml --output-dir jobs``, then inspect the
scripts and run ``bash jobs/submit.sh --account=PROJECT``. Commands are argv
lists, never interpolated shell programs. See docs/guides/hpc.md for the schema.
"""
from __future__ import annotations

import argparse
from importlib.resources import files
from pathlib import Path
import re
import shlex

import yaml


_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")
_PART = re.compile(r"(\d+)(?:-(\d+)(?::(\d+))?)?\Z")
_PLACEHOLDER = re.compile(r"(\{root\}|\{work_dir\}|\{task_id\})")
_RESERVED = {"array", "dependency", "chdir", "output", "error", "signal",
             "requeue", "no-requeue", "job-name", "wrap", "parsable", "export"}
_DEFAULT_RESOURCES = {"ntasks": 1, "cpus-per-task": 4, "mem": "8G", "time": "01:00:00"}
_BLUEBEAR_MODULES = ["bear-apps/2024a/live", "Python/3.12.3-GCCcore-13.3.0"]
_DEPENDENCIES = {"afterok", "afterany", "afternotok", "aftercorr"}


def _mapping(value, label, allowed=None):
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{label} must be a mapping with string keys")
    if allowed is not None and (unknown := value.keys() - allowed):
        raise ValueError(f"Unknown {label} field(s): {', '.join(sorted(unknown))}")
    return value


def _text(value, label):
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be text or a number")
    result = str(value)
    if any(c in result for c in "\n\r\0"):
        raise ValueError(f"{label} cannot contain newlines or NUL")
    return result


def _resources(value):
    value = _mapping(value, "resources")
    for key, item in value.items():
        if not re.fullmatch(r"[a-z][a-z0-9-]*", key) or key in _RESERVED:
            raise ValueError(f"Unsupported resource directive: {key}")
        if item is not None and not isinstance(item, bool):
            if not _text(item, key):
                raise ValueError(f"Empty resource directive: {key}")
    return value


def array_indices(expression):
    """Validate Slurm ranges, lists, strides and a concurrency limit."""
    expression = _text(expression, "array")
    parts = expression.split("%")
    if len(parts) > 2 or (len(parts) == 2 and not re.fullmatch(r"[1-9][0-9]*", parts[1])):
        raise ValueError(f"Invalid array concurrency limit: {expression}")
    indices = set()
    for part in parts[0].split(","):
        match = _PART.fullmatch(part)
        if not match:
            raise ValueError(f"Invalid array range: {part}")
        start = int(match[1])
        end = int(match[2]) if match[2] is not None else start
        step = int(match[3]) if match[3] is not None else 1
        if end < start or step < 1 or end > 4_000_000:
            raise ValueError(f"Invalid array bounds: {part}")
        indices.update(range(start, end + 1, step))
    return frozenset(indices)


def _topological(jobs):
    ordered, visiting, visited = [], set(), set()
    by_name = {job["name"]: job for job in jobs}

    def visit(name):
        if name in visiting:
            raise ValueError(f"Dependency cycle involving {name}")
        if name in visited:
            return
        if name not in by_name:
            raise ValueError(f"Unknown dependency: {name}")
        visiting.add(name)
        for parent in by_name[name]["needs"]:
            visit(parent)
        visiting.remove(name)
        visited.add(name)
        ordered.append(by_name[name])

    for job in jobs:
        visit(job["name"])
    for job in ordered:
        if job["dependency"] == "aftercorr":
            if not job.get("array") or not job["needs"]:
                raise ValueError("aftercorr requires an array and at least one dependency")
            for parent in job["needs"]:
                upstream = by_name[parent]
                if not upstream.get("array") or array_indices(job["array"]) != array_indices(upstream["array"]):
                    raise ValueError(f"aftercorr arrays must have matching indices: {parent}, {job['name']}")
    return ordered


def load_workflow(path, profile=None):
    """Load and validate a workflow, resolving root relative to its YAML file."""
    path = Path(path).resolve()
    data = _mapping(yaml.safe_load(path.read_text()), "workflow", {
        "version", "profile", "root", "resources", "environment", "checkpoint", "jobs"})
    if type(data.get("version", 1)) is not int or data.get("version", 1) != 1:
        raise ValueError("Only workflow version 1 is supported")
    profile = profile or data.get("profile", "generic")
    if not isinstance(profile, str) or profile not in {"generic", "bluebear"}:
        raise ValueError(f"Unknown profile: {profile}")
    root = (path.parent / _text(data.get("root", "."), "root")).resolve()
    env = _mapping(data.get("environment", {}), "environment", {"modules", "venv"})
    modules = env.get("modules", _BLUEBEAR_MODULES if profile == "bluebear" else [])
    if not isinstance(modules, list) or any(not isinstance(m, str) or not m for m in modules):
        raise ValueError("environment.modules must be a list of module names")
    modules = [_text(m, "module") for m in modules]
    venv = env.get("venv")
    if venv is not None:
        venv = str((root / _text(venv, "venv")).resolve())
    checkpoint = {"signal_seconds": 120, "requeue": False}
    checkpoint.update(_mapping(data.get("checkpoint", {}), "checkpoint", checkpoint.keys()))
    seconds = checkpoint["signal_seconds"]
    if type(seconds) is not int or not 1 <= seconds <= 65535:
        raise ValueError("checkpoint.signal_seconds must be an integer from 1 to 65535")
    if type(checkpoint["requeue"]) is not bool:
        raise ValueError("checkpoint.requeue must be true or false")
    resources = {**_DEFAULT_RESOURCES, **_resources(data.get("resources", {}))}
    raw_jobs = data.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("jobs must be a nonempty list")
    jobs, names = [], set()
    for raw in raw_jobs:
        raw = _mapping(raw, "job", {"name", "work_dir", "commands", "array", "resources", "needs", "dependency"})
        name = raw.get("name")
        if not isinstance(name, str) or not _NAME.fullmatch(name) or name in names:
            raise ValueError(f"Job names must be unique identifiers: {name!r}")
        names.add(name)
        job = {**raw, "needs": raw.get("needs", []), "dependency": raw.get("dependency", "afterok")}
        if not isinstance(job["needs"], list) or any(not isinstance(n, str) for n in job["needs"]):
            raise ValueError(f"{name}.needs must be a list of job names")
        if not isinstance(job["dependency"], str) or job["dependency"] not in _DEPENDENCIES:
            raise ValueError(f"Unsupported dependency type: {job['dependency']}")
        work_dir = _text(raw.get("work_dir", f"runs/{name}"), "work_dir")
        if not work_dir or "{work_dir}" in work_dir:
            raise ValueError(f"Invalid work_dir for {name}")
        work_dir = work_dir.replace("{root}", str(root))
        job["work_dir"] = str((root / work_dir).resolve())
        if raw.get("array") is not None:
            job["array"] = _text(raw["array"], "array")
            array_indices(job["array"])
            if "{task_id}" not in job["work_dir"]:
                raise ValueError(f"Array job {name} needs {{task_id}} in work_dir to isolate outputs")
        commands = raw.get("commands")
        if not isinstance(commands, list) or not commands:
            raise ValueError(f"{name}.commands must be a nonempty list of argv lists")
        job["commands"] = []
        for command in commands:
            if not isinstance(command, list) or not command:
                raise ValueError(f"{name}.commands entries must be nonempty argv lists")
            argv = [_text(token, "command argument") for token in command]
            if not argv[0]:
                raise ValueError(f"Empty command in {name}")
            job["commands"].append(argv)
        job_resources = {**resources, **_resources(raw.get("resources", {}))}
        if profile == "bluebear" and "qos" not in job_resources:
            gpu = any(job_resources.get(k) for k in ("gres", "gpus", "gpus-per-node", "gpus-per-task"))
            job_resources["qos"] = "bbgpu" if gpu else "bbdefault"
        job["resources"] = job_resources
        jobs.append(job)
    return {"root": str(root), "profile": profile, "modules": modules, "venv": venv,
            "checkpoint": checkpoint, "jobs": _topological(jobs)}


def _argument(value, root):
    """Quote literal fragments while allowing only our three placeholders."""
    replacements = {"{root}": shlex.quote(root), "{work_dir}": '"${AMORPHGEN_JOB_DIR}"',
                    "{task_id}": '"${SLURM_ARRAY_TASK_ID:-0}"'}
    return "".join(replacements.get(part, shlex.quote(part))
                   for part in _PLACEHOLDER.split(value) if part) or "''"


def render_job(workflow, job):
    """Render a self-contained batch script; no repository helper is needed."""
    root = workflow["root"]
    name = job["name"]
    log_suffix = "%A_%a" if job.get("array") else "%j"
    directives = {"job-name": name, **job["resources"], "chdir": root,
                  "output": f"{root}/logs/{name}_{log_suffix}.out",
                  "error": f"{root}/logs/{name}_{log_suffix}.err",
                  "signal": f"B:USR1@{workflow['checkpoint']['signal_seconds']}",
                  "requeue": True}
    if job.get("array"):
        directives["array"] = job["array"]
    lines = ["#!/bin/bash", "# Generated by amorphgen-slurm. Edit the workflow YAML to regenerate."]
    for key, value in directives.items():
        if value is True:
            lines.append(f"#SBATCH --{key}")
        elif value is not None and value is not False:
            lines.append(f"#SBATCH --{key}={shlex.quote(str(value))}")
    lines += ["", "set -euo pipefail", "export PYTHONUNBUFFERED=1",
              f"export AMORPHGEN_SLURM_REQUEUE={int(workflow['checkpoint']['requeue'])}",
              files("amorphgen").joinpath("slurm_runtime.sh").read_text(),
              'amorphgen_slurm_main "$0" "$@"', f"cd -- {shlex.quote(root)}"]
    if workflow["modules"]:
        lines += ["module purge", "module load " + shlex.join(workflow["modules"])]
    if workflow["venv"]:
        lines.append(f"source {shlex.quote(workflow['venv'] + '/bin/activate')}")
    elif workflow["profile"] == "bluebear":
        lines.append('source "${AMORPHGEN_VENV:?Set AMORPHGEN_VENV to your virtualenv directory}/bin/activate"')
    else:
        lines += ['if [[ -n "${AMORPHGEN_VENV:-}" ]]; then',
                  '    source "${AMORPHGEN_VENV}/bin/activate"', "fi"]
    if job.get("array"):
        lines += [': "${SLURM_ARRAY_TASK_ID:?Submit this script as its configured Slurm array}"',
                  '[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]] || { echo "Invalid array task ID" >&2; exit 2; }']
    lines += [f"export AMORPHGEN_JOB_DIR={_argument(job['work_dir'], root)}",
              'mkdir -p -- "$AMORPHGEN_JOB_DIR"']
    for command in job["commands"]:
        lines.append(" \\\n    ".join(_argument(token, root) for token in command))
    return "\n".join(lines) + "\n"


def render_submission(workflow):
    """Render dependency submission, failing immediately on sbatch errors."""
    lines = ["#!/bin/bash", "# Pass account/partition overrides as sbatch arguments to this script.",
             "set -euo pipefail", 'SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)',
             f"mkdir -p -- {shlex.quote(workflow['root'] + '/logs')}",
             "# Each successful submission is recorded even if a later submission fails.",
             # BSD/macOS mktemp only fills X's at the END of the template, so
             # the .tsv suffix is added by a rename (GNU and BSD alike).
             'RECEIPT=$(mktemp "$SCRIPT_DIR/submitted.XXXXXXXX")',
             'mv -- "$RECEIPT" "$RECEIPT.tsv"; RECEIPT="$RECEIPT.tsv"',
             'echo "Submission receipt: $RECEIPT" >&2',
             'submit_job() {', '    local result',
             '    result=$(sbatch --parsable "$@") || return $?',
             r'    if [[ ! "$result" =~ ^[0-9]+(\;[A-Za-z0-9_.-]+)?$ ]]; then',
             '        echo "Unexpected sbatch response: $result" >&2; return 1',
             '    fi', '    printf "%s\\n" "${result%%;*}"', '}']
    variables = {job["name"]: f"job_{index}" for index, job in enumerate(workflow["jobs"])}
    for job in workflow["jobs"]:
        variable = variables[job["name"]]
        dependency = ""
        if job["needs"]:
            ids = ":".join('${' + variables[name] + '}' for name in job["needs"])
            dependency = f' --dependency="{job["dependency"]}:{ids}"'
        lines += [f'{variable}=$(submit_job "$@"{dependency} "$SCRIPT_DIR/{job["name"]}.slurm")',
                  f'printf "%s\\t%s\\n" {shlex.quote(job["name"])} "${{{variable}}}" | tee -a "$RECEIPT"']
    return "\n".join(lines) + "\n"


def generate(workflow, output_dir, force=False):
    """Write validated scripts, refusing to overwrite existing files by default."""
    output_dir = Path(output_dir).resolve()
    rendered = {f"{job['name']}.slurm": render_job(workflow, job) for job in workflow["jobs"]}
    rendered["submit.sh"] = render_submission(workflow)
    existing = [name for name in rendered if (output_dir / name).exists()]
    if existing and not force:
        raise ValueError("Refusing to overwrite existing scripts (use --force): " + ", ".join(existing))
    output_dir.mkdir(parents=True, exist_ok=True)
    # Slurm opens its log before running the batch script.
    (Path(workflow["root"]) / "logs").mkdir(parents=True, exist_ok=True)
    paths = []
    for name, content in rendered.items():
        path = output_dir / name
        path.write_text(content)
        path.chmod(0o755)
        paths.append(path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path, help="Workflow YAML file")
    parser.add_argument("--output-dir", type=Path, default=Path("slurm_jobs"))
    parser.add_argument("--profile", choices=("generic", "bluebear"))
    parser.add_argument("--force", action="store_true", help="Overwrite existing generated scripts")
    args = parser.parse_args(argv)
    try:
        workflow = load_workflow(args.workflow, args.profile)
        paths = generate(workflow, args.output_dir, args.force)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    for path in paths:
        print(path)
    print(f"Inspect these scripts, then submit with: bash {shlex.quote(str(paths[-1]))} --account=PROJECT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
