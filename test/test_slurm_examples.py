"""Exercise the real shell supervisor without a Slurm installation or MLIP."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "amorphgen" / "slurm_runtime.sh"
LINUX_SLURM = pytest.mark.skipif(
    not sys.platform.startswith("linux") or not shutil.which("setsid"),
    reason="Slurm supervisor requires Linux setsid and ps",
)


@pytest.mark.parametrize("script", sorted((ROOT / "examples").glob("*.slurm")), ids=lambda p: p.name)
def test_legacy_scripts_share_checkpoint_supervisor(script):
    text = script.read_text()
    assert "#SBATCH --signal=B:USR1@120" in text
    assert "#SBATCH --requeue" in text
    assert "set -euo pipefail" in text
    assert 'amorphgen_slurm_main "$0" "$@"' in text
    assert "amorphgen_slurm_bluebear_environment" in text
    assert "module load" not in text
    subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)


def _wait_for(path, proc):
    deadline = time.monotonic() + 10
    while not path.exists():
        if proc.poll() is not None:
            raise AssertionError(f"Worker exited early: {proc.communicate()}")
        if time.monotonic() > deadline:
            raise AssertionError(f"Timed out waiting for {path}")
        time.sleep(0.01)


def _job(tmp_path, body, env=None):
    script = tmp_path / "job.sh"
    script.write_text(f'#!/bin/bash\nset -euo pipefail\nsource "{RUNTIME}"\namorphgen_slurm_main "$0" "$@"\n{body}\n')
    return subprocess.Popen(["bash", str(script)], cwd=tmp_path,
                            env={**os.environ, **(env or {})},
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


@LINUX_SLURM
@pytest.mark.parametrize("received", [signal.SIGUSR1, signal.SIGTERM])
@pytest.mark.parametrize("background", [False, True])
def test_preemption_waits_for_checkpoint_and_keeps_pipeline_alive(tmp_path, received, background):
    worker = tmp_path / "worker.py"
    worker.write_text('''import os, pathlib, signal, sys, time
assert os.environ["AMORPHGEN_CHECKPOINT_ON_SIGNAL"] == "1"
assert "_AMORPHGEN_SLURM_WORKER" not in os.environ
assert "_AMORPHGEN_SLURM_READY" not in os.environ
def checkpoint(signum, frame):
    pathlib.Path("checkpoint-started").touch()
    time.sleep(0.25)
    pathlib.Path("checkpoint-complete").touch()
    print("checkpoint written", flush=True)
    sys.exit(75)
signal.signal(signal.SIGUSR1, checkpoint)
pathlib.Path("ready").touch()
while True:
    time.sleep(0.1)
''')
    run = f'"{sys.executable}" "{worker}" | tee worker.log'
    if background:
        run += ' &\nwait'
    proc = _job(tmp_path, f'{run}\ntouch next-stage')
    try:
        _wait_for(tmp_path / "ready", proc)
        proc.send_signal(received)
        _wait_for(tmp_path / "checkpoint-started", proc)
        # Repeated scheduler signals must neither kill the checkpoint nor
        # allow the shell to launch the next scientific stage.
        proc.send_signal(received)
        time.sleep(0.05)
        assert proc.poll() is None, "Supervisor exited before checkpoint completion"
        stdout, stderr = proc.communicate(timeout=10)
        assert proc.returncode == 75, (stdout, stderr)
        assert (tmp_path / "checkpoint-complete").exists()
        assert "checkpoint written" in (tmp_path / "worker.log").read_text()
        assert not (tmp_path / "next-stage").exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()


@LINUX_SLURM
def test_supervisor_preserves_pipeline_failure(tmp_path):
    proc = _job(tmp_path, "bash -c 'exit 23' | cat\ntouch next-stage")
    stdout, stderr = proc.communicate(timeout=10)
    assert proc.returncode == 23, (stdout, stderr)
    assert not (tmp_path / "next-stage").exists()


@LINUX_SLURM
@pytest.mark.parametrize("received,requeue", [(signal.SIGUSR1, True), (signal.SIGTERM, False)])
@pytest.mark.parametrize("checkpoint_status", [75, 23])
@pytest.mark.parametrize("background", [False, True])
def test_optional_requeue_uses_array_element_and_never_requeues_term(tmp_path, received, requeue, checkpoint_status, background):
    scontrol = tmp_path / "scontrol"
    scontrol.write_text('#!/bin/bash\nprintf "%s\\n" "$*" > requeued\n')
    scontrol.chmod(0o755)
    worker = tmp_path / "worker.py"
    worker.write_text(f'''import pathlib, signal, sys, time
def stop(*args):
    time.sleep(0.1)
    sys.exit({checkpoint_status})
signal.signal(signal.SIGUSR1, stop)
pathlib.Path("ready").touch()
while True:
    time.sleep(0.1)
''')
    command = f'"{sys.executable}" "{worker}"'
    if background:
        command += ' &\nwait'
    proc = _job(tmp_path, command, {
        "PATH": f'{tmp_path}:{os.environ["PATH"]}', "AMORPHGEN_SLURM_REQUEUE": "1",
        "SLURM_JOB_ID": "12345", "SLURM_ARRAY_JOB_ID": "12340", "SLURM_ARRAY_TASK_ID": "7",
    })
    try:
        _wait_for(tmp_path / "ready", proc)
        proc.send_signal(received)
        stdout, stderr = proc.communicate(timeout=10)
        assert proc.returncode == (75 if background else checkpoint_status), (stdout, stderr)
        # Interrupted bash wait does not prove a background child's checkpoint
        # succeeded, even once the process has finished. Requeue fails closed.
        should_requeue = requeue and checkpoint_status == 75 and not background
        assert (tmp_path / "requeued").exists() == should_requeue
        if should_requeue:
            assert (tmp_path / "requeued").read_text().strip() == "requeue 12340_7"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()


@LINUX_SLURM
def test_sio2_array_tasks_use_independent_checkpoints_and_publish_disjoint_indices(tmp_path):
    """Run two real legacy array tasks with a lightweight placement command."""
    script = ROOT / "examples" / "run_sio2_gen_array_bluebear.slurm"
    if not script.exists():
        pytest.skip("examples/ is not in this source tree")   # an sdist
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "activate").write_text(":\n")
    command = bin_dir / "amorphgen"
    command.write_text(f'''#!{sys.executable}
import pathlib, sys
args = sys.argv[1:]
out = pathlib.Path(args[args.index("-o") + 1])
out.mkdir(parents=True, exist_ok=True)
lock = out / "checkpoint.lock"
with lock.open("x"):
    first, last = map(int, args[args.index("--indices") + 1].split("-"))
    initial = out / "random_initial"
    initial.mkdir(exist_ok=True)
    for index in range(first, last + 1):
        (initial / f"random_{{index:04d}}.xyz").write_text(str(index))
lock.unlink()
''')
    command.chmod(0o755)
    processes = []
    for task in (0, 1):
        processes.append(subprocess.Popen(
            ["bash", str(script)],
            cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**os.environ, "PATH": f'{bin_dir}:{os.environ["PATH"]}',
                 "AMORPHGEN_ROOT": str(ROOT), "AMORPHGEN_VENV": str(tmp_path),
                 "AMORPHGEN_SKIP_MODULES": "1", "SLURM_ARRAY_TASK_ID": str(task)},
        ))
    for proc in processes:
        stdout, stderr = proc.communicate(timeout=10)
        assert proc.returncode == 0, (stdout, stderr)
    output = tmp_path / "sio2_100"
    assert len(list((output / "random_initial").glob("*.xyz"))) == 50
    for task in (0, 1):
        task_files = list((output / "placements" / f"task_{task}" / "random_initial").glob("*.xyz"))
        assert len(task_files) == 25
        for placement in task_files:
            published = output / "random_initial" / placement.name
            assert published.is_symlink()
            assert published.resolve() == placement
