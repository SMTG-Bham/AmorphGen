"""Real signals must stop at resumable boundaries and never report success."""
import json
import os
import signal

import numpy as np
import pytest
from ase.io import read, write

from amorphgen.utils import preemption


@pytest.mark.parametrize("signum", [signal.SIGUSR1, signal.SIGTERM])
def test_handlers_only_request_stop_and_restore(monkeypatch, signum):
    monkeypatch.setenv("AMORPHGEN_CHECKPOINT_ON_SIGNAL", "1")
    original = signal.getsignal(signum)
    with preemption.checkpoint_signals():
        os.kill(os.getpid(), signum)
        # No exception or I/O occurs in the signal handler itself.
        with pytest.raises(preemption.PreemptionRequested) as caught:
            preemption.stop_if_requested()
        assert caught.value.code == 75
        assert not isinstance(caught.value, Exception)
    assert signal.getsignal(signum) == original
    preemption.stop_if_requested()


def test_handlers_are_opt_in(monkeypatch):
    monkeypatch.delenv("AMORPHGEN_CHECKPOINT_ON_SIGNAL", raising=False)
    original = signal.getsignal(signal.SIGTERM)
    with preemption.checkpoint_signals():
        assert signal.getsignal(signal.SIGTERM) == original


@pytest.mark.parametrize("explicit_exit", [False, True])
def test_cli_reports_preemption_exit_code(monkeypatch, capsys, explicit_exit):
    from amorphgen import cli
    monkeypatch.setenv("AMORPHGEN_CHECKPOINT_ON_SIGNAL", "1")

    def work():
        os.kill(os.getpid(), signal.SIGTERM)
        if explicit_exit:
            raise SystemExit(0)

    monkeypatch.setattr(cli, "_main", work)
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 75
    assert "SIGTERM" in capsys.readouterr().err


@pytest.mark.parametrize("fmt", ["extxyz", "traj"])
@pytest.mark.parametrize("signum", [signal.SIGUSR1, signal.SIGTERM])
def test_ase_signal_preserves_checkpoint_and_resumes(
        tmp_path, cu_bulk, emt_calc, monkeypatch, fmt, signum):
    from amorphgen.pipeline import equilibrate
    from amorphgen.pipeline.run_pipeline import MeltQuenchPipeline
    from amorphgen.utils.common import read_md_checkpoint
    from amorphgen.utils.run_lock import run_lock

    monkeypatch.setenv("AMORPHGEN_CHECKPOINT_ON_SIGNAL", "1")
    write(tmp_path / "input.xyz", cu_bulk)
    real_build = equilibrate.build_md_dynamics
    dynamics = []

    def build(*args, **kwargs):
        dyn = real_build(*args, **kwargs)
        if not dynamics:
            # Interrupt mid-block; the saved checkpoint must still be exactly
            # step 100, not a partial frame miscounted as another 100 steps.
            dyn.attach(lambda: os.kill(os.getpid(), signum), interval=-17)
        dynamics.append(dyn)
        return dyn

    monkeypatch.setattr(equilibrate, "build_md_dynamics", build)
    pipe = MeltQuenchPipeline(
        str(tmp_path / "input.xyz"), str(tmp_path / "run"), calc=emt_calc,
        cfg_override={"seed": 4, "traj_format": fmt,
                      "eq_high": {"ensemble": "NVT", "T": 300,
                                  "steps": 220, "timestep": 0.5}},
    )
    with preemption.checkpoint_signals():
        with pytest.raises(preemption.PreemptionRequested):
            pipe.run(stages=[4])
    run_dir = tmp_path / "run"
    assert dynamics[0].nsteps == 100
    assert not (run_dir / "stage4_eq.xyz").exists()
    checkpoint, elapsed = read_md_checkpoint(str(run_dir / "stage4_eq_traj.xyz"))
    assert elapsed == 100
    assert np.abs(checkpoint.get_momenta()).sum() > 0
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["attempts"][-1]["status"] == "interrupted"
    assert manifest["attempts"][-1]["stages"][0]["status"] == "interrupted"
    with run_lock(str(run_dir)):
        pass  # Cleanup releases ownership before the next Slurm attempt.

    with preemption.checkpoint_signals():
        pipe.run(stages=[4], resume=True)
    assert dynamics[1].nsteps == 120
    assert (run_dir / "stage4_eq.xyz").exists()
    assert len(read(run_dir / "stage4_eq_traj.xyz", index=":")) == 3
    assert read_md_checkpoint(str(run_dir / "stage4_eq_traj.xyz"))[1] == 200
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["attempts"][-1]["status"] == "completed"


def test_optimisation_does_not_publish_unfinished_structure(
        tmp_path, cu_bulk, emt_calc, monkeypatch):
    from amorphgen.pipeline import opt_cell
    from ase.optimize import FIRE

    monkeypatch.setenv("AMORPHGEN_CHECKPOINT_ON_SIGNAL", "1")

    class InterruptedFIRE(FIRE):
        def step(self, *args, **kwargs):
            super().step(*args, **kwargs)
            os.kill(os.getpid(), signal.SIGUSR1)

    monkeypatch.setattr(opt_cell, "_get_optimizer", lambda name: InterruptedFIRE)
    with preemption.checkpoint_signals():
        with pytest.raises(preemption.PreemptionRequested):
            opt_cell.run(cu_bulk, calc=emt_calc, work_dir=tmp_path,
                         cfg_override={"opt": {"cell_filter": "none"}})
    assert not (tmp_path / "stage1_opt.xyz").exists()
    assert len(read(tmp_path / "stage1_opt.traj", index=":")) == 1


def test_torchsim_signal_saves_whole_batch_before_stopping(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_sim")
    from ase import Atoms
    from torch_sim.models.interface import ModelInterface
    from amorphgen.utils.torchsim_md import batch_nvt, _RunWriter
    from amorphgen.pipeline.batch_quench import _batched_stage_checkpoint

    monkeypatch.setenv("AMORPHGEN_CHECKPOINT_ON_SIGNAL", "1")

    class SignallingModel(ModelInterface):
        def __init__(self):
            super().__init__()
            self._device = torch.device("cpu")
            self._dtype = torch.float64
            self._compute_forces = self._compute_stress = True
            self.calls = 0

        def forward(self, state, **kwargs):
            self.calls += 1
            if self.calls == 3:
                os.kill(os.getpid(), signal.SIGUSR1)
            return {"energy": torch.zeros(state.n_systems, dtype=self.dtype),
                    "forces": torch.zeros_like(state.positions),
                    "stress": torch.zeros_like(state.cell)}

    atoms = Atoms("Cu2", positions=[[2, 2, 2], [4, 2, 2]], cell=[8] * 3, pbc=True)
    dirs = [str(tmp_path / str(i)) for i in range(2)]
    writers = [_RunWriter(d, "eq.log", "traj.xyz") for d in dirs]
    with preemption.checkpoint_signals():
        with pytest.raises(preemption.PreemptionRequested):
            batch_nvt([atoms, atoms.copy()], SignallingModel(), 300,
                      n_steps=25, interval=10, writers=writers, seed=8)
    frames, done, complete = _batched_stage_checkpoint(
        dirs, "eq.log", "traj.xyz", "final.xyz", 25, 10, True)
    assert len(frames) == 2
    assert done == 10
    assert not complete
    for d in dirs:
        assert len(read(os.path.join(d, "traj.xyz"), index=":")) == 1
        assert not os.path.exists(os.path.join(d, "final.xyz"))


def test_ase_batch_resume_retains_completed_structures(tmp_path, cu_bulk, monkeypatch):
    from amorphgen.pipeline import opt_cell

    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    for name in ("a", "b"):
        write(source / f"{name}.xyz", cu_bulk)
    write(output / "a_opt.xyz", cu_bulk)
    saved = (output / "a_opt.xyz").read_bytes()
    # A torn write from a hard kill must not count as complete.
    (output / "b_opt.xyz").write_text("incomplete frame")
    called = []

    def optimize(path, **kwargs):
        called.append(path)
        write("b_opt.xyz", cu_bulk)
        return cu_bulk

    monkeypatch.setattr(opt_cell, "run", optimize)
    result = opt_cell.batch_optimize(str(source), str(output), resume=True)
    assert called == [str(source / "b.xyz")]
    assert len(result) == 2
    assert (output / "a_opt.xyz").read_bytes() == saved
