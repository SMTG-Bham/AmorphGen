"""Optional torch-sim batched relaxation engine (Tier 2: torch-sim's LJ model on CPU).

Skipped when torch-sim is not installed (pip install "amorphgen[torchsim]").
"""
import os
import sys

import numpy as np
import pytest
from ase.build import bulk
from ase.io import write, read

ts = pytest.importorskip("torch_sim")

from amorphgen.utils.torchsim_engine import build_model, batch_relax, resolve_torch_device  # noqa: E402
from amorphgen.pipeline.opt_cell import batch_optimize  # noqa: E402


def _rattled_cu(k, scale=1.05):
    a = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    a.rattle(0.15, seed=k); a.set_cell(a.cell * scale, scale_atoms=True)
    return a


LJ = {"params": {"Cu-Cu": {"sigma": 2.3, "epsilon": 0.1}}, "cutoff": 6.0}


class TestEngine:
    def test_batch_relax_cubic_keeps_cubic_and_converges(self):
        model = build_model("lennard-jones", device="cpu", classical_params=LJ)
        out = batch_relax([_rattled_cu(k) for k in range(3)], model, fmax=0.05,
                          max_steps=500, cell_filter="cubic", log=lambda *a: None)
        assert len(out) == 3
        for o in out:
            assert np.allclose(o.cell.lengths(), o.cell.lengths()[0], atol=1e-6)
            assert o.info["max_force"] < 0.05
            assert o.calc is not None and np.isfinite(o.get_potential_energy())

    def test_fixed_cell(self):
        model = build_model("lennard-jones", device="cpu", classical_params=LJ)
        ats = [_rattled_cu(k) for k in range(2)]
        out = batch_relax(ats, model, fmax=0.05, max_steps=300, cell_filter="none", log=lambda *a: None)
        assert all(np.allclose(o.cell[:], a.cell[:]) for o, a in zip(out, ats))

    def test_unsupported_models_raise_clearly(self):
        with pytest.raises(ValueError, match="no torch-sim implementation"):
            build_model("chgnet", device="cpu")
        with pytest.raises(ValueError, match="no torch-sim implementation"):
            build_model("buckingham", device="cpu", classical_params=LJ)
        with pytest.raises(ValueError, match="MPS"):
            resolve_torch_device("mps")


class TestBatchOptimizeEngine:
    def test_batch_optimize_torchsim_writes_same_outputs(self, tmp_path):
        src = tmp_path / "in"; src.mkdir()
        for k in range(3):
            write(str(src / f"s{k}.xyz"), _rattled_cu(k), format="extxyz")
        cfg = {"model": "lennard-jones", "device": "cpu", "classical_params": LJ,
               "opt": {"fmax": 0.05, "max_steps": 300, "cell_filter": "cubic"}}
        out = batch_optimize(str(src), str(tmp_path / "out"), cfg_override=cfg, engine="torchsim")
        assert len(out) == 3 and all(os.path.exists(p) for p in out)
        assert sorted(os.path.basename(p) for p in out) == ["s0_opt.xyz", "s1_opt.xyz", "s2_opt.xyz"]
        a = read(out[0]); assert np.allclose(a.cell.lengths(), a.cell.lengths()[0], atol=1e-6)
        assert (tmp_path / "out" / "s0_opt.log").exists()

    def test_cli_engine_flag(self, tmp_path, monkeypatch, capsys):
        import yaml
        from amorphgen.cli import main
        src = tmp_path / "in"; src.mkdir()
        for k in range(2):
            write(str(src / f"s{k}.xyz"), _rattled_cu(k), format="extxyz")
        y = tmp_path / "lj.yaml"
        y.write_text(yaml.safe_dump({"model": "lennard-jones", "device": "cpu", "classical_params": LJ,
                                     "opt": {"fmax": 0.05, "max_steps": 200, "cell_filter": "cubic"}}))
        monkeypatch.setattr(sys, "argv", ["amorphgen", "--batch-opt", "--input-dir", str(src),
                                          "--config", str(y), "--engine", "torchsim", "-o", str(tmp_path / "o")])
        main()
        out = capsys.readouterr().out
        assert "torch-sim engine" in out and "[torch-sim] done" in out
        assert (tmp_path / "o" / "s1_opt.xyz").exists()


class TestCellConvergenceAndOptimizer:
    def test_cell_forces_converged_and_optimizer_choice(self):
        import torch_sim as ts
        model = build_model("lennard-jones", device="cpu", classical_params=LJ)
        for opt in ("lbfgs", "fire"):
            out = batch_relax([_rattled_cu(k) for k in range(2)], model, fmax=0.02, max_steps=800,
                              cell_filter="cubic", optimizer=opt, log=lambda *a: None)
            st = ts.io.atoms_to_state(out, device=model.device, dtype=model.dtype)
            stress = model(st)["stress"].detach().cpu().numpy()
            for s in stress:                       # cell converged: |P| < 0.02 GPa
                assert abs(np.trace(s) / 3) * 160.21766 < 0.02
        with pytest.raises(ValueError, match="optimizer"):
            batch_relax([_rattled_cu(0)], model, optimizer="nelder-mead", log=lambda *a: None)


class TestChunkingAndResume:
    def _inputs(self, tmp_path, n=5):
        src = tmp_path / "in"; src.mkdir()
        for k in range(n):
            write(str(src / f"s{k}.xyz"), _rattled_cu(k), format="extxyz")
        return src

    _CFG = {"model": "lennard-jones", "device": "cpu", "classical_params": LJ,
            "opt": {"fmax": 0.05, "max_steps": 300, "cell_filter": "cubic"}}

    def test_chunks_write_all_outputs(self, tmp_path, capsys):
        src = self._inputs(tmp_path)
        out = batch_optimize(str(src), str(tmp_path / "o"), cfg_override=self._CFG,
                             engine="torchsim", batch_size=2)
        assert len(out) == 5 and all(os.path.exists(p) for p in out)
        assert "chunk 3/3" in capsys.readouterr().out

    def test_resume_skips_existing_and_finishes_the_rest(self, tmp_path, capsys):
        src = self._inputs(tmp_path)
        batch_optimize(str(src), str(tmp_path / "o"), cfg_override=self._CFG, engine="torchsim", batch_size=2)
        os.remove(tmp_path / "o" / "s3_opt.xyz")               # simulate a kill mid-chunk
        out = batch_optimize(str(src), str(tmp_path / "o"), cfg_override=self._CFG,
                             engine="torchsim", batch_size=2, resume=True)
        text = capsys.readouterr().out
        assert "[Resume] 4 already relaxed, 1 to do" in text
        assert len(out) == 5 and (tmp_path / "o" / "s3_opt.xyz").exists()


def test_oom_splits_chunk_and_finishes(tmp_path, monkeypatch, capsys):
    """A CUDA out-of-memory error on a chunk must split it, not kill the batch."""
    import amorphgen.pipeline.opt_cell as oc
    from amorphgen.utils import torchsim_engine as eng
    src = tmp_path / "in"; src.mkdir()
    for k in range(4):
        write(str(src / f"s{k}.xyz"), _rattled_cu(k), format="extxyz")
    real = eng.batch_relax
    def flaky(atoms_list, *a, **kw):
        if len(atoms_list) > 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.74 GiB")
        return real(atoms_list, *a, **kw)
    monkeypatch.setattr(oc, "batch_relax", flaky, raising=False)
    monkeypatch.setattr("amorphgen.utils.torchsim_engine.batch_relax", flaky)
    cfg = {"model": "lennard-jones", "device": "cpu", "classical_params": LJ,
           "opt": {"fmax": 0.05, "max_steps": 200, "cell_filter": "cubic"}}
    out = batch_optimize(str(src), str(tmp_path / "o"), cfg_override=cfg, engine="torchsim", batch_size=4)
    text = capsys.readouterr().out
    assert "GPU out of memory with 4 structures; retrying as 2 + 2" in text
    assert len(out) == 4 and all(os.path.exists(p) for p in out)


# ─── phase 2: batched MD ───────────────────────────────────────────────────

class TestBatchNVT:
    def _model(self):
        return build_model("lennard-jones", device="cpu", classical_params=LJ)

    def _cells(self, n=2):
        return [bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2)) for _ in range(n)]

    def test_temperature_momenta_files_and_seed(self, tmp_path):
        from amorphgen.utils.torchsim_md import batch_nvt, _RunWriter
        m = self._model()
        ws = [_RunWriter(str(tmp_path / f"run_{k:04d}"), "stage4_eq.log", "stage4_eq_traj.xyz") for k in range(2)]
        out = batch_nvt(self._cells(), m, 300.0, n_steps=400, timestep_fs=1.0, seed=1,
                        writers=ws, interval=100, log=lambda *a: None)
        assert all(np.abs(a.get_momenta()).sum() > 0 for a in out)
        assert all(150 < a.get_temperature() < 450 for a in out)            # 32-atom cells: wide band
        assert len(read(str(tmp_path / "run_0000" / "stage4_eq_traj.xyz"), index=":")) == 4
        assert sum(1 for _ in open(tmp_path / "run_0000" / "stage4_eq.log")) == 6   # header(2) + 4 frames
        again = batch_nvt(self._cells(), m, 300.0, n_steps=400, timestep_fs=1.0, seed=1, log=lambda *a: None)
        assert np.allclose(out[0].positions, again[0].positions)              # seeded noise
        other = batch_nvt(self._cells(), m, 300.0, n_steps=400, timestep_fs=1.0, seed=2, log=lambda *a: None)
        assert not np.allclose(out[0].positions, other[0].positions)          # the seed matters
        tagged = batch_nvt(self._cells(), m, 300.0, n_steps=400, timestep_fs=1.0, seed=1, tag=1, log=lambda *a: None)
        assert not np.allclose(out[0].positions, tagged[0].positions)         # so does the chunk / resume tag
        free1 = batch_nvt(self._cells(), m, 300.0, n_steps=400, timestep_fs=1.0, log=lambda *a: None)
        free2 = batch_nvt(self._cells(), m, 300.0, n_steps=400, timestep_fs=1.0, log=lambda *a: None)
        assert not np.allclose(free1[0].positions, free2[0].positions)        # unseeded runs are independent

    def test_ramp_and_momenta_carry_over(self):
        from amorphgen.utils.torchsim_md import batch_nvt
        m = self._model()
        hot = batch_nvt(self._cells(), m, 600.0, n_steps=300, timestep_fs=1.0, seed=2, log=lambda *a: None)
        cold = batch_nvt(hot, m, np.linspace(600, 100, 400), n_steps=400, timestep_fs=1.0, seed=2, stage=5,
                         log=lambda *a: None)
        assert all(a.get_temperature() < 300 for a in cold)


class TestHybridTorchsim:
    def _setup(self, tmp_path, n=3):
        import yaml
        src = tmp_path / "in"; src.mkdir()
        for k in range(n):
            a = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2)); a.rattle(0.1, seed=k)
            write(str(src / f"s{k}.xyz"), a, format="extxyz")
        y = tmp_path / "lj.yaml"
        y.write_text(yaml.safe_dump({"model": "lennard-jones", "device": "cpu", "seed": 5, "classical_params": LJ,
                                     "opt": {"fmax": 0.05, "max_steps": 200, "cell_filter": "cubic"}}))
        return src, y

    _ARGS = ["--timestep", "1.0", "--eq-high-ensemble", "NVT", "--eq-high-T", "600", "--eq-high-steps", "200",
             "--quench-ensemble", "NVT", "--quench-T-start", "600", "--quench-T-end", "300",
             "--quench-T-step", "-150", "--quench-steps-per-T", "50",
             "--eq-low-ensemble", "NVT", "--eq-low-T", "300", "--eq-low-steps", "100", "-C", "cubic"]

    def test_hybrid_ensemble_torchsim_end_to_end_and_resume(self, tmp_path, monkeypatch, capsys):
        from amorphgen.cli import main
        src, y = self._setup(tmp_path)
        base = ["amorphgen", "--hybrid-ensemble", "--input-dir", str(src), "--config", str(y),
                "--engine", "torchsim", "-o", str(tmp_path / "out")] + self._ARGS
        monkeypatch.setattr(sys, "argv", base); main()
        out = capsys.readouterr().out
        assert "Hybrid ensemble complete (torch-sim)" in out
        run0 = tmp_path / "out" / "quench_runs" / "run_0000"
        for f in ("stage4_eq.log", "stage4_eq_traj.xyz", "stage5_quench.log", "stage6_eq.log",
                  "stage7_opt.xyz", "final_amorphous.xyz"):
            assert (run0 / f).exists(), f
        assert sorted(p.name for p in (tmp_path / "out" / "final").glob("*.xyz")) == \
            ["hybrid_0000.xyz", "hybrid_0001.xyz", "hybrid_0002.xyz"]
        monkeypatch.setattr(sys, "argv", base + ["--resume"]); main()
        assert "3 run(s) already complete, 0 to do" in capsys.readouterr().out

    def test_hybrid_defaults_to_nvt_and_refuses_explicit_npt(self, tmp_path, monkeypatch, capsys):
        """DEFAULT_CONFIG stage 4 is NPT; the torch-sim hybrid path switches
        unset stages to NVT and refuses an explicit NPT before starting."""
        from amorphgen.cli import main
        src, y = self._setup(tmp_path, n=1)
        base = ["amorphgen", "--hybrid-ensemble", "--input-dir", str(src), "--config", str(y),
                "--engine", "torchsim", "-o", str(tmp_path / "o1"), "--eq-high-steps", "20",
                "--quench-T-start", "500", "--quench-T-end", "300", "--quench-T-step", "-100",
                "--quench-steps-per-T", "10", "--eq-low-steps", "10", "-C", "cubic"]
        monkeypatch.setattr(sys, "argv", base); main()
        out = capsys.readouterr().out
        assert "MD stages run NVT" in out and "Hybrid ensemble complete (torch-sim)" in out
        monkeypatch.setattr(sys, "argv", base + ["--eq-high-ensemble", "NPT"])
        with pytest.raises(SystemExit):
            main()
        assert "runs NVT only" in capsys.readouterr().out

    def test_npt_stage_is_rejected(self, tmp_path):
        from amorphgen.pipeline.batch_quench import run_torchsim
        src, _ = self._setup(tmp_path, n=1)
        with pytest.raises(ValueError, match="must be NVT"):
            run_torchsim([str(src / "s0.xyz")], cfg_override={"model": "lennard-jones", "device": "cpu",
                         "classical_params": LJ, "eq_high": {"ensemble": "NPT"}}, work_dir=str(tmp_path / "o"))


class TestPhase3:
    """Auto batch size (CPU fallback) and frame-level resume of batched MD."""

    def test_estimate_batch_size_cpu_fallback(self):
        from amorphgen.utils.torchsim_engine import estimate_batch_size
        m = build_model("lennard-jones", device="cpu", classical_params=LJ)
        assert estimate_batch_size(m, [_rattled_cu(0)], fallback=7) == 7
        assert estimate_batch_size(m, [_rattled_cu(0)], md=True) == 16

    def test_cli_batch_size_auto_and_int(self, tmp_path, monkeypatch, capsys):
        """--batch-size accepts 'auto' (CPU -> 16 per chunk) or an integer."""
        from amorphgen.cli import main
        y = tmp_path / "lj.yaml"
        y.write_text("model: lennard-jones\ndevice: cpu\nclassical_params:\n  cutoff: 6.0\n  params:\n"
                     "    Cu-Cu: {sigma: 2.3, epsilon: 0.1}\n")
        for bs, expect in (("auto", "batch size 16 -> 1 chunk(s)"), ("2", "batch size 2 -> 2 chunk(s)")):
            monkeypatch.setattr(sys, "argv", ["amorphgen", "--random-gen", "--composition", "Cu=16", "-n", "3", "--relax",
                                              "--config", str(y), "--engine", "torchsim", "--batch-size", bs,
                                              "-C", "none", "--opt-steps", "5", "-o", str(tmp_path / f"o_{bs}")])
            main()
            assert expect in capsys.readouterr().out

    def _stage4_runs(self, tmp_path, steps=300):
        from amorphgen.pipeline.batch_quench import run_torchsim
        src = [str(tmp_path / f"s{k}.xyz") for k in range(2)]
        for k, p in enumerate(src):
            write(p, _rattled_cu(k, scale=1.0), format="extxyz")
        cfg = {"model": "lennard-jones", "device": "cpu", "seed": 3, "classical_params": LJ,
               "eq_high": {"ensemble": "NVT", "T": 500, "steps": steps, "timestep": 1.0},
               "quench": {"ensemble": "NVT", "T_start": 500, "T_end": 300, "rate": 1000.0, "timestep": 1.0}}
        return run_torchsim, src, cfg, str(tmp_path / "w")

    @staticmethod
    def _log_steps(path):
        return [int(l.split()[0]) for l in open(path) if l.strip() and l.split()[0].isdigit()]

    def test_frame_level_resume_continues_from_common_frame(self, tmp_path, capsys):
        run_torchsim, src, cfg, w = self._stage4_runs(tmp_path)
        run_torchsim(src, cfg_override=cfg, work_dir=w, stages=[4], batch_size=2)
        r0, r1 = (os.path.join(w, f"run_000{k}") for k in range(2))
        for r in (r0, r1):                                   # "killed" before the stage end file was written
            os.remove(os.path.join(r, "final_amorphous.xyz")); os.remove(os.path.join(r, "stage4_eq.xyz"))
        fr = read(os.path.join(r1, "stage4_eq_traj.xyz"), index=":")   # run 1 only reached 200 steps
        write(os.path.join(r1, "stage4_eq_traj.xyz"), fr[:2], format="extxyz")
        frame2 = read(os.path.join(r0, "stage4_eq_traj.xyz"), index=1)
        cfg["eq_high"]["steps"] = 500
        run_torchsim(src, cfg_override=cfg, work_dir=w, stages=[4, 5, 7], batch_size=2, resume=True)
        out = capsys.readouterr().out
        assert "[Stage 4] resuming from step 200/500" in out and "NVT 500 K, 300 steps" in out
        assert "momenta carried over" in out
        for r in (r0, r1):
            assert len(read(os.path.join(r, "stage4_eq_traj.xyz"), index=":")) == 5
            assert self._log_steps(os.path.join(r, "stage4_eq.log")) == [100, 200, 300, 400, 500]
            assert os.path.isfile(os.path.join(r, "final_amorphous.xyz"))
        # run 0's frames beyond the common point were discarded, the common frame kept intact
        assert np.allclose(frame2.get_positions(), read(os.path.join(r0, "stage4_eq_traj.xyz"), index=1).get_positions())
        # second kill inside stage 5: stage 4 is skipped, the ramp continues from the common frame
        for r in (r0, r1):
            for f in ("final_amorphous.xyz", "stage5_quenched.xyz", "stage7_opt.xyz"):
                os.remove(os.path.join(r, f))
            fr = read(os.path.join(r, "stage5_quench_traj.xyz"), index=":")
            write(os.path.join(r, "stage5_quench_traj.xyz"), fr[:1], format="extxyz")
        run_torchsim(src, cfg_override=cfg, work_dir=w, stages=[4, 5, 7], batch_size=2, resume=True)
        out = capsys.readouterr().out
        assert "[Stage 4] already complete for this chunk -- skipping" in out
        assert "[Stage 5] resuming from step 100/200" in out
        for r in (r0, r1):
            assert self._log_steps(os.path.join(r, "stage5_quench.log")) == [100, 200]
            assert os.path.isfile(os.path.join(r, "final_amorphous.xyz"))

    def test_no_resume_starts_fresh(self, tmp_path, capsys):
        run_torchsim, src, cfg, w = self._stage4_runs(tmp_path)
        run_torchsim(src, cfg_override=cfg, work_dir=w, stages=[4], batch_size=2)
        r0 = os.path.join(w, "run_0000"); os.remove(os.path.join(r0, "final_amorphous.xyz"))
        run_torchsim(src, cfg_override=cfg, work_dir=w, stages=[4], batch_size=2)   # no resume: rewrite everything
        assert "resuming" not in capsys.readouterr().out
        assert self._log_steps(os.path.join(r0, "stage4_eq.log")) == [100, 200, 300]

    def test_partial_last_block_and_lost_end_file(self, tmp_path, capsys):
        """A 350-step stage writes frames at 100/200/300/350; resume must clamp to
        the stage length, and a complete trajectory whose end file was lost
        counts as complete (the end file is rewritten from the last frame)."""
        run_torchsim, src, cfg, w = self._stage4_runs(tmp_path, steps=350)
        run_torchsim(src, cfg_override=cfg, work_dir=w, stages=[4], batch_size=2)
        r0, r1 = (os.path.join(w, f"run_000{k}") for k in range(2))
        for r in (r0, r1):
            assert self._log_steps(os.path.join(r, "stage4_eq.log")) == [100, 200, 300, 350]
            os.remove(os.path.join(r, "final_amorphous.xyz")); os.remove(os.path.join(r, "stage4_eq.xyz"))
        run_torchsim(src, cfg_override=cfg, work_dir=w, stages=[4], batch_size=2, resume=True)
        out = capsys.readouterr().out
        assert "[Stage 4] already complete for this chunk -- skipping" in out
        for r in (r0, r1):
            assert os.path.isfile(os.path.join(r, "stage4_eq.xyz"))
            assert self._log_steps(os.path.join(r, "stage4_eq.log")) == [100, 200, 300, 350]
        # now cut both back to 3 frames: 50 steps remain, not -50
        for r in (r0, r1):
            os.remove(os.path.join(r, "final_amorphous.xyz")); os.remove(os.path.join(r, "stage4_eq.xyz"))
            fr = read(os.path.join(r, "stage4_eq_traj.xyz"), index=":")
            write(os.path.join(r, "stage4_eq_traj.xyz"), fr[:3], format="extxyz")
        run_torchsim(src, cfg_override=cfg, work_dir=w, stages=[4], batch_size=2, resume=True)
        assert "resuming from step 300/350" in capsys.readouterr().out
        for r in (r0, r1):
            assert self._log_steps(os.path.join(r, "stage4_eq.log")) == [100, 200, 300, 350]

    def test_auto_chunk_size_is_persisted_for_resume(self, tmp_path, capsys):
        import json
        run_torchsim, src, cfg, w = self._stage4_runs(tmp_path, steps=100)
        run_torchsim(src, cfg_override=cfg, work_dir=w, stages=[4], batch_size="auto")
        assert json.load(open(os.path.join(w, "batch_size.json")))["batch_size"] == 16
        json.dump({"batch_size": 1}, open(os.path.join(w, "batch_size.json"), "w"))   # pretend the probe said 1
        os.remove(os.path.join(w, "run_0000", "final_amorphous.xyz"))
        run_torchsim(src, cfg_override=cfg, work_dir=w, stages=[4], batch_size="auto", resume=True)
        out = capsys.readouterr().out
        assert "[Resume] chunk size 1 taken from" in out and "chunks of 1" in out
