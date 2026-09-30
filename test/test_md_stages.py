"""
tests/test_md_stages.py
------------------------
Tier 2 integration tests for MD pipeline stages using EMT.
Tests equilibrate, melt, quench, and full pipeline end-to-end.
"""

import os
import pytest
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import read, write

from amorphgen.utils.common import merge_config


# ── Minimal config for fast EMT tests ──

EMT_CFG = {
    "model": "mace-mpa-0",  # not used — calc is passed directly
    "device": "cpu",
    "seed": 0,
    "traj_format": "extxyz",
    "opt": {
        "fmax": 0.5,
        "max_steps": 10,
        "cell_filter": "none",
    },
    "eq_premelt": {
        "ensemble": "NVT",
        "T": 300,
        "steps": 20,
        "timestep": 1.0,
        "friction": 0.01,
    },
    "melt": {
        "ensemble": "NVT",
        "T_start": 300,
        "T_end": 600,
        "T_step": 100,
        "steps_per_T": 10,
        "timestep": 1.0,
        "friction": 0.01,
        "make_cubic": False,
    },
    "eq_high": {
        "ensemble": "NVT",
        "T": 600,
        "steps": 20,
        "timestep": 1.0,
        "friction": 0.01,
    },
    "quench": {
        "ensemble": "NVT",
        "T_start": 600,
        "T_end": 300,
        "T_step": -100,
        "steps_per_T": 10,
        "timestep": 1.0,
        "friction": 0.01,
    },
    "eq_low": {
        "ensemble": "NVT",
        "T": 300,
        "steps": 20,
        "timestep": 1.0,
        "friction": 0.01,
    },
}


class TestEquilibrate:
    """Test equilibration stage (stages 2, 4, 6)."""

    def test_eq_high_runs(self, cu_supercell, emt_calc, tmp_work_dir):
        from amorphgen.pipeline.equilibrate import run
        result = run(cu_supercell, cfg_override=EMT_CFG, calc=emt_calc, stage="high")
        assert result is not None
        assert len(result) == len(cu_supercell)
        assert os.path.isfile("stage4_eq.xyz")

    def test_eq_low_runs(self, cu_supercell, emt_calc, tmp_work_dir):
        from amorphgen.pipeline.equilibrate import run
        result = run(cu_supercell, cfg_override=EMT_CFG, calc=emt_calc, stage="low")
        assert result is not None
        assert os.path.isfile("stage6_eq.xyz")

    def test_eq_premelt_runs(self, cu_supercell, emt_calc, tmp_work_dir):
        from amorphgen.pipeline.equilibrate import run
        result = run(cu_supercell, cfg_override=EMT_CFG, calc=emt_calc, stage="premelt")
        assert result is not None
        assert os.path.isfile("stage2_eq.xyz")

    def test_eq_high_make_cubic(self, cu_supercell, emt_calc, tmp_work_dir):
        """Stage 4 reshapes the cell to a cube when make_cubic flag is on
        (eq_high.make_cubic taking precedence over melt.make_cubic)."""
        from amorphgen.pipeline.equilibrate import run
        # Distort the input cell so reshape is observable.
        atoms = cu_supercell.copy()
        skew_cell = atoms.cell.array.copy()
        skew_cell[0, 1] += 0.5      # tilt a vector toward y -> gamma != 90
        atoms.set_cell(skew_cell, scale_atoms=True)
        # Sanity: gamma (angle between a and b) is no longer 90.
        a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
        assert not np.isclose(gamma, 90.0, atol=0.1)

        cfg = dict(EMT_CFG)
        cfg["eq_high"] = dict(EMT_CFG["eq_high"])
        cfg["eq_high"]["make_cubic"] = True
        result = run(atoms, cfg_override=cfg, calc=emt_calc, stage="high")
        a, b, c, alpha, beta, gamma = result.cell.cellpar()
        assert np.isclose(a, b, rtol=1e-3) and np.isclose(b, c, rtol=1e-3)
        assert np.allclose([alpha, beta, gamma], 90.0, atol=0.1)

    def test_eq_high_make_cubic_disabled(self, cu_supercell, emt_calc, tmp_work_dir):
        """Stage 4 leaves the cell shape alone when make_cubic flag is off."""
        from amorphgen.pipeline.equilibrate import run
        atoms = cu_supercell.copy()
        skew_cell = atoms.cell.array.copy()
        skew_cell[0, 1] += 0.5
        atoms.set_cell(skew_cell, scale_atoms=True)
        cell_before = atoms.cell.array.copy()

        cfg = dict(EMT_CFG)
        cfg["eq_high"] = dict(EMT_CFG["eq_high"])
        cfg["eq_high"]["make_cubic"] = False
        result = run(atoms, cfg_override=cfg, calc=emt_calc, stage="high")
        # NVT preserves volume + shape exactly; check the off-diagonal stayed.
        assert np.isclose(result.cell.array[0, 1], cell_before[0, 1], atol=1e-6)

    def test_eq_invalid_stage_raises(self, cu_supercell, emt_calc, tmp_work_dir):
        from amorphgen.pipeline.equilibrate import run
        with pytest.raises(ValueError, match="Unknown equilibration stage"):
            run(cu_supercell, cfg_override=EMT_CFG, calc=emt_calc, stage="invalid")

    def test_eq_from_file(self, cu_supercell, emt_calc, tmp_work_dir):
        from amorphgen.pipeline.equilibrate import run
        write("input.xyz", cu_supercell)
        result = run("input.xyz", cfg_override=EMT_CFG, calc=emt_calc, stage="high")
        assert result is not None

    def test_eq_output_is_readable(self, cu_supercell, emt_calc, tmp_work_dir):
        from amorphgen.pipeline.equilibrate import run
        run(cu_supercell, cfg_override=EMT_CFG, calc=emt_calc, stage="high")
        atoms = read("stage4_eq.xyz")
        assert len(atoms) == len(cu_supercell)


class TestMelt:
    """Test melt (heating ramp) stage."""

    def test_melt_runs(self, cu_supercell, emt_calc, tmp_work_dir):
        from amorphgen.pipeline.melt_cell import run
        result = run(cu_supercell, cfg_override=EMT_CFG, calc=emt_calc)
        assert result is not None
        assert len(result) == len(cu_supercell)
        assert os.path.isfile("stage3_melted.xyz")

    def test_melt_from_file(self, cu_supercell, emt_calc, tmp_work_dir):
        from amorphgen.pipeline.melt_cell import run
        write("input.xyz", cu_supercell)
        result = run("input.xyz", cfg_override=EMT_CFG, calc=emt_calc)
        assert result is not None

    def test_melt_trajectory_written(self, cu_supercell, emt_calc, tmp_work_dir):
        from amorphgen.pipeline.melt_cell import run
        run(cu_supercell, cfg_override=EMT_CFG, calc=emt_calc)
        frames = read("stage3_melt_traj.xyz", index=":")
        assert frames
        assert all(len(frame) == len(cu_supercell) for frame in frames)
        assert os.path.isfile("stage3_melt.log")


class TestQuench:
    """Test quench (cooling ramp) stage."""

    def test_quench_runs(self, cu_supercell, emt_calc, tmp_work_dir):
        from amorphgen.pipeline.quench import run
        result = run(cu_supercell, cfg_override=EMT_CFG, calc=emt_calc)
        assert result is not None
        assert len(result) == len(cu_supercell)
        assert os.path.isfile("stage5_quenched.xyz")

    def test_quench_from_file(self, cu_supercell, emt_calc, tmp_work_dir):
        from amorphgen.pipeline.quench import run
        write("input.xyz", cu_supercell)
        result = run("input.xyz", cfg_override=EMT_CFG, calc=emt_calc)
        assert result is not None

    def test_quench_output_readable(self, cu_supercell, emt_calc, tmp_work_dir):
        from amorphgen.pipeline.quench import run
        run(cu_supercell, cfg_override=EMT_CFG, calc=emt_calc)
        atoms = read("stage5_quenched.xyz")
        assert len(atoms) == len(cu_supercell)


class TestRateConfig:
    """Test rate (K/ps) auto-calculation of steps_per_T."""

    @pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
    @pytest.mark.parametrize("stage,rate,timestep,expected_steps", [
        ("melt", 100, 1.0, 1000),
        ("quench", 200, 0.5, 1000),
        ("quench", 10000, 1.0, 10),
    ])
    def test_rate_drives_stage_schedule(self, cu_supercell, emt_calc,
                                       tmp_work_dir, monkeypatch, sign,
                                       stage, rate, timestep, expected_steps):
        """The stage uses rate magnitude and timestep, overriding steps_per_T."""
        from unittest.mock import Mock, call
        from amorphgen.pipeline import melt_cell, quench

        module = melt_cell if stage == "melt" else quench
        temperatures = [300, 400, 500, 600]
        if stage == "quench":
            temperatures.reverse()
        temperatures = temperatures[1:]  # initial temperature is set at construction
        cfg = merge_config(EMT_CFG, {stage: {
            "rate": sign * rate,
            "timestep": timestep,
            "steps_per_T": 99999,
        }})
        dyn = Mock(spec=["set_temperature", "run"])
        monkeypatch.setattr(module, "build_md_dynamics", Mock(return_value=dyn))
        monkeypatch.setattr(module, "attach_outputs",
                            Mock(return_value=(Mock(), Mock())))

        module.run(cu_supercell, cfg_override=cfg, calc=emt_calc)

        expected_calls = []
        for temperature in temperatures:
            expected_calls.extend([
                call.set_temperature(temperature_K=temperature),
                call.run(expected_steps),
            ])
        assert dyn.mock_calls == expected_calls


class TestFinalOpt:
    """Test final optimisation stage."""

    def test_final_opt_runs(self, cu_supercell, emt_calc, tmp_work_dir):
        from amorphgen.pipeline.final_opt import run
        # Slightly distort
        cu_supercell.positions[0] += [0.1, 0.0, 0.0]
        result = run(cu_supercell, cfg_override=EMT_CFG, calc=emt_calc)
        assert result is not None
        assert os.path.isfile("stage7_opt.xyz")


class TestBatchQuench:
    """Test batch quench on multiple snapshots."""

    def test_batch_quench_runs(self, cu_supercell, emt_calc, tmp_work_dir):
        from amorphgen.pipeline.batch_quench import run

        # Create fake snapshot files
        snap_dir = tmp_work_dir / "snapshots"
        snap_dir.mkdir()
        files = []
        for i in range(3):
            f = str(snap_dir / f"snap_{i}.xyz")
            atoms = cu_supercell.copy()
            atoms.rattle(0.05, seed=i)
            write(f, atoms)
            files.append(f)

        results = run(
            snapshot_files=files,
            n_runs=3,
            cfg_override=EMT_CFG,
            work_dir=str(tmp_work_dir / "bq_out"),
            calc=emt_calc,
        )
        assert len(results) == 3

    def test_batch_quench_resume(self, cu_supercell, emt_calc, tmp_work_dir):
        from amorphgen.pipeline.batch_quench import run

        snap_dir = tmp_work_dir / "snapshots"
        snap_dir.mkdir()
        files = []
        for i in range(2):
            f = str(snap_dir / f"snap_{i}.xyz")
            write(f, cu_supercell)
            files.append(f)

        work = str(tmp_work_dir / "bq_resume")

        # First run
        run(snapshot_files=files, cfg_override=EMT_CFG,
            work_dir=work, calc=emt_calc)

        # Second run with resume — should skip
        results = run(snapshot_files=files, cfg_override=EMT_CFG,
                      work_dir=work, calc=emt_calc, resume=True)
        assert len(results) == 2


class TestFullPipelineEMT:
    """End-to-end pipeline test with EMT — all 7 stages."""

    def test_full_7_stages(self, cu_supercell, emt_calc, tmp_work_dir):
        from amorphgen.pipeline.run_pipeline import MeltQuenchPipeline

        input_file = str(tmp_work_dir / "input.xyz")
        write(input_file, cu_supercell)

        pipe = MeltQuenchPipeline(
            input_file=input_file,
            work_dir=str(tmp_work_dir / "full_run"),
            cfg_override=EMT_CFG,
        )
        # Override calculator to EMT
        pipe._calc = emt_calc
        pipe.share_calc = True

        result = pipe.run(stages=[1, 2, 3, 4, 5, 6, 7])
        assert result is not None
        assert len(result) == len(cu_supercell)

        # Check all stage outputs exist
        run_dir = tmp_work_dir / "full_run"
        for f in ["stage1_opt.xyz", "stage2_eq.xyz", "stage3_melted.xyz",
                   "stage4_eq.xyz", "stage5_quenched.xyz", "stage6_eq.xyz",
                   "stage7_opt.xyz"]:
            assert os.path.isfile(run_dir / f), f"Missing {f}"

    def test_hybrid_stages_1_4_5_6_7(self, cu_supercell, emt_calc, tmp_work_dir):
        """Test the hybrid AIRSS workflow (skip stages 2, 3)."""
        from amorphgen.pipeline.run_pipeline import MeltQuenchPipeline

        input_file = str(tmp_work_dir / "input.xyz")
        write(input_file, cu_supercell)

        pipe = MeltQuenchPipeline(
            input_file=input_file,
            work_dir=str(tmp_work_dir / "hybrid_run"),
            cfg_override=EMT_CFG,
        )
        pipe._calc = emt_calc
        pipe.share_calc = True

        result = pipe.run(stages=[1, 4, 5, 6, 7])
        assert result is not None

        run_dir = tmp_work_dir / "hybrid_run"
        for f in ["stage1_opt.xyz", "stage4_eq.xyz", "stage5_quenched.xyz",
                   "stage6_eq.xyz", "stage7_opt.xyz"]:
            assert os.path.isfile(run_dir / f), f"Missing {f}"

    def test_pipeline_summary_log(self, cu_supercell, emt_calc, tmp_work_dir):
        from amorphgen.pipeline.run_pipeline import MeltQuenchPipeline

        input_file = str(tmp_work_dir / "input.xyz")
        write(input_file, cu_supercell)

        pipe = MeltQuenchPipeline(
            input_file=input_file,
            work_dir=str(tmp_work_dir / "log_run"),
            cfg_override=EMT_CFG,
        )
        pipe._calc = emt_calc
        pipe.share_calc = True

        pipe.run(stages=[1, 4, 5, 6, 7])
        logfile = tmp_work_dir / "log_run" / "pipeline_summary.log"
        assert os.path.isfile(logfile)

    def test_resume_partial_pipeline(self, cu_supercell, emt_calc, tmp_work_dir):
        """Run stages 1,4 first, then resume to get 5,6,7."""
        from amorphgen.pipeline.run_pipeline import MeltQuenchPipeline

        input_file = str(tmp_work_dir / "input.xyz")
        write(input_file, cu_supercell)
        work = str(tmp_work_dir / "resume_run")

        # Run first two stages
        pipe = MeltQuenchPipeline(
            input_file=input_file, work_dir=work, cfg_override=EMT_CFG,
        )
        pipe._calc = emt_calc
        pipe.share_calc = True
        pipe.run(stages=[1, 4])

        assert os.path.isfile(os.path.join(work, "stage1_opt.xyz"))
        assert os.path.isfile(os.path.join(work, "stage4_eq.xyz"))

        # Resume — should run 5, 6, 7
        pipe2 = MeltQuenchPipeline(
            input_file=input_file, work_dir=work, cfg_override=EMT_CFG,
        )
        pipe2._calc = emt_calc
        pipe2.share_calc = True
        result = pipe2.run(stages=[1, 4, 5, 6, 7], resume=True)

        assert result is not None
        assert os.path.isfile(os.path.join(work, "stage5_quenched.xyz"))
        assert os.path.isfile(os.path.join(work, "stage7_opt.xyz"))


# ── Frame-level resume (kill mid-stage, resume, verify) ──

class TestFrameLevelResume:
    """An interrupted MD stage resumes from the last trajectory frame with
    only the remaining steps, appending (not duplicating) frames."""

    def _cu(self):
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2)
        atoms.rattle(0.05, seed=1)
        return atoms

    def test_read_md_checkpoint_edge_cases(self, tmp_work_dir):
        from amorphgen.utils.common import read_md_checkpoint
        assert read_md_checkpoint("does_not_exist.xyz") is None
        write("single.xyz", self._cu(), format="extxyz")
        assert read_md_checkpoint("single.xyz") is None      # 1 frame = step 0
        write("two.xyz", self._cu(), format="extxyz")
        write("two.xyz", self._cu(), format="extxyz", append=True)
        ck = read_md_checkpoint("two.xyz")
        assert ck is not None and ck[1] == 100               # (frames-1)*100

    def test_trajectory_truncated_on_fresh_run(self, tmp_work_dir):
        """A NON-resume rerun must not append to a stale trajectory."""
        from amorphgen.utils.common import TrajectoryWriter
        write("stale.xyz", self._cu(), format="extxyz")
        write("stale.xyz", self._cu(), format="extxyz", append=True)
        TrajectoryWriter("stale.xyz", fmt="extxyz")          # fresh => truncate
        assert not os.path.exists("stale.xyz")

    def test_equilibrate_kill_and_resume(self, tmp_work_dir, capsys):
        from amorphgen.pipeline import equilibrate

        def cfg(steps):
            return {"device": "cpu", "traj_format": "extxyz", "seed": 0,
                    "eq_premelt": {"ensemble": "NVT", "T": 300,
                                   "steps": steps, "timestep": 1.0,
                                   "friction": 0.01}}

        # "Kill" at step 200: run a 200-step stage => frames at 0, 100, 200
        equilibrate.run(self._cu(), cfg(200), EMT(), stage="premelt")
        assert len(read("stage2_eq_traj.xyz", index=":")) == 3
        capsys.readouterr()

        # Resume the FULL 450-step stage: must do only the remaining 250
        equilibrate.run(self._cu(), cfg(450), EMT(), stage="premelt",
                        resume=True)
        out = capsys.readouterr().out
        assert "Frame-level resume: 200 steps" in out
        # appended without duplicating the resume point: 0,100,200 + 300,400
        assert len(read("stage2_eq_traj.xyz", index=":")) == 5

    def test_quench_ramp_resume_position(self, tmp_work_dir, capsys):
        from amorphgen.pipeline import quench

        def cfg():
            return {"device": "cpu", "traj_format": "extxyz", "seed": 0,
                    "quench": {"ensemble": "NVT", "T_start": 600,
                               "T_end": 300, "T_step": -100,
                               "steps_per_T": 60, "timestep": 1.0,
                               "friction": 0.01}}

        # Partial run: only 140 of the 4x60=240 ramp steps -> frames 0,100
        partial = cfg(); partial["quench"]["steps_per_T"] = 35  # 4*35=140
        quench.run(self._cu(), partial, EMT())
        assert len(read("stage5_quench_traj.xyz", index=":")) == 2   # 0, 100
        capsys.readouterr()

        # Resume the full ramp: elapsed=100 -> segment k0=1, 20 steps left
        quench.run(self._cu(), cfg(), EMT(), resume=True)
        out = capsys.readouterr().out
        assert "Frame-level resume: 100 steps" in out
        assert "(resumed, 20 steps left)" in out   # elapsed 100 = 1*60 + 40


# ── work_dir= on a stage run by itself ──

class TestStageWorkDir:
    """A stage called on its own writes its files into ``work_dir=``.

    The keyword used to fall into ``**kwargs``: Tutorial 7 passed it to
    equilibrate.run, the trajectory landed in the notebook's directory, and
    the notebook's search of work_dir found nothing to analyse."""

    def test_md_stages(self, cu_supercell, emt_calc, tmp_work_dir):
        from amorphgen.pipeline import equilibrate, melt_cell, quench
        write("input.xyz", cu_supercell)
        # a relative input path is still read from the caller's directory
        equilibrate.run("input.xyz", EMT_CFG, emt_calc, stage="high",
                        work_dir="eq")
        melt_cell.run(cu_supercell, EMT_CFG, emt_calc,
                      work_dir=tmp_work_dir / "melt")
        quench.run(cu_supercell, EMT_CFG, emt_calc, work_dir="a/quench")
        assert sorted(os.listdir(".")) == ["a", "eq", "input.xyz", "melt"]
        assert sorted(os.listdir("eq")) == [
            "stage4_eq.log", "stage4_eq.xyz", "stage4_eq_traj.xyz"]
        assert sorted(os.listdir("melt")) == [
            "stage3_melt.log", "stage3_melt_traj.xyz", "stage3_melted.xyz"]
        assert sorted(os.listdir("a/quench")) == [
            "stage5_quench.log", "stage5_quench_traj.xyz", "stage5_quenched.xyz"]

    def test_opt_stages(self, cu_supercell, emt_calc, tmp_work_dir):
        from amorphgen.pipeline import opt_cell, final_opt
        cfg = merge_config(EMT_CFG, {"opt": {"output_format": "vasp"}})
        opt_cell.run(cu_supercell, cfg, emt_calc, work_dir="opt")
        final_opt.run(cu_supercell, cfg, emt_calc, work_dir="final")
        assert sorted(os.listdir(".")) == ["final", "opt"]
        assert sorted(os.listdir("opt")) == [
            "stage1_opt.cif", "stage1_opt.log", "stage1_opt.traj",
            "stage1_opt.vasp", "stage1_opt.xyz"]
        assert sorted(os.listdir("final")) == [
            "stage7_opt.cif", "stage7_opt.log", "stage7_opt.traj",
            "stage7_opt.vasp", "stage7_opt.xyz"]

    def test_resume_reads_the_work_dir_trajectory(self, tmp_work_dir, capsys):
        from amorphgen.pipeline import equilibrate
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2)

        def cfg(steps):
            return {"device": "cpu", "traj_format": "extxyz", "seed": 0,
                    "eq_premelt": {"ensemble": "NVT", "T": 300,
                                   "steps": steps, "timestep": 1.0,
                                   "friction": 0.01}}

        equilibrate.run(atoms, cfg(200), EMT(), stage="premelt", work_dir="eq")
        capsys.readouterr()
        equilibrate.run(atoms, cfg(300), EMT(), stage="premelt", work_dir="eq",
                        resume=True)
        assert "Frame-level resume: 200 steps" in capsys.readouterr().out
        assert len(read("eq/stage2_eq_traj.xyz", index=":")) == 4
        assert not os.path.exists("stage2_eq_traj.xyz")

    def test_absolute_names_are_kept(self, tmp_path):
        from amorphgen.utils.common import stage_file
        assert stage_file("x.log") == "x.log"
        assert stage_file("x.log", tmp_path / "d") == os.path.join(tmp_path / "d", "x.log")
        assert (tmp_path / "d").is_dir()
        absolute = str(tmp_path / "elsewhere.log")
        assert stage_file(absolute, tmp_path / "d") == absolute
