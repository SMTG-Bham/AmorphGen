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

from amorphgen.configs import DEFAULT_CONFIG
from amorphgen.utils.common import merge_config


# ── Minimal config for fast EMT tests ──

EMT_CFG = {
    "model": "mace-mpa-0",  # not used — calc is passed directly
    "device": "cpu",
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
        import numpy as np
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
        import numpy as np
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

    def test_melt_with_rate(self, cu_supercell, emt_calc, tmp_work_dir):
        from amorphgen.pipeline.melt_cell import run
        cfg = dict(EMT_CFG)
        cfg["melt"] = {
            "ensemble": "NVT",
            "T_start": 300, "T_end": 600, "T_step": 100,
            "rate": 100,         # 100 K/ps
            "timestep": 1.0,     # 1 fs
            "friction": 0.01,
            "make_cubic": False,
        }
        # rate=100 K/ps, T_step=100 K, timestep=1 fs
        # steps_per_T = 100 / (100 * 0.001) = 1000
        result = run(cu_supercell, cfg_override=cfg, calc=emt_calc)
        assert result is not None

    def test_quench_with_rate(self, cu_supercell, emt_calc, tmp_work_dir):
        from amorphgen.pipeline.quench import run
        cfg = dict(EMT_CFG)
        cfg["quench"] = {
            "ensemble": "NVT",
            "T_start": 600, "T_end": 300, "T_step": -100,
            "rate": 200,         # 200 K/ps
            "timestep": 0.5,     # 0.5 fs
            "friction": 0.01,
        }
        # rate=200 K/ps, T_step=100 K, timestep=0.5 fs
        # steps_per_T = 100 / (200 * 0.0005) = 1000
        result = run(cu_supercell, cfg_override=cfg, calc=emt_calc)
        assert result is not None

    def test_rate_overrides_steps_per_T(self, cu_supercell, emt_calc, tmp_work_dir):
        """When both rate and steps_per_T are given, rate wins."""
        from amorphgen.pipeline.quench import run
        cfg = dict(EMT_CFG)
        cfg["quench"] = {
            "ensemble": "NVT",
            "T_start": 600, "T_end": 300, "T_step": -100,
            "steps_per_T": 99999,  # should be ignored
            "rate": 10000,         # very fast → few steps
            "timestep": 1.0,
            "friction": 0.01,
        }
        # rate=10000 K/ps → steps_per_T = 100 / (10000 * 0.001) = 10
        result = run(cu_supercell, cfg_override=cfg, calc=emt_calc)
        assert result is not None
        assert os.path.isfile("stage5_quenched.xyz")


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
            return {"device": "cpu", "traj_format": "extxyz",
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
            return {"device": "cpu", "traj_format": "extxyz",
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
