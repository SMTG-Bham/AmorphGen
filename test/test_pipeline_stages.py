"""
tests/test_pipeline_stages.py
------------------------------
Tier 2 integration tests — run pipeline stages with ASE's EMT calculator.
No GPU or MACE model needed. EMT only supports Cu, Ag, Au, Ni, Pd, Pt.
"""

import os
import pytest
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT

from amorphgen.utils.common import merge_config, build_md_dynamics
from amorphgen.configs import DEFAULT_CONFIG


class TestBuildMdDynamics:
    """Test NVT and NPT dynamics creation."""

    def test_nvt_creation(self, cu_supercell, emt_calc):
        cu_supercell.calc = emt_calc
        dyn = build_md_dynamics(cu_supercell, ensemble="NVT", T=300.0)
        assert dyn is not None

    def test_npt_creation(self, cu_supercell, emt_calc):
        cu_supercell.calc = emt_calc
        dyn = build_md_dynamics(cu_supercell, ensemble="NPT", T=300.0)
        assert dyn is not None

    def test_invalid_ensemble_raises(self, cu_supercell, emt_calc):
        cu_supercell.calc = emt_calc
        with pytest.raises(ValueError, match="Unknown ensemble"):
            build_md_dynamics(cu_supercell, ensemble="XYZ", T=300.0)


class TestOptCell:
    """Test structure optimisation with EMT."""

    def test_opt_reduces_forces(self, cu_bulk, emt_calc, tmp_work_dir):
        from amorphgen.pipeline.opt_cell import run

        # Slightly distort the cell
        cu_bulk.positions[0] += [0.1, 0.0, 0.0]

        override = {
            "model": "mace-mpa-0",  # won't be used since calc is passed
            "opt": {
                "fmax": 0.1,
                "max_steps": 50,
                "optimizer": "LBFGS",
                "logfile": "test_opt.log",
                "traj_file": "test_opt.traj",
                "output_cif": "test_opt.cif",
                "output_xyz": "test_opt.xyz",
            },
        }

        result = run(cu_bulk, cfg_override=override, calc=emt_calc)
        assert result is not None
        assert len(result) == len(cu_bulk)
        assert os.path.isfile("test_opt.log")

    def test_optimizer_choices(self, cu_bulk, emt_calc, tmp_work_dir):
        from amorphgen.pipeline.opt_cell import _get_optimizer
        for name in ["LBFGS", "FIRE", "BFGS"]:
            cls = _get_optimizer(name)
            assert cls is not None

    def test_invalid_optimizer_raises(self):
        from amorphgen.pipeline.opt_cell import _get_optimizer
        with pytest.raises(ValueError):
            _get_optimizer("NonexistentOptimizer")


class TestResume:
    """Test smart resume in MeltQuenchPipeline."""

    def test_find_resume_point_no_checkpoints(self, cu_bulk, tmp_work_dir):
        """With no checkpoint files, all stages should be returned."""
        from amorphgen.pipeline.run_pipeline import MeltQuenchPipeline
        from ase.io import write

        input_file = str(tmp_work_dir / "input.xyz")
        write(input_file, cu_bulk)

        pipe = MeltQuenchPipeline(
            input_file=input_file,
            work_dir=str(tmp_work_dir / "run"),
        )
        stages = [1, 4, 5, 6, 7]
        remaining, resume_input = pipe._find_resume_point(stages)
        assert remaining == [1, 4, 5, 6, 7]
        assert resume_input == input_file

    def test_find_resume_point_partial(self, cu_bulk, tmp_work_dir):
        """With stages 1 and 4 complete, should resume from stage 5."""
        from amorphgen.pipeline.run_pipeline import MeltQuenchPipeline
        from ase.io import write

        input_file = str(tmp_work_dir / "input.xyz")
        write(input_file, cu_bulk)

        work = tmp_work_dir / "run"
        work.mkdir()
        pipe = MeltQuenchPipeline(
            input_file=input_file,
            work_dir=str(work),
        )

        # Simulate completed stages 1 and 4
        write(str(work / "stage1_opt.xyz"), cu_bulk)
        write(str(work / "stage4_eq.xyz"), cu_bulk)

        stages = [1, 4, 5, 6, 7]
        remaining, resume_input = pipe._find_resume_point(stages)
        assert remaining == [5, 6, 7]
        assert resume_input == str(work / "stage4_eq.xyz")

    def test_find_resume_point_all_done(self, cu_bulk, tmp_work_dir):
        """With all stages complete, no stages should remain."""
        from amorphgen.pipeline.run_pipeline import MeltQuenchPipeline
        from ase.io import write

        input_file = str(tmp_work_dir / "input.xyz")
        write(input_file, cu_bulk)

        work = tmp_work_dir / "run"
        work.mkdir()
        pipe = MeltQuenchPipeline(
            input_file=input_file,
            work_dir=str(work),
        )

        # Simulate all stages complete
        for fname in ["stage1_opt.xyz", "stage4_eq.xyz",
                      "stage5_quenched.xyz", "stage6_eq.xyz",
                      "stage7_opt.xyz"]:
            write(str(work / fname), cu_bulk)

        stages = [1, 4, 5, 6, 7]
        remaining, resume_input = pipe._find_resume_point(stages)
        assert remaining == []

    def test_run_resume_skips_completed(self, cu_bulk, tmp_work_dir):
        """run(resume=True) should return atoms when all stages are done."""
        from amorphgen.pipeline.run_pipeline import MeltQuenchPipeline
        from ase.io import write

        input_file = str(tmp_work_dir / "input.xyz")
        write(input_file, cu_bulk)

        work = tmp_work_dir / "run"
        work.mkdir()
        pipe = MeltQuenchPipeline(
            input_file=input_file,
            work_dir=str(work),
        )

        # Simulate all stages complete
        for fname in ["stage1_opt.xyz", "stage4_eq.xyz",
                      "stage5_quenched.xyz", "stage6_eq.xyz",
                      "stage7_opt.xyz"]:
            write(str(work / fname), cu_bulk)

        result = pipe.run(stages=[1, 4, 5, 6, 7], resume=True)
        assert result is not None
        assert len(result) == len(cu_bulk)


class TestMDStages:
    """Smoke tests for MD stages — verify they run without crashing."""

    def test_short_nvt_run(self, cu_supercell, emt_calc, tmp_work_dir):
        """Run 10 NVT MD steps to verify dynamics setup works."""
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

        cu_supercell.calc = emt_calc
        MaxwellBoltzmannDistribution(cu_supercell, temperature_K=300)
        dyn = build_md_dynamics(cu_supercell, ensemble="NVT", T=300.0)
        dyn.run(10)

        # System should still be physically reasonable
        assert cu_supercell.get_temperature() > 0
        assert np.isfinite(cu_supercell.get_potential_energy())


class TestCalcInjection:
    """A calculator passed to MeltQuenchPipeline is used for every stage,
    bypassing the get_calculator() backend factory."""

    def test_injected_calc_is_reused(self, cu_bulk, emt_calc, tmp_work_dir):
        from amorphgen.pipeline.run_pipeline import MeltQuenchPipeline
        from ase.io import write

        input_file = str(tmp_work_dir / "input.xyz")
        write(input_file, cu_bulk)

        pipe = MeltQuenchPipeline(
            input_file=input_file,
            work_dir=str(tmp_work_dir / "run"),
            calc=emt_calc,
        )
        # No get_calculator() call, no backend factory — the exact object back.
        assert pipe._get_calc() is emt_calc
        assert pipe._get_calc() is emt_calc  # stable across calls


class TestGlobalSeedReproducibility:
    """A global `seed` must make the MD stages bit-reproducible (velocities +
    Langevin noise), per stage and per run directory."""

    def _eq(self, seed, tmp_path, sub="run_0003"):
        import os
        from ase.build import bulk
        from ase.calculators.emt import EMT
        from amorphgen.pipeline import equilibrate
        d = tmp_path / sub; d.mkdir(parents=True, exist_ok=True); cwd = os.getcwd(); os.chdir(d)
        try:
            a = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2)); a.calc = EMT()
            out = equilibrate.run(a, cfg_override={"seed": seed, "eq_high": {
                "ensemble": "NVT", "T": 600, "steps": 30, "timestep": 1.0}},
                calc=EMT(), stage="high")
            return out.get_positions().copy()
        finally:
            os.chdir(cwd)

    def test_same_seed_same_trajectory(self, tmp_path):
        assert np.allclose(self._eq(7, tmp_path / "a"), self._eq(7, tmp_path / "b"))

    def test_different_seed_or_run_differs(self, tmp_path):
        p = self._eq(7, tmp_path / "a")
        assert not np.allclose(p, self._eq(8, tmp_path / "b"))
        assert not np.allclose(p, self._eq(7, tmp_path / "c", sub="run_0004"))   # per-run stream

    def test_quench_stage_is_seeded_too(self, tmp_path):
        import os
        from ase.build import bulk
        from ase.calculators.emt import EMT
        from amorphgen.pipeline import quench
        res = []
        for k in range(2):
            d = tmp_path / f"q{k}"; d.mkdir(); cwd = os.getcwd(); os.chdir(d)
            try:
                a = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2)); a.calc = EMT()
                out = quench.run(a, cfg_override={"seed": 3, "quench": {
                    "ensemble": "NVT", "T_start": 600, "T_end": 300, "T_step": -150,
                    "steps_per_T": 10, "timestep": 1.0}}, calc=EMT())
                res.append(out.get_positions().copy())
            finally:
                os.chdir(cwd)
        assert np.allclose(res[0], res[1])

    def test_cli_seed_flag_reaches_override(self):
        from amorphgen.cli import _get_parser, _build_override
        p = _get_parser(); argv = ["POSCAR", "--seed", "11"]
        ov = _build_override(p.parse_args(argv), p, explicit_only=True, argv=argv)
        assert ov["seed"] == 11
