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


def test_run_seed_index_bands_and_stability(tmp_path, monkeypatch):
    """Review round 8: the seed index of one MD run. Its LOCAL identity is the
    snapshot_NNNN number when there is one (so a run keeps its seed when the
    input set changes), and that is banded by where the scope came from, so two
    runs collide only within one source. Snapshot numbers here are deliberately
    not equal to the loop positions."""
    from amorphgen.pipeline.batch_quench import _run_seed_index as idx
    from amorphgen.utils.common import scoped_run_index, _INDEX_BAND, _INDEX_STRIDE
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    # local identity: the filename number, not the loop position
    assert idx("snapshot_0007_frame01.xyz", 0) == 7
    assert [idx(f"snapshot_{n:04d}.xyz", i) for i, n in enumerate((7, 3, 11))] == [7, 3, 11]
    assert [idx(f"s{k}.xyz", k) for k in range(3)] == [0, 1, 2]   # else the loop position
    # a run keeps its seed when another file appears beside it
    assert idx("snapshot_0007.xyz", 0, 5) == idx("snapshot_0007.xyz", 1, 5)
    # the three sources occupy disjoint bands
    local = {idx(f"s{k}.xyz", k) for k in range(4)} | {idx("snapshot_0003.xyz", 0)}
    explicit = {idx(f"s{k}.xyz", k, b) for b in (0, 2) for k in range(4)}
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "2")
    slurm = {idx(f"s{k}.xyz", k) for k in range(4)}
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID")
    assert not (local & explicit) and not (local & slurm) and not (explicit & slurm)
    assert max(local) < _INDEX_BAND
    # two jobs that both use --run-index cannot overlap either
    assert not ({idx(f"s{k}.xyz", k, 0) for k in range(3)} & {idx("s.xyz", 0, 1)})
    # and the pipeline path bands a bare --run-index the same way
    # the value a stage actually receives, i.e. through run_index_for, is what
    # matters: the earlier tests stopped at _run_seed_index and missed a double
    # banding downstream that put three unrelated runs on one stream
    from amorphgen.utils.common import run_index_for
    through = [run_index_for({"seed_index": idx("snapshot_0003.xyz", 0)}),        # batch, no flag
               run_index_for({"seed_index": idx("snapshot_0000.xyz", 0, 3)}),     # batch, --run-index 3
               run_index_for({"run_index": 3})]                                   # pipeline, --run-index 3
    assert len(set(through)) == 3, through
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    through += [run_index_for({"seed_index": idx("snapshot_0000.xyz", 0)}),       # batch under SLURM 3
                run_index_for({})]                                                # pipeline under SLURM 3
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID")
    assert len(set(through)) == 5, through


def test_batch_quench_assigns_distinct_seed_indices(tmp_path, monkeypatch):
    """The loop itself: every run of a batch gets its own index, and a
    single-snapshot run picks up the SLURM array task."""
    from unittest.mock import patch
    from ase.build import bulk
    from ase.io import write
    from amorphgen.pipeline import batch_quench as bq
    src = tmp_path / "in"; src.mkdir()
    # numbered out of order on purpose: the index must come from the FILENAME,
    # not from the position in the sorted loop
    for n in (7, 3, 11):
        write(str(src / f"snapshot_{n:04d}_f.xyz"), bulk("Cu", cubic=True), format="extxyz")
    seen = []

    def fake_eq(atoms, cfg_override=None, **kw):
        seen.append(cfg_override.get("seed_index"))
        return atoms

    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    with patch.object(bq.equilibrate, "run", fake_eq), \
         patch.object(bq, "get_calculator", lambda **kw: None):
        bq.run([str(p) for p in sorted(src.glob("*.xyz"))],
               cfg_override={"model": "lennard-jones"}, work_dir=str(tmp_path / "w"),
               stages=[4])
    assert seen == [3, 7, 11], seen          # the snapshot numbers, not the loop positions


def test_batch_quench_single_snapshot_uses_the_slurm_array_task(tmp_path, monkeypatch):
    """A single-structure run whose file carries no snapshot number must take
    its seed index from SLURM_ARRAY_TASK_ID, so array tasks sharing a --seed do
    not start from identical velocities. Driven through bq.run, not the helper."""
    from unittest.mock import patch
    from ase.build import bulk
    from ase.io import write
    from amorphgen.pipeline import batch_quench as bq
    from amorphgen.utils.common import scoped_run_index
    src = tmp_path / "one.xyz"
    write(str(src), bulk("Cu", cubic=True), format="extxyz")
    seen = []

    def fake_eq(atoms, cfg_override=None, **kw):
        seen.append(cfg_override.get("seed_index"))
        return atoms

    for task in ("4", "9"):
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", task)
        with patch.object(bq.equilibrate, "run", fake_eq), \
             patch.object(bq, "get_calculator", lambda **kw: None):
            bq.run([str(src)], cfg_override={"model": "lennard-jones"},
                   work_dir=str(tmp_path / f"w{task}"), stages=[4])
    assert seen == [scoped_run_index(0, 4, "slurm"), scoped_run_index(0, 9, "slurm")], seen


def test_batch_quench_single_snapshot_with_explicit_run_index(tmp_path, monkeypatch):
    """An explicit --run-index on a single snapshot, through bq.run: two such
    jobs must not share a seed index, and neither may collide with a plain run."""
    from unittest.mock import patch
    from ase.build import bulk
    from ase.io import write
    from amorphgen.pipeline import batch_quench as bq
    src = tmp_path / "one.xyz"
    write(str(src), bulk("Cu", cubic=True), format="extxyz")
    seen = []

    def fake_eq(atoms, cfg_override=None, **kw):
        seen.append(cfg_override.get("seed_index"))
        return atoms

    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    for base in (None, 0, 1):
        cfg = {"model": "lennard-jones"}
        if base is not None:
            cfg["run_index"] = base
        with patch.object(bq.equilibrate, "run", fake_eq), \
             patch.object(bq, "get_calculator", lambda **kw: None):
            bq.run([str(src)], cfg_override=cfg,
                   work_dir=str(tmp_path / f"w{base}"), stages=[4])
    assert len(set(seen)) == 3, seen


def test_scoped_run_index_refuses_out_of_range_values():
    """Wrapping would be silent: snapshot_100003 folding onto snapshot_0003, or
    --run-index 100000 spilling into the next band, puts two runs on one seed
    stream. Both are refused instead."""
    import pytest
    from amorphgen.utils.common import scoped_run_index, _INDEX_STRIDE
    with pytest.raises(ValueError, match="outside"):
        scoped_run_index(_INDEX_STRIDE + 3)
    with pytest.raises(ValueError, match="outside"):
        scoped_run_index(0, _INDEX_STRIDE, "batch")
    with pytest.raises(ValueError, match="unknown run-index source"):
        scoped_run_index(0, 1, "nonsense")
    assert scoped_run_index(_INDEX_STRIDE - 1) == _INDEX_STRIDE - 1
