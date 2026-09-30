"""
tests/test_batch_opt.py
------------------------
Tests for batch_optimize and CLI --batch-opt, --analyse.
"""

import os
import pytest
from ase.io import read, write
from ase.build import bulk
from ase.calculators.emt import EMT


class TestBatchOptimize:

    @pytest.fixture
    def input_dir(self, tmp_path):
        """Create a directory with Cu structures for testing."""
        d = tmp_path / "input"
        d.mkdir()
        for i in range(3):
            atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
            # Add small perturbation
            atoms.positions += 0.1 * (i + 1)
            write(str(d / f"struct_{i:04d}.extxyz"), atoms, format="extxyz")
        return str(d)

    def test_batch_optimize_writes_outputs_for_each_structure(self, input_dir, tmp_path):
        from amorphgen.pipeline.opt_cell import batch_optimize
        calc = EMT()
        out_dir = str(tmp_path / "output")
        paths = batch_optimize(
            input_dir=input_dir,
            output_dir=out_dir,
            calc=calc,
        )
        assert len(paths) == 3
        for p in paths:
            assert os.path.exists(p)
            assert read(p).get_chemical_formula() == "Cu4"
        output = tmp_path / "output"
        for index in range(3):
            for extension in ("cif", "xyz", "log"):
                path = output / f"struct_{index:04d}_opt.{extension}"
                assert path.is_file(), f"Missing {path.name}"
                assert path.stat().st_size > 0

    def test_batch_optimize_empty_dir(self, tmp_path):
        from amorphgen.pipeline.opt_cell import batch_optimize
        empty = str(tmp_path / "empty")
        os.makedirs(empty)
        paths = batch_optimize(input_dir=empty, output_dir=str(tmp_path / "out"))
        assert paths == []


class TestCLIAnalyse:

    def test_cli_parse_analyse_flag(self):
        from amorphgen.cli import _get_parser
        parser = _get_parser()
        args = parser.parse_args(["--analyse", "--input-dir", "test/"])
        assert args.analyse is True
        assert args.input_dir == "test/"

    def test_cli_parse_cutoff_auto(self):
        from amorphgen.cli import _get_parser
        parser = _get_parser()
        args = parser.parse_args(["--analyse", "--input-dir", "test/",
                                  "--cutoff", "auto-rdf"])
        assert args.cutoff == "auto-rdf"

    def test_cli_parse_cutoff_float(self):
        from amorphgen.cli import _get_parser
        parser = _get_parser()
        args = parser.parse_args(["--analyse", "--input-dir", "test/",
                                  "--cutoff", "2.5"])
        assert args.cutoff == "2.5"

    def test_cli_parse_batch_opt(self):
        from amorphgen.cli import _get_parser
        parser = _get_parser()
        args = parser.parse_args(["--batch-opt", "--input-dir", "input/",
                                  "--work-dir", "output/"])
        assert args.batch_opt is True
        assert args.input_dir == "input/"

    def test_cli_parse_config(self):
        from amorphgen.cli import _get_parser
        parser = _get_parser()
        args = parser.parse_args(["--random-gen", "--config", "test.yaml",
                                  "--composition", "Si=16,O=32"])
        assert args.config == "test.yaml"

    def test_cli_parse_timestep(self):
        from amorphgen.cli import _get_parser
        parser = _get_parser()
        args = parser.parse_args(["POSCAR", "--timestep", "0.5"])
        assert args.timestep == 0.5

    def test_cli_parse_target_cn(self):
        from amorphgen.cli import _get_parser, _parse_target_cn
        parser = _get_parser()
        args = parser.parse_args(["--random-gen", "--composition", "Si=16,O=32",
                                  "--target-cn", "Si=4,O=2"])
        cn = _parse_target_cn(args.target_cn)
        assert cn == {"Si": 4, "O": 2}

    def test_cli_parse_dmax(self):
        from amorphgen.cli import _parse_dmax
        dmax = _parse_dmax("Si-O=2.0,O-O=3.2")
        assert dmax == {"Si-O": 2.0, "O-O": 3.2}


class TestRandomGenOutputDir:
    """--random-gen writes to <dir>/random_initial/ (and random_opt/ with
    --relax), never to <dir>/ itself. The next step is pointed at the
    subdirectory; given the work dir it must say where to look and exit 1."""

    # Lennard-Jones: torch-free, so these run on the bare install too
    LJ_YAML = ("model: lj\ndevice: cpu\n"
               "classical_params:\n  params:\n    Cu-Cu: {sigma: 2.3, epsilon: 0.1}\n"
               "  cutoff: 6.0\n"
               "opt:\n  fmax: 0.5\n  max_steps: 5\n  cell_filter: none\n")

    def _cli(self, monkeypatch, *argv):
        import sys
        from amorphgen.cli import main
        monkeypatch.setattr(sys, "argv", ["amorphgen", *argv])
        main()

    @pytest.fixture
    def gen_dir(self, tmp_path, monkeypatch):
        """A real --random-gen work dir: two Cu8 placements, not relaxed."""
        out = tmp_path / "gen"
        self._cli(monkeypatch, "--random-gen", "--composition", "Cu=8", "-n", "2",
                  "--seed", "1", "-o", str(out))
        return out

    @pytest.fixture
    def lj_cfg(self, tmp_path):
        cfg = tmp_path / "lj.yaml"
        cfg.write_text(self.LJ_YAML)
        return str(cfg)

    def test_hint_lists_subdirs_holding_structures(self, gen_dir, tmp_path):
        from amorphgen.pipeline.random_gen import random_gen_dir_hint
        (gen_dir / "random_opt").mkdir()
        (gen_dir / "random_opt" / "random_0000_opt.log").write_text("")  # not a structure
        hint = random_gen_dir_hint(str(gen_dir))
        assert os.path.join(str(gen_dir), "random_initial") in hint
        assert "random_opt" not in hint
        assert random_gen_dir_hint(str(tmp_path / "gen" / "random_initial")) == ""

    def test_batch_optimize_on_work_dir_prints_hint(self, gen_dir, tmp_path, capsys):
        from amorphgen.pipeline.opt_cell import batch_optimize
        assert batch_optimize(input_dir=str(gen_dir), output_dir=str(tmp_path / "o"),
                              calc=EMT()) == []
        assert os.path.join(str(gen_dir), "random_initial") in capsys.readouterr().out

    def test_cli_batch_opt_on_work_dir_exits_1(self, gen_dir, lj_cfg, tmp_path,
                                               monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            self._cli(monkeypatch, "--batch-opt", "--input-dir", str(gen_dir),
                      "--config", lj_cfg, "-o", str(tmp_path / "opt"))
        assert exc.value.code == 1
        assert os.path.join(str(gen_dir), "random_initial") in capsys.readouterr().out

    def test_cli_batch_opt_on_random_initial_runs(self, gen_dir, lj_cfg, tmp_path,
                                                  monkeypatch):
        """The documented two-step workflow: generate, then --batch-opt the
        random_initial/ subdirectory."""
        self._cli(monkeypatch, "--batch-opt", "--input-dir", str(gen_dir / "random_initial"),
                  "--config", lj_cfg, "-o", str(tmp_path / "opt"))
        assert sorted(p.name for p in (tmp_path / "opt").glob("*_opt.xyz")) == [
            "random_0000_opt.xyz", "random_0001_opt.xyz"]

    def test_cli_hybrid_ensemble_on_work_dir_exits_1(self, gen_dir, lj_cfg, tmp_path,
                                                     monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            self._cli(monkeypatch, "--hybrid-ensemble", "--input-dir", str(gen_dir),
                      "--config", lj_cfg, "-o", str(tmp_path / "hyb"))
        assert exc.value.code == 1
        assert os.path.join(str(gen_dir), "random_initial") in capsys.readouterr().out
