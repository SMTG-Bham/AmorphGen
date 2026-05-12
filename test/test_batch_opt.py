"""
tests/test_batch_opt.py
------------------------
Tests for batch_optimize and CLI --batch-opt, --analyse.
"""

import os
import pytest
from ase.io import write
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

    def test_batch_optimize_runs(self, input_dir, tmp_path):
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

    def test_batch_optimize_output_files(self, input_dir, tmp_path):
        from amorphgen.pipeline.opt_cell import batch_optimize
        calc = EMT()
        out_dir = str(tmp_path / "output")
        batch_optimize(input_dir=input_dir, output_dir=out_dir, calc=calc)
        # Should have .cif, .xyz, .log for each structure
        files = os.listdir(out_dir)
        assert any(f.endswith(".cif") for f in files)
        assert any(f.endswith(".xyz") for f in files)
        assert any(f.endswith(".log") for f in files)

    def test_batch_optimize_empty_dir(self, tmp_path):
        from amorphgen.pipeline.opt_cell import batch_optimize
        empty = str(tmp_path / "empty")
        os.makedirs(empty)
        paths = batch_optimize(input_dir=empty, output_dir=str(tmp_path / "out"))
        assert paths == []


class TestCLIAnalyse:

    @pytest.fixture
    def structure_dir(self, tmp_path):
        """Create structures for --analyse testing."""
        d = tmp_path / "structures"
        d.mkdir()
        for i in range(2):
            atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
            write(str(d / f"cu_{i:04d}.xyz"), atoms, format="extxyz")
        return str(d)

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
