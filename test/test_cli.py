"""
tests/test_cli.py
-----------------
Tier 1 tests for CLI argument parsing and input validation.
"""

import pytest
import sys
from unittest.mock import patch

from amorphgen.cli import (
    parse_args, _parse_composition, _parse_minsep,
    _parse_target_cn, _parse_dmax,
)


class TestParseComposition:

    def test_simple(self):
        assert _parse_composition("In=32,O=48") == {"In": 32, "O": 48}

    def test_single_element(self):
        assert _parse_composition("Cu=100") == {"Cu": 100}

    def test_with_spaces(self):
        assert _parse_composition("Si = 8 , O = 16") == {"Si": 8, "O": 16}

    def test_ternary(self):
        result = _parse_composition("In=16,Ga=16,O=48")
        assert result == {"In": 16, "Ga": 16, "O": 48}

    def test_missing_equals(self):
        with pytest.raises(ValueError, match="Invalid composition"):
            _parse_composition("In32,O=48")

    def test_empty_string(self):
        with pytest.raises(ValueError, match="Empty composition"):
            _parse_composition("")

    def test_non_integer_count(self):
        with pytest.raises(ValueError, match="Invalid count"):
            _parse_composition("Si=3.5")

    def test_zero_count(self):
        with pytest.raises(ValueError, match="must be positive"):
            _parse_composition("Si=0")

    def test_negative_count(self):
        with pytest.raises(ValueError, match="must be positive"):
            _parse_composition("Si=-4")

    def test_invalid_element_symbol(self):
        with pytest.raises(ValueError, match="Unknown element symbol 'Ox'"):
            _parse_composition("In=16,Ox=24")

    def test_case_sensitive_element(self):
        with pytest.raises(ValueError, match="Unknown element symbol"):
            _parse_composition("si=16,o=32")

    # -- Formula format --

    def test_formula_with_multiplier(self):
        result = _parse_composition("In2O3*16")
        assert result == {"In": 32, "O": 48}

    def test_formula_simple(self):
        result = _parse_composition("SiO2*16")
        assert result == {"Si": 16, "O": 32}

    def test_formula_no_multiplier(self):
        result = _parse_composition("Si64")
        assert result == {"Si": 64}

    def test_formula_single_unit(self):
        result = _parse_composition("In2O3*1")
        assert result == {"In": 2, "O": 3}

    def test_formula_invalid(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            _parse_composition("XyzAbc*10")

    def test_formula_zero_multiplier(self):
        with pytest.raises(ValueError, match="must be positive"):
            _parse_composition("SiO2*0")

    def test_trailing_comma(self):
        assert _parse_composition("Si=8,O=16,") == {"Si": 8, "O": 16}


class TestParseMinsep:

    def test_simple(self):
        result = _parse_minsep("Si-O=1.6,O-O=2.2")
        assert result == {"Si-O": 1.6, "O-O": 2.2}

    def test_missing_equals(self):
        with pytest.raises(ValueError, match="Invalid minsep"):
            _parse_minsep("Si-O 1.6")

    def test_missing_dash(self):
        with pytest.raises(ValueError, match="Invalid pair format"):
            _parse_minsep("SiO=1.6")

    def test_non_numeric(self):
        with pytest.raises(ValueError, match="Invalid minsep value"):
            _parse_minsep("Si-O=abc")

    def test_negative(self):
        with pytest.raises(ValueError, match="must be positive"):
            _parse_minsep("Si-O=-1.0")

    def test_zero(self):
        with pytest.raises(ValueError, match="must be positive"):
            _parse_minsep("Si-O=0")


class TestParseTargetCn:

    def test_simple(self):
        assert _parse_target_cn("Si=4,O=2") == {"Si": 4, "O": 2}

    def test_missing_equals(self):
        with pytest.raises(ValueError, match="Invalid target-cn"):
            _parse_target_cn("Si4")

    def test_non_integer(self):
        with pytest.raises(ValueError, match="Invalid CN"):
            _parse_target_cn("Si=4.5")

    def test_negative(self):
        with pytest.raises(ValueError, match="must be positive"):
            _parse_target_cn("Si=-1")


class TestParseDmax:

    def test_simple(self):
        assert _parse_dmax("Si-O=2.0") == {"Si-O": 2.0}

    def test_missing_dash(self):
        with pytest.raises(ValueError, match="Invalid pair format"):
            _parse_dmax("SiO=2.0")

    def test_negative(self):
        with pytest.raises(ValueError, match="must be positive"):
            _parse_dmax("Si-O=-1.0")


class TestArgParsing:

    def test_default_model(self):
        with patch("sys.argv", ["amorphgen", "POSCAR"]):
            args = parse_args()
            assert args.model == "mace-mpa-0"
            assert args.device == "auto"

    def test_custom_model(self):
        with patch("sys.argv", ["amorphgen", "POSCAR", "--model", "chgnet"]):
            args = parse_args()
            assert args.model == "chgnet"

    def test_model_path(self):
        with patch("sys.argv", ["amorphgen", "POSCAR",
                                "--model-path", "/tmp/my.model"]):
            args = parse_args()
            assert args.model_path == "/tmp/my.model"

    def test_stages(self):
        with patch("sys.argv", ["amorphgen", "POSCAR", "--stages", "1", "2", "3"]):
            args = parse_args()
            assert args.stages == [1, 2, 3]

    def test_random_gen_flag(self):
        with patch("sys.argv", ["amorphgen", "--random-gen",
                                "--composition", "Cu=10"]):
            args = parse_args()
            assert args.random_gen is True
            assert args.composition == "Cu=10"

    def test_batch_quench_flag(self):
        with patch("sys.argv", ["amorphgen", "--batch-quench",
                                "--snapshot-dir", "snaps/",
                                "--n-runs", "5", "--resume"]):
            args = parse_args()
            assert args.batch_quench is True
            assert args.resume is True
            assert args.n_runs == 5


class TestConvertMode:
    """End-to-end smoke test for `amorphgen --convert`."""

    def test_convert_directory_to_vasp(self, tmp_path):
        """Converts every xyz/extxyz file in a directory to .vasp,
        sorted by species, into the user-supplied --work-dir."""
        from ase.build import bulk
        from ase.io import read, write
        from amorphgen.cli import _run_convert

        src = tmp_path / "snaps"
        src.mkdir()
        for i in range(3):
            atoms = bulk("Cu", "fcc", a=3.6 + 0.001 * i, cubic=True) * (2, 2, 2)
            write(str(src / f"snap_{i:04d}.xyz"), atoms, format="extxyz")

        out = tmp_path / "snaps_vasp"

        class A:  # minimal args duck-type
            convert = str(src)
            format = "vasp"
            work_dir = str(out)

        _run_convert(A())

        outputs = sorted(out.glob("*.vasp"))
        assert len(outputs) == 3
        # Each file is a valid VASP POSCAR readable by ASE.
        for f in outputs:
            atoms = read(str(f))
            assert len(atoms) == 32  # 4 atoms/fcc * 2*2*2

    def test_convert_single_file(self, tmp_path):
        """Single-file mode writes one converted output next to or in --work-dir."""
        from ase.build import bulk
        from ase.io import read, write
        from amorphgen.cli import _run_convert

        src = tmp_path / "in.xyz"
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
        write(str(src), atoms, format="extxyz")

        out_dir = tmp_path / "converted"

        class A:
            convert = str(src)
            format = "vasp"
            work_dir = str(out_dir)

        _run_convert(A())
        outputs = sorted(out_dir.glob("*.vasp"))
        assert len(outputs) == 1
        assert read(str(outputs[0])).get_chemical_symbols().count("Cu") == 32

    def test_python_api(self, tmp_path):
        """Public Python API: amorphgen.convert() returns list of paths."""
        from ase.build import bulk
        from ase.io import write
        from amorphgen import convert

        src = tmp_path / "snaps"
        src.mkdir()
        for i in range(2):
            atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
            write(str(src / f"snap_{i:04d}.xyz"), atoms, format="extxyz")

        out = tmp_path / "snaps_vasp"
        paths = convert(str(src), output_format="vasp",
                        output_dir=str(out), verbose=False)
        assert len(paths) == 2
        assert all(p.endswith(".vasp") for p in paths)

    def test_yaml_driven(self, tmp_path):
        """YAML-driven convert: --config foo.yaml with a convert: block,
        no --convert CLI flag, runs convert mode using YAML values."""
        from ase.build import bulk
        from ase.io import write
        from amorphgen.cli import _run_convert

        src = tmp_path / "snaps"
        src.mkdir()
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
        write(str(src / "snap_0000.xyz"), atoms, format="extxyz")

        out_dir = tmp_path / "yaml_out"
        yaml_cfg = {
            "convert": {
                "input": str(src),
                "format": "vasp",
                "output_dir": str(out_dir),
            }
        }

        class A:
            convert = None       # CLI flag NOT set; comes from YAML
            format = None
            work_dir = None

        _run_convert(A(), yaml_cfg=yaml_cfg)
        assert sorted(out_dir.glob("*.vasp"))
