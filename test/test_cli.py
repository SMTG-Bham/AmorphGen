"""
tests/test_cli.py
-----------------
Tier 1 tests for CLI argument parsing.
"""

import pytest
import sys
from unittest.mock import patch

from amorphgen.cli import parse_args, _parse_composition


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


class TestArgParsing:

    def test_default_model(self):
        with patch("sys.argv", ["amorphgen", "POSCAR"]):
            args = parse_args()
            assert args.model == "mace-mpa-0"
            assert args.device == "cuda"

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
