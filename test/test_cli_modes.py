"""End-to-end smoke tests for each CLI mode.

These exercise the full CLI dispatch path (parse_args → main) for each
top-level mode flag, verifying that the mode runs to completion and
produces the expected output artefacts.  EMT-only (no MLIP), so these
run on Tier 2 hardware (no GPU).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import read, write

from amorphgen.cli import main


# ─── helpers ───────────────────────────────────────────────────────────────

def _run_cli(args_list, monkeypatch):
    """Invoke amorphgen.cli.main() with sys.argv set to args_list."""
    monkeypatch.setattr(sys, "argv", ["amorphgen"] + args_list)
    main()


def _make_si_traj(path: Path, n_frames: int = 5):
    """Create a minimal extxyz trajectory of n_frames Si4 frames."""
    frames = []
    for k in range(n_frames):
        a = Atoms("Si4",
                  positions=[(0, 0, 0), (1.4, 1.4, 0),
                             (1.4, 0, 1.4), (0, 1.4, 1.4)],
                  cell=[3, 3, 3], pbc=True)
        a.positions += 0.01 * k
        a.info["energy"] = -10.0 + 0.1 * k
        frames.append(a)
    write(str(path), frames, format="extxyz")


def _make_si_xyz(path: Path):
    """Single-frame Si4 in extxyz."""
    a = Atoms("Si4",
              positions=[(0, 0, 0), (1.4, 1.4, 0),
                         (1.4, 0, 1.4), (0, 1.4, 1.4)],
              cell=[3, 3, 3], pbc=True)
    write(str(path), a, format="extxyz")


# ─── --list-models ─────────────────────────────────────────────────────────

class TestListModels:
    def test_runs_and_exits_zero(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_cli(["--list-models"], monkeypatch)
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "MACE" in out or "model" in out.lower()


# ─── --random-gen (no relax, no calculator) ───────────────────────────────

class TestRandomGenMode:
    def test_minimal_run_produces_files(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "rand"
        _run_cli([
            "--random-gen",
            "--composition", "Si=8",
            "-n", "2",
            "-o", str(out_dir),
            "--format", "vasp",
        ], monkeypatch)
        files = sorted(out_dir.glob("random_*.vasp"))
        assert len(files) == 2
        for f in files:
            atoms = read(f)
            assert len(atoms) == 8
            assert atoms.get_chemical_formula() == "Si8"


# ─── --extract-snapshots ──────────────────────────────────────────────────

class TestExtractSnapshotsMode:
    def test_extracts_n_frames_default_xyz(self, tmp_path, monkeypatch):
        traj = tmp_path / "traj.xyz"
        _make_si_traj(traj, n_frames=10)
        out_dir = tmp_path / "snaps"
        _run_cli([
            "--extract-snapshots", str(traj),
            "-n", "3",                       # unified count flag
            "--burn-in-frames", "0",
            "-o", str(out_dir),
        ], monkeypatch)
        files = sorted(out_dir.glob("snapshot_*.xyz"))
        assert len(files) == 3

    def test_extracts_n_frames_vasp_format(self, tmp_path, monkeypatch):
        """--format vasp now writes POSCAR-style .vasp files (was hardcoded
        to .xyz before this fix)."""
        traj = tmp_path / "traj.xyz"
        _make_si_traj(traj, n_frames=8)
        out_dir = tmp_path / "snaps_vasp"
        _run_cli([
            "--extract-snapshots", str(traj),
            "-n", "3",
            "--burn-in-frames", "0",
            "-o", str(out_dir),
            "--format", "vasp",
        ], monkeypatch)
        files = sorted(out_dir.glob("snapshot_*.vasp"))
        assert len(files) == 3
        # Confirm it's actually a POSCAR (ASE-readable) and not just a renamed extxyz.
        atoms = read(files[0])
        assert len(atoms) == 4

    def test_n_runs_back_compat(self, tmp_path, monkeypatch):
        """Old --n-runs flag still works for users with existing scripts."""
        traj = tmp_path / "traj.xyz"
        _make_si_traj(traj, n_frames=8)
        out_dir = tmp_path / "snaps_back"
        _run_cli([
            "--extract-snapshots", str(traj),
            "--n-runs", "2",
            "--burn-in-frames", "0",
            "-o", str(out_dir),
        ], monkeypatch)
        files = sorted(out_dir.glob("snapshot_*.xyz"))
        assert len(files) == 2


# ─── --convert ─────────────────────────────────────────────────────────────

class TestConvertMode:
    def test_xyz_to_vasp(self, tmp_path, monkeypatch):
        src = tmp_path / "in.xyz"
        _make_si_xyz(src)
        out_dir = tmp_path / "converted"
        _run_cli([
            "--convert", str(src),
            "--format", "vasp",
            "-o", str(out_dir),
        ], monkeypatch)
        files = list(out_dir.glob("*.vasp"))
        assert len(files) >= 1
        atoms = read(files[0])
        assert len(atoms) == 4


# ─── --analyse (single structure, no reference) ───────────────────────────

class TestAnalyseMode:
    def test_single_structure(self, tmp_path, monkeypatch, capsys):
        src = tmp_path / "struct.xyz"
        a = Atoms("Si64",
                  positions=[(i * 1.4 % 11, (i // 4) * 1.4 % 11,
                              (i // 16) * 1.4 % 11) for i in range(64)],
                  cell=[11, 11, 11], pbc=True)
        write(str(src), a, format="extxyz")
        _run_cli(["--analyse", str(src)], monkeypatch)
        out = capsys.readouterr().out
        # Output should mention pair distances or coordination.
        assert any(tok in out
                   for tok in ("Pair", "Bond", "Coordination",
                                "Coord", "Si-Si"))


# ─── --analyse with --reference ───────────────────────────────────────────

class TestAnalyseWithReference:
    def test_reference_yaml_runs(self, tmp_path, monkeypatch, capsys):
        src = tmp_path / "struct.xyz"
        a = Atoms("Si8",
                  positions=[(i * 1.2, 0, 0) for i in range(8)],
                  cell=[10, 10, 10], pbc=True)
        write(str(src), a, format="extxyz")

        ref = tmp_path / "ref.yaml"
        ref.write_text(
            "system: a-Si\n"
            "references:\n"
            "  - 'Synthetic test reference'\n"
            "bond_distances:\n"
            "  Si-Si:\n"
            "    expected: [1.0, 2.0]\n"
            "    units: 'A'\n"
        )
        _run_cli(["--analyse", str(src), "--reference", str(ref)], monkeypatch)
        out = capsys.readouterr().out
        assert "Validation" in out or "match" in out or "Synthetic" in out


# ─── --rank-from-log ──────────────────────────────────────────────────────

class TestRankFromLogMode:
    def test_parses_log(self, tmp_path, monkeypatch, capsys):
        log = tmp_path / "random_gen.log"
        # Format must match random_gen.batch_random's actual output:
        #   "  Composition: <formula> (<n> atoms)"
        #   LBFGS step rows: "  step  energy  fmax  ..."
        #   "  Converged after N steps!"
        #   "[k/N] <formula> -> .../random_NNNN_opt.<ext>"
        log.write_text(
            "  Composition: Si16 (16 atoms)\n"
            "       0   -52.345    0.95\n"
            "       5   -52.789    0.04\n"
            "  Converged after 5 steps!\n"
            "[1/3] Si16 -> /tmp/fake/random_0000_opt.vasp\n"
            "  Composition: Si16 (16 atoms)\n"
            "       0   -52.500    0.92\n"
            "       6   -53.123    0.03\n"
            "  Converged after 6 steps!\n"
            "[2/3] Si16 -> /tmp/fake/random_0001_opt.vasp\n"
            "  Composition: Si16 (16 atoms)\n"
            "       0   -51.500    0.99\n"
            "       4   -51.876    0.04\n"
            "  Converged after 4 steps!\n"
            "[3/3] Si16 -> /tmp/fake/random_0002_opt.vasp\n"
        )
        _run_cli(["--rank-from-log", str(log)], monkeypatch)
        out = capsys.readouterr().out
        # Lowest-energy entry should be 0001 (-53.123).
        assert "0001" in out
