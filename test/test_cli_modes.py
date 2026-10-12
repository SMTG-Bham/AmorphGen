"""End-to-end smoke tests for each CLI mode.

These exercise the full CLI dispatch path (parse_args → main) for each
top-level mode flag, verifying that the mode runs to completion and
produces the expected output artefacts.  EMT-only (no MLIP), so these
run on Tier 2 hardware (no GPU).
"""

from __future__ import annotations


import numpy as np
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import read, write

from amorphgen_test_helpers import run_cli


# ─── helpers ───────────────────────────────────────────────────────────────

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
            run_cli(monkeypatch, ["--list-models"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "MACE" in out or "model" in out.lower()


# ─── --random-gen (no relax, no calculator) ───────────────────────────────

class TestRandomGenMode:
    def test_minimal_run_produces_files(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "rand"
        run_cli(monkeypatch, [
            "--random-gen",
            "--composition", "Si=8",
            "--seed", "0",
            "-n", "2",
            "-o", str(out_dir),
            "--format", "vasp",
        ])
        # v1.0.0rc2: initial structures land in random_initial/ subdir
        initial_dir = out_dir / "random_initial"
        assert initial_dir.is_dir()
        files = sorted(initial_dir.glob("random_*.vasp"))
        assert len(files) == 2
        for f in files:
            atoms = read(f)
            assert len(atoms) == 8
            assert atoms.get_chemical_formula() == "Si8"

    @pytest.mark.parametrize("engine", ["ase", "torchsim"])
    @pytest.mark.parametrize("opt_yaml, cli_args, expected", [
        ("opt:\n  fmax: 0.123\n  max_steps: 7\n  optimizer: FIRE\n",
         [], (0.123, 7, "FIRE")),
        ("opt:\n  fmax: 0.123\n  max_steps: 7\n  optimizer: FIRE\n",
         ["--fmax", "0.02", "--opt-steps", "3", "--optimizer", "BFGS"],
         (0.02, 3, "BFGS")),
        ("opt:\n  fmax: 0.123\n  max_steps: 7\n  optimizer: FIRE\n",
         ["-f", "0.01", "--opt-steps", "1000", "-O", "LBFGS"],
         (0.01, 1000, "LBFGS")),
        ("opt:\n  fmax: 0.123\n  max_steps: 7\n  optimizer: FIRE\n",
         ["--fmax=0.01", "--opt-steps=1000", "--optimizer=LBFGS"],
         (0.01, 1000, "LBFGS")),
        ("opt:\n  max_steps: 7\n", [], (0.05, 7, "LBFGS")),
        ("model: lj\n", [], (0.05, 1000, "LBFGS")),
        (None, [], (0.05, 1000, "LBFGS")),
    ], ids=["yaml", "cli", "cli-defaults", "cli-equals-defaults",
            "partial-opt", "no-opt", "no-config"])
    def test_relax_settings_precedence(self, tmp_path, monkeypatch, engine,
                                       opt_yaml, cli_args, expected):
        """Both engines receive CLI > YAML > random-gen defaults."""
        from unittest.mock import Mock

        generate = Mock(return_value=[])
        optimize = Mock(return_value=[])
        calculator = Mock(return_value=object())
        monkeypatch.setattr("amorphgen.pipeline.random_gen._batch_random_unlocked", generate)
        monkeypatch.setattr("amorphgen.pipeline.opt_cell.batch_optimize", optimize)
        monkeypatch.setattr("amorphgen.utils.get_calculator", calculator)
        config_args = []
        if opt_yaml is not None:
            config_path = tmp_path / "settings.yaml"
            config_path.write_text(opt_yaml)
            config_args = ["--config", str(config_path)]

        run_cli(monkeypatch, [
            "--random-gen", "--relax", "--composition", "Cu=4",
            "--model", "lj", "--engine", engine, "-o", str(tmp_path / "run"),
            *config_args, *cli_args,
        ])

        generate.assert_called_once()
        generation = generate.call_args.kwargs
        assert generation["relax"] is (engine == "ase")
        assert (generation["fmax"], generation["max_relax_steps"],
                generation["optimizer"]) == expected
        if engine == "torchsim":
            calculator.assert_not_called()
            optimize.assert_called_once()
            relaxation = optimize.call_args.kwargs
            assert relaxation["engine"] == "torchsim"
            assert (relaxation["fmax"], relaxation["max_steps"],
                    relaxation["optimizer"]) == expected
        else:
            calculator.assert_called_once()
            optimize.assert_not_called()

    def test_relax_uses_yaml_optimizer_and_step_limit(self, tmp_path, monkeypatch):
        from ase.calculators.emt import EMT

        config_path = tmp_path / "settings.yaml"
        config_path.write_text(
            "model: lj\n"
            "opt:\n  optimizer: FIRE\n  fmax: 1.0e-12\n"
            "  max_steps: 2\n  cell_filter: none\n"
        )
        monkeypatch.setattr("amorphgen.utils.get_calculator", lambda **kw: EMT())
        out_dir = tmp_path / "run"
        run_cli(monkeypatch, [
            "--random-gen", "--relax", "--composition", "Cu=4", "--no-sc",
            "--seed", "0", "-n", "1", "--config", str(config_path),
            "-o", str(out_dir),
        ])

        log = (out_dir / "random_gen.log").read_text()
        assert "Optimizer: FIRE  fmax=1e-12  max_steps=2" in log
        assert "WARNING: did not converge in 2 steps." in log
        assert (out_dir / "random_opt" / "random_0000_opt.xyz").is_file()


# ─── --extract-snapshots ──────────────────────────────────────────────────

class TestExtractSnapshotsMode:
    def test_extracts_n_frames_default_xyz(self, tmp_path, monkeypatch):
        traj = tmp_path / "traj.xyz"
        _make_si_traj(traj, n_frames=10)
        out_dir = tmp_path / "snaps"
        run_cli(monkeypatch, [
            "--extract-snapshots", str(traj),
            "-n", "3",                       # unified count flag
            "--burn-in-frames", "0",
            "-o", str(out_dir),
        ])
        files = sorted(out_dir.glob("snapshot_*.xyz"))
        assert len(files) == 3

    def test_extracts_n_frames_vasp_format(self, tmp_path, monkeypatch):
        """--format vasp now writes POSCAR-style .vasp files (was hardcoded
        to .xyz before this fix)."""
        traj = tmp_path / "traj.xyz"
        _make_si_traj(traj, n_frames=8)
        out_dir = tmp_path / "snaps_vasp"
        run_cli(monkeypatch, [
            "--extract-snapshots", str(traj),
            "-n", "3",
            "--burn-in-frames", "0",
            "-o", str(out_dir),
            "--format", "vasp",
        ])
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
        run_cli(monkeypatch, [
            "--extract-snapshots", str(traj),
            "--n-runs", "2",
            "--burn-in-frames", "0",
            "-o", str(out_dir),
        ])
        files = sorted(out_dir.glob("snapshot_*.xyz"))
        assert len(files) == 2


# ─── --convert ─────────────────────────────────────────────────────────────

class TestConvertMode:
    def test_xyz_to_vasp(self, tmp_path, monkeypatch):
        src = tmp_path / "in.xyz"
        _make_si_xyz(src)
        out_dir = tmp_path / "converted"
        run_cli(monkeypatch, [
            "--convert", str(src),
            "--format", "vasp",
            "-o", str(out_dir),
        ])
        files = list(out_dir.glob("*.vasp"))
        assert len(files) == 1
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
        run_cli(monkeypatch, ["--analyse", str(src)])
        out = capsys.readouterr().out
        # Output should mention pair distances or coordination.
        assert any(tok in out
                   for tok in ("Pair", "Bond", "Coordination",
                                "Coord", "Si-Si"))

    @pytest.mark.parametrize("method", ["direct", "ft"])
    def test_sq_method_flag(self, tmp_path, monkeypatch, capsys, method):
        """--sq-method selects the S(q) route; direct (default) reports
        q-vectors per shell, ft prints the truncation note and has no
        n_per_bin column."""
        src = tmp_path / "struct.xyz"
        a = Atoms("Si64",
                  positions=[(i * 1.4 % 11, (i // 4) * 1.4 % 11,
                              (i // 16) * 1.4 % 11) for i in range(64)],
                  cell=[11, 11, 11], pbc=True)
        write(str(src), a, format="extxyz")
        plots = tmp_path / "plots"
        run_cli(monkeypatch, ["--analyse", str(src), "--sq", "--sq-method", method,
                  "--save-plot", str(plots)])
        out = capsys.readouterr().out
        assert f"S(q): {method} method" in out
        header = (plots / "analysis_sq.csv").read_text().splitlines()[0]
        if method == "direct":
            assert "n_per_bin" in header
            assert "truncated" not in out
        else:
            assert "n_per_bin" not in header
            assert "truncated at r = L/2" in out

    def test_sq_partials_flag(self, tmp_path, monkeypatch, capsys):
        """--sq-partials adds Faber-Ziman S_ab(q) columns, a partials PNG and
        a printed first-peak table; skipped with a note for --sq-method ft."""
        rng = np.random.default_rng(0)
        a = Atoms("Si32O64", positions=rng.uniform(0, 12, (96, 3)),
                  cell=[12, 12, 12], pbc=True)
        src = tmp_path / "sio.xyz"; write(str(src), a, format="extxyz")
        plots = tmp_path / "plots"
        run_cli(monkeypatch, ["--analyse", str(src), "--sq", "--sq-partials",
                  "--save-plot", str(plots)])
        out = capsys.readouterr().out
        assert "Faber-Ziman partials S_ab(q)" in out and "O-Si" in out
        header = (plots / "analysis_sq.csv").read_text().splitlines()[0].split(",")
        assert {"s_Si-Si", "s_O-Si", "s_O-O"} <= set(header)
        assert (plots / "analysis_sq_partials.png").exists()
        run_cli(monkeypatch, ["--analyse", str(src), "--sq", "--sq-partials", "--pair-panels",
                  "--save-plot", str(tmp_path / "p3")])
        assert (tmp_path / "p3" / "analysis_rdf_panels.png").exists()
        assert (tmp_path / "p3" / "analysis_sq_partials_panels.png").exists()
        run_cli(monkeypatch, ["--analyse", str(src), "--sq", "--sq-partials", "--sq-method", "ft",
                  "--save-plot", str(tmp_path / "p2")])
        assert "partials skipped" in capsys.readouterr().out

    def test_cutoff_overrides(self, tmp_path, monkeypatch, capsys):
        """Per-pair overrides on top of auto-rdf, with a base prefix, and the
        total-coordination block for an element with several bonded partners."""
        from amorphgen.analysis.cutoff import parse_cutoff_spec
        assert parse_cutoff_spec("auto-rdf") == "auto-rdf"
        assert parse_cutoff_spec("2.4") == 2.4
        assert parse_cutoff_spec("In-O=2.6, Zn-O=2.3") == {"In-O": 2.6, "Zn-O": 2.3}
        assert parse_cutoff_spec("auto,In-O=2.6") == {"default": "auto", "In-O": 2.6}
        assert parse_cutoff_spec({"default": 2.4, "In-O": 2.6}) == {"default": 2.4, "In-O": 2.6}
        with pytest.raises(ValueError):
            parse_cutoff_spec("In-O-Zn=2")
        rng = np.random.default_rng(1)
        a = Atoms("Ga16Zn16O48", positions=rng.uniform(0, 11, (80, 3)),
                  cell=[11, 11, 11], pbc=True)
        src = tmp_path / "gzo.xyz"; write(str(src), a, format="extxyz")
        run_cli(monkeypatch, ["--analyse", str(src), "--cutoff", "2.2,Ga-O=1.9"])
        out = capsys.readouterr().out
        assert "Cutoff mode: fixed 2.20 A + overrides Ga-O=1.90" in out
        assert "Ga-O: 1.90 A" in out and "O-Zn: 2.20 A" in out
        assert "Total coordination (all bonded partners):" in out
        assert "O-(Ga+Zn):" in out
        with pytest.raises(SystemExit):
            run_cli(monkeypatch, ["--analyse", str(src), "--cutoff", "Ga-O=abc"])
        assert "Error: could not convert" in capsys.readouterr().out
        plots = tmp_path / "cnplots"
        run_cli(monkeypatch, ["--analyse", str(src), "--total-cn", "O", "--total-cn", "O:Ga",
                  "--total-cn", "Xe", "--save-plot", str(plots)])
        out = capsys.readouterr().out
        assert "Total coordination (requested):" in out
        assert "O-(all bonded): mean=" in out and "O-(Ga): mean=" in out
        assert "Xe: not present" in out
        assert (plots / "analysis_cn_total.png").exists()
        assert "g(r)_Total" in (plots / "analysis_rdf.csv").read_text().splitlines()[0]
        cn_csv = (plots / "analysis_cn.csv").read_text()
        # multi-cation: bonded pairs + the anion total, no cation-cation pair
        assert "O-(Ga+Zn)," in cn_csv and "Ga-O," in cn_csv and "Ga-Zn," not in cn_csv
        assert (plots / "analysis_cn_total.csv").read_text().count("O-(Ga)") > 0


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
        run_cli(monkeypatch, ["--analyse", str(src), "--reference", str(ref)])
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
        run_cli(monkeypatch, ["--rank-from-log", str(log)])
        out = capsys.readouterr().out
        # Lowest-energy entry should be 0001 (-53.123).
        assert "Best : random_0001_opt" in out
        assert "Worst: random_0002_opt" in out
        rows = [line.split() for line in out.splitlines()
                if line.strip() and line.split()[0].isdigit()]
        assert [int(row[1]) for row in rows] == [1, 0, 2]
        assert [float(row[2]) for row in rows] == [-53.123, -52.789, -51.876]


class TestRingsAndVoronoiFlags:
    """--rings / --voronoi print sections, extend the report and write CSVs."""

    def _sio2(self, tmp_path):
        from ase.spacegroup import crystal
        crist = crystal(["Si", "O"], basis=[(0, 0, 0), (0.125, 0.125, 0.125)], spacegroup=227,
                        cellpar=[7.16, 7.16, 7.16, 90, 90, 90]).repeat((2, 2, 2))
        crist.rattle(0.05, seed=1)
        src = tmp_path / "in"; src.mkdir()
        write(str(src / "s.xyz"), crist, format="extxyz")
        return src

    def test_rings_and_voronoi(self, tmp_path, monkeypatch, capsys):
        src = self._sio2(tmp_path); plots = tmp_path / "plots"; rep = tmp_path / "r" / "report.txt"
        run_cli(monkeypatch, ["--analyse", "--input-dir", str(src), "--rings", "--voronoi", "Si",
                  "--save-plot", str(plots), "--save-report", str(rep)])
        out = capsys.readouterr().out
        assert "Ring statistics" in out and " 6-ring:" in out       # cristobalite: 6-rings, Si nodes
        assert "Voronoi indices" in out
        text = rep.read_text()
        assert "Ring statistics" in text and "Voronoi indices" in text
        assert (plots / "analysis_rings.csv").exists() and (plots / "analysis_rings.png").exists()
        assert (plots / "analysis_voronoi.csv").exists()
        assert "6," in (plots / "analysis_rings.csv").read_text()

    def test_rings_explicit_pair(self, tmp_path, monkeypatch, capsys):
        src = self._sio2(tmp_path)
        run_cli(monkeypatch, ["--analyse", "--input-dir", str(src), "--rings", "Si-O"])
        assert "nodes-bridge: Si-O" in capsys.readouterr().out


def test_connectivity_flag(tmp_path, monkeypatch, capsys):
    from ase.spacegroup import crystal
    crist = crystal(["Si", "O"], basis=[(0, 0, 0), (0.125, 0.125, 0.125)], spacegroup=227,
                    cellpar=[7.16, 7.16, 7.16, 90, 90, 90]).repeat((2, 2, 2))
    src = tmp_path / "in"; src.mkdir(); write(str(src / "s.xyz"), crist, format="extxyz")
    plots = tmp_path / "plots"; rep = tmp_path / "report.txt"
    run_cli(monkeypatch, ["--analyse", "--input-dir", str(src), "--connectivity", "--cutoff", "auto",
              "--save-plot", str(plots), "--save-report", str(rep)])
    out = capsys.readouterr().out
    assert "Polyhedral connectivity" in out and "corner 100.0%" in out
    assert "Polyhedral connectivity" in rep.read_text()
    assert (plots / "analysis_connectivity.csv").exists()


def test_run_index_flag_reaches_the_config(monkeypatch):
    """Review round 11: --run-index must arrive in the merged config. Dropping
    the mapping line in cli.py makes the flag silently do nothing."""
    import sys
    from amorphgen.cli import _get_parser, _build_override
    parser = _get_parser()
    argv = ["--random-gen", "--composition", "Si=8", "--run-index", "7", "--seed", "3"]
    args = parser.parse_args(argv)
    monkeypatch.setattr(sys, "argv", ["amorphgen"] + argv)
    override = _build_override(args, parser, explicit_only=True, argv=argv)
    assert override.get("run_index") == 7
    assert override.get("seed") == 3
    # and it is absent when not given, so it cannot shadow another source
    args2 = parser.parse_args(["--random-gen", "--composition", "Si=8"])
    o2 = _build_override(args2, parser, explicit_only=True,
                         argv=["--random-gen", "--composition", "Si=8"])
    assert o2.get("run_index") is None
