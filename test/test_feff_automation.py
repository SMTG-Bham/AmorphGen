"""From structures to spectra without a FEFF binary: plots, tables and the
automated write -> run -> collect path, driven by a stand-in executable."""
import csv
import json
import sys
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from amorphgen.analysis.feff import (
    collect_feff_results, feff_pipeline, feff_uncertainty, find_feff_trees,
    plot_feff, run_feff_jobs, write_feff_jobs,
)

# Writes a damped single-shell chi.dat. The shell distance depends on the
# input text, so different sites give different spectra.
FAKE_FEFF = """
import math
r0 = 1.55 + (len(open('feff.inp').read()) % 13) * 0.01
with open('chi.dat', 'w') as fh:
    fh.write('# fake feff\\n#  k  chi  mag  phase\\n')
    for i in range(401):
        k = 0.05 * i
        c = math.exp(-0.01 * k * k) * math.sin(2 * k * r0) / max(k, 1e-9) ** 2
        fh.write(f'{k:11.4f}{c:14.6e}{abs(c):14.6e}{0.0:14.6e}\\n')
"""


def executable(tmp_path, body=FAKE_FEFF, name="fake-feff"):
    path = tmp_path / name
    path.write_text(f"#!{sys.executable}\n{body}\n")
    path.chmod(0o755)
    return str(path)


def cell(shift=0.0):
    """SiO2-ish 8 A cell, 2 Si and 4 O; ``shift`` distorts it slightly."""
    return Atoms(symbols=["Si", "Si", "O", "O", "O", "O"],
                 positions=[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0],
                            [1.6 + shift, 0.0, 0.0], [0.0, 1.6, 0.0],
                            [0.0, 0.0, 1.6 - shift], [5.6, 0.0, 0.0]],
                 cell=[8.0, 8.0, 8.0], pbc=True)


def ensemble():
    return [cell(0.0), cell(0.04), cell(-0.03)]


def rows(path):
    with open(path, newline="") as fh:
        return list(csv.reader(fh))


# ─── run_feff_jobs: workers and resume ────────────────────────────────────

class TestRunner:
    def test_parallel_workers_run_every_job(self, tmp_path):
        tree = tmp_path / "jobs"
        manifest = write_feff_jobs(ensemble(), "Si", str(tree), radius=4.0)
        out = run_feff_jobs(tree, executable(tmp_path), workers=3)
        assert out == {"ran": manifest["n_jobs"], "skipped": 0, "failed": []}
        states = {json.loads((tree / j["dir"] / "run_status.json").read_text())["state"]
                  for j in manifest["jobs"]}
        assert states == {"complete"}

    def test_skip_complete_reruns_only_what_needs_it(self, tmp_path):
        tree = tmp_path / "jobs"
        manifest = write_feff_jobs(ensemble(), "Si", str(tree), radius=4.0)
        binary = executable(tmp_path)
        run_feff_jobs(tree, binary)
        altered = tree / manifest["jobs"][0]["dir"] / "chi.dat"
        altered.write_text(altered.read_text() + "\n")
        out = run_feff_jobs(tree, binary, skip_complete=True)
        assert out == {"ran": 1, "skipped": manifest["n_jobs"] - 1, "failed": []}
        # without skip_complete every job runs again, as before
        assert run_feff_jobs(tree, binary)["ran"] == manifest["n_jobs"]

    @pytest.mark.parametrize("workers", [0, -1, 1.5, True])
    def test_workers_must_be_a_positive_integer(self, tmp_path, workers):
        tree = tmp_path / "jobs"
        write_feff_jobs([cell()], "Si", str(tree), radius=4.0)
        with pytest.raises(ValueError, match="workers"):
            run_feff_jobs(tree, executable(tmp_path), workers=workers)


# ─── figures and tables ───────────────────────────────────────────────────

class TestPlotAndTables:
    @pytest.fixture
    def result(self, tmp_path):
        tree = tmp_path / "jobs"
        write_feff_jobs(ensemble(), "Si", str(tree), radius=4.0)
        run_feff_jobs(tree, executable(tmp_path))
        return collect_feff_results(tree)

    def test_per_structure_curves_are_carried(self, result):
        per = result["chi_per_structure"]
        assert per.shape == (3, len(result["k"]))
        assert np.allclose(per.mean(axis=0), result["chi"])

    def test_bootstrap_bands_bracket_the_curves(self, result):
        chik, chir = feff_uncertainty(result)
        kw = result["kweight"]
        mean = np.asarray(chik["mean"], dtype=float)
        assert np.allclose(mean, result["chi"] * result["k"] ** kw)
        low, high = (np.asarray(chir[key], dtype=float)
                     for key in ("bootstrap_low", "bootstrap_high"))
        assert low.shape == high.shape == result["r"].shape
        assert np.all(low <= high + 1e-15)
        assert chik["sampling_unit"] == chir["sampling_unit"] == "structure"

    def test_figures_tables_and_uncertainty_files(self, result, tmp_path):
        out = tmp_path / "plots"
        plot_feff(result, output_dir=str(out), prefix="run", save_pdf=True)
        for base in ("run_feff_chik", "run_feff_chir"):
            assert (out / f"{base}.png").exists() and (out / f"{base}.pdf").exists()
        for name in ("run_feff_chik_uncertainty.csv", "run_feff_chik_uncertainty.json",
                     "run_feff_chik_per_structure.csv"):
            assert (out / name).exists()
        table = rows(out / "run_feff_chir.csv")
        assert table[1][-2:] == ["chir_mag_bootstrap_low", "chir_mag_bootstrap_high"]
        assert all(value != "" for value in table[2][-2:])
        assert len(table) == 2 + len(result["r"])

    def test_save_csv_false_writes_figures_only(self, result, tmp_path):
        out = tmp_path / "plots"
        plot_feff(result, output_dir=str(out), save_csv=False)
        assert sorted(p.name for p in out.iterdir()) == [
            "analysis_feff_chik.png", "analysis_feff_chir.png"]


# ─── feff_pipeline: one call from structures to spectra ───────────────────

class TestPipeline:
    def test_several_absorbers_end_to_end(self, tmp_path):
        out = feff_pipeline(ensemble(), "Si,O", str(tmp_path / "feff"),
                            executable=executable(tmp_path), workers=2,
                            radius=4.0, plot_dir=str(tmp_path / "plots"))
        assert list(out) == ["Si", "O"]
        for element in ("Si", "O"):
            entry = out[element]
            assert entry["jobs_dir"] == str(tmp_path / "feff" / element)
            assert entry["run"]["failed"] == [] and entry["result"] is not None
            assert f"{element} K edge" in entry["report"]
            assert (tmp_path / "plots" / f"analysis_{element}_feff_chik.png").exists()
            assert (tmp_path / "plots" / f"analysis_{element}_feff_chir.csv").exists()
        # restarting keeps every finished job
        again = feff_pipeline(ensemble(), ["Si", "O"], str(tmp_path / "feff"),
                              executable=executable(tmp_path), radius=4.0)
        assert all(e["run"]["ran"] == 0 and e["run"]["skipped"] == e["n_jobs"]
                   for e in again.values())

    def test_single_absorber_uses_output_dir_itself(self, tmp_path):
        out = feff_pipeline([cell()], "Si", str(tmp_path / "feff"),
                            executable=executable(tmp_path), radius=4.0)
        assert out["Si"]["jobs_dir"] == str(tmp_path / "feff")
        assert (tmp_path / "feff" / "manifest.json").exists()

    def test_failed_tree_is_not_averaged(self, tmp_path):
        out = feff_pipeline([cell()], "Si", str(tmp_path / "feff"),
                            executable=executable(tmp_path, "raise SystemExit(2)"),
                            radius=4.0, plot_dir=str(tmp_path / "plots"))
        assert out["Si"]["result"] is None and out["Si"]["run"]["failed"]
        assert not (tmp_path / "plots").exists()

    def test_cards_reach_every_input(self, tmp_path):
        feff_pipeline([cell()], "Si", str(tmp_path / "feff"),
                      executable=executable(tmp_path), radius=4.0,
                      cards={"RPATH": "3.5", "CRITERIA": "1.0 0.5"})
        text = next((tmp_path / "feff").glob("s*/feff.inp")).read_text()
        assert any(line.split()[:2] == ["RPATH", "3.5"] for line in text.splitlines())
        assert any(line.split() == ["CRITERIA", "1.0", "0.5"] for line in text.splitlines())

    def test_no_absorber_is_an_error(self, tmp_path):
        with pytest.raises(ValueError, match="absorbing element"):
            feff_pipeline([cell()], " , ", str(tmp_path / "feff"))

    def test_find_feff_trees(self, tmp_path):
        feff_pipeline(ensemble(), "Si,O", str(tmp_path / "feff"),
                      executable=executable(tmp_path), radius=4.0)
        assert find_feff_trees(tmp_path / "feff") == [
            str(tmp_path / "feff" / "O"), str(tmp_path / "feff" / "Si")]
        assert find_feff_trees(tmp_path / "feff" / "Si") == [str(tmp_path / "feff" / "Si")]
        with pytest.raises(FileNotFoundError, match="manifest.json"):
            find_feff_trees(tmp_path)


# ─── CLI and YAML ─────────────────────────────────────────────────────────

class TestCLI:
    @pytest.fixture
    def structures(self, tmp_path):
        d = tmp_path / "structs"
        d.mkdir()
        for n, atoms in enumerate(ensemble()):
            write(str(d / f"s_{n}.xyz"), atoms, format="extxyz")
        return d

    def run(self, argv, capsys):
        from amorphgen.cli import main
        with patch("sys.argv", ["amorphgen", *argv]):
            main()
        return capsys.readouterr().out

    def test_one_command_from_structures_to_spectra(self, structures, tmp_path, capsys):
        jobs, plots = tmp_path / "feff", tmp_path / "plots"
        argv = ["--analyse", "--input-dir", str(structures),
                "--feff-write", str(jobs), "--feff-absorber", "Si,O",
                "--feff-radius", "4.0", "--feff-run",
                "--feff-executable", executable(tmp_path), "--feff-workers", "2",
                "--feff-card", "RPATH=3.5", "--save-plot", str(plots)]
        out = self.run(argv, capsys)
        assert out.count("Configurationally averaged EXAFS") == 2
        assert "Cards:       RPATH 3.5" in out
        for element in ("Si", "O"):
            assert (jobs / element / "manifest.json").exists()
            for base in (f"analysis_{element}_feff_chik", f"analysis_{element}_feff_chir"):
                assert (plots / f"{base}.png").exists()
                assert (plots / f"{base}.csv").exists()
        out = self.run([*argv, "--resume"], capsys)
        assert "0 ok, " in out and "already complete" in out

    def test_collect_reads_a_directory_of_trees(self, structures, tmp_path, capsys):
        feff_pipeline(ensemble(), "Si,O", str(tmp_path / "feff"),
                      executable=executable(tmp_path), radius=4.0)
        out = self.run(["--analyse", "--input-dir", str(structures),
                        "--feff-collect", str(tmp_path / "feff")], capsys)
        assert "O K edge" in out and "Si K edge" in out

    def test_executable_from_environment(self, structures, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("FEFF_EXECUTABLE", executable(tmp_path))
        out = self.run(["--analyse", "--input-dir", str(structures),
                        "--feff-write", str(tmp_path / "feff"), "--feff-absorber", "Si",
                        "--feff-radius", "4.0", "--feff-run"], capsys)
        assert "Configurationally averaged EXAFS" in out

    def test_malformed_card_is_rejected(self, structures, tmp_path):
        from amorphgen.cli import main
        argv = ["amorphgen", "--analyse", "--input-dir", str(structures),
                "--feff-write", str(tmp_path / "feff"), "--feff-absorber", "Si",
                "--feff-card", "RPATH"]
        with patch("sys.argv", argv), pytest.raises(ValueError, match="KEY=VALUE"):
            main()


@pytest.mark.parametrize("analysis, ok", [
    ({"feff_cards": {"RPATH": 8, "CRITERIA": "1.0 0.5"}, "feff_workers": 4,
      "feff_executable": "/opt/feff/bin/feff8l"}, True),
    ({"feff_cards": {"RPATH": True}}, False),
    ({"feff_cards": ["RPATH=8"]}, False),
    ({"feff_workers": 0}, False),
])
def test_yaml_automation_keys_are_validated(analysis, ok):
    from amorphgen.configs.yaml_config import _validate_config
    _, errors = _validate_config({"analysis": analysis}, "test.yaml")
    assert (errors == []) is ok
