"""CLI argument validation, YAML/CLI precedence, mode dispatch and their error exits."""

import json
import re
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import read, write
from ase.spacegroup import crystal

from amorphgen_test_helpers import run_cli

from amorphgen import cli
from amorphgen.analysis import StructureAnalyser


def _cristobalite(seed=None, energy=None):
    atoms = crystal(["Si", "O"], basis=[(0, 0, 0), (0.125, 0.125, 0.125)], spacegroup=227,
                    cellpar=[7.16, 7.16, 7.16, 90, 90, 90])
    if seed is not None:
        atoms.rattle(0.05, seed=seed)
    if energy is not None:
        atoms.info["potential_energy"] = energy
    return atoms


def _config(path, data):
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def no_plots(monkeypatch):
    """Record plot requests instead of rendering figures."""
    import amorphgen.analysis.descriptors as descriptors
    import amorphgen.analysis.plotting as plotting

    calls = {"plot": [], "sq": [], "tr": []}
    monkeypatch.setattr(StructureAnalyser, "plot",
                        lambda self, **kwargs: calls["plot"].append(kwargs))
    monkeypatch.setattr(plotting, "plot_sq", lambda result, **kwargs: calls["sq"].append(kwargs))
    monkeypatch.setattr(plotting, "plot_tr", lambda result, **kwargs: calls["tr"].append(kwargs))
    monkeypatch.setattr(descriptors, "save_descriptor", lambda *args, **kwargs: None)
    return calls


# ── Parsers and small helpers ───────────────────────────────────────────────

@pytest.mark.parametrize("spec,message", [
    ("SiO2*two", "Invalid multiplier in 'SiO2\\*two'"),
    ("=4", "Empty element symbol in: '=4'"),
    ("Si=4,=3", "Empty element symbol in: '=3'"),
])
def test_composition_parser_rejects_malformed_entries(spec, message):
    with pytest.raises(ValueError, match=message):
        cli._parse_composition(spec)


def test_target_cn_parser_skips_empty_entries():
    assert cli._parse_target_cn(" Si=4,, O = 2 ,") == {"Si": 4, "O": 2}


def test_full_override_carries_every_parser_default():
    parser = cli._get_parser()
    args = parser.parse_args(["--timestep", "2.0", "--pair-style", "sw"])
    override = cli._build_override(args, parser)
    assert override["lammps_params"] == {"pair_style": "sw"}
    assert override["seed"] is None and override["engine"] is None
    assert override["opt"] == override["final_opt"] == {
        "fmax": 0.01, "max_steps": 1000, "optimizer": "LBFGS",
        "cell_filter": "FrechetCellFilter", "output_format": "xyz"}
    for stage in ("eq_premelt", "melt", "eq_high", "quench", "eq_low"):
        assert override[stage]["timestep"] == 2.0
    assert override["melt"]["T_end"] == parser.get_default("melt_T_end")


def test_cubic_default_leaves_a_non_mapping_override_untouched(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["amorphgen", "--batch-opt"])
    args = SimpleNamespace(batch_opt=True, hybrid_ensemble=False)
    assert cli._apply_amorphous_cubic_default(args, None, "FrechetCellFilter") is None


def test_convergence_summaries_flatten_to_public_names():
    leaf = {"per_structure": [1.0, 2.0], "sem": 0.5}
    found = {}
    cli._collect_convergence_summaries(found, "rings", {
        ("Si", "O"): leaf, "missing": None, "count": 3, "nested": {"six": leaf}})
    cli._collect_convergence_summaries(found, "voronoi", None)
    assert found == {"rings.Si-O": leaf, "rings.nested.six": leaf}


def test_module_entry_point_runs_main(monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "amorphgen.cli")
    monkeypatch.setattr(sys, "argv", ["amorphgen", "--examples"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("amorphgen.cli", run_name="__main__")
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == cli._EXAMPLES.strip()


# ── Sequential convergence contract ─────────────────────────────────────────

_SEQUENTIAL = ["--random-gen", "--relax", "--engine", "torchsim", "--until-converged",
               "--composition", "Si=4", "--seed", "42"]


@pytest.mark.parametrize("extra", [
    ["--descriptor-bounds", "density=0,10"],
    ["--convergence-batch-size", "4"],
    ["--convergence-min-structures", "3"],
])
def test_sampling_controls_without_until_converged_fail_early(tmp_path, monkeypatch, capsys, extra):
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, ["--random-gen", "--composition", "Si=4",
                              "-o", tmp_path / "out", *extra])
    assert exc.value.code == 1
    assert "require --until-converged" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_until_converged_without_any_tolerance_fails_early(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, [*_SEQUENTIAL, "-o", tmp_path / "out"])
    assert exc.value.code == 1
    assert "requires at least one declared --tolerance" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("analysis,message", [
    ({"tolerances": {1: .1}, "descriptor_bounds": {1: [0, 1]}}, "names must be strings"),
    ({"tolerances": {"density": float("nan")}, "descriptor_bounds": {"density": [0, 1]}},
     "Tolerance for 'density' must be finite and positive"),
    ({"tolerances": {"density": True}, "descriptor_bounds": {"density": [0, 1]}},
     "Tolerance for 'density' must be finite and positive"),
    ({"tolerances": {"density": .1}, "descriptor_bounds": {"density": [1, 0]}},
     "Descriptor bounds for 'density' require finite lower < upper"),
    ({"tolerances": {"density": .1}, "descriptor_bounds": {"density": [0, float("inf")]}},
     "Descriptor bounds for 'density'"),
    ({"tolerances": {"density": .1}, "descriptor_bounds": {"density": "0,1"}},
     "Descriptor bounds for 'density'"),
])
def test_sequential_contract_rejects_unvalidated_mappings(monkeypatch, analysis, message):
    """Python callers can bypass YAML validation; the contract still holds."""
    monkeypatch.setattr(sys, "argv", ["amorphgen", *_SEQUENTIAL])
    args = cli._get_parser().parse_args(_SEQUENTIAL)
    with pytest.raises(ValueError, match=message):
        cli._until_convergence_options(args, {"engine": "torchsim", "analysis": analysis})


# ── Default work directories and missing mode inputs ────────────────────────

@pytest.fixture
def mode_recorder(monkeypatch):
    """Replace every mode's expensive entry point by a recorder of its output dir."""
    from amorphgen.pipeline import batch_quench, opt_cell, random_gen, run_pipeline
    import amorphgen.utils

    seen = []

    class Pipeline:
        def __init__(self, input_file, work_dir, cfg_override):
            seen.append(work_dir)

        def run(self, stages, resume):
            pass

    def optimise(**kwargs):
        seen.append(kwargs["output_dir"])
        return ["optimised.xyz"]

    monkeypatch.setattr(random_gen, "_batch_random_unlocked",
                        lambda **kwargs: seen.append(kwargs["output_dir"]) or [])
    monkeypatch.setattr(batch_quench, "run", lambda **kwargs: seen.append(kwargs["work_dir"]))
    monkeypatch.setattr(opt_cell, "batch_optimize", optimise)
    monkeypatch.setattr(run_pipeline, "MeltQuenchPipeline", Pipeline)
    monkeypatch.setattr(cli, "_run_mq_ensemble", lambda args, *a, **k: seen.append(args.work_dir))
    monkeypatch.setattr(cli, "_run_hybrid_ensemble", lambda args, *a: seen.append(args.work_dir))
    monkeypatch.setattr(amorphgen.utils, "get_calculator", lambda **kwargs: object())
    return seen


@pytest.mark.parametrize("arguments,expected", [
    (["--random-gen", "--composition", "Si=4,O=8"], "random_O8Si4"),
    (["--random-gen", "--composition", "SiO2*4"], "random_structures"),
    (["--batch-quench", "--snapshot-dir", "inputs"], "batch_quench"),
    (["--batch-opt", "--input-dir", "inputs"], "batch_opt"),
    (["crystal.xyz", "--mq-ensemble"], "mq_ensemble_run"),
    (["--hybrid-ensemble", "--input-dir", "inputs"], "hybrid_run"),
    (["crystal.xyz"], "melt_quench_run"),
])
def test_each_mode_gets_its_own_default_work_dir(tmp_path, monkeypatch, mode_recorder,
                                                 arguments, expected):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "inputs").mkdir()
    write(tmp_path / "inputs" / "s.xyz", bulk("Cu", cubic=True))
    run_cli(monkeypatch, [*arguments, "-m", "lj"])
    assert mode_recorder == [expected]


def test_extract_snapshots_defaults_to_twenty_in_snapshots_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frames = [bulk("Cu", "fcc", a=3.6 + 0.01 * i, cubic=True) for i in range(25)]
    write(tmp_path / "traj.xyz", frames)
    run_cli(monkeypatch, ["--extract-snapshots", "traj.xyz"])
    snapshots = sorted((tmp_path / "snapshots").glob("snapshot_*.xyz"))
    assert len(snapshots) == 20
    # Uniform selection spans the whole trajectory, first to last frame.
    assert read(snapshots[0]).cell[0, 0] == pytest.approx(3.6)
    assert read(snapshots[-1]).cell[0, 0] == pytest.approx(3.84)


@pytest.mark.parametrize("arguments,message", [
    (["--mq-ensemble"], "input_file (crystal structure) is required for --mq-ensemble"),
    (["--hybrid-ensemble"], "--input-dir is required for --hybrid-ensemble"),
    (["--batch-opt"], "--input-dir is required for --batch-opt mode"),
    (["--analyse"], "--input-dir or input_file is required for --analyse"),
    (["--random-gen"], "--composition is required for --random-gen mode"),
    ([], "input_file is required for melt-quench pipeline mode"),
])
def test_modes_without_their_input_exit_before_any_work(tmp_path, monkeypatch, capsys,
                                                        arguments, message):
    import amorphgen.utils

    monkeypatch.chdir(tmp_path)
    calculator = Mock(side_effect=AssertionError("no calculator before validation"))
    monkeypatch.setattr(amorphgen.utils, "get_calculator", calculator)
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, [*arguments, "-m", "lj"])
    assert exc.value.code == 1
    assert message in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []
    calculator.assert_not_called()


def test_batch_quench_on_a_random_gen_root_points_to_its_structures(tmp_path, monkeypatch, capsys):
    import amorphgen.utils

    root = tmp_path / "gen"
    (root / "random_initial").mkdir(parents=True)
    write(root / "random_initial" / "random_0000.xyz", bulk("Cu", cubic=True))
    calculator = Mock()
    monkeypatch.setattr(amorphgen.utils, "get_calculator", calculator)
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, ["--batch-quench", "--snapshot-dir", root, "-m", "lj",
                              "-o", tmp_path / "out"])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert f"no snapshot files found in {root}/" in out
    assert "is a --random-gen output directory" in out
    assert f"{root / 'random_initial'}/   (as placed)" in out
    calculator.assert_not_called()


def test_batch_quench_directory_forwards_first_format_group(tmp_path, monkeypatch):
    from amorphgen.pipeline import batch_quench
    import amorphgen.utils

    source = tmp_path / "snaps"
    source.mkdir()
    for name in ("b.xyz", "a.xyz", "c.vasp"):
        write(source / name, bulk("Cu", cubic=True))
    quench = Mock()
    monkeypatch.setattr(batch_quench, "run", quench)
    monkeypatch.setattr(amorphgen.utils, "get_calculator", lambda **kwargs: "calc")
    run_cli(monkeypatch, ["--batch-quench", "--snapshot-dir", source, "--n-runs", "2",
                          "--batch-stages", "6", "7", "-m", "lj", "-o", tmp_path / "out"])
    kwargs = quench.call_args.kwargs
    assert kwargs["snapshot_files"] == [str(source / "a.xyz"), str(source / "b.xyz")]
    assert (kwargs["n_runs"], kwargs["select"], kwargs["stages"]) == (2, "uniform", [6, 7])
    assert kwargs["calc"] == "calc" and kwargs["work_dir"] == str(tmp_path / "out")


# ── Convert mode ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cli_format,expected", [([], "cif"), (["--format", "vasp"], "vasp")])
def test_yaml_convert_without_output_dir_uses_input_named_default(tmp_path, monkeypatch,
                                                                 cli_format, expected):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "snaps").mkdir()
    structure = bulk("Cu", cubic=True)
    write(tmp_path / "snaps" / "s0.xyz", structure)
    _config(tmp_path / "convert.yaml", {"convert": {"input": "snaps", "format": "cif"}})
    run_cli(monkeypatch, ["--config", "convert.yaml", *cli_format])
    converted = read(tmp_path / f"snaps_{expected}" / f"s0.{expected}")
    np.testing.assert_allclose(converted.positions, structure.positions, atol=1e-6)
    assert not (tmp_path / "melt_quench_run").exists()


def test_convert_helper_without_any_input_exits(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["amorphgen"])
    with pytest.raises(SystemExit) as exc:
        cli._run_convert(SimpleNamespace(convert=None, format="xyz", work_dir=None), yaml_cfg={})
    assert exc.value.code == 1
    assert "--convert needs an input PATH" in capsys.readouterr().out


# ── MQ ensemble dispatch ────────────────────────────────────────────────────

@pytest.fixture
def mq_shared(tmp_path, monkeypatch):
    """Stand in for stages 1-4; the test decides which files they leave behind."""
    from amorphgen.pipeline import run_pipeline

    crystal_path = tmp_path / "crystal.xyz"
    write(crystal_path, bulk("Cu", cubic=True))
    state = {"outputs": {}, "override": None}

    class SharedPipeline:
        def __init__(self, input_file, work_dir, cfg_override):
            self.work_dir = Path(work_dir)
            state["override"] = cfg_override

        def run(self, stages, resume):
            assert stages == [1, 2, 3, 4]
            self.work_dir.mkdir(parents=True, exist_ok=True)
            for name, frames in state["outputs"].items():
                write(self.work_dir / name, frames)

    monkeypatch.setattr(run_pipeline, "MeltQuenchPipeline", SharedPipeline)
    state["crystal"] = crystal_path
    state["work"] = tmp_path / "mq"
    return state


def _cu_frames(n):
    return [bulk("Cu", "fcc", a=3.6 + 0.02 * i, cubic=True) for i in range(n)]


def test_mq_ensemble_runs_through_main_to_collected_finals(monkeypatch, mq_shared):
    from amorphgen.pipeline import batch_quench
    import amorphgen.utils

    mq_shared["outputs"] = {"stage4_eq_traj.xyz": _cu_frames(3)}
    monkeypatch.setattr(amorphgen.utils, "get_calculator", lambda **kwargs: EMT())
    for stage in (batch_quench.quench, batch_quench.equilibrate, batch_quench.final_opt):
        monkeypatch.setattr(stage, "run", lambda atoms, **kwargs: atoms)
    assert run_cli(monkeypatch, [mq_shared["crystal"], "--mq-ensemble", "-m", "lj",
                                 "--select", "last", "-o", mq_shared["work"]]) is None
    assert mq_shared["override"]["model"] == "lj"
    final = read(mq_shared["work"] / "final" / "mq_0000.xyz")
    assert final.cell[0, 0] == pytest.approx(3.64)
    assert (mq_shared["work"] / "melt_memory.json").is_file()


def test_mq_ensemble_falls_back_to_legacy_stage4_trajectory(monkeypatch, mq_shared, capsys):
    from amorphgen.pipeline import batch_quench
    import amorphgen.utils

    mq_shared["outputs"] = {"stage4_eq.xyz": _cu_frames(3)}
    quench = Mock()
    monkeypatch.setattr(batch_quench, "run", quench)
    monkeypatch.setattr(amorphgen.utils, "get_calculator", lambda **kwargs: object())
    monkeypatch.setattr(cli, "_collect_ensemble_final", lambda *args, **kwargs: None)
    run_cli(monkeypatch, [mq_shared["crystal"], "--mq-ensemble", "-m", "lj",
                          "--select", "last", "-o", mq_shared["work"]])
    legacy = mq_shared["work"] / "shared" / "stage4_eq.xyz"
    assert f"Extracting 1 snapshots from {legacy}" in capsys.readouterr().out
    (snapshot,) = quench.call_args.kwargs["snapshot_files"]
    assert read(snapshot).cell[0, 0] == pytest.approx(3.64)


def test_mq_ensemble_without_stage4_trajectory_saves_diagnostics_and_exits(
        monkeypatch, mq_shared, capsys):
    from amorphgen.pipeline import batch_quench

    quench = Mock(side_effect=AssertionError("must not quench"))
    monkeypatch.setattr(batch_quench, "run", quench)
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, [mq_shared["crystal"], "--mq-ensemble", "-m", "lj",
                              "-o", mq_shared["work"]])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "stage 4 trajectory not found" in out
    assert "stage4_eq_traj.xyz" in out and "stage4_eq.xyz)" in out
    report = json.loads((mq_shared["work"] / "melt_memory.json").read_text())
    assert [row["kind"] for row in report["comparisons"]] == ["stage3", "stage4"]
    assert all(row["status"] == "unavailable" for row in report["comparisons"])


def test_mq_extraction_error_survives_a_failing_diagnostic_report(monkeypatch, mq_shared, capsys):
    import amorphgen.analysis.melt_memory as melt_memory

    mq_shared["outputs"] = {"stage4_eq_traj.xyz": _cu_frames(2)}

    def failing_report(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(melt_memory, "report_melt_memory", failing_report)
    with pytest.raises(ValueError, match=r"burn_in_frames \(2\)"):
        run_cli(monkeypatch, [mq_shared["crystal"], "--mq-ensemble", "-m", "lj",
                              "--burn-in-frames", "2", "-o", mq_shared["work"]])
    assert "Warning: could not save melt-memory report: disk full" in capsys.readouterr().out


def test_collect_final_falls_back_to_xyz_for_an_unknown_format(tmp_path, capsys):
    run = tmp_path / "quench" / "run_0003"
    run.mkdir(parents=True)
    structure = bulk("Cu", cubic=True)
    write(run / "final_amorphous.xyz", structure)
    cli._collect_ensemble_final(str(run.parent), str(tmp_path / "final"), "pdb", "hybrid",
                                {"xyz": ("extxyz", ".xyz")})
    assert "Warning: unknown format 'pdb', using 'xyz'" in capsys.readouterr().out
    np.testing.assert_allclose(read(tmp_path / "final" / "hybrid_0003.xyz").positions,
                               structure.positions)


# ── Analysis mode ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("rings,voronoi,bond_pair,header", [
    (True, True, None, "(all atoms)"),
    (["Si", "O"], "Si", ("Si", "O"), "(Si)"),
])
def test_yaml_analysis_settings_reach_every_consumer(tmp_path, monkeypatch, capsys, no_plots,
                                                     rings, voronoi, bond_pair, header):
    source = tmp_path / "silica.xyz"
    write(source, _cristobalite())
    plots = tmp_path / "plots"
    config = _config(tmp_path / "analysis.yaml", {"analysis": {
        "total_cn": "O", "rdf_pairs": ["Si-O"], "angle_triplets": ["O-Si-O"],
        "angle_style": "line", "rmax": 5.0, "total_rdf": True, "show_title": True,
        "save_plot": str(plots), "sq": True, "sq_weighting": "neutron", "sq_method": "ft",
        "tr": True, "rings": rings, "voronoi": voronoi,
    }})
    ring_pairs = []
    original = StructureAnalyser.ring_statistics

    def ring_statistics(self, bond_pair=None, **kwargs):
        ring_pairs.append(bond_pair)
        return original(self, bond_pair=bond_pair, **kwargs)

    monkeypatch.setattr(StructureAnalyser, "ring_statistics", ring_statistics)
    run_cli(monkeypatch, ["--analyse", source, "--config", config, "--cutoff", "auto"])
    (plot,) = no_plots["plot"]
    assert {key: plot[key] for key in ("rdf_pairs", "angle_triplets", "angle_style", "rmax",
                                        "show_total_rdf", "show_title", "total_cn")} == {
        "rdf_pairs": ["Si-O"], "angle_triplets": ["O-Si-O"], "angle_style": "line",
        "rmax": 5.0, "show_total_rdf": True, "show_title": True, "total_cn": ["O"]}
    (sq,) = no_plots["sq"]
    assert (sq["weighting"], sq["method"], sq["show_title"]) == ("neutron", "ft", True)
    assert len(no_plots["tr"]) == 1
    assert ring_pairs == [bond_pair]
    out = capsys.readouterr().out
    assert "S(q): ft method, neutron weighting" in out
    assert "T(r): neutron weighting" in out
    assert "O-(all bonded): mean=" in out
    assert f"Voronoi indices <n3 n4 n5 n6> {header}" in out
    assert (plots / "analysis_voronoi.csv").read_text().startswith("voronoi_index,count,percent")


@pytest.mark.parametrize("flags", [["--sq-qmax", "0.1"], ["--sq-qmax", "nan"], ["--sq-nq", "1"]])
def test_invalid_sq_grid_is_a_cli_error(tmp_path, monkeypatch, capsys, flags):
    source = tmp_path / "si.xyz"
    write(source, bulk("Si", "diamond", a=5.43, cubic=True))
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, ["--analyse", source, "--cutoff", "2.6", "--sq", *flags])
    assert exc.value.code == 1
    assert "--sq-qmax must exceed 0.1 and --sq-nq must be at least 2" in capsys.readouterr().out


def test_tr_without_usable_q_points_blocks_experimental_comparison(tmp_path, monkeypatch, capsys):
    source = tmp_path / "si.xyz"
    write(source, bulk("Si", "diamond", a=5.43, cubic=True))
    measured = tmp_path / "measured_tr.dat"
    measured.write_text("1.0 0.5\n2.0 1.5\n3.0 2.0\n")
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, ["--analyse", source, "--cutoff", "2.6", "--tr-qrange", "0.3", "0.8",
                              "--experiment-tr", measured])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "T(r) skipped: compute_total_correlation: only 0 usable S(Q) points" in out
    assert ("Error: scattering comparison: cannot compare experimental tr: "
            "calculation unavailable") in out


def test_tr_reports_density_when_no_shell_is_resolved(tmp_path, monkeypatch, capsys):
    """An H2 molecule's only distance (0.74 A) lies in the excluded r < 1 A region."""
    source = tmp_path / "h2.xyz"
    write(source, Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[12] * 3, pbc=True))
    run_cli(monkeypatch, ["--analyse", source, "--cutoff", "1.0", "--tr"])
    out = capsys.readouterr().out
    assert f"no resolved first peak; rho = {2 / 12 ** 3:.4f} atoms/A^3" in out


def test_tr_peak_and_qmax_scan_are_appended_to_the_report(tmp_path, monkeypatch):
    from amorphgen.analysis.rdf import format_Tr_scan, scan_Tr_qmax

    diamond = bulk("Si", "diamond", a=5.43, cubic=True)
    source = tmp_path / "si.xyz"
    write(source, diamond)
    report = tmp_path / "report.txt"
    run_cli(monkeypatch, ["--analyse", source, "--cutoff", "2.6", "--tr", "--tr-scan",
                          "--save-report", report])
    text = report.read_text()
    peak = float(re.search(r"first peak at r = ([0-9.]+) A", text).group(1))
    assert peak == pytest.approx(5.43 * np.sqrt(3) / 4, abs=0.02)  # Si-Si bond length
    expected = format_Tr_scan(scan_Tr_qmax([diamond], weighting="xray", qmin=0.3))
    assert text.rstrip().endswith(expected.rstrip())


def test_reference_validation_is_appended_to_the_report(tmp_path, monkeypatch):
    from amorphgen.analysis.validate import format_validation_report, validate_against_reference

    source = tmp_path / "si.xyz"
    write(source, bulk("Si", "diamond", a=5.43, cubic=True))
    reference = {"system": "a-Si test", "bond_distances": {"Si-Si": {"expected": [2.3, 2.4]}}}
    reference_path = _config(tmp_path / "reference.yaml", reference)
    report = tmp_path / "report.txt"
    run_cli(monkeypatch, ["--analyse", source, "--cutoff", "2.6",
                          "--reference", reference_path, "--save-report", report])
    expected = format_validation_report(validate_against_reference(
        StructureAnalyser(str(source), cutoff=2.6), reference))
    assert "Validation: a-Si test" in expected
    assert report.read_text().rstrip().endswith(expected.rstrip())


@pytest.fixture
def screening_candidates(tmp_path):
    directory = tmp_path / "candidates"
    directory.mkdir()
    for index, energy in enumerate([-10.0, 10.0]):
        atoms = Atoms("SiO2", positions=[[0, 0, 0], [1.6, 0, 0], [0, 1.6, 0]],
                      cell=[8, 8, 8], pbc=True)
        atoms.info["potential_energy"] = energy
        write(directory / f"candidate_{index}.xyz", atoms)
    return directory


def test_fully_excluded_screening_still_writes_the_requested_report(
        tmp_path, monkeypatch, capsys, screening_candidates):
    config = _config(tmp_path / "config.yaml", {"analysis": {
        "screening": {"energy": {"min": 100, "exclude": True}}}})
    report = tmp_path / "report.txt"
    run_cli(monkeypatch, ["--analyse", "--input-dir", screening_candidates, "--config", config,
                          "--cutoff", "1.8", "--save-report", report, "-o", tmp_path / "out"])
    out = capsys.readouterr().out
    text = report.read_text()
    assert text.startswith("Screening")
    assert "energy (excluded)" in text
    # The saved report is the screening block printed before "No structures retained".
    assert text.strip() in out.split("No structures retained")[0]


def test_screening_failure_stops_analysis_without_partial_audit(
        tmp_path, monkeypatch, capsys, screening_candidates):
    import amorphgen.analysis.screening as screening

    def failure(*args, **kwargs):
        raise ValueError("source_names must match the number of structures")

    monkeypatch.setattr(screening, "screen_structures", failure)
    monkeypatch.setattr(StructureAnalyser, "summary", Mock(side_effect=AssertionError))
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, ["--analyse", "--input-dir", screening_candidates, "--screen",
                              "--cutoff", "1.8", "-o", tmp_path / "out"])
    assert exc.value.code == 1
    assert "Error: screening: source_names must match" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


@pytest.fixture
def captured_convergence(monkeypatch):
    import amorphgen.analysis.convergence_output as convergence_output

    captured = []
    monkeypatch.setattr(convergence_output, "save_convergence_report",
                        lambda report, *args, **kwargs: captured.append(report))
    return captured


def test_optional_descriptors_join_the_convergence_report(
        tmp_path, monkeypatch, capsys, no_plots, captured_convergence):
    from scipy import stats

    ensemble = tmp_path / "ensemble"
    ensemble.mkdir()
    energies = [-240.0, -252.0, -246.0]
    for index, energy in enumerate(energies):
        write(ensemble / f"s{index}.xyz", _cristobalite(seed=index, energy=energy))
    config = _config(tmp_path / "config.yaml", {"analysis": {"energy_ranking": True}})
    run_cli(monkeypatch, [
        "--analyse", "--input-dir", ensemble, "--config", config, "--cutoff", "auto",
        "--connectivity", "--voronoi", "Si", "--rings", "--sq", "--tr", "--total-rdf",
        "--tolerance", "rings.mean_ring_size=0.5",
        "--tolerance", "connectivity.edge_or_face_percent=1",
        "--save-plot", tmp_path / "plots"])
    (report,) = captured_convergence
    descriptors = report["descriptors"]
    for name in ("sq.total", "tr.T_r", "rdf.total", "voronoi.mean_faces",
                 "connectivity.link_percent.corner", "energy.total"):
        assert descriptors[name]["n_structures"] == 3
    # Cristobalite: corner-sharing tetrahedra in six-membered rings.
    assert descriptors["rings.mean_ring_size"]["mean"] == pytest.approx(6.0)
    assert descriptors["rings.mean_ring_size"]["tolerance"] == 0.5
    assert descriptors["rings.mean_ring_size"]["status"] == "met"
    assert descriptors["connectivity.link_percent.corner"]["mean"] == pytest.approx(100.0)
    assert descriptors["connectivity.edge_or_face_percent"]["status"] == "met"
    per_atom = np.array(energies) / 24
    assert descriptors["energy.per_atom"]["mean"] == pytest.approx(per_atom.mean())
    expected_half_width = stats.t.ppf(0.975, 2) * per_atom.std(ddof=1) / np.sqrt(3)
    assert descriptors["energy.per_atom"]["half_width"] == pytest.approx(expected_half_width)
    out = capsys.readouterr().out
    assert f"Best: {per_atom.min():.4f} eV/atom" in out
    assert f"Worst: {per_atom.max():.4f} eV/atom" in out
    assert f"Spread: {np.ptp(per_atom):.4f} eV/atom" in out


def test_dimer_pairs_join_the_convergence_report(tmp_path, monkeypatch, no_plots,
                                                 captured_convergence):
    ensemble = tmp_path / "ensemble"
    ensemble.mkdir()
    distances = [1.20, 1.25, 1.30]
    for index, distance in enumerate(distances):
        write(ensemble / f"d{index}.xyz", Atoms(
            "SiO2", positions=[[0, 0, 0], [4, 4, 4], [4 + distance, 4, 4]],
            cell=[9] * 3, pbc=True))
    run_cli(monkeypatch, ["--analyse", "--input-dir", ensemble, "--cutoff", "2.0",
                          "--check-dimers", "--convergence", "--save-plot", tmp_path / "plots"])
    (report,) = captured_convergence
    descriptors = report["descriptors"]
    assert descriptors["dimers.count.O-O"]["mean"] == pytest.approx(1.0)
    assert descriptors["dimers.min_distance.O-O"]["mean"] == pytest.approx(np.mean(distances))
    assert descriptors["dimers.fraction_of_sites"]["mean"] == pytest.approx(2 / 3)


# ── Random generation settings ──────────────────────────────────────────────

_RANDOM_YAML = {"random_gen": {
    "composition": {"Si": 4, "O": 8}, "n_structures": 3, "target_density": 2.2,
    "density_scale": 1.1, "output_format": "vasp", "minsep": {"Si-O": 1.5, "O-O": 2.2},
    "target_cn": {"Si": 4, "O": 2}, "dmax": {"Si-O": 2.0}, "cn_tolerance": 1,
    "dmax_factor": 1.3, "cell_filter": "none",
}}


@pytest.mark.parametrize("flags,expected", [
    ([], {"composition": {"Si": 4, "O": 8}, "n_structures": 3, "target_density": 2.2,
          "density_scale": 1.1, "output_format": "vasp",
          "minsep": {"Si-O": 1.5, "O-O": 2.2}, "target_cn": {"Si": 4, "O": 2},
          "dmax": {"Si-O": 2.0}, "cn_tolerance": 1, "dmax_factor": 1.3,
          "cell_filter": "none"}),
    (["--composition", "Si=2,O=4", "-n", "2", "--target-density", "2.5",
      "--density-scale", "1.0", "--format", "xyz", "--minsep", "Si-O=1.6",
      "--target-cn", "Si=3", "--dmax", "Si-O=2.2", "--cn-tolerance", "0",
      "--dmax-factor", "1.4", "-C", "UnitCellFilter"],
     {"composition": {"Si": 2, "O": 4}, "n_structures": 2, "target_density": 2.5,
      "density_scale": 1.0, "output_format": "xyz", "minsep": {"Si-O": 1.6},
      "target_cn": {"Si": 3}, "dmax": {"Si-O": 2.2}, "cn_tolerance": 0,
      "dmax_factor": 1.4, "cell_filter": "UnitCellFilter"}),
], ids=["yaml", "cli-overrides-yaml"])
def test_random_gen_settings_follow_cli_then_yaml(tmp_path, monkeypatch, flags, expected):
    from amorphgen.pipeline import random_gen

    generate = Mock(return_value=[])
    monkeypatch.setattr(random_gen, "_batch_random_unlocked", generate)
    config = _config(tmp_path / "random.yaml", _RANDOM_YAML)
    run_cli(monkeypatch, ["--random-gen", "--config", config, "-o", tmp_path / "out", *flags])
    kwargs = generate.call_args.kwargs
    assert {key: kwargs[key] for key in expected} == expected
    assert kwargs["relax"] is False and kwargs["calc"] is None


def test_random_gen_relax_keeps_lammps_settings_for_resume(tmp_path, monkeypatch):
    from amorphgen.pipeline import random_gen
    import amorphgen.utils
    import amorphgen.utils.calculators as calculators

    generate = Mock(return_value=[])
    factory = Mock(return_value="lammps-calculator")
    monkeypatch.setattr(random_gen, "_batch_random_unlocked", generate)
    monkeypatch.setattr(amorphgen.utils, "get_calculator", factory)
    # LAMMPS itself is optional; the fail-fast probes are not under test here.
    monkeypatch.setattr(calculators, "require_backend", lambda *args, **kwargs: "lammps")
    monkeypatch.setattr(calculators, "require_potential", lambda *args, **kwargs: None)
    run_cli(monkeypatch, ["--random-gen", "--relax", "--composition", "Si=8",
                          "--pair-style", "sw", "--pair-coeff", "* * Si.sw Si",
                          "-o", tmp_path / "out"])
    lammps = {"pair_style": "sw", "pair_coeff": ["* * Si.sw Si"]}
    settings = generate.call_args.kwargs["resume_settings"]
    assert (settings["model"], settings["engine"]) == ("lammps", "ase")
    assert settings["lammps_params"] == lammps
    assert "ace_params" not in settings
    assert factory.call_args.kwargs["lammps_params"] == lammps
    assert generate.call_args.kwargs["calc"] == "lammps-calculator"


# ── Batch optimisation and single-stage pipeline ────────────────────────────

def test_torchsim_batch_opt_with_nothing_matched_exits_nonzero(tmp_path, monkeypatch):
    from amorphgen.pipeline import opt_cell

    optimise = Mock(return_value=[])
    monkeypatch.setattr(opt_cell, "batch_optimize", optimise)
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, ["--batch-opt", "--input-dir", tmp_path, "--engine", "torchsim",
                              "-m", "lj", "--batch-size", "4", "--pattern", "*.vasp",
                              "--indices", "0-2", "--resume", "-o", tmp_path / "out"])
    assert exc.value.code == 1
    kwargs = optimise.call_args.kwargs
    assert (kwargs["engine"], kwargs["calc"], kwargs["batch_size"]) == ("torchsim", None, 4)
    assert (kwargs["pattern"], kwargs["indices"], kwargs["resume"]) == ("*.vasp", "0-2", True)
    assert kwargs["cfg_override"]["final_opt"]["cell_filter"] == "cubic"


@pytest.mark.parametrize("stage,expected", [
    ("1", "Optimizer: FIRE  fmax=0.123  max_steps=2"),
    ("7", "Optimizer: FIRE  fmax=0.234  max_steps=3"),
])
def test_single_optimisation_stage_writes_input_named_outputs(tmp_path, monkeypatch,
                                                              stage, expected):
    import amorphgen.utils

    monkeypatch.chdir(tmp_path)
    rattled = bulk("Cu", cubic=True)
    rattled.rattle(0.02, seed=1)
    write(tmp_path / "crystal.xyz", rattled)
    config = _config(tmp_path / "settings.yaml", {
        "opt": {"optimizer": "FIRE", "fmax": 0.123, "max_steps": 2, "cell_filter": "none"},
        "final_opt": {"fmax": 0.234, "max_steps": 3}})
    monkeypatch.setattr(amorphgen.utils, "get_calculator", lambda **kwargs: EMT())
    run_cli(monkeypatch, ["crystal.xyz", "--stages", stage, "--config", config, "-m", "lj",
                          "-o", "run"])
    assert Path.cwd().resolve() == tmp_path.resolve()
    assert expected in (tmp_path / "run" / "crystal_opt.log").read_text()
    assert len(read(tmp_path / "run" / "crystal_opt.xyz")) == 4
    # The direct optimisation path bypasses the pipeline and its manifest.
    assert not (tmp_path / "run" / "run_manifest.json").exists()
