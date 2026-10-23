"""Regression tests for FEFF provenance, spectral coverage and uncertainty."""
import json
import sys

import numpy as np
import pytest
from ase.build import bulk

from amorphgen.analysis.feff import (
    average_chi, chi_fourier_transform, collect_feff_results,
    format_feff_report, read_chi, run_feff_jobs, write_feff_jobs,
)


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "jobs"
    atoms = bulk("Cu", "fcc", a=3.61)
    manifest = write_feff_jobs([atoms], "Cu", str(root), radius=4)
    job = root / manifest["jobs"][0]["dir"]
    return root, job, atoms


def executable(tmp_path, body):
    path = tmp_path / "fake-feff"
    path.write_text(f"#!{sys.executable}\n{body}\n")
    path.chmod(0o755)
    return str(path)


def spectrum(job, kmax=12):
    k = np.arange(0, kmax + .01, .05)
    np.savetxt(job / "chi.dat", np.column_stack((k, np.sin(4 * k))))


def test_rewrite_changed_input_is_rejected_without_modification(tree):
    root, job, atoms = tree
    spectrum(job)
    before = {p: p.read_bytes() for p in (job / "feff.inp", job / "chi.dat",
                                        root / "manifest.json")}
    changed = atoms.copy()
    changed.set_cell(atoms.cell * 1.1, scale_atoms=True)
    with pytest.raises(ValueError, match="input changed"):
        write_feff_jobs([changed], "Cu", str(root), radius=4)
    assert all(p.read_bytes() == value for p, value in before.items())


def test_identical_write_retains_valid_results(tree, tmp_path):
    root, job, atoms = tree
    binary = executable(tmp_path, "open('chi.dat','w').write('0 0\\n5 1\\n12 0\\n')")
    assert run_feff_jobs(root, binary)["ran"] == 1
    before = (job / "feff.inp").stat().st_mtime_ns
    write_feff_jobs([atoms], "Cu", str(root), radius=4)
    assert (job / "feff.inp").stat().st_mtime_ns == before
    assert collect_feff_results(root)["n_sites"] == 1


@pytest.mark.parametrize("body", ["pass", "raise SystemExit(2)",
                                "open('chi.dat','w').write('0 1\\n1 invalid\\n')"])
def test_failed_rerun_never_accepts_previous_spectrum(tree, tmp_path, body):
    root, job, _ = tree
    spectrum(job)
    collect_feff_results(root, trust_external=True)
    result = run_feff_jobs(root, executable(tmp_path, body))
    assert result["ran"] == 0 and len(result["failed"]) == 1
    assert json.loads((job / "run_status.json").read_text())["state"] == "failed"
    with pytest.raises(ValueError, match="unusable"):
        collect_feff_results(root, trust_external=True)


def test_external_results_require_explicit_import(tree):
    root, job, _ = tree
    spectrum(job)
    with pytest.raises(ValueError, match="no run provenance"):
        collect_feff_results(root)
    result = collect_feff_results(root, trust_external=True)
    assert result["external_imports"] == [job.name]
    assert collect_feff_results(root)["external_imports"] == [job.name]
    assert "trusted by caller" in format_feff_report(result)


@pytest.mark.parametrize("filename", ["feff.inp", "chi.dat"])
def test_edits_after_import_are_detected(tree, filename):
    root, job, _ = tree
    spectrum(job)
    collect_feff_results(root, trust_external=True)
    with (job / filename).open("a") as fh:
        fh.write("\n# altered\n")
    with pytest.raises(ValueError, match="differs"):
        collect_feff_results(root, trust_external=True)


def test_success_records_output_and_preserves_partial_stage_inputs(tree, tmp_path):
    root, job, _ = tree
    (job / "phase.pad").write_text("intermediate")
    binary = executable(tmp_path,
        "assert open('phase.pad').read() == 'intermediate'\n"
        "open('chi.dat','w').write('0 0\\n5 1\\n12 0\\n')")
    assert run_feff_jobs(root, binary) == {"ran": 1, "skipped": 0, "failed": []}
    status = json.loads((job / "run_status.json").read_text())
    assert status["state"] == "complete"
    assert "chi.dat" in status["outputs"]
    assert collect_feff_results(root)["external_imports"] == []


def path_sum_output(listed, header):
    """Fake FEFF body writing list.dat with ``listed`` paths and a chi.dat
    whose ``paths used`` line is ``header``."""
    return (
        "rows = ''.join(f'{i:9d}  0.0  50.0  1.0  2  2.0\\n' "
        f"for i in range(1, {listed} + 1))\n"
        "open('list.dat','w').write('PATH  Rmax= 8.0\\n ----\\n pathindex\\n' + rows)\n"
        f"open('chi.dat','w').write('# {header} paths used\\n0 0\\n5 1\\n12 0\\n')")


def test_complete_path_sum_records_listed_paths(tree, tmp_path):
    root, job, _ = tree
    binary = executable(tmp_path, path_sum_output(3, "   2/   3"))
    assert run_feff_jobs(root, binary) == {"ran": 1, "skipped": 0, "failed": []}
    assert json.loads((job / "run_status.json").read_text())["paths_listed"] == 3


@pytest.mark.parametrize("listed, header", [(4, "   2/   3"),
                                            (15001, "****/****")])
def test_truncated_path_sum_is_rejected(tree, tmp_path, listed, header):
    # FEFF8.5L's final sum reads at most 15000 paths and says nothing
    # about the rest; its four-digit count overflows long before that.
    root, job, _ = tree
    result = run_feff_jobs(root, executable(tmp_path, path_sum_output(listed, header)))
    assert result["ran"] == 0 and "truncated" in result["failed"][0][1]
    assert json.loads((job / "run_status.json").read_text())["state"] == "failed"
    assert not (job / "chi.dat").exists()


def test_timeout_marks_failed_and_releases_lock(tree, tmp_path):
    root, job, _ = tree
    result = run_feff_jobs(root, executable(tmp_path, "import time; time.sleep(10)"),
                           timeout=.1)
    assert "timed out" in result["failed"][0][1]
    assert not (job / ".amorphgen_feff.lock").exists()
    assert json.loads((job / "run_status.json").read_text())["state"] == "failed"


def test_running_job_is_not_collected_or_overwritten(tree, tmp_path):
    root, job, _ = tree
    spectrum(job)
    (job / ".amorphgen_feff.lock").touch()
    assert run_feff_jobs(root, executable(tmp_path, "pass"))["failed"]
    with pytest.raises(ValueError, match="currently running"):
        collect_feff_results(root, trust_external=True)


def test_job_selection_cannot_escape_manifest(tree, tmp_path):
    root, _, _ = tree
    with pytest.raises(ValueError, match="not in the manifest"):
        run_feff_jobs(root, executable(tmp_path, "pass"), jobs=["../other"])
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["jobs"][0]["dir"] = "../other"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="directly inside"):
        collect_feff_results(root)


@pytest.mark.parametrize("body,match", [
    ("0 1\n1 invalid\n2 3\n", "malformed"),
    ("0 1\n1\n2 3\n", "malformed"),
    ("0 1\n1 NaN\n", "non-finite"),
    ("0 1\n1 inf\n", "non-finite"),
    ("0 1\n0 2\n", "strictly increasing"),
    ("2 1\n1 2\n", "strictly increasing"),
    ("0 1\n", "at least two"),
])
def test_invalid_spectra_fail_at_read(tmp_path, body, match):
    path = tmp_path / "chi.dat"
    path.write_text(body)
    with pytest.raises(ValueError, match=match):
        read_chi(path)


def test_fortran_exponents_and_header_are_supported(tmp_path):
    path = tmp_path / "chi.dat"
    path.write_text("FEFF output\n# k chi\n0D0 1D-2\n1D0 2D-2\n")
    k, chi = read_chi(path)
    np.testing.assert_allclose(k, [0, 1])
    np.testing.assert_allclose(chi, [.01, .02])


def test_unequal_coverage_does_not_reduce_amplitude():
    result = average_chi([([0, 10], [1, 1]), ([0, 20], [1, 1])])
    assert result["k"].max() == 10
    np.testing.assert_allclose(result["chi"], 1)


def test_no_overlap_is_rejected():
    with pytest.raises(ValueError, match="no shared"):
        average_chi([([0, 5], [1, 1]), ([6, 10], [1, 1])])


def test_collection_tapers_within_real_data(tree):
    root, job, _ = tree
    spectrum(job, kmax=10)
    result = collect_feff_results(root, trust_external=True)
    assert result["k"].max() == 10
    assert result["ft_kmax"] == 9.5
    assert np.isnan(result["chi_sem"]).all()
    assert "SEM unavailable" in format_feff_report(result)


def test_requested_window_keeps_data_for_entire_upper_taper(tree):
    root, job, _ = tree
    spectrum(job, kmax=20)
    result = collect_feff_results(root, kmax=12, trust_external=True)
    assert result["k"].max() == 12.5
    assert result["ft_kmax"] == 12


def test_correlated_sites_do_not_create_false_precision():
    sources = [([0, 20], [1, 1])] * 50 + [([0, 20], [3, 3])] * 50
    result = average_chi(sources, structure_ids=[0] * 50 + [1] * 50)
    np.testing.assert_allclose(result["chi"], 2)
    np.testing.assert_allclose(result["chi_sem"], 1)
    assert result["n_structures"] == 2


def test_unequal_site_counts_give_equal_structure_weight():
    sources = [([0, 20], [1, 1])] * 50 + [([0, 20], [3, 3])]
    result = average_chi(sources, structure_ids=[0] * 50 + [1])
    np.testing.assert_allclose(result["chi"], 2)
    np.testing.assert_allclose(result["chi_sem"], 1)


def test_ungrouped_spectra_do_not_claim_ensemble_sem():
    result = average_chi([([0, 20], [1, 1]), ([0, 20], [3, 3])])
    assert result["n_structures"] == 0
    assert np.isnan(result["chi_sem"]).all()


def test_fft_cannot_silently_truncate_data():
    with pytest.raises(ValueError, match="nfft is too small"):
        chi_fourier_transform([0, 20], [1, 1], nfft=16)


def test_collection_preserves_structure_grouping(tmp_path):
    root = tmp_path / "jobs"
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
    manifest = write_feff_jobs([atoms, atoms], "Cu", root, radius=4)
    for job in manifest["jobs"]:
        np.savetxt(root / job["dir"] / "chi.dat",
                   [[0, 1 + 2 * job["structure"]], [20, 1 + 2 * job["structure"]]])
    result = collect_feff_results(root, trust_external=True)
    assert result["n_sites"] == 8 and result["n_structures"] == 2
    np.testing.assert_allclose(result["chi_sem"], 1)


def test_cli_yaml_collection_settings_and_explicit_override(tree, tmp_path, monkeypatch):
    from ase.io import write
    from amorphgen.cli import main
    import yaml
    root, job, atoms = tree
    spectrum(job)
    structures = tmp_path / "structures"
    structures.mkdir()
    write(structures / "cu.xyz", atoms, format="extxyz")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"analysis": {
        "feff_collect": str(root), "feff_trust_external": True,
        "feff_kmin": 4, "feff_kmax": 10, "feff_dk": 2, "feff_kweight": 3,
    }}))
    report = tmp_path / "report.txt"
    monkeypatch.setattr(sys, "argv", [
        "amorphgen", "--analyse", "--input-dir", str(structures),
        "--config", str(config), "--feff-kmax", "8", "--save-report", str(report),
    ])
    main()
    text = report.read_text()
    assert "k = 4.0-8.0" in text and "k^3 weighted" in text
    assert "SEM unavailable" in text


def test_cli_feff_failure_sets_failing_exit_status(tree, tmp_path, monkeypatch):
    from ase.io import write
    from amorphgen.cli import main
    root, _, atoms = tree
    structures = tmp_path / "structures"
    structures.mkdir()
    write(structures / "cu.xyz", atoms, format="extxyz")
    monkeypatch.setattr("amorphgen.analysis.feff.run_feff_jobs",
                        lambda *_, **__: {"ran": 0, "skipped": 0,
                                          "failed": [("site", "no output")]})
    monkeypatch.setattr(sys, "argv", [
        "amorphgen", "--analyse", "--input-dir", str(structures),
        "--feff-write", str(root), "--feff-absorber", "Cu", "--feff-radius", "4",
        "--feff-run",
    ])
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 1


def test_manifest_records_requested_cards_and_effective_edge(tmp_path):
    atoms = bulk("Cu", "fcc", a=3.61)
    manifest = write_feff_jobs([atoms], "Cu", tmp_path, radius=4,
                               cards={"EDGE": "L3", "EXAFS": "12"},
                               extra_cards=["CRITERIA 4.0 2.5"])
    assert manifest["edge"] == "L3"
    assert manifest["cards"]["EXAFS"] == "12"
    assert manifest["extra_cards"] == ["CRITERIA 4.0 2.5"]


@pytest.mark.parametrize("name,value", [
    ("feff_sites", 0), ("feff_radius", -1), ("feff_kmax", float("inf")),
    ("feff_dk", -1), ("harmonic_bootstrap", -1), ("exafs_mfp", 0),
    ("feff_trust_external", "yes"), ("feff_kweight", True),
])
def test_invalid_exafs_yaml_is_rejected(name, value):
    from amorphgen.configs.yaml_config import _validate_config
    _, errors = _validate_config({"analysis": {name: value}}, "test.yaml")
    assert any(f"analysis.{name}" in error for error in errors)
