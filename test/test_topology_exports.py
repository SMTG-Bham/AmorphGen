"""Ring and void artifacts retain definitions, structure identity and units."""

import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from amorphgen.analysis.descriptors import format_descriptor, save_descriptor
from amorphgen.analysis.plotting import plot_rings
from amorphgen.analysis.rings import compute_ring_statistics
from amorphgen.analysis.voids import compute_void_distribution


def read_rows(path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def test_ring_exports_keep_edge_assignments_and_structure_identity(tmp_path):
    diamond = bulk("C", "diamond", a=3.57)
    chain = Atoms("C2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[10] * 3)
    result = compute_ring_statistics([diamond, chain], cutoff=1.6)
    result["structure_files"] = ["diamond.xyz", "chain, sample.xyz"]
    save_descriptor("rings", result, tmp_path, dpi=40, save_pdf=True)

    exported = json.loads((tmp_path / "analysis_rings.json").read_text())
    assert exported["n_network_edges"] == 5
    assert exported["n_ring_edges"] == 4
    assert exported["n_unresolved_edges"] == 1
    assert exported["ring_edge_fraction"] == pytest.approx(0.8)
    assert exported["mean_ring_size"] == 6
    assert exported["structure_files"] == result["structure_files"]
    header, rows = read_rows(tmp_path / "analysis_rings.csv")
    assert header == ["ring_size", "count", "fraction_percent"]
    assert rows == [{"ring_size": "6", "count": "4", "fraction_percent": "100.0"}]
    _, structures = read_rows(tmp_path / "analysis_rings_structures.csv")
    assert [row["source_file"] for row in structures] == result["structure_files"]
    assert structures[1]["mean_ring_size"] == ""
    assert structures[1]["n_unresolved_edges"] == "1"
    _, rows = read_rows(tmp_path / "analysis_rings_per_structure.csv")
    assert [row["count"] for row in rows] == ["4", "0"]
    assert [row["fraction_percent"] for row in rows] == ["100.0", ""]
    for suffix in ("png", "pdf"):
        assert (tmp_path / f"analysis_rings.{suffix}").stat().st_size > 0
    text = format_descriptor("rings", result)
    assert " 6-ring:" in text
    assert "nodes-bridge: C-C" in text
    assert "Counts are edge assignments, not unique cycles" in text
    assert "no closure found with size <= 12" in text


def test_no_ring_plot_is_explicit_and_does_not_open_pyplot_figures(tmp_path, monkeypatch):
    import amorphgen.analysis.plotting as plotting

    result = compute_ring_statistics([Atoms("C")], cutoff=1.6, max_ring=4)
    assert "No ring closures resolved" in format_descriptor("rings", result)
    captured = {}
    original_save = plotting._save_fig

    def capture(fig, *args, **kwargs):
        captured["ylabel"] = fig.axes[0].get_ylabel()
        captured["text"] = [artist.get_text() for artist in fig.axes[0].texts]
        original_save(fig, *args, **kwargs)

    monkeypatch.setattr(plotting, "_save_fig", capture)
    before = plt.get_fignums()
    output = plot_rings(result, tmp_path, dpi=40)
    assert output == str(tmp_path / "analysis_rings.png")
    assert captured["ylabel"] == "Fraction of resolved edges (%)"
    assert captured["text"] == ["No ring closures resolved\nwith size ≤ 4"]
    assert plt.get_fignums() == before
    save_descriptor("rings", result, tmp_path, dpi=40)
    _, rows = read_rows(tmp_path / "analysis_rings_per_structure.csv")
    assert rows == []
    _, rows = read_rows(tmp_path / "analysis_rings_structures.csv")
    assert rows[0]["ring_edge_fraction"] == ""


def test_legacy_ring_plot_result_remains_supported(tmp_path):
    # External callers may have saved a result produced by an older release.
    path = plot_rings({"ring_sizes": [6], "fractions": [100]}, tmp_path,
                      label="C", dpi=40)
    assert path == str(tmp_path / "analysis_rings.png")


def test_void_exports_preserve_histogram_and_probe_curve_units(tmp_path, monkeypatch):
    import amorphgen.analysis.plotting as plotting

    atoms = Atoms("Si", cell=[5] * 3, pbc=True)
    result = compute_void_distribution([atoms], n_samples=96, nbins=8,
                                       probe_radii=[0.0, 0.5, 1.0, 10.0], seed=8)
    result["structure_files"] = ["sample.xyz"]
    captured = {}
    original_save = plotting._save_fig

    def capture(fig, base, **kwargs):
        if base.endswith("_probe"):
            captured["labels"] = fig.axes[0].get_legend_handles_labels()[1]
            captured["values"] = fig.axes[0].lines[0].get_ydata()
        original_save(fig, base, **kwargs)

    monkeypatch.setattr(plotting, "_save_fig", capture)
    save_descriptor("voids", result, tmp_path, dpi=40, save_pdf=True)
    exported = json.loads((tmp_path / "analysis_voids.json").read_text())
    assert exported["probe_curve"] == result["probe_curve"]
    header, rows = read_rows(tmp_path / "analysis_voids.csv")
    assert header == ["clearance_A", "probability_density_per_A", "bin_volume_fraction"]
    assert len(rows) == 8
    header, rows = read_rows(tmp_path / "analysis_voids_probe.csv")
    assert header == ["probe_radius_A", "accessible_fraction", "accessible_fraction_stderr",
                      "accessible_volume_A3", "accessible_volume_stderr_A3"]
    fractions = [float(row["accessible_fraction"]) for row in rows]
    assert fractions == result["probe_curve"]["accessible_fraction"]
    assert np.all(np.diff(fractions) <= 0)
    assert fractions[-1] == 0
    assert "±1 Monte Carlo sampling SE" in captured["labels"]
    np.testing.assert_allclose(captured["values"], fractions)
    _, rows = read_rows(tmp_path / "analysis_voids_structures.csv")
    assert rows[0]["source_file"] == "sample.xyz"
    assert float(rows[0]["cell_volume_A3"]) == pytest.approx(125)
    assert float(rows[0]["clearance_p50_A"]) == pytest.approx(
        result["per_structure"][0]["clearance_quantiles"]["p50"])
    for name in ("analysis_voids", "analysis_voids_probe"):
        for suffix in ("png", "pdf"):
            assert (tmp_path / f"{name}.{suffix}").stat().st_size > 0
    text = format_descriptor("voids", result)
    assert "Accessible-point clearance quantiles (A)" in text
    assert "Sample maxima are lower bounds" in text
    assert "exclude between-structure variation" in text


def test_void_unobserved_clearances_export_as_missing(tmp_path):
    atoms = Atoms("Si", cell=[4] * 3, pbc=True)
    result = compute_void_distribution([atoms], n_samples=8, radii={"Si": 10})
    save_descriptor("voids", result, tmp_path, dpi=40)
    _, rows = read_rows(tmp_path / "analysis_voids_structures.csv")
    assert rows[0]["mean_clearance_A"] == ""
    assert rows[0]["clearance_p90_A"] == ""
    assert rows[0]["accessible_fraction"] == "0.0"
    assert float(rows[0]["accessible_fraction_interval_95_high"]) > 0
    assert "p50=unavailable" in format_descriptor("voids", result)
