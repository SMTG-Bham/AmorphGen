"""Convergence results remain explicit and usable in reports and exports."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from amorphgen.analysis.convergence import convergence_report
from amorphgen.analysis.convergence_output import (
    format_convergence_report, save_convergence_report,
)


def test_text_reports_declared_precision_counts_forecasts_and_limits():
    report = convergence_report(
        {"density": [1, 2, 3, 4], "curve": [[1, None], [2, 1], [3, None], [4, None]],
         "constant": [2] * 4, "undeclared": [None] * 4},
        {"density": .5, "curve": .1, "constant": .1},
    )
    report["descriptors"]["density"]["units"] = "g/cm^3"
    text = format_convergence_report(report)
    assert "confidence: 95%" in text
    assert "density: tolerance=0.5 g/cm^3" in text
    assert "available structures=4/4 (n=1–4 per point)" in text
    assert "status=insufficient_data" in text
    assert "constant: tolerance=0.1" in text and "additional=0" in text
    assert "undeclared: tolerance=undeclared" in text
    assert "estimated total=" in text
    assert "all subsets" in text and "generation-order" in text
    assert "not simultaneous" in text and "Zero observed variance" in text
    limited = convergence_report({"density": [1, 2, 3]}, {"density": .0001}, max_structures=5)
    assert "estimated total exceeds search limit 5" in format_convergence_report(limited)


def test_exports_preserve_missing_values_pointwise_data_and_projection(tmp_path):
    report = convergence_report(
        {"density": [1, 2, 3, 4], "curve": [[1, None], [2, 1], [3, None], [4, None]]},
        {"density": .5, "curve": .1}, sizes=[1, 2, 4],
    )
    report["descriptors"]["density"]["units"] = "g/cm^3"
    paths = save_convergence_report(report, tmp_path, dpi=40, save_pdf=True)
    assert json.loads(Path(paths["json"]).read_text()) == report
    assert Path(paths["report"]).read_text() == format_convergence_report(report)
    with Path(paths["summary"]).open() as handle:
        summary = {row["descriptor"]: row for row in csv.DictReader(handle)}
    assert summary["density"]["units"] == "g/cm^3"
    assert summary["curve"]["n_per_point"] == "[4, 1]"
    assert summary["curve"]["half_width"] == ""
    with Path(paths["curves"]).open() as handle:
        curves = list(csv.DictReader(handle))
    missing = [row for row in curves if row["descriptor"] == "curve" and row["point_index"] == "1"]
    assert missing and all(row["half_width"] == "" for row in missing)
    target = report["descriptors"]["density"]["estimated_total_structures"]
    endpoint = next(row for row in curves if row["descriptor"] == "density"
                    and int(row["ensemble_size"]) == target)
    assert endpoint["kind"] == "projection"
    assert float(endpoint["half_width"]) <= .5
    for files in paths["figures"].values():
        assert len(files) == 2
        assert all(Path(path).stat().st_size > 500 for path in files)


def test_figures_distinguish_planning_tolerance_and_observed_endpoint(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from amorphgen.analysis import plotting

    report = convergence_report({"density": [1, 2, 3, 4]}, {"density": .5})
    captured = []
    monkeypatch.setattr(plotting, "_save_fig", lambda figure, *args, **kwargs: captured.append(figure))
    original = plt.get_fignums()
    save_convergence_report(report, tmp_path)
    assert plt.get_fignums() == original
    axis = captured[0].axes[0]
    lines = {line.get_label(): line for line in axis.lines}
    assert "All-subset RMS planning" in lines
    assert lines["Projected uncertainty"].get_linestyle() == "--"
    tolerance = next(line for label, line in lines.items() if label.startswith("Declared tolerance"))
    np.testing.assert_equal(tolerance.get_ydata(), [.5, .5])
    np.testing.assert_allclose(axis.collections[0].get_offsets(),
                               [[4, report["descriptors"]["density"]["half_width"]]])
    assert any("Estimated total:" in label for label in lines)


def test_empty_data_and_colliding_descriptor_names_are_exportable(tmp_path):
    report = convergence_report({"a/b": [], "a.b": []}, {"a/b": .1})
    result = save_convergence_report(report, tmp_path, dpi=40)
    figures = [Path(paths[0]) for paths in result["figures"].values()]
    assert len(set(figures)) == 2
    assert all(path.parent == tmp_path and path.exists() for path in figures)
    empty_curve = convergence_report({"empty": [[], []]}, {"empty": .1})
    assert "n=no components" in format_convergence_report(empty_curve)


def test_invalid_json_numbers_do_not_write_partial_report(tmp_path):
    report = convergence_report({"density": [1, 2]}, {"density": .5})
    report["descriptors"]["density"]["half_width"] = float("nan")
    with pytest.raises(ValueError, match="JSON"):
        save_convergence_report(report, tmp_path)
    assert not list(tmp_path.iterdir())
