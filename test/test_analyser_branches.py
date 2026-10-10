"""StructureAnalyser option and fallback branches, and analysis plot titles and styles."""

import os
import warnings

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from amorphgen.analysis import StructureAnalyser
from amorphgen.analysis import plotting
from amorphgen.analysis.plotting import plot_analysis, plot_rings, plot_sq, plot_tr

BOND = 1.65


def _mol(symbols, offsets, cell=10.0):
    return Atoms(symbols, positions=np.asarray(offsets, float) + 5.0,
                 cell=[cell] * 3, pbc=True)


def _three_fold():
    # Si bonded to three O: O-Si-O angles 180, 90, 90
    return _mol("SiO3", [[0, 0, 0], [BOND, 0, 0], [-BOND, 0, 0], [0, BOND, 0]])


def _two_fold():
    # same composition, one O left unbonded: Si CN 2, O CN 1, 1, 0; one 90-degree angle
    return _mol("SiO3", [[0, 0, 0], [BOND, 0, 0], [0, BOND, 0], [-3.5, -3.5, 0]])


def _ensemble():
    return StructureAnalyser([_three_fold(), _two_fold()], cutoff=2.0)


def _table(text):
    """Per-structure table rows keyed by index, plus the Mean and Std rows."""
    rows = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 5 and (parts[0].isdigit() or parts[0] in ("Mean", "Std")):
            rows[int(parts[0]) if parts[0].isdigit() else parts[0]] = parts
    return rows


def test_empty_source_list_raises():
    with pytest.raises(ValueError, match="No structures loaded"):
        StructureAnalyser([])


def test_pair_missing_from_cutoff_table_is_unbonded_and_warned_once():
    # The cutoff table comes from the canonical first frame (pure Si), so the
    # Si-O pair of the second frame has no entry.
    lone = Atoms("Si", positions=[[5, 5, 5]], cell=[10] * 3, pbc=True)
    bonded = _mol("SiO", [[0, 0, 0], [BOND, 0, 0]])
    with pytest.warns(UserWarning, match="different compositions"):
        sa = StructureAnalyser([bonded, lone], cutoff={"default": 2.0, "Si-Si": 2.5})
    assert sa.cutoff == {"Si-Si": 2.5}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = sa.coordination()
        second = sa.coordination()
    messages = sorted(str(w.message) for w in caught if "no cutoff" in str(w.message))
    assert [m.split(";")[0] for m in messages] == ["no cutoff for pair O-Si",
                                                   "no cutoff for pair Si-O"]
    assert "treating it as not bonded" in messages[0]
    for result in (first, second):
        assert result["Si-O"]["per_structure"] == [0.0, 0.0]
        assert result["O-Si"]["per_structure"] == [0.0, None]


@pytest.mark.parametrize("bins", [True, 2.5, 0, -4])
def test_angle_distribution_rejects_non_positive_integer_bins(bins):
    with pytest.raises(ValueError, match="bins must be a positive integer or bin edges"):
        _ensemble().angle_distribution(bins=bins)


@pytest.mark.parametrize("edges", [[[0, 90, 180]], [0.0], [0, np.nan, 180],
                                   [0, 120, 90, 180], [10, 180], [0, 170]])
def test_angle_distribution_rejects_edges_not_covering_0_to_180(edges):
    with pytest.raises(ValueError, match="angle bin edges must increase and cover 0–180"):
        _ensemble().angle_distribution(bins=edges)


def test_angle_distribution_explicit_edges():
    sa = _ensemble()
    # structure 1: 90, 90 | 180; structure 2: 90 | none (edges 0, 100, 180)
    density = sa.angle_distribution(bins=[0, 100, 180])["O-Si-O"]
    assert density["bin_edges"] == [0, 100, 180]
    assert density["angle"] == [50, 140]
    np.testing.assert_allclose(density["per_structure"], [[2 / 300, 1 / 240], [1 / 100, 0]])
    np.testing.assert_allclose(density["distribution"], [1 / 120, 1 / 480])
    counts = sa.angle_distribution(bins=4.0, normalise=False)["O-Si-O"]
    assert counts["bin_edges"] == [0, 45, 90, 135, 180]
    assert counts["normalization"] == "count"
    np.testing.assert_allclose(counts["per_structure"], [[0, 0, 2, 1], [0, 0, 1, 0]])


@pytest.mark.parametrize("kwargs, message", [
    ({"load_options": {"skiprows": 1}}, "load_options requires an experimental data path"),
    ({"method": "debye"}, "method must be 'direct' or 'ft'"),
])
def test_compare_experiment_rejects_bad_arguments(kwargs, message):
    measured = {"kind": "sq", "x": [1.0, 2.0], "observed": [1.0, 1.0]}
    with pytest.raises(ValueError, match=message):
        _ensemble().compare_experiment(measured, **kwargs)


def test_averaged_rdf_matches_the_analytic_histogram():
    result = _ensemble().averaged_rdf(pair="Si-O", rmax=3.0, nbins=30)
    r = np.asarray(result["r"])
    assert r[16] == pytest.approx(BOND)
    # one Si with three (then two) O at 1.65 A in 1000 A^3, O density 3/1000
    shell = (3 / 1000) * 4 * np.pi * BOND ** 2 * 0.1
    mean, spread = np.zeros(30), np.zeros(30)
    mean[16], spread[16] = 2.5 / shell, 0.5 / shell
    np.testing.assert_allclose(result["g_r_mean"], mean)
    np.testing.assert_allclose(result["g_r_std"], spread, atol=1e-9)
    assert result["n_structures"] == 2


def test_averaged_cn_uses_structure_means_and_skips_absent_centres():
    oxygen_only = Atoms("O3", positions=[[1, 1, 1], [4, 4, 4], [7, 7, 7]],
                        cell=[10] * 3, pbc=True)
    with pytest.warns(UserWarning, match="different compositions"):
        sa = StructureAnalyser([_three_fold(), _two_fold(), oxygen_only], cutoff=2.0)
    result = sa.averaged_cn("Si-O")
    assert set(result) == {"Si-O", "O-Si"}
    si = result["Si-O"]
    assert si["mean_per_structure"] == [3.0, 2.0, None]
    assert si["overall_mean"] == pytest.approx(2.5)
    assert si["overall_std"] == pytest.approx(0.5)
    assert si["n_structures"] == 2
    assert si["uncertainty"]["sem"] == pytest.approx(0.5)
    o = result["O-Si"]
    assert o["mean_per_structure"] == pytest.approx([1.0, 2 / 3, 0.0])
    assert o["overall_std"] == pytest.approx(np.std([1.0, 2 / 3, 0.0]))
    assert o["n_structures"] == 3


def test_convergence_report_rejects_non_mapping_descriptors():
    with pytest.raises(ValueError, match="descriptors must be a mapping"):
        _ensemble().convergence_report(descriptors=[("rdf.total", [1.0, 2.0])])


def test_per_structure_energy_prefers_stored_info_then_calculator():
    stored = _three_fold()
    stored.info["energy"] = -10.0
    stored.calc = SinglePointCalculator(stored, energy=-99.0)
    calculated = _three_fold()
    calculated.calc = SinglePointCalculator(calculated, energy=-12.0)
    text = StructureAnalyser([stored, calculated, _three_fold()],
                             cutoff=2.0).per_structure_summary()
    rows = _table(text)
    assert [rows[i][2] for i in range(3)] == ["-2.5000", "-3.0000", "N/A"]
    assert rows["Mean"][2] == "-2.7500" and rows["Std"][2] == "0.2500"
    assert "E/atom: mean=-2.75; SEM=0.25;" in text and "(n=2)" in text


def _random_gen_layout(tmp_path, names, log):
    structures = tmp_path / "run" / "random_opt"
    structures.mkdir(parents=True)
    for name in names:
        write(structures / f"{name}.vasp", _three_fold(), format="vasp")
    (tmp_path / "run" / "random_gen.log").write_bytes(log)
    return [str(structures / f"{name}.vasp") for name in names]


def _log_entry(index, energy):
    return (f"  Composition: O3Si (4 atoms)\n"
            f"       0   {energy + 1:.3f}    0.90\n"
            f"       7   {energy:.3f}    0.03\n"
            f"  Converged after 7 steps!\n"
            f"[{index + 1}/3] O3Si -> out/random_{index:04d}_opt.vasp\n")


def test_per_structure_energy_falls_back_to_random_gen_log(tmp_path):
    # VASP files carry no energy; the log in the parent directory does.
    # File names select log entries; a name without an index uses its position.
    log = "".join(_log_entry(i, e) for i, e in enumerate([-20.0, -22.0, -18.0]))
    files = _random_gen_layout(
        tmp_path, ["random_0001_opt", "random_0000_opt", "plain", "random_0007_opt"],
        log.encode())
    text = StructureAnalyser(files, cutoff=2.0).per_structure_summary()
    rows = _table(text)
    assert [rows[i][2] for i in range(4)] == ["-5.5000", "-5.0000", "-4.5000", "N/A"]
    assert rows["Mean"][2] == "-5.0000"
    assert rows["Std"][2] == f"{np.std([-5.5, -5.0, -4.5]):.4f}"


@pytest.mark.parametrize("log", [
    b"\xff\xfe not UTF-8\n",
    b"  Composition: O3Si (4 atoms)\n       0   -20.000    0.90\n",
])
def test_unusable_random_gen_log_gives_no_energies(tmp_path, log):
    files = _random_gen_layout(tmp_path, ["random_0000_opt", "random_0001_opt"], log)
    rows = _table(StructureAnalyser(files, cutoff=2.0).per_structure_summary())
    assert [rows[key][2] for key in (0, 1, "Mean", "Std")] == ["N/A"] * 4


# ── plotting.py ───────────────────────────────────────────────────────────


@pytest.fixture
def figures(monkeypatch):
    """Keep each analysis figure by file stem instead of rendering it."""
    saved = {}

    def capture(fig, base_path, *args, **kwargs):
        saved[os.path.basename(base_path)] = fig

    monkeypatch.setattr(plotting, "_save_fig", capture)
    return saved


def test_plot_analysis_titles_selected_pair_and_panel_bands(tmp_path, figures):
    plot_analysis(_ensemble(), output_dir=str(tmp_path), rdf_pairs=["O-Si"],
                  show_title=True, pair_panels=True, total_cn=["O"], save_csv=False)
    rdf = figures["analysis_rdf"].axes[0]
    assert rdf.get_title() == "RDF — O3Si"
    assert [line.get_label() for line in rdf.get_lines()
            if not line.get_label().startswith("_")] == ["O-Si"]

    panels = figures["analysis_rdf_panels"]
    assert "Partial RDFs — O3Si" in [text.get_text() for text in panels.texts]
    assert [text.get_text() for text in panels.axes[0].get_legend().get_texts()] == [
        "95% pointwise bootstrap CI"]
    assert panels.axes[0].texts[0].get_text() == "O-Si"

    total = figures["analysis_cn_total"]
    assert "CN Distribution — O3Si" in [text.get_text() for text in total.texts]
    assert total.axes[0].get_xlabel() == "O-(all bonded) CN"
    assert figures["analysis_angles"].axes[0].get_title() == "Bond Angle Distribution — O3Si"
    assert figures["analysis_density"].axes[0].get_title() == "Density — O3Si"


def test_rdf_pair_filter_limits_the_exported_columns(tmp_path, figures):
    plot_analysis(_ensemble(), output_dir=str(tmp_path), rdf_pairs=["O-Si", "O-O"])
    header = (tmp_path / "analysis_rdf.csv").read_text().splitlines()[0]
    assert header == "r(A),g(r)_Total,g(r)_O-O,g(r)_O-Si"


@pytest.mark.parametrize("style", ["histogram", "both"])
def test_angle_histogram_styles(tmp_path, figures, style):
    plot_analysis(_ensemble(), output_dir=str(tmp_path), angle_style=style, save_csv=False)
    ax = figures["analysis_angles"].axes[0]
    # 2-degree bins: structure 1 has 2/3 of its angles at 90 and 1/3 at 180,
    # structure 2 has its single angle at 90; densities are fractions / 2.
    expected = np.zeros(90)
    expected[45], expected[89] = (1 / 3 + 1 / 2) / 2, (1 / 6) / 2
    np.testing.assert_allclose([bar.get_height() for bar in ax.patches], expected)
    assert {round(bar.get_width(), 9) for bar in ax.patches} == {2.0}
    line_labels = [line.get_label() for line in ax.get_lines()]
    assert ("O-Si-O" in line_labels) == (style == "both")
    assert "O-Si-O" in [text.get_text() for text in ax.get_legend().get_texts()]


def test_scattering_plots_without_uncertainty(tmp_path, figures):
    sq = {"q": [1.0, 2.0, 3.0, 4.0], "s_q": [0.4, np.nan, 1.3, 0.9],
          "n_per_bin": [3, 2, 0, 5], "partials": {"O-Si": [0.2, 0.5, 0.7, 1.1]}}
    plot_sq(sq, output_dir=str(tmp_path), weighting="neutron", show_title=True)
    total = figures["analysis_sq"].axes[0]
    assert total.get_title() == "Total S(q) — direct method, neutron weighting"
    assert total.get_legend() is None
    # NaN values and shells without q-vectors are not drawn
    assert list(total.get_lines()[0].get_xdata()) == [1.0, 4.0]
    partial = figures["analysis_sq_partials"].axes[0]
    assert partial.get_title() == "Faber-Ziman partial structure factors"
    assert list(partial.get_lines()[0].get_xdata()) == [1.0, 2.0, 4.0]

    tr = {"r": [1.0, 2.0, 3.0], "T_r": [0.1, 0.5, 0.2], "g_r": [0.0, 1.2, 1.0],
          "G_r": [0.0, 0.3, 0.1], "weighting": "xray", "qmin": 0.3, "qmax": 20.0,
          "window": "lorch", "rho": 0.07}
    plot_tr(tr, output_dir=str(tmp_path), show_title=True)
    ax = figures["analysis_tr"].axes[0]
    assert ax.get_title() == r"T(r) — xray weighting, Q = 0.3-20.0 $\mathrm{\AA}^{-1}$"
    assert ax.get_legend() is None
    assert sorted(p.name for p in tmp_path.iterdir()) == ["analysis_sq.csv", "analysis_tr.csv"]


def test_ring_plot_title(tmp_path, monkeypatch):
    seen = {}

    def capture(fig, base_path, *args, **kwargs):
        ax = fig.axes[0]
        seen.update(title=ax.get_title(), heights=[p.get_height() for p in ax.patches])

    monkeypatch.setattr(plotting, "_save_fig", capture)
    plot_rings({"ring_sizes": [5, 6], "fractions": [40.0, 60.0]}, str(tmp_path),
               label="Si", show_title=True)
    assert seen == {"title": "Ring statistics (Si)", "heights": [40.0, 60.0]}
