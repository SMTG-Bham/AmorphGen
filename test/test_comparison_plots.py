"""Multi-ensemble comparison panels: drawn data, legends and exported CSV values."""

import csv
import json

import numpy as np
import pytest
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.io import write
from matplotlib.collections import LineCollection, PathCollection
from scipy.stats import t

from amorphgen.analysis import StructureAnalyser
from amorphgen.analysis import comparison_plots as cp

BOND = 1.65


def _mol(symbols, offsets, cell=10.0):
    return Atoms(symbols, positions=np.asarray(offsets, float) + 5.0,
                 cell=[cell] * 3, pbc=True)


def _three_fold(cell=10.0):
    # Si bonded to three O: O-Si-O angles 180, 90, 90
    return _mol("SiO3", [[0, 0, 0], [BOND, 0, 0], [-BOND, 0, 0], [0, BOND, 0]], cell)


def _two_fold():
    # same composition, one O left unbonded: Si CN 2, O CN 1, 1, 0; one 90-degree angle
    return _mol("SiO3", [[0, 0, 0], [BOND, 0, 0], [0, BOND, 0], [-3.5, -3.5, 0]])


def _bridge():
    # linear Si-O-Si: Si CN 1, O CN 2, Si-O-Si 180
    return _mol("SiOSi", [[-BOND, 0, 0], [0, 0, 0], [BOND, 0, 0]])


def _spec(label, frames):
    return cp.EnsembleSpec.from_analyser(label, StructureAnalyser(frames, cutoff=2.0))


def _rows(path):
    with open(path, newline="") as handle:
        return list(csv.reader(handle))[1:]


def _density(atoms):
    mass = sum(atomic_masses[atomic_numbers[s]] for s in atoms.get_chemical_symbols())
    return mass * 1.66053906660e-24 / (atoms.get_volume() * 1e-24)


@pytest.fixture
def figures(monkeypatch):
    """Record each panel's figure while still writing small image files."""
    saved = {}
    original = cp._save

    def capture(fig, output_dir, prefix, name, save_pdf=True, dpi=300):
        saved[name] = fig
        original(fig, output_dir, prefix, name, save_pdf=save_pdf, dpi=30)

    monkeypatch.setattr(cp, "_save", capture)
    return saved


def _data_lines(ax):
    return [line for line in ax.get_lines() if len(line.get_xdata()) > 2]


def test_glob_spec_resolves_sorted_files_once(tmp_path):
    cells = {"c": 11.0, "a": 9.0, "b": 10.0}
    for name, cell in cells.items():
        write(tmp_path / f"{name}.xyz", _three_fold(cell))
    spec = cp.EnsembleSpec("G", str(tmp_path / "*.xyz"), cutoff=2.0)
    files = spec.resolve_files()
    assert files == [str(tmp_path / f"{n}.xyz") for n in "abc"]
    write(tmp_path / "d.xyz", _three_fold())
    assert spec.resolve_files() == files
    analyser = spec.analyser()
    assert spec.analyser() is analyser
    assert [a.cell.lengths()[0] for a in analyser.atoms_list] == [9.0, 10.0, 11.0]


@pytest.mark.parametrize("files", ["missing/*.vasp", []])
def test_unmatched_inputs_raise(tmp_path, files):
    if isinstance(files, str):
        files = str(tmp_path / files)
    with pytest.raises(FileNotFoundError, match="No files matched for ensemble 'Empty'"):
        cp.EnsembleSpec("Empty", files).analyser()


def test_missing_colours_follow_palette_then_grey(tmp_path, figures):
    specs = [_spec(f"E{i}", [_three_fold()]) for i in range(8)]
    specs[2].color = "red"
    cp.plot_density(specs, None, str(tmp_path), save_pdf=False)
    expected = cp.DEFAULT_COLORS[:2] + ["red"] + cp.DEFAULT_COLORS[2:] + ["#888888"]
    assert [s.color for s in specs] == expected
    ax = figures["density"].axes[0]
    annotations = [text for text in ax.texts if "CI unavailable" in text.get_text()]
    assert [text.get_color() for text in annotations] == expected


def test_rdf_panel_curves_integrate_to_coordination(tmp_path, figures):
    a = _spec("A", [_three_fold(), _two_fold()])
    b = _spec("B", [_bridge()])
    # "SiO" is not an A-B label; the panel skips it instead of failing.
    cp.plot_partial_rdf([a, b], [("Si-O", "-"), ("SiO", ":")], str(tmp_path),
                        prefix="cmp", save_pdf=True)
    for suffix in ("png", "pdf", "csv"):
        assert (tmp_path / f"cmp_rdf.{suffix}").is_file()
    rows = _rows(tmp_path / "cmp_rdf.csv")
    assert {(row[0], row[1]) for row in rows} == {("A", "Si-O"), ("B", "Si-O")}

    # First-shell integral 4 pi rho_O int g r^2 dr is the mean Si-O coordination.
    for label, rho_o, cn in (("A", 3 / 1000, 2.5), ("B", 1 / 1000, 1.0)):
        r, g = np.array([[float(row[2]), float(row[3])] for row in rows
                         if row[0] == label]).T
        shell = r < 3.0
        integral = 4 * np.pi * rho_o * np.sum(g[shell] * r[shell] ** 2) * (r[1] - r[0])
        assert integral == pytest.approx(cn, rel=0.02)
        assert r[np.argmax(g)] == pytest.approx(BOND, abs=0.03)

    ax = figures["rdf"].axes[0]
    lines = _data_lines(ax)
    assert [line.get_color() for line in lines] == [a.color, b.color]
    assert ax.get_xlim() == pytest.approx((0, r[-1]))
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert labels[:3] == ["A", "B", "Si-O"]
    descriptors = json.loads((tmp_path / "cmp_rdf_uncertainty.json").read_text())["descriptors"]
    assert descriptors["A: Si-O"]["n_structures"] == 2
    assert descriptors["B: Si-O"]["n_structures"] == 1


def _bars(ax):
    return [(round(p.get_x() + p.get_width() / 2, 6), round(p.get_height(), 6))
            for p in ax.patches]


def test_coordination_panel_mirrors_the_second_site(tmp_path, figures):
    a = _spec("A", [_three_fold(), _two_fold()])
    b = _spec("B", [_bridge()])
    cp.plot_coordination([a, b], "Si-O", "O-Si", str(tmp_path), prefix="p", save_pdf=False)
    # Si-O: A has one CN-3 and one CN-2 Si; O-Si: A has 5 CN-1 and 1 CN-0 sites.
    expected = [
        ["A", "Si-O", 0, 0.0, 0], ["A", "Si-O", 1, 0.0, 0],
        ["A", "Si-O", 2, 50.0, 50.0], ["A", "Si-O", 3, 50.0, 50.0],
        ["B", "Si-O", 0, 0.0, 0], ["B", "Si-O", 1, 100.0, 100.0],
        ["B", "Si-O", 2, 0.0, 0], ["B", "Si-O", 3, 0.0, 0],
        ["A", "O-Si", 0, 16.7, 50.0], ["A", "O-Si", 1, 83.3, 100.0],
        ["A", "O-Si", 2, 0.0, 0], ["A", "O-Si", 3, 0.0, 0],
        ["B", "O-Si", 0, 0.0, 0], ["B", "O-Si", 1, 0.0, 0],
        ["B", "O-Si", 2, 100.0, 100.0], ["B", "O-Si", 3, 0.0, 0],
    ]
    rows = _rows(tmp_path / "p_coordination.csv")
    assert [[r[0], r[1], int(r[2]), float(r[3]), float(r[4])] for r in rows] == expected

    ax = figures["coordination"].axes[0]
    # two ensembles share each CN slot: offsets -0.2 / +0.2, bars 0.4 wide
    assert _bars(ax) == [
        (-0.2, 0), (0.8, 0), (1.8, 50), (2.8, 50),
        (0.2, 0), (1.2, 100), (2.2, 0), (3.2, 0),
        (-0.2, -16.7), (0.8, -83.3), (1.8, 0), (2.8, 0),
        (0.2, 0), (1.2, 0), (2.2, -100), (3.2, 0)]
    assert {round(p.get_width(), 6) for p in ax.patches} == {0.4}
    assert ax.get_ylim() == pytest.approx((-120, 120))
    assert list(ax.get_xticks()) == [0, 1, 2, 3]
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["A", "B"]
    assert {"Si-O", "O-Si", "(b)"} <= {text.get_text() for text in ax.texts}
    assert ax.yaxis.get_major_formatter()(-83.3, 0) == "83"


def test_coordination_panel_single_site(tmp_path, figures):
    a = _spec("A", [_three_fold(), _two_fold()])
    cp.plot_coordination([a], "Si-O", None, str(tmp_path), save_pdf=False)
    rows = _rows(tmp_path / "coordination.csv")
    assert [(r[1], int(r[2]), float(r[3])) for r in rows] == [
        ("Si-O", 2, 50.0), ("Si-O", 3, 50.0)]
    ax = figures["coordination"].axes[0]
    assert _bars(ax) == [(2, 50), (3, 50)]
    assert ax.get_ylim() == pytest.approx((0, 60))
    texts = {text.get_text() for text in ax.texts}
    assert "Si-O" in texts and "O-Si" not in texts


def test_angle_panel_normalises_over_the_full_range(tmp_path, figures):
    a = _spec("A", [_three_fold(), _two_fold()])
    b = _spec("B", [_bridge()])
    cp.plot_bond_angles([a, b], [("O-Si-O", "-"), ("Si-O-Si", "--")], str(tmp_path),
                        bins=[80, 100, 120], save_pdf=False)
    # A: structure 1 has angles 90, 90, 180 -> 2 / (3 * 20) in the 80-100 bin;
    # structure 2 has one 90 -> 1 / 20. Each structure is normalised over
    # 0-180 before averaging, so the shown 90-degree value is 1/24, not 1/20.
    # B's only angle is 180, outside the shown range.
    rows = [(r[0], r[1], float(r[2]), float(r[3]))
            for r in _rows(tmp_path / "angles.csv")]
    assert rows == [("A", "O-Si-O", 90.0, pytest.approx(1 / 24)),
                    ("A", "O-Si-O", 110.0, 0.0),
                    ("B", "Si-O-Si", 90.0, 0.0),
                    ("B", "Si-O-Si", 110.0, 0.0)]
    ax = figures["angles"].axes[0]
    assert ax.get_xlim() == (80.0, 120.0)
    lines = [line for line in ax.get_lines() if len(line.get_xdata()) == 4]
    assert [(line.get_color(), line.get_linestyle()) for line in lines] == [
        (a.color, "-"), (b.color, "--")]
    np.testing.assert_allclose(lines[1].get_ydata(), [0, 0, 0, 1 / 60])
    descriptors = json.loads((tmp_path / "angles_uncertainty.json").read_text())["descriptors"]
    assert set(descriptors) == {"A: O-Si-O", "B: Si-O-Si"}


@pytest.mark.parametrize("bins", [[100, 80], [-10, 90], [90, 190], [90], [[0, 90]],
                                  [0, np.nan]])
def test_angle_bins_must_increase_within_0_to_180(tmp_path, bins):
    with pytest.raises(ValueError, match="angle bins must increase within 0–180 degrees"):
        cp.plot_bond_angles([_spec("A", [_three_fold()])], [("O-Si-O", "-")],
                            str(tmp_path), bins=bins, save_pdf=False)


def test_density_panel_draws_experiment_and_intervals(tmp_path, figures):
    dense_frames = [_three_fold(), _three_fold(10.5)]
    a = _spec("Dense", dense_frames)
    b = _spec("Single", [_bridge()])
    lo, hi = 0.115, 0.125
    cp.plot_density([a, b], (lo, hi), str(tmp_path), exp_label="Neutron", save_pdf=False)

    rho = [_density(f) for f in dense_frames]
    rows = _rows(tmp_path / "density.csv")
    assert [r[:2] for r in rows] == [["Neutron", "exp_lo"], ["Neutron", "exp_hi"],
                                     ["Dense", "0"], ["Dense", "1"], ["Single", "0"]]
    # the package rounds Avogadro's number to 6.022e23 (2e-5 relative)
    np.testing.assert_allclose([float(r[2]) for r in rows],
                               [lo, hi, *rho, _density(_bridge())], rtol=1e-4)

    ax = figures["density"].axes[0]
    assert list(ax.get_xticks()) == [1, 2, 3]
    assert [label.get_text() for label in ax.get_xticklabels()] == ["Neutron", "Dense", "Single"]
    segments = [s.tolist() for c in ax.collections if isinstance(c, LineCollection)
                for s in c.get_segments()]
    assert [[1, lo], [1, hi]] in segments
    for y in (lo, hi):
        assert [[pytest.approx(0.82), y], [pytest.approx(1.18), y]] in segments
    markers = [c.get_offsets().tolist() for c in ax.collections
               if isinstance(c, PathCollection)]
    assert [[1, pytest.approx((lo + hi) / 2)]] in markers

    mean = np.mean(rho)
    half = t.ppf(0.975, 1) * abs(rho[0] - rho[1]) / 2
    assert [[2, pytest.approx(mean - half, rel=1e-4)],
            [2, pytest.approx(mean + half, rel=1e-4)]] in segments
    texts = [text.get_text() for text in ax.texts]
    assert f"{(lo + hi) / 2:.2f}" in texts
    assert f"{mean:.2f}\n95% t CI [{mean - half:.2f}, {mean + half:.2f}]" in texts
    assert f"{_density(_bridge()):.2f}\nCI unavailable (n < 2)" in texts
    ymin, ymax = ax.get_ylim()
    assert ymin < min(mean - half, lo) and ymax > max(mean + half, hi)


def test_compare_ensembles_writes_only_requested_descriptors(tmp_path, figures):
    for i, atoms in enumerate([_three_fold(), _two_fold()]):
        write(tmp_path / f"a_{i}.xyz", atoms)
    write(tmp_path / "b_0.xyz", _bridge())
    specs = [cp.EnsembleSpec("A", str(tmp_path / "a_*.xyz"), cutoff=2.0),
             cp.EnsembleSpec("B", [str(tmp_path / "b_0.xyz")], cutoff=2.0)]
    out = tmp_path / "all"
    cp.compare_ensembles(specs, rdf_pairs=[("Si-O", "-")], cn_top_key="Si-O",
                         cn_bot_key="O-Si", angle_keys=[("O-Si-O", "-")],
                         exp_density=(0.1, 0.13), output_dir=str(out),
                         prefix="sio2", save_pdf=False)
    assert [s.color for s in specs] == cp.DEFAULT_COLORS[:2]
    names = {p.name for p in out.iterdir()}
    # B has no O-Si-O triplet, so it has no angle rows
    for descriptor, labels in (("rdf", {"A", "B"}), ("coordination", {"A", "B"}),
                               ("angles", {"A"}), ("density", {"A", "B", "Expt."})):
        assert {f"sio2_{descriptor}.png", f"sio2_{descriptor}.csv"} <= names
        assert {row[0] for row in _rows(out / f"sio2_{descriptor}.csv")} == labels
    assert not any(name.endswith(".pdf") for name in names)

    only = tmp_path / "density_only"
    cp.compare_ensembles(specs, output_dir=str(only))
    assert {p.name for p in only.iterdir()} == {
        "density.png", "density.pdf", "density.csv", "density_uncertainty.json",
        "density_uncertainty.csv", "density_per_structure.csv"}
