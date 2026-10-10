"""Equilibration statistics against analytic oracles: MSD, D fits, blocks, RDF/CN windows, reports."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
from ase import Atoms, units
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import atomic_numbers, covalent_radii
from ase.io import Trajectory, write

from amorphgen.utils import equilibration as eq


def opposing_pair(displacements, energies=None, info_key=None):
    """Two Si atoms moving apart by +-d: the centre of mass is fixed, so MSD = d**2."""
    frames = []
    for i, d in enumerate(displacements):
        atoms = Atoms("Si2", positions=[[-2 - d, 0, 0], [2 + d, 0, 0]],
                      cell=[100, 100, 100], pbc=False)
        if energies is not None:
            if info_key:
                atoms.info[info_key] = energies[i]
            else:
                atoms.calc = SinglePointCalculator(atoms, energy=energies[i])
        frames.append(atoms)
    return frames


def write_log(path, energies, temperature=300.0):
    rows = ["Step Time_ps T_K Epot_eV Ekin_eV Etot_eV Vol_A3", "-" * 50]
    rows += [f"{i * 100} {i * 0.1:.4f} {temperature} {e:.8f} 1.0 {e + 1:.8f} 500.0"
             for i, e in enumerate(energies)]
    path.write_text("\n".join(rows) + "\n")
    return path


def patch_extent(patch, axis):
    """Data extent of an axhspan/axvspan patch (Polygon in old, Rectangle in new matplotlib)."""
    vertices = patch.get_patch_transform().transform(patch.get_path().vertices)
    return vertices[:, axis].min(), vertices[:, axis].max()


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ── Trajectory input and MSD ────────────────────────────────────────────────

@pytest.mark.parametrize("kind", ["str", "pathlike", "file_list", "ase_trajectory", "generator"])
def test_every_trajectory_input_gives_the_analytic_msd(tmp_path, kind):
    displacements = 0.3 * np.arange(6)
    frames = opposing_pair(displacements)
    if kind in ("str", "pathlike"):
        write(tmp_path / "traj.xyz", frames, format="extxyz")
        source = tmp_path / "traj.xyz" if kind == "pathlike" else str(tmp_path / "traj.xyz")
    elif kind == "file_list":
        source = []
        for i, atoms in enumerate(frames):
            write(tmp_path / f"frame{i}.xyz", atoms, format="extxyz")
            source.append(str(tmp_path / f"frame{i}.xyz"))
    elif kind == "ase_trajectory":
        with Trajectory(str(tmp_path / "traj.traj"), "w") as handle:
            for atoms in frames:
                handle.write(atoms)
        source = Trajectory(str(tmp_path / "traj.traj"))
    else:
        source = (atoms for atoms in frames)
    time_ps, msd = eq.compute_msd(source, timestep_fs=2.0, frame_stride=5)
    if kind == "ase_trajectory":
        source.close()
    np.testing.assert_allclose(time_ps, np.arange(6) * 0.01)
    np.testing.assert_allclose(msd["all"], displacements ** 2, atol=1e-12)
    np.testing.assert_allclose(msd["Si"], displacements ** 2, atol=1e-12)


def test_triclinic_boundary_crossings_are_unwrapped():
    cell = np.array([[6.0, 0, 0], [2.0, 5.0, 0], [1.0, 1.0, 4.0]])
    step = np.array([0.7, 0.4, -0.3])
    frames = []
    for i in range(12):
        atoms = Atoms("Si2", positions=[[1, 1, 1] + i * step, [3, 3, 2] - i * step],
                      cell=cell, pbc=True)
        atoms.wrap()
        frames.append(atoms)
    _, msd = eq.compute_msd(frames, by_element=False)
    assert set(msd) == {"all"}
    np.testing.assert_allclose(msd["all"], np.arange(12) ** 2 * (step @ step))


def test_msd_of_empty_or_atomless_trajectory_is_empty():
    times, msd = eq.compute_msd([])
    assert len(times) == 0 and msd == {}
    times, msd = eq.compute_msd([Atoms(cell=[5, 5, 5], pbc=True)] * 3,
                                timestep_fs=1.0, frame_stride=1)
    np.testing.assert_allclose(times, [0, 0.001, 0.002])
    assert msd == {}


@pytest.mark.parametrize("change, message", [
    ("symbols", "unchanged atom order and composition"),
    ("pbc", "unchanged periodic boundary conditions"),
])
def test_msd_rejects_frames_that_are_not_the_same_system(change, message):
    frames = opposing_pair([0.0, 0.1, 0.2])
    if change == "symbols":
        frames[2].set_chemical_symbols(["Si", "Ge"])
    else:
        frames[1].pbc = [True, False, False]
    with pytest.raises(ValueError, match=message):
        eq.compute_msd(frames)


@pytest.mark.parametrize("timestep_fs, frame_stride", [(0.0, 100), (-0.5, 100), (0.5, 0)])
def test_nonpositive_timestep_or_stride_is_rejected(timestep_fs, frame_stride):
    with pytest.raises(ValueError, match="timestep_fs and frame_stride must be positive"):
        eq.compute_msd(opposing_pair([0.0, 0.1]), timestep_fs=timestep_fs,
                       frame_stride=frame_stride)


# 24 frames 0.1 ps apart; plot_msd fits the last 12 (t = 1.2 ... 2.3 ps).
# Diffusive: MSD = 0.6 t -> D = 0.1 A^2/ps = 1e-5 cm^2/s. Ballistic: MSD = v^2 t^2,
# whose least-squares slope over a symmetric grid is v^2 (t_first + t_last).
@pytest.mark.parametrize("motion, expected_cm2_s", [
    ("diffusive", 1e-5),
    ("ballistic", 0.25 * (1.2 + 2.3) / 6 * 1e-4),
])
def test_plot_msd_fits_the_late_half_slope(motion, expected_cm2_s):
    t = np.arange(24) * 0.1
    d = np.sqrt(0.6 * t) if motion == "diffusive" else 0.5 * t
    fig, ax = plt.subplots()
    returned, diffusion = eq.plot_msd(opposing_pair(d), timestep_fs=1.0, ax=ax)
    assert returned is fig
    assert diffusion == pytest.approx({"all": expected_cm2_s, "Si": expected_cm2_s})
    assert {line.get_label() for line in ax.lines} == {"all", "Si"}
    assert [text.get_text() for text in ax.texts] == [f"D(Si) = {expected_cm2_s:.2e} cm$^2$/s"]


# ── Energies, drift and block averaging ─────────────────────────────────────

@pytest.mark.parametrize("key", ["energy", "Energy"])
def test_energies_fall_back_to_info_without_a_calculator(key):
    energies = [-8.0, -8.5, -9.0]
    total, per_atom = eq.extract_energies(opposing_pair([0, 0, 0], energies, info_key=key))
    np.testing.assert_allclose(total, energies)
    np.testing.assert_allclose(per_atom, np.array(energies) / 2)


def test_energies_of_an_empty_trajectory_are_empty():
    total, per_atom = eq.extract_energies([])
    assert total.shape == per_atom.shape == (0,)


@pytest.mark.parametrize("call, n_atoms, message", [
    (eq.extract_energies, None, "n_atoms required"),
    (eq.plot_energy_convergence, None, "n_atoms required"),
    (eq.convergence_report, None, "positive n_atoms required"),
    (eq.convergence_report, 0, "positive n_atoms required"),
])
def test_log_sources_need_a_positive_atom_count(tmp_path, call, n_atoms, message):
    log = write_log(tmp_path / "md.log", [-10.0] * 4)
    with pytest.raises(ValueError, match=message):
        call(str(log), n_atoms=n_atoms)


def test_energy_drift_from_frames_uses_timestep_and_stride():
    per_atom = -3.0 + 0.02 * np.arange(20)
    frames = opposing_pair(np.zeros(20), 2 * per_atom)
    fig, ax = plt.subplots()
    # One frame = 2 fs * 50 steps = 0.1 ps, so 0.02 eV/atom per frame is 0.2 eV/atom/ps.
    returned, drift = eq.plot_energy_convergence(frames, timestep_fs=2.0, frame_stride=50, ax=ax)
    assert returned is fig
    assert drift == pytest.approx(0.2)
    raw, average, fit = ax.lines
    np.testing.assert_allclose(raw.get_xdata(), np.arange(20) * 0.1)
    np.testing.assert_allclose(raw.get_ydata(), per_atom)
    # A centred 5-frame (0.5 ps) average reproduces a straight line away from the ends.
    np.testing.assert_allclose(average.get_ydata()[2:-2], per_atom[2:-2])
    np.testing.assert_allclose(fit.get_ydata(), per_atom)
    _, drift_total = eq.plot_energy_convergence(frames, timestep_fs=2.0, frame_stride=50,
                                                per_atom=False)
    assert drift_total == pytest.approx(0.4)


# Production [0,2,0,2 | 1,3,1,3]: block means 1 and 2, mean 1.5, std sqrt(5)/2,
# so the block threshold 2*std/sqrt(4) is sqrt(5)/2 and the largest deviation 0.5.
BLOCKS = [0, 2, 0, 2, 1, 3, 1, 3]


@pytest.mark.parametrize("discard, passed", [(0.5, True), (0.0, False)])
def test_block_average_matches_hand_computed_statistics(discard, passed):
    energies = [5.0] * 8 + BLOCKS
    frames = [Atoms("Si", cell=[5, 5, 5], pbc=True) for _ in energies]
    for atoms, energy in zip(frames, energies):
        atoms.calc = SinglePointCalculator(atoms, energy=energy)
    is_eq, data = eq.block_average_test(frames, n_blocks=2, discard_fraction=discard)
    assert is_eq is passed
    if passed:
        np.testing.assert_allclose(data["block_means"], [1.0, 2.0])
        assert data["overall_mean"] == pytest.approx(1.5)
        assert data["overall_std"] == pytest.approx(np.sqrt(5) / 2)
        assert data["sem"] == pytest.approx(np.sqrt(5) / 2 / np.sqrt(8))
        assert data["threshold"] == pytest.approx(np.sqrt(5) / 2)
        assert data["max_deviation"] == pytest.approx(0.5)
    else:
        # Keeping the 5 eV plateau: means 5 and 1.5 deviate by 1.75 from 3.25, beyond
        # 2 * sqrt(1.25 / 2 + 1.75**2) / sqrt(8).
        np.testing.assert_allclose(data["block_means"], [5.0, 1.5])
        assert data["max_deviation"] == pytest.approx(1.75)
        assert data["threshold"] == pytest.approx(2 * np.sqrt(0.625 + 1.75 ** 2) / np.sqrt(8))


@pytest.mark.parametrize("n_blocks, discard", [(1, 0.1), (4, 1.0), (4, -0.1)])
def test_block_average_rejects_invalid_parameters(n_blocks, discard):
    with pytest.raises(ValueError, match="n_blocks must be >= 2"):
        eq.block_average_test(opposing_pair(np.zeros(20), np.zeros(20)),
                              n_blocks=n_blocks, discard_fraction=discard)


def test_plot_block_averages_from_log_places_blocks_in_time(tmp_path):
    log = write_log(tmp_path / "md.log", 2 * np.array([100.0, 100.0] + BLOCKS))
    fig, ax = plt.subplots()
    returned, is_eq, data = eq.plot_block_averages(
        str(log), n_blocks=2, discard_fraction=0.2, timestep_fs=1.0, n_atoms=2, ax=ax)
    assert returned is fig and is_eq is True
    np.testing.assert_allclose(data["block_means"], [1.0, 2.0])
    # 0.1 ps per frame; two discarded frames, then blocks of four.
    segments = [collection.get_segments()[0] for collection in ax.collections]
    np.testing.assert_allclose(segments, [[[0.2, 1.0], [0.6, 1.0]], [[0.6, 2.0], [1.0, 2.0]]])
    discarded, = [patch for patch in ax.patches if patch.get_label() == "Discarded"]
    assert patch_extent(discarded, 0) == pytest.approx((0.0, 0.2))
    assert ax.get_title() == "Block average test: EQUILIBRATED"


# ── Temperature ─────────────────────────────────────────────────────────────

def test_plot_temperature_from_frames_draws_the_canonical_band():
    temperatures = [300.0, 320.0, 280.0, 310.0]
    frames = []
    for temperature in temperatures:
        atoms = Atoms("Ar4", positions=np.eye(4, 3) * 3, cell=[10] * 3, pbc=True)
        masses = atoms.get_masses()
        # Each atom carries 3/2 kB T along x, i.e. a kinetic temperature of T.
        momenta = np.zeros((4, 3))
        momenta[:, 0] = np.sqrt(3 * units.kB * temperature * masses)
        atoms.set_momenta(momenta)
        frames.append(atoms)
    fig, ax = plt.subplots()
    returned = eq.plot_temperature(frames, timestep_fs=1.0, T_target=300, ax=ax,
                                   frame_stride=100)
    assert returned is fig
    raw = ax.lines[0]
    np.testing.assert_allclose(raw.get_xdata(), [0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(raw.get_ydata(), temperatures)
    sigma = 300 * np.sqrt(2 / (3 * 4))
    band, = ax.patches
    assert patch_extent(band, 1) == pytest.approx((300 - 2 * sigma, 300 + 2 * sigma))
    assert band.get_label() == "Expected +/-2s (245 K)"


# ── RDF and coordination windows ────────────────────────────────────────────

def nacl_frames(lattice_constants):
    return [bulk("NaCl", "rocksalt", a=a, cubic=True) for a in lattice_constants]


def running_cn(g, frame, target, rmax, nbins, same):
    """n(r) = integral of rho g 4 pi r^2 dr with the module's directed pair density."""
    dr = rmax / nbins
    r = np.linspace(dr / 2, rmax - dr / 2, nbins)
    n_target = frame.get_chemical_symbols().count(target) - (1 if same else 0)
    return np.sum(g * n_target / frame.get_volume() * 4 * np.pi * r ** 2 * dr)


def test_rdf_windows_track_lattice_expansion():
    # Nearest Na-Cl at a/2 (6 neighbours, bin centres every 0.05 A); like
    # pairs start at a/sqrt(2) > 3.5 A.
    constants = [5.65, 5.85, 6.05, 6.25]
    frames = nacl_frames(np.repeat(constants, 2))
    fig = eq.plot_rdf_time_windows(frames, rmax=3.5, nbins=70, timestep_fs=1.0,
                                   frame_stride=100)
    assert [ax.get_title() for ax in fig.axes] == ["Cl-Cl", "Cl-Na", "Na-Na"]
    labels = ["0.0-0.2 ps", "0.2-0.4 ps", "0.4-0.6 ps", "0.6-0.8 ps"]
    for ax in fig.axes:
        assert [line.get_label() for line in ax.lines] == labels
    for ax in (fig.axes[0], fig.axes[2]):
        assert all(not line.get_ydata().any() for line in ax.lines)
    for window, line in enumerate(fig.axes[1].lines):
        g = line.get_ydata()
        assert line.get_xdata()[np.argmax(g)] == pytest.approx(constants[window] / 2)
        assert running_cn(g, frames[2 * window], "Na", 3.5, 70, False) == pytest.approx(6)


@pytest.mark.parametrize("pair, rmax, expected_cn", [
    (("Na", "K"), 4.2, 0),
    (("Na", "Na"), 3.5, 0),
    (("Na", "Na"), 4.2, 12),
    (("Cl", "Na"), 4.2, 6),
])
def test_partial_rdf_integrates_to_the_coordination_number(pair, rmax, expected_cn):
    frame, = nacl_frames([5.65])
    g = eq._compute_partial_rdf_frame(frame, *pair, rmax=rmax, nbins=84)
    if expected_cn == 0:
        assert not g.any()
    else:
        assert running_cn(g, frame, pair[1], rmax, 84, pair[0] == pair[1]) == pytest.approx(expected_cn)


def si_o_frames(n_frames):
    """One O just inside the default Si-O cutoff; a second O alternately just outside and inside."""
    cutoff = 1.3 * (covalent_radii[atomic_numbers["Si"]] + covalent_radii[atomic_numbers["O"]])
    frames = []
    for i in range(n_frames):
        second = 1.02 * cutoff if i % 2 == 0 else 0.97 * cutoff
        frames.append(Atoms("SiO2", positions=[[10, 10, 10], [10 + 0.98 * cutoff, 10, 10],
                                               [10, 10 + second, 10]],
                            cell=[20, 20, 20], pbc=False))
    return frames


def test_cn_windows_use_the_scaled_covalent_cutoff():
    times, cn, spread = eq.compute_cn_vs_time(si_o_frames(10), "Si", "O", window_size=4,
                                              timestep_fs=1.0, frame_stride=100)
    # Two full windows of four frames; the last two frames are not a window.
    np.testing.assert_allclose(times, [0.2, 0.6])
    np.testing.assert_allclose(cn, [1.5, 1.5])
    np.testing.assert_allclose(spread, [0.5, 0.5])


def test_cn_is_zero_when_the_centre_species_is_absent():
    _, cn, spread = eq.compute_cn_vs_time(si_o_frames(4), "Ge", "O", window_size=2)
    np.testing.assert_array_equal(cn, [0.0, 0.0])
    np.testing.assert_array_equal(spread, [0.0, 0.0])


def test_plot_cn_vs_time_draws_each_pair_with_its_expectation():
    fig, ax = plt.subplots()
    returned = eq.plot_cn_vs_time(si_o_frames(8), [("Si", "O", 1.5), ("O", "Si", 1.0)],
                                  window_size=4, timestep_fs=1.0, ax=ax)
    assert returned is fig
    si_o, o_si = ax.containers
    assert si_o.get_label() == "Si-O (expect 1.5)"
    assert o_si.get_label() == "O-Si (expect 1.0)"
    np.testing.assert_allclose(si_o.lines[0].get_xdata(), [0.2, 0.6])
    np.testing.assert_allclose(si_o.lines[0].get_ydata(), [1.5, 1.5])
    # Each O sees the Si in every frame (first O) or every other frame (second O).
    np.testing.assert_allclose(o_si.lines[0].get_ydata(), [0.75, 0.75])
    np.testing.assert_allclose([segment[:, 1] for segment in o_si.lines[2][0].get_segments()],
                               [[0.5, 1.0], [0.5, 1.0]])
    assert [line.get_ydata()[0] for line in ax.lines if line.get_linestyle() == "--"] == [1.5, 1.0]


# ── Convergence report ──────────────────────────────────────────────────────

def test_json_values_are_strict_and_native():
    converted = eq._json_values({"n": np.int64(3), "ok": np.bool_(True),
                                 "bad": np.float32(np.nan), "pair": (np.inf, np.float64(1.5)),
                                 "array": np.array([1.0, -np.inf])})
    assert converted == {"n": 3, "ok": True, "bad": None, "pair": [None, 1.5],
                         "array": [1.0, None]}
    assert type(converted["n"]) is int and type(converted["ok"]) is bool
    json.dumps(converted, allow_nan=False)


def test_report_accepts_pathlike_trajectory_and_log(tmp_path):
    per_atom = -4.0 + 0.01 * np.arange(10)
    write(tmp_path / "md.xyz", opposing_pair(np.zeros(10), 2 * per_atom), format="extxyz")
    report = eq.convergence_report(tmp_path / "md.xyz", make_plots=False)
    assert report["n_frames"] == 10 and report["n_atoms"] == 2
    assert report["elements"] == ["Si"]
    # Default 0.5 fs x 100 steps per frame: 0.01 eV/atom per 0.05 ps.
    assert report["energy_drift_eV_per_atom_per_ps"] == pytest.approx(0.2)
    assert report["liquid_test"]["is_liquid"] is False

    log = write_log(tmp_path / "md.log", 2 * per_atom)
    from_log = eq.convergence_report(tmp_path / "md.log", n_atoms=2, make_plots=False)
    assert from_log["energy_drift_eV_per_atom_per_ps"] == pytest.approx(0.1)
    assert from_log == eq.convergence_report(str(log), n_atoms=2, make_plots=False)
    assert any("log only supplied" in warning for warning in from_log["warnings"])


@pytest.mark.parametrize("temperatures", [[1000.0] * 3, [1000.0, np.nan, 900.0, 900.0]])
def test_report_rejects_misaligned_or_nonfinite_targets(temperatures):
    with pytest.raises(ValueError, match="one finite target temperature per frame"):
        eq.convergence_report(opposing_pair(np.zeros(4)), temperatures=temperatures,
                              make_plots=False)


def test_log_ramp_tests_energy_stationarity_within_each_hold(tmp_path):
    steady = -10 + np.tile([-0.1, 0.1], 4)
    drifting = -10 - 0.5 * np.arange(8)
    log = write_log(tmp_path / "ramp.log", np.r_[steady, drifting])
    report = eq.convergence_report(str(log), n_atoms=2, make_plots=False,
                                   temperatures=np.repeat([1000.0, 500.0], 8))
    windows = report["temperature_windows"]
    assert [window["temperature_K"] for window in windows] == [1000, 500]
    assert [window["stationarity"]["energy"]["passed"] for window in windows] == [True, False]
    assert windows[1]["stationarity"]["energy"]["difference_eV_per_atom"] == pytest.approx(-1.0)
    assert all(window["liquid_test"]["is_liquid"] is None for window in windows)
    assert report["status"] == "insufficient_data"
    assert report["diffusion_freezing"]["status"] == "insufficient_data"
    assert report["liquid_test"]["status"] == "not_applicable_temperature_ramp"


def test_nonfinite_positions_make_diffusion_inconclusive_not_nan():
    frames = opposing_pair(0.1 * np.arange(32), np.full(32, -10.0), info_key="energy")
    frames[20].positions[0, 0] = np.nan
    report = eq.convergence_report(frames, make_plots=False)
    assert report["liquid_test"]["status"] == "insufficient_data"
    assert report["liquid_test"]["is_liquid"] is None
    assert report["msd_final_A2"] == {"Si": None, "all": None}
    assert report["stationarity"]["diffusion"] == {"status": "insufficient_data", "passed": None}
    assert report["stationarity"]["energy"]["passed"] is True
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("targets, status", [
    ([300.0, 600.0], "not_a_cooling_ramp"),
    ([1000.0, 600.0], "not_observed"),
])
def test_ramp_without_a_freezing_crossing(targets, status):
    # 1 A/ps ballistic separation: every 16-frame hold is far above 1e-6 cm^2/s.
    times = np.arange(32) * 0.1
    report = eq.convergence_report(opposing_pair(times), time_ps=times,
                                   temperatures=np.repeat(targets, 16), make_plots=False)
    assert report["diffusion_freezing"] == {"status": status, "temperature_K": None,
                                            "bracket_K": None}
    assert report["freezing_temperature_K"] is None
    assert [w["liquid_test"]["is_liquid"] for w in report["temperature_windows"]] == [True, True]


@pytest.mark.parametrize("save", [False, True])
def test_report_plots_ramp_figures(tmp_path, capsys, save):
    times = np.arange(16) * 0.1
    targets = np.repeat([1000.0, 600.0], 8)
    energies = -10.0 - 0.01 * np.arange(16)
    out = tmp_path / "plots" if save else None
    report = eq.convergence_report(
        opposing_pair(times, energies), time_ps=times, temperatures=targets,
        pairs_cn=[("Si", "Si", 1.0)], cn_cutoffs=[10.0], prefix="run",
        output_dir=None if out is None else str(out))
    assert capsys.readouterr().out == report["summary_text"] + "\n"
    names = ["energy", "blocks", "temperature", "msd", "rdf_windows", "cn"]
    if save:
        assert not any(key.startswith("fig_") for key in report)
        assert sorted(path.name for path in out.iterdir()) == sorted(
            [f"run_{name}.png" for name in names] + ["run_report.txt"])
        return
    assert sorted(key for key in report if key.startswith("fig_")) == sorted(
        f"fig_{name}" for name in names)
    np.testing.assert_allclose(report["fig_energy"].axes[0].lines[0].get_ydata(), energies / 2)
    instantaneous, target = report["fig_temperature"].axes[0].lines
    assert target.get_label() == "Target"
    np.testing.assert_allclose(target.get_ydata(), targets)
    msd_lines = {line.get_label(): line for line in report["fig_msd"].axes[0].lines}
    np.testing.assert_allclose(msd_lines["all"].get_xdata(), times)
    np.testing.assert_allclose(msd_lines["all"].get_ydata(), times ** 2, atol=1e-12)
    cn_line = report["fig_cn"].axes[0].containers[0].lines[0]
    np.testing.assert_allclose(cn_line.get_ydata(), [1.0])
