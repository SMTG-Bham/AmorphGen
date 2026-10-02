"""Physical and statistical checks for melt-ensemble snapshot sampling."""

import json

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator

from amorphgen.utils.snapshot_sampling import analyze_snapshot_sampling


def _brownian_frames(n_frames=400, *, seed=15, energies=None, slow_scale=0.3):
    """Paired walks keep each species' centre fixed independently."""
    rng = np.random.default_rng(seed)
    base = np.array([
        [3, 3, 3], [5, 3, 3], [3, 5, 3], [5, 5, 3],
        [3, 3, 5], [5, 3, 5], [3, 5, 5], [5, 5, 5],
    ], dtype=float)
    walk = np.cumsum(rng.normal(size=(n_frames, 4, 3)), axis=0)
    walk[:, :2] *= slow_scale
    walk[:, 2:] *= 0.3
    displacements = np.empty((n_frames, 8, 3))
    displacements[:, ::2] = walk
    displacements[:, 1::2] = -walk
    frames = []
    for i, displacement in enumerate(displacements):
        atoms = Atoms("Si4O4", positions=base + displacement,
                      cell=[100, 100, 100], pbc=False)
        if energies is not None:
            atoms.calc = SinglePointCalculator(atoms, energy=float(energies[i]))
        frames.append(atoms)
    return frames


def _assert_report_is_consistent(report, n_frames, requested):
    indices = report["selected_frame_indices"]
    assert indices == sorted(set(indices))
    assert 1 <= len(indices) <= requested
    assert all(isinstance(index, int) for index in indices)
    assert all(report["burn_in_frames"] <= index < n_frames for index in indices)
    assert report["trajectory_frames"] == n_frames
    assert report["requested_snapshots"] == requested
    assert 1 <= report["effective_independent_snapshots"] <= len(indices)
    assert isinstance(report["warnings"], list)
    # Prohibit numpy objects and NaNs in the persisted provenance report.
    json.dumps(report, allow_nan=False)


def test_energy_correlation_increases_spacing_for_same_atomic_diffusion():
    rng = np.random.default_rng(410)
    independent = rng.normal(size=800)
    correlated = np.zeros(800)
    for i in range(1, len(correlated)):
        correlated[i] = 0.98 * correlated[i - 1] + independent[i]
    fast = analyze_snapshot_sampling(
        _brownian_frames(800, energies=independent), n_snapshots=80,
        burn_in_frames=0, decorrelation_distance=0.2,
    )
    slow = analyze_snapshot_sampling(
        _brownian_frames(800, energies=correlated), n_snapshots=80,
        burn_in_frames=0, decorrelation_distance=0.2,
    )
    _assert_report_is_consistent(fast, 800, 80)
    _assert_report_is_consistent(slow, 800, 80)
    assert slow["spacing_frames"] > fast["spacing_frames"]
    assert len(slow["selected_frame_indices"]) < len(fast["selected_frame_indices"])
    assert fast["effective_independent_snapshots"] > 1


@pytest.mark.parametrize("period", [50, 100])
def test_undamped_energy_oscillation_is_not_independent_at_matching_phases(period):
    # Correlation becomes negative, then returns to one each period. Its
    # first zero crossing must not erase those later correlated recurrences.
    energies = np.cos(2 * np.pi * np.arange(1001) / period)
    frames = _brownian_frames(1001, energies=energies)
    n_snapshots = 1000 // period + 1
    uniform = analyze_snapshot_sampling(
        frames, n_snapshots=n_snapshots, select="uniform", burn_in_frames=0,
        decorrelation_distance=0.2,
    )
    indices = uniform["selected_frame_indices"]
    assert indices == list(range(0, 1001, period))
    np.testing.assert_allclose(energies[indices], 1)
    assert uniform["effective_independent_snapshots"] <= 1.1
    _assert_report_is_consistent(uniform, 1001, n_snapshots)

    decorrelated = analyze_snapshot_sampling(
        frames, n_snapshots=n_snapshots, burn_in_frames=0,
        decorrelation_distance=0.2,
    )
    assert decorrelated["status"] == "unresolved"
    assert decorrelated["selected_frame_indices"] == [1000]
    assert decorrelated["effective_independent_snapshots"] == 1
    assert decorrelated["spacing_frames"] is None
    assert decorrelated["autocorrelation"]["energy_per_atom"]["status"] == "unresolved"


def test_slowest_species_limits_sampling_even_when_other_species_diffuses():
    mobile = analyze_snapshot_sampling(
        _brownian_frames(), burn_in_frames=0, decorrelation_distance=0.2,
    )
    pinned_silicon = analyze_snapshot_sampling(
        _brownian_frames(slow_scale=0.0), burn_in_frames=0,
        decorrelation_distance=0.2,
    )
    assert len(mobile["selected_frame_indices"]) > 1
    assert mobile["effective_independent_snapshots"] > 1
    assert pinned_silicon["selected_frame_indices"] == [399]
    assert pinned_silicon["effective_independent_snapshots"] == 1
    assert pinned_silicon["warnings"]


def test_auto_burn_in_removes_initial_energy_relaxation():
    rng = np.random.default_rng(24)
    energies = np.r_[np.linspace(50, 5, 100), rng.normal(0, 0.1, 300)]
    report = analyze_snapshot_sampling(
        _brownian_frames(energies=energies), decorrelation_distance=0.2,
    )
    assert 80 <= report["burn_in_frames"] <= 200
    _assert_report_is_consistent(report, 400, 20)


def test_automatic_burn_in_has_nonzero_minimum():
    report = analyze_snapshot_sampling(
        _brownian_frames(100), decorrelation_distance=0.2,
    )
    assert report["burn_in_frames"] >= 10
    _assert_report_is_consistent(report, 100, 20)


@pytest.mark.parametrize("n_frames", [1, 2, 3])
def test_short_trajectory_reports_one_snapshot_conservatively(n_frames):
    report = analyze_snapshot_sampling(_brownian_frames(n_frames))
    assert report["selected_frame_indices"] == [n_frames - 1]
    assert report["effective_independent_snapshots"] == 1
    assert report["warnings"]
    _assert_report_is_consistent(report, n_frames, 20)


def test_missing_cached_energy_never_runs_a_calculator():
    class NeverCalculate(Calculator):
        implemented_properties = ["energy"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            raise AssertionError("Snapshot analysis must not evaluate the potential")

    frames = _brownian_frames(100)
    for frame in frames:
        frame.calc = NeverCalculate()
    report = analyze_snapshot_sampling(frames, decorrelation_distance=0.2)
    assert len(report["selected_frame_indices"]) > 1
    assert report["warnings"]
    _assert_report_is_consistent(report, 100, 20)


def test_constant_energy_uses_diffusion_and_warns():
    report = analyze_snapshot_sampling(
        _brownian_frames(100, energies=np.ones(100)), decorrelation_distance=0.2,
    )
    assert len(report["selected_frame_indices"]) > 1
    assert report["warnings"]
    _assert_report_is_consistent(report, 100, 20)


def test_larger_diffusion_distance_requires_larger_snapshot_spacing():
    frames = _brownian_frames()
    near = analyze_snapshot_sampling(
        frames, n_snapshots=80, burn_in_frames=0, decorrelation_distance=0.2,
    )
    far = analyze_snapshot_sampling(
        frames, n_snapshots=80, burn_in_frames=0, decorrelation_distance=1.0,
    )
    assert far["spacing_frames"] > near["spacing_frames"]
    assert len(far["selected_frame_indices"]) < len(near["selected_frame_indices"])


def test_effective_count_penalizes_dense_samples_of_the_same_trajectory():
    frames = _brownian_frames()
    decorrelated = analyze_snapshot_sampling(
        frames, burn_in_frames=0, decorrelation_distance=1.0,
    )
    consecutive = analyze_snapshot_sampling(
        frames, select="last", burn_in_frames=0, decorrelation_distance=1.0,
    )
    assert len(decorrelated["selected_frame_indices"]) == 20
    assert consecutive["selected_frame_indices"] == list(range(380, 400))
    assert (consecutive["effective_independent_snapshots"]
            < decorrelated["effective_independent_snapshots"])
    assert consecutive["effective_independent_snapshots"] < 10


def test_default_distance_is_median_nearest_neighbor_distance():
    atoms = Atoms("Si4", positions=[[0, 0, 0], [1, 0, 0], [4, 0, 0], [9, 0, 0]],
                  cell=[20, 20, 20], pbc=False)
    report = analyze_snapshot_sampling([atoms.copy() for _ in range(20)])
    # Per-atom nearest neighbours are 1, 1, 3 and 5 Angstrom.
    assert report["decorrelation_distance_angstrom"] == pytest.approx(2)
    assert report["distance_method"] == "median_nearest_neighbor"


def test_frame_stride_and_timestep_change_time_units_not_selected_indices():
    frames = _brownian_frames(150)
    short = analyze_snapshot_sampling(
        frames, burn_in_frames=0, decorrelation_distance=0.2,
        timestep_fs=0.5, frame_stride=100,
    )
    long = analyze_snapshot_sampling(
        frames, burn_in_frames=0, decorrelation_distance=0.2,
        timestep_fs=2.0, frame_stride=200,
    )
    assert short["selected_frame_indices"] == long["selected_frame_indices"]
    assert short["frame_interval_ps"] == pytest.approx(0.05)
    assert long["frame_interval_ps"] == pytest.approx(0.4)
    assert long["spacing_ps"] == pytest.approx(8 * short["spacing_ps"])
    assert short["effective_independent_snapshots"] == pytest.approx(
        long["effective_independent_snapshots"])


def test_unresolved_diffusion_returns_last_frame_and_count_shortfall_warning():
    report = analyze_snapshot_sampling(
        _brownian_frames(60), n_snapshots=50, burn_in_frames=10,
        decorrelation_distance=100,
    )
    assert report["selected_frame_indices"] == [59]
    assert report["effective_independent_snapshots"] == 1
    assert report["warnings"]
    _assert_report_is_consistent(report, 60, 50)


@pytest.mark.parametrize("cell", [
    np.diag([8.0, 9.0, 10.0]),
    np.array([[8.0, 0, 0], [3.0, 9.0, 0], [1.0, 2.0, 10.0]]),
])
def test_fractional_unwrapping_recovers_periodic_crossings(cell):
    from amorphgen.utils.snapshot_sampling import _relative_positions

    base = np.array([[0.9, 0.1, 0.3], [0.1, 0.8, 0.5]])
    velocity = np.array([[0.2, 0.1, 0], [-0.2, -0.1, 0]])
    frames = [Atoms("Si2", scaled_positions=(base + i * velocity) % 1,
                    cell=cell, pbc=True) for i in range(15)]
    relative = _relative_positions(frames)
    expected = np.arange(15)[:, None, None] * (velocity @ cell)[None, :, :]
    np.testing.assert_allclose(relative - relative[0], expected, atol=1e-12)


def test_nonperiodic_motion_is_not_wrapped_at_half_a_box():
    from amorphgen.utils.snapshot_sampling import _relative_positions

    frames = [Atoms("Si2", positions=[[3 * i, 0, 0], [-3 * i, 0, 0]],
                    cell=[4, 4, 4], pbc=False) for i in range(8)]
    relative = _relative_positions(frames)
    expected = np.array([frame.positions for frame in frames])
    np.testing.assert_allclose(relative - relative[0], expected)


def test_rigid_translation_and_affine_cell_expansion_are_not_diffusion():
    from amorphgen.utils.snapshot_sampling import _relative_positions

    base = np.array([[0.1, 0.2, 0.3], [0.8, 0.4, 0.6], [0.5, 0.7, 0.9]])
    cell = np.array([[8.0, 0, 0], [2.0, 9.0, 0], [1.0, 2.0, 10.0]])
    frames = [
        Atoms("SiO2", scaled_positions=(base + i * np.array([0.11, 0.03, 0])) % 1,
              cell=cell * (1 + 0.02 * i), pbc=True)
        for i in range(60)
    ]
    relative = _relative_positions(frames)
    np.testing.assert_allclose(np.diff(relative, axis=0), 0, atol=1e-12)
    report = analyze_snapshot_sampling(frames, burn_in_frames=0,
                                       decorrelation_distance=0.2)
    assert report["selected_frame_indices"] == [59]
    assert report["effective_independent_snapshots"] == 1


def test_lag_msd_averages_over_all_time_origins():
    from amorphgen.utils.snapshot_sampling import _lag_msd

    positions = np.zeros((5, 2, 3))
    positions[:, 0, 0] = [0, 1, 3, 6, 10]
    positions[:, 1, 0] = -positions[:, 0, 0]
    actual = _lag_msd(positions, max_lag=3)
    expected = [0] + [
        np.mean(np.sum((positions[lag:] - positions[:-lag]) ** 2, axis=-1))
        for lag in range(1, 4)
    ]
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("kwargs", [
    {"burn_in_frames": -1},
    {"burn_in_frames": 10},
    {"n_snapshots": 0},
    {"timestep_fs": 0},
    {"timestep_fs": np.inf},
    {"timestep_fs": np.nan},
    {"frame_stride": 0},
    {"decorrelation_distance": 0},
    {"decorrelation_distance": np.nan},
])
def test_invalid_sampling_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        analyze_snapshot_sampling(_brownian_frames(10), **kwargs)


def test_empty_trajectory_is_rejected():
    with pytest.raises(ValueError):
        analyze_snapshot_sampling([])
