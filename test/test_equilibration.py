"""
tests/test_equilibration.py
----------------------------
Tests for equilibration convergence analysis.
"""

import pytest
import numpy as np


@pytest.fixture
def write_energy_log(tmp_path):
    """Write deterministic energy samples in the stage log format."""
    def write_log(energies):
        logfile = tmp_path / "energy.log"
        rows = [
            "Step Time_ps T_K Epot_eV Ekin_eV Etot_eV Vol_A3",
            "-" * 80,
        ]
        rows.extend(
            f"{i * 100} {i * 0.1:.4f} 3000.0 {energy:.8f} "
            f"10.0 {energy + 10:.8f} 500.0"
            for i, energy in enumerate(energies)
        )
        logfile.write_text("\n".join(rows) + "\n")
        return str(logfile)

    return write_log


class TestParseMdLog:

    @pytest.fixture
    def sample_log(self, tmp_path):
        logfile = tmp_path / "test.log"
        logfile.write_text(
            "    Step     Time_ps       T_K       Epot_eV       Ekin_eV       Etot_eV      Vol_A3\n"
            "------------------------------------------------------------------------------------\n"
            "       0      0.0000    3000.0     -200.0000       10.0000     -190.0000      500.00\n"
            "     100      0.1000    2800.0     -201.0000        9.5000     -191.5000      500.00\n"
            "     200      0.2000    3100.0     -200.5000       10.5000     -190.0000      500.00\n"
            "     300      0.3000    2900.0     -201.5000        9.8000     -191.7000      500.00\n"
            "     400      0.4000    3050.0     -201.0000       10.2000     -190.8000      500.00\n"
            "     500      0.5000    2950.0     -201.2000        9.9000     -191.3000      500.00\n"
        )
        return str(logfile)

    def test_parse_basic(self, sample_log):
        from amorphgen.utils.equilibration import parse_md_log
        data = parse_md_log(sample_log)
        assert len(data["step"]) == 6
        assert data["step"][0] == 0
        assert data["step"][-1] == 500
        assert abs(data["time_ps"][-1] - 0.5) < 0.001

    def test_parse_energies(self, sample_log):
        from amorphgen.utils.equilibration import parse_md_log
        data = parse_md_log(sample_log)
        assert data["Epot_eV"][0] == -200.0
        assert data["T_K"][0] == 3000.0

    def test_parse_with_temperature_markers(self, tmp_path):
        """Log with '-> T = ...' lines (from quench stage)."""
        logfile = tmp_path / "quench.log"
        logfile.write_text(
            "    Step     Time_ps       T_K       Epot_eV       Ekin_eV       Etot_eV      Vol_A3\n"
            "------------------------------------------------------------------------------------\n"
            "       0      0.0000    3000.0     -200.0000       10.0000     -190.0000      500.00\n"
            "  -> T =  2800 K\n"
            "     100      0.1000    2800.0     -201.0000        9.5000     -191.5000      500.00\n"
        )
        from amorphgen.utils.equilibration import parse_md_log
        data = parse_md_log(str(logfile))
        np.testing.assert_array_equal(data["step"], [0, 100])
        np.testing.assert_allclose(data["T_K"], [3000.0, 2800.0])


class TestBlockAverage:

    def test_equilibrated_signal(self, write_energy_log):
        """Equal block means with finite fluctuations should pass."""
        from amorphgen.utils.equilibration import block_average_test
        logfile = write_energy_log(-200.0 + np.tile([-0.1, 0.1], 40))
        is_eq, bd = block_average_test(
            logfile, n_atoms=40, n_blocks=4, discard_fraction=0.0,
        )
        assert is_eq is True
        assert bd["is_equilibrated"] is True
        np.testing.assert_allclose(bd["block_means"], [-5.0] * 4)
        assert bd["overall_mean"] == pytest.approx(-5.0)
        assert bd["max_deviation"] < bd["threshold"]

    def test_drifting_signal_fails(self, write_energy_log):
        """Linearly drifting energy should fail block average test."""
        from amorphgen.utils.equilibration import block_average_test
        logfile = write_energy_log(-200.0 - np.arange(100) * 0.5)
        is_eq, bd = block_average_test(logfile, n_atoms=40, n_blocks=4)
        assert is_eq is False
        assert bd["is_equilibrated"] is False
        assert bd["max_deviation"] > bd["threshold"]
        assert np.all(np.diff(bd["block_means"]) < 0)


class TestRunningAverage:

    def test_basic(self):
        from amorphgen.utils.equilibration import running_average
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        avg = running_average(data, 3)
        np.testing.assert_allclose(avg, [2.0, 2.0, 3.0, 4.0, 4.0])

    def test_window_larger_than_data(self):
        from amorphgen.utils.equilibration import running_average
        data = np.array([1.0, 2.0, 3.0])
        avg = running_average(data, 10)
        assert np.allclose(avg, np.mean(data))


class TestMSD:

    def test_stationary_atoms_zero_msd(self):
        """Atoms that don't move should have MSD = 0."""
        from amorphgen.utils.equilibration import compute_msd
        from ase import Atoms

        atoms = Atoms("Si4", positions=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],
                      cell=[5,5,5], pbc=True)
        frames = [atoms.copy() for _ in range(10)]

        time_ps, msd = compute_msd(frames, timestep_fs=1.0)
        assert len(time_ps) == 10
        assert np.allclose(msd["all"], 0.0)

    def test_moving_atoms_positive_msd(self):
        """Atoms moving linearly should have increasing MSD."""
        from amorphgen.utils.equilibration import compute_msd
        from ase import Atoms

        frames = []
        for i in range(20):
            atoms = Atoms("Si4",
                         positions=[[i*0.1,0,0],[1+i*0.1,0,0],
                                    [0,1,0],[0,0,1]],
                         cell=[10,10,10], pbc=True)
            frames.append(atoms)

        time_ps, msd = compute_msd(frames, timestep_fs=1.0)
        # Half the equal-mass atoms move by d, so each is displaced by d/2
        # relative to the centre of mass.
        np.testing.assert_allclose(msd["all"], (np.arange(20) * 0.05) ** 2)

    def test_rigid_drift_is_not_diffusion(self):
        """A drift of the whole system (the Langevin thermostat leaves the
        centre of mass free) must not read as diffusion."""
        from amorphgen.utils.equilibration import compute_msd
        from ase import Atoms

        base = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        frames = [Atoms("SiO3", positions=base + [0.3 * i, 0.1 * i, 0.0],
                        cell=[10, 10, 10], pbc=True) for i in range(20)]
        _, msd = compute_msd(frames, timestep_fs=1.0)
        for key in ("all", "Si", "O"):
            assert np.allclose(msd[key], 0.0)

    def test_displacement_is_relative_to_the_mass_weighted_centre(self):
        """Moving a light H by d beside a heavy Pb moves the centre of mass by
        d*m_H/M, so relative to it the H moves d*m_Pb/M and the Pb d*m_H/M."""
        from amorphgen.utils.equilibration import compute_msd
        from ase import Atoms

        frames = [Atoms("PbH", positions=[[0, 0, 0], [2 + 0.5 * i, 0, 0]],
                        cell=[20, 20, 20], pbc=True) for i in range(5)]
        _, msd = compute_msd(frames, timestep_fs=1.0)
        m_pb, m_h = frames[0].get_masses()
        d = 0.5 * 4
        assert msd["H"][-1] == pytest.approx((d * m_pb / (m_pb + m_h)) ** 2)
        assert msd["Pb"][-1] == pytest.approx((d * m_h / (m_pb + m_h)) ** 2)

    def test_frame_stride_scales_time_axis(self):
        """Trajectories are written every TRAJ_LOG_INTERVAL MD steps, so the
        time axis must scale with frame_stride (regression: it assumed one
        step per frame, making D wrong by ~100x)."""
        from amorphgen.utils.equilibration import compute_msd
        from amorphgen.utils.common import TRAJ_LOG_INTERVAL
        from ase import Atoms

        frames = [Atoms("Si", positions=[[i * 0.1, 0, 0]],
                        cell=[10, 10, 10], pbc=True) for i in range(8)]
        t_default, _ = compute_msd(frames, timestep_fs=0.5)
        t_stride1, _ = compute_msd(frames, timestep_fs=0.5, frame_stride=1)
        assert t_default[-1] == pytest.approx(t_stride1[-1] * TRAJ_LOG_INTERVAL)


class TestPartialRdfNormalisation:

    def test_same_element_asymptotes_to_one(self):
        """Same-element partial g(r) must approach 1 (not 0.5) at large r.
        Regression: undirected (i<j) counting with directed-pair-density
        normalisation halved g_AA."""
        import numpy as np
        from ase import Atoms
        from amorphgen.utils.equilibration import _compute_partial_rdf_frame

        rng = np.random.default_rng(0)
        L = 30.0
        n = 1500
        gas = Atoms("Ar" + str(n), positions=rng.uniform(0, L, (n, 3)),
                    cell=[L, L, L], pbc=True)
        g = _compute_partial_rdf_frame(gas, "Ar", "Ar", rmax=8.0, nbins=80)
        assert abs(g[len(g) // 2:].mean() - 1.0) < 0.1


# ─── Extra coverage: plot helpers + compute_cn_vs_time + convergence_report

class TestPlotEnergyConvergence:
    """Log times and atom counts set the units of the reported drift."""

    @pytest.mark.parametrize("per_atom, expected_drift", [(True, -0.125), (False, -5.0)])
    def test_drift_from_log(self, write_energy_log, per_atom, expected_drift):
        import matplotlib.pyplot as plt
        from amorphgen.utils.equilibration import plot_energy_convergence
        energies = -200.0 - np.arange(50) * 0.5
        log = write_energy_log(energies)
        fig, drift = plot_energy_convergence(
            log, timestep_fs=1.0, window_ps=0.5, n_atoms=40,
            per_atom=per_atom,
        )
        try:
            assert drift == pytest.approx(expected_drift)
            raw = fig.axes[0].lines[0]
            np.testing.assert_allclose(raw.get_xdata(), np.arange(50) * 0.1)
            np.testing.assert_allclose(raw.get_ydata(), energies / (40 if per_atom else 1))
        finally:
            plt.close(fig)


class TestPlotBlockAverages:
    def test_renders_without_error(self):
        import matplotlib.pyplot as plt
        from ase import Atoms
        from ase.calculators.singlepoint import SinglePointCalculator
        from amorphgen.utils.equilibration import plot_block_averages

        frames = []
        for energy in -50.0 + np.tile([-0.1, 0.1], 40):
            atoms = Atoms("Si4", positions=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],
                          cell=[5,5,5], pbc=True)
            atoms.calc = SinglePointCalculator(atoms, energy=energy)
            frames.append(atoms)
        fig, is_eq, data = plot_block_averages(
            frames, n_blocks=4, discard_fraction=0.0, timestep_fs=1.0,
        )
        try:
            assert is_eq is True
            np.testing.assert_allclose(data["block_means"], [-12.5] * 4)
            assert fig.axes[0].get_title() == "Block average test: EQUILIBRATED"
            assert len(fig.axes[0].collections) == 4
        finally:
            plt.close(fig)


class TestPlotMsd:
    def test_runs_on_drifting_traj(self):
        import matplotlib.pyplot as plt
        from ase import Atoms
        from amorphgen.utils.equilibration import plot_msd

        frames = []
        for k in range(30):
            pos = np.array([[0,0,0],[2,0,0],[0,2,0],[0,0,2]], dtype=float) + 0.1 * k
            frames.append(Atoms("Si4", positions=pos, cell=[10,10,10], pbc=True))
        fig, diffusion = plot_msd(frames, timestep_fs=1.0)
        try:
            assert set(diffusion) == {"all", "Si"}
            assert diffusion == pytest.approx({"all": 0.0, "Si": 0.0}, abs=1e-12)
            assert {line.get_label() for line in fig.axes[0].lines} == {"all", "Si"}
            for line in fig.axes[0].lines:
                np.testing.assert_allclose(line.get_ydata(), 0.0, atol=1e-12)
        finally:
            plt.close(fig)


class TestPlotTemperature:
    def test_runs_on_log(self, tmp_path):
        import matplotlib.pyplot as plt
        from amorphgen.utils.equilibration import plot_temperature
        log = tmp_path / "tlog.log"
        log.write_text(
            "Step  Time(ps)  T(K)  Epot(eV)  Ekin(eV)  Etot(eV)  Vol(A^3)\n"
            + "\n".join(f"  {k:3d}  {k*0.001:.3f}  {300+k*5}  -50.0  10.0  -40.0  500"
                       for k in range(20))
        )
        fig = plot_temperature(str(log), timestep_fs=1.0, T_target=400)
        try:
            raw, _, target = fig.axes[0].lines
            np.testing.assert_allclose(raw.get_xdata(), np.arange(20) * 0.001)
            np.testing.assert_allclose(raw.get_ydata(), 300 + np.arange(20) * 5)
            np.testing.assert_allclose(target.get_ydata(), [400, 400])
        finally:
            plt.close(fig)


class TestComputeCnVsTime:
    def test_constant_cn_returns_target(self):
        import numpy as _np
        from ase import Atoms
        from amorphgen.utils.equilibration import compute_cn_vs_time

        # SiO4 tetrahedron with all 4 O atoms at exactly 1.6 A from Si.
        d = 1.6
        positions = [
            (0, 0, 0),
            (d, 0, 0),
            (-d/3, d * (8/9)**0.5, 0),
            (-d/3, -d * (2/9)**0.5, d * (2/3)**0.5),
            (-d/3, -d * (2/9)**0.5, -d * (2/3)**0.5),
        ]
        frames = []
        for _ in range(8):
            frames.append(Atoms("SiO4", positions=positions,
                                 cell=[8, 8, 8], pbc=True))
        # Returns (time_centres_ps, cn_avg, cn_std)
        time_ps, cn_avg, cn_std = compute_cn_vs_time(
            frames, "Si", "O", cutoff=2.0, window_size=2, timestep_fs=1.0,
        )
        _np.testing.assert_allclose(cn_avg, 4.0)
        _np.testing.assert_allclose(cn_std, 0.0, atol=1e-12)


class TestConvergenceReport:
    def test_runs_on_log_file(self, tmp_path):
        from amorphgen.utils.equilibration import convergence_report

        # convergence_report needs energies — supply via a log file (the
        # canonical AmorphGen MD stage log format).
        log = tmp_path / "stage_eq.log"
        log.write_text(
            "Step  Time(ps)  T(K)  Epot(eV)  Ekin(eV)  Etot(eV)  Vol(A^3)\n"
            + "\n".join(
                f"  {k:3d}  {k*0.001:.3f}  {300+k:.0f}  {-50.0 - 0.005*k:.4f}  10.0  -40.0  500"
                for k in range(80)
            )
        )

        out_dir = tmp_path / "report"
        report = convergence_report(
            str(log),
            timestep_fs=1.0,
            T_target=300,
            n_atoms=6,
            output_dir=str(out_dir),
        )
        assert report["n_frames"] == 80
        assert report["n_atoms"] == 6
        assert report["total_time_ps"] == pytest.approx(0.079)
        assert report["energy_drift_eV_per_atom_per_ps"] == pytest.approx(-5.0 / 6)
        assert report["block_test_passed"] is False
        assert {path.name for path in out_dir.iterdir()} == {
            "convergence_energy.png", "convergence_blocks.png",
            "convergence_temperature.png", "convergence_report.txt",
        }
        assert (out_dir / "convergence_report.txt").read_text() == report["summary_text"]


@pytest.fixture
def diagnostic_frames():
    """Opposing equal-mass displacements with a known, linear total MSD."""
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    def make(displacements, energies=None):
        frames = []
        if energies is None:
            energies = np.full(len(displacements), -10.0)
        for displacement, energy in zip(displacements, energies):
            atoms = Atoms("Si2", positions=[[-2 - displacement, 0, 0],
                                           [2 + displacement, 0, 0]],
                          cell=[100, 100, 100], pbc=False)
            atoms.calc = SinglePointCalculator(atoms, energy=energy)
            frames.append(atoms)
        return frames

    return make


class TestNumericalDiagnostics:
    def test_nonuniform_times_and_diffusion_units(self, diagnostic_frames, capsys):
        import json
        from amorphgen.utils.equilibration import convergence_report

        # 0.06 A^2/ps / 6 * 1e-4 = 1e-6 cm^2/s; exceed it tenfold.
        times = np.r_[np.arange(23) * 0.1, 2.25]
        frames = diagnostic_frames(np.sqrt(0.6 * times))
        report = convergence_report(frames, time_ps=times, make_plots=False)
        assert report["diffusion_coefficients_cm2_s"] == pytest.approx({"all": 1e-5, "Si": 1e-5})
        assert report["msd_final_A2"]["all"] == pytest.approx(0.6 * 2.25)
        assert report["total_time_ps"] == pytest.approx(2.25)
        assert report["liquid_test"]["is_liquid"] is True
        assert report["block_test_passed"] is True
        assert report["stationarity"]["energy"]["passed"] is True
        json.dumps(report, allow_nan=False)
        assert capsys.readouterr().out == ""
        assert not any(key.startswith("fig_") for key in report)

    @pytest.mark.parametrize("n_frames", [0, 1, 2, 7])
    def test_short_trajectories_are_inconclusive(self, diagnostic_frames, n_frames):
        import json
        from amorphgen.utils.equilibration import convergence_report

        report = convergence_report(diagnostic_frames(np.zeros(n_frames)), make_plots=False)
        assert report["status"] == "insufficient_data"
        assert report["liquid_test"]["is_liquid"] is None
        assert report["diffusion_coefficients_cm2_s"] == {}
        assert report["freezing_temperature_K"] is None
        assert report["stationarity"]["diffusion"]["passed"] is None
        json.dumps(report, allow_nan=False)

    def test_early_late_energy_and_diffusion_detect_drift(self, diagnostic_frames):
        from amorphgen.utils.equilibration import convergence_report

        times = np.arange(32) * 0.1
        motion = np.r_[np.sqrt(6 * times[:16]), np.full(16, np.sqrt(6 * times[15]))]
        energies = np.r_[np.full(16, -10.0), np.full(16, -5.0)]
        report = convergence_report(diagnostic_frames(motion, energies), time_ps=times,
                                    make_plots=False)
        assert report["stationarity"]["energy"]["passed"] is False
        diffusion = report["stationarity"]["diffusion"]
        assert diffusion["passed"] is False
        assert diffusion["early_cm2_s"]["all"] == pytest.approx(1e-4)
        assert diffusion["late_cm2_s"]["all"] == pytest.approx(0)

    def test_freezing_from_local_temperature_windows(self, diagnostic_frames):
        from amorphgen.utils.equilibration import convergence_report

        times = np.arange(48) * 0.1
        motion = np.r_[np.sqrt(6 * times[:16]), np.full(32, np.sqrt(6 * times[15]))]
        report = convergence_report(diagnostic_frames(motion), time_ps=times,
                                    temperatures=np.repeat([1000, 600, 300], 16),
                                    make_plots=False)
        assert report["diffusion_coefficients_cm2_s"] == {}
        assert report["liquid_test"]["is_liquid"] is None
        assert report["freezing_temperature_K"] == 600
        assert report["diffusion_freezing"]["bracket_K"] == [600, 1000]
        assert report["diffusion_freezing"]["status"] == "observed"
        windows = report["temperature_windows"]
        assert [window["liquid_test"]["is_liquid"] for window in windows] == [True, False, False]
        assert windows[0]["diffusion_coefficients_cm2_s"]["all"] == pytest.approx(1e-4)
        assert report["stationarity"]["energy"]["passed"] is None

    def test_short_holds_do_not_manufacture_freezing(self, diagnostic_frames):
        from amorphgen.utils.equilibration import convergence_report

        report = convergence_report(diagnostic_frames(np.zeros(21)),
                                    temperatures=np.repeat([1000, 600, 300], 7),
                                    make_plots=False)
        assert report["freezing_temperature_K"] is None
        assert report["diffusion_freezing"]["status"] == "insufficient_data"
        assert all(window["liquid_test"]["is_liquid"] is None
                   for window in report["temperature_windows"])

    def test_frozen_entire_ramp_has_no_transition(self, diagnostic_frames):
        from amorphgen.utils.equilibration import convergence_report

        report = convergence_report(diagnostic_frames(np.zeros(32)),
                                    temperatures=np.repeat([1000, 300], 16),
                                    make_plots=False)
        assert report["freezing_temperature_K"] is None
        assert report["diffusion_freezing"]["status"] == "no_liquid_reference"

    def test_distinct_undersampled_targets_are_never_combined(self, diagnostic_frames):
        from amorphgen.utils.equilibration import convergence_report

        report = convergence_report(diagnostic_frames(np.zeros(80)),
                                    temperatures=np.linspace(2000, 300, 80),
                                    make_plots=False)
        assert report["status"] == "insufficient_data"
        assert len(report["temperature_windows"]) == 80
        assert all(window["liquid_test"]["is_liquid"] is None
                   for window in report["temperature_windows"])
        assert report["freezing_temperature_K"] is None

    def test_missing_energies_still_reports_diffusion(self, diagnostic_frames):
        from amorphgen.utils.equilibration import convergence_report

        frames = diagnostic_frames(np.zeros(16))
        for frame in frames:
            frame.calc = None
        report = convergence_report(frames, make_plots=False)
        assert report["block_test_passed"] is None
        assert report["stationarity"]["energy"]["status"] == "insufficient_data"
        assert report["liquid_test"]["is_liquid"] is False

    @pytest.mark.parametrize("times", [[0, 0], [0, np.nan], [0], [1, 0]])
    def test_invalid_sample_times_rejected(self, diagnostic_frames, times):
        from amorphgen.utils.equilibration import convergence_report

        with pytest.raises(ValueError, match="time_ps"):
            convergence_report(diagnostic_frames([0, 0]), time_ps=times, make_plots=False)


class TestMSDBoundariesAndCell:
    def test_nonperiodic_motion_is_not_wrapped(self):
        from ase import Atoms
        from amorphgen.utils.equilibration import compute_msd

        frames = [Atoms("Si2", positions=[[0, 0, 7 * i], [1, 0, 0]],
                        cell=[10, 10, 10], pbc=[True, False, False]) for i in range(4)]
        _, msd = compute_msd(frames)
        np.testing.assert_allclose(msd["all"], (np.arange(4) * 3.5) ** 2)

    def test_periodic_crossing_is_unwrapped(self):
        from ase import Atoms
        from amorphgen.utils.equilibration import compute_msd

        frames = [Atoms("Si2", positions=[[x, 0, 0], [5, 0, 0]],
                        cell=[10, 10, 10], pbc=True) for x in [9.5, 0, 0.5]]
        _, msd = compute_msd(frames)
        np.testing.assert_allclose(msd["all"], [0, 0.0625, 0.25])

    def test_affine_npt_expansion_is_not_diffusion(self):
        from ase import Atoms
        from amorphgen.utils.equilibration import compute_msd

        cell = np.array([[10, 0, 0], [1, 9, 0], [2, 1, 10]], float)
        frames = [Atoms("Si2", scaled_positions=[[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]],
                        cell=cell * scale, pbc=True) for scale in np.linspace(1, 2, 16)]
        _, msd = compute_msd(frames)
        np.testing.assert_allclose(msd["all"], 0, atol=1e-25)


class TestExpandedLogParser:
    def test_pressure_density_and_malformed_rows_are_aligned(self, tmp_path):
        from amorphgen.utils.equilibration import parse_md_log

        path = tmp_path / "stage.log"
        path.write_text("0 0 300 -10 1 -9 500 0.125 2.5\n"
                        "100 0.1 300 -10 1 -9 broken 0.1 2.5\n"
                        "200 0.2 300 -10 1 -9 500\n"
                        "300 0.3 300 -10 1 -9 500 nan 2.5\n")
        data = parse_md_log(path)
        assert all(len(values) == 3 for values in data.values())
        np.testing.assert_array_equal(data["step"], [0, 200, 300])
        np.testing.assert_allclose(data["P_GPa"], [0.125, np.nan, np.nan], equal_nan=True)
        np.testing.assert_allclose(data["density_g_cm3"], [2.5, np.nan, 2.5], equal_nan=True)
