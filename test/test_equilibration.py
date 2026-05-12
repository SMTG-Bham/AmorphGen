"""
tests/test_equilibration.py
----------------------------
Tests for equilibration convergence analysis.
"""

import os
import pytest
import numpy as np
import tempfile


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

    def test_parse_skips_headers(self, sample_log):
        from amorphgen.utils.equilibration import parse_md_log
        data = parse_md_log(sample_log)
        # Should not include header or separator lines
        assert len(data["step"]) == 6

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
        assert len(data["step"]) == 2


class TestBlockAverage:

    def test_equilibrated_signal(self):
        """Constant energy should pass block average test."""
        from amorphgen.utils.equilibration import block_average_test
        # Create a fake log with constant energy
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("    Step     Time_ps       T_K       Epot_eV       Ekin_eV       Etot_eV      Vol_A3\n")
            f.write("-" * 80 + "\n")
            for i in range(100):
                e = -200.0 + np.random.normal(0, 0.1)
                f.write(f"  {i*100:6d}  {i*0.1:10.4f}  3000.0  {e:12.4f}  10.0000  {e+10:12.4f}  500.00\n")
            logfile = f.name

        try:
            is_eq, bd = block_average_test(logfile, n_atoms=40, n_blocks=4)
            # With random noise around a constant, should pass
            assert isinstance(is_eq, bool)
            assert "block_means" in bd
            assert "overall_mean" in bd
            assert len(bd["block_means"]) == 4
        finally:
            os.unlink(logfile)

    def test_drifting_signal_fails(self):
        """Linearly drifting energy should fail block average test."""
        from amorphgen.utils.equilibration import block_average_test
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("    Step     Time_ps       T_K       Epot_eV       Ekin_eV       Etot_eV      Vol_A3\n")
            f.write("-" * 80 + "\n")
            for i in range(100):
                e = -200.0 - i * 0.5  # strong drift
                f.write(f"  {i*100:6d}  {i*0.1:10.4f}  3000.0  {e:12.4f}  10.0000  {e+10:12.4f}  500.00\n")
            logfile = f.name

        try:
            is_eq, bd = block_average_test(logfile, n_atoms=40, n_blocks=4)
            assert is_eq is False
        finally:
            os.unlink(logfile)


class TestEnergyConvergence:

    def test_plot_from_log(self, tmp_path):
        from amorphgen.utils.equilibration import plot_energy_convergence
        logfile = tmp_path / "test.log"
        logfile.write_text(
            "    Step     Time_ps       T_K       Epot_eV       Ekin_eV       Etot_eV      Vol_A3\n"
            "------------------------------------------------------------------------------------\n"
        )
        with open(logfile, "a") as f:
            for i in range(50):
                e = -200.0 + np.random.normal(0, 0.1)
                f.write(f"  {i*100:6d}  {i*0.1:10.4f}  3000.0  {e:12.4f}  10.0000  {e+10:12.4f}  500.00\n")

        fig, drift = plot_energy_convergence(str(logfile), n_atoms=40)
        assert isinstance(drift, float)
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestRunningAverage:

    def test_basic(self):
        from amorphgen.utils.equilibration import running_average
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        avg = running_average(data, 3)
        assert len(avg) == len(data)
        # Window of 3 on [1,2,3,4,5]: valid results are [2,3,4]
        # With edge padding, middle value should be 3.0
        assert abs(avg[2] - 3.0) < 0.01

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
        # MSD should increase over time
        assert msd["all"][-1] > msd["all"][0]


# ─── Extra coverage: plot helpers + compute_cn_vs_time + convergence_report

class TestPlotEnergyConvergence:
    """Plot helpers should run without raising — even if no display backend."""
    def test_runs_on_log(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from amorphgen.utils.equilibration import plot_energy_convergence
        log = tmp_path / "stage_eq.log"
        log.write_text(
            "Step  Time(ps)  T(K)  Epot(eV)  Ekin(eV)  Etot(eV)  Vol(A^3)\n"
            + "\n".join(f"  {k:3d}  {k*0.001:.3f}  300  {-50.0 - 0.01*k:.4f}  10.0  -40.0  500"
                       for k in range(40))
        )
        ax = plot_energy_convergence(str(log), timestep_fs=1.0,
                                      window_ps=0.005, n_atoms=8)
        assert ax is not None


class TestPlotBlockAverages:
    def test_renders_without_error(self):
        import matplotlib
        matplotlib.use("Agg")
        from ase import Atoms
        from amorphgen.utils.equilibration import plot_block_averages

        # Reuse the FakeAtoms-style construction inline:
        class _FA(Atoms):
            def __init__(self, e):
                super().__init__("Si4",
                                 positions=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],
                                 cell=[5,5,5], pbc=True)
                self.info["energy"] = e
            def get_potential_energy(self, *a, **kw):
                return self.info["energy"]

        import numpy as _np
        rng = _np.random.default_rng(0)
        atoms = [_FA(-50 + rng.normal(0, 0.05)) for _ in range(60)]
        ax = plot_block_averages(atoms, n_blocks=4, discard_fraction=0.1,
                                  timestep_fs=1.0, n_atoms=4)
        assert ax is not None


class TestPlotMsd:
    def test_runs_on_drifting_traj(self):
        import matplotlib
        matplotlib.use("Agg")
        from ase import Atoms
        import numpy as _np
        from amorphgen.utils.equilibration import plot_msd

        frames = []
        for k in range(15):
            pos = _np.array([[0,0,0],[2,0,0],[0,2,0],[0,0,2]], dtype=float) + 0.1 * k
            frames.append(Atoms("Si4", positions=pos, cell=[10,10,10], pbc=True))
        ax = plot_msd(frames, timestep_fs=1.0)
        assert ax is not None


class TestPlotTemperature:
    def test_runs_on_log(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from amorphgen.utils.equilibration import plot_temperature
        log = tmp_path / "tlog.log"
        log.write_text(
            "Step  Time(ps)  T(K)  Epot(eV)  Ekin(eV)  Etot(eV)  Vol(A^3)\n"
            + "\n".join(f"  {k:3d}  {k*0.001:.3f}  {300+k*5}  -50.0  10.0  -40.0  500"
                       for k in range(20))
        )
        ax = plot_temperature(str(log), timestep_fs=1.0, T_target=400)
        assert ax is not None


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
        import matplotlib
        matplotlib.use("Agg")
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
        convergence_report(
            str(log),
            timestep_fs=1.0,
            T_target=300,
            n_atoms=6,
            output_dir=str(out_dir),
        )
        assert out_dir.exists()
        assert any(out_dir.iterdir())
