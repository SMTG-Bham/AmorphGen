"""Tests for amorphgen.analysis.feff (forward-modelled EXAFS chi(k)).

Tier 1 — no FEFF binary required. The writer, the parsers, the
configurational average and the Fourier transform are all exercised
against closed-form expectations; only the runner needs the external
code, and what is tested there is that its absence fails fast and says
so.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from amorphgen.analysis.feff import (
    average_chi,
    chi_fourier_transform,
    collect_feff_results,
    feff_cluster,
    format_feff_report,
    ft_window,
    read_chi,
    read_manifest,
    run_feff_jobs,
    structure_neighbours,
    write_feff_csv,
    write_feff_input,
    write_feff_jobs,
)

A_SI = 5.43
D_SI = A_SI * np.sqrt(3) / 4          # diamond first-neighbour distance


def _sio2_cell():
    """Small periodic SiO2-ish cell: 2 Si, 4 O."""
    return Atoms(
        symbols=["Si", "Si", "O", "O", "O", "O"],
        positions=[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0],
                   [1.6, 0.0, 0.0], [0.0, 1.6, 0.0],
                   [0.0, 0.0, 1.6], [5.6, 0.0, 0.0]],
        cell=[8.0, 8.0, 8.0], pbc=True,
    )


def _atoms_rows(text):
    """The atom lines of a feff.inp, as lists of fields."""
    body = text.split("ATOMS")[1]
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("*") or line.startswith("END"):
            continue
        rows.append(line.split())
    return rows


def _potential_rows(text):
    """The POTENTIALS lines of a feff.inp, as (ipot, Z, tag) tuples."""
    body = text.split("POTENTIALS")[1].split("ATOMS")[0]
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("*"):
            continue
        ipot, z, tag = line.split()
        rows.append((int(ipot), int(z), tag))
    return rows


def _write_fake_chi(path, r0=1.62, sigma2=0.005, kmax=20.0, kstep=0.05,
                    style="chi"):
    """A synthetic single-shell FEFF output file.

    Not a physical spectrum — just a damped sinusoid with the column
    layout and comment header of the real thing, so the parsers and the
    averaging have something well defined to work on.
    """
    k = np.arange(0.0, kmax + 0.5 * kstep, kstep)
    chi = np.exp(-2 * sigma2 * k ** 2) * np.sin(2 * k * r0) / np.maximum(
        k, 1e-9) ** 2
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        if style == "chi":
            fh.write("# Feff synthetic fixture\n#  k  chi  mag  phase\n")
            for kk, cc in zip(k, chi):
                fh.write(f"{kk:11.4f}{cc:14.6e}{abs(cc):14.6e}{0.0:14.6e}\n")
        else:
            fh.write("# Feff synthetic fixture\n"
                     "#  omega  e  k  mu  mu0  chi\n")
            for kk, cc in zip(k, chi):
                fh.write(f"{0.0:10.3f}{0.0:10.3f}{kk:11.4f}"
                         f"{1.0:10.4f}{1.0:10.4f}{cc:14.6e}\n")
    return k, chi


# ─── cluster extraction ───────────────────────────────────────────────────

class TestCluster:
    @pytest.fixture
    def si(self):
        return bulk("Si", "diamond", a=A_SI, cubic=True)

    def test_absorber_first_and_at_origin(self, si):
        cl = feff_cluster(si, 0, radius=5.0)
        assert cl["symbols"][0] == "Si"
        assert np.allclose(cl["positions"][0], 0.0)
        assert cl["distances"][0] == 0.0

    def test_sorted_by_distance(self, si):
        cl = feff_cluster(si, 0, radius=6.0)
        assert np.all(np.diff(cl["distances"]) >= -1e-12)

    def test_periodic_images_included(self, si):
        """An 8-atom cell has 4 first neighbours only via its own images."""
        cl = feff_cluster(si, 0, radius=5.0)
        assert len(cl["symbols"]) > len(si)
        assert np.allclose(cl["distances"][1:5], D_SI)

    def test_radius_respected(self, si):
        cl = feff_cluster(si, 0, radius=4.0)
        assert cl["distances"].max() <= 4.0
        assert len(feff_cluster(si, 0, radius=6.0)["symbols"]) > len(
            cl["symbols"])

    def test_positions_match_distances(self, si):
        cl = feff_cluster(si, 0, radius=5.0)
        assert np.allclose(np.linalg.norm(cl["positions"], axis=1),
                           cl["distances"])

    def test_bad_index_raises(self, si):
        with pytest.raises(IndexError, match="absorber_index"):
            feff_cluster(si, len(si), radius=4.0)

    def test_shared_neighbour_list_gives_identical_cluster(self, si):
        """write_feff_jobs builds one neighbour list per structure and
        reuses it across absorbers; that must change nothing."""
        nl = structure_neighbours(si, 5.0)
        own = feff_cluster(si, 1, radius=5.0)
        shared = feff_cluster(si, 1, radius=5.0, neighbours=nl)
        assert own["symbols"] == shared["symbols"]
        assert np.allclose(own["positions"], shared["positions"])
        assert np.allclose(own["distances"], shared["distances"])


# ─── feff.inp ─────────────────────────────────────────────────────────────

class TestWriteInput:
    def test_required_blocks_present(self):
        text = write_feff_input(_sio2_cell(), 0, radius=4.0)
        for card in ("TITLE", "EDGE", "S02", "POTENTIALS", "ATOMS", "END"):
            assert card in text

    def test_absorber_row_is_ipot_zero_at_origin(self):
        rows = _atoms_rows(write_feff_input(_sio2_cell(), 0, radius=4.0))
        x, y, z, ipot = rows[0][0], rows[0][1], rows[0][2], rows[0][3]
        assert (float(x), float(y), float(z)) == (0.0, 0.0, 0.0)
        assert ipot == "0"

    def test_every_atom_ipot_is_declared(self):
        text = write_feff_input(_sio2_cell(), 0, radius=4.0)
        declared = {ipot for ipot, _, _ in _potential_rows(text)}
        used = {int(r[3]) for r in _atoms_rows(text)}
        assert used <= declared

    def test_no_unused_potential(self):
        """Every declared potential must be used by at least one atom.

        FEFF objects to a potential with no atoms assigned to it, which is
        what a naive "one ipot per element plus ipot 0" mapping produces
        for a lone dopant absorber.
        """
        lone = Atoms(symbols=["Ge", "O", "O", "O", "O"],
                     positions=[[6, 6, 6], [7.7, 6, 6], [4.3, 6, 6],
                                [6, 7.7, 6], [6, 4.3, 6]],
                     cell=[12] * 3, pbc=True)
        text = write_feff_input(lone, 0, radius=4.0)
        declared = {ipot for ipot, _, _ in _potential_rows(text)}
        used = {int(r[3]) for r in _atoms_rows(text)}
        assert declared == used
        assert [tag for _, _, tag in _potential_rows(text)] == ["Ge", "O"]

    def test_absorber_element_also_gets_scatterer_potential(self):
        """When other atoms share the absorber's element they need their own
        ipot: ipot 0 is the excited atom and must not be reused."""
        text = write_feff_input(_sio2_cell(), 0, radius=5.0)
        rows = _potential_rows(text)
        si_ipots = sorted(ipot for ipot, _, tag in rows if tag == "Si")
        assert si_ipots == [0, 1]

    def test_potentials_ordered_by_decreasing_z(self):
        text = write_feff_input(_sio2_cell(), 0, radius=5.0)
        zs = [z for ipot, z, _ in _potential_rows(text) if ipot > 0]
        assert zs == sorted(zs, reverse=True)

    def test_atom_count_matches_cluster(self):
        atoms = _sio2_cell()
        text = write_feff_input(atoms, 0, radius=4.0)
        assert len(_atoms_rows(text)) == len(
            feff_cluster(atoms, 0, radius=4.0)["symbols"])

    def test_edge_written(self):
        assert "EDGE      L3" in write_feff_input(_sio2_cell(), 0, edge="L3",
                                                  radius=4.0)

    def test_s02_is_unreduced_by_default(self):
        """A fitted S0^2 would be a silent fudge in a forward model."""
        text = write_feff_input(_sio2_cell(), 0, radius=4.0)
        assert "S02       1.0" in text

    def test_card_override(self):
        text = write_feff_input(_sio2_cell(), 0, radius=4.0,
                                cards={"EXAFS": "18.0", "SCF": "4.0"})
        assert "EXAFS     18.0" in text
        assert "SCF       4.0" in text

    def test_extra_cards_verbatim(self):
        text = write_feff_input(_sio2_cell(), 0, radius=4.0,
                                extra_cards=["CRITERIA 4.0 2.5"])
        assert "CRITERIA 4.0 2.5" in text

    def test_custom_title(self):
        assert "TITLE my run" in write_feff_input(_sio2_cell(), 0, radius=4.0,
                                                  title="my run")

    def test_written_to_disk(self, tmp_path):
        path = tmp_path / "job" / "feff.inp"
        text = write_feff_input(_sio2_cell(), 0, str(path), radius=4.0)
        assert path.exists()
        assert path.read_text() == text

    def test_path_none_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        text = write_feff_input(_sio2_cell(), 0, radius=4.0)
        assert "ATOMS" in text
        assert list(tmp_path.iterdir()) == []


# ─── job tree ─────────────────────────────────────────────────────────────

class TestJobTree:
    @pytest.fixture
    def ensemble(self):
        return [_sio2_cell() for _ in range(3)]

    def test_job_count_and_layout(self, ensemble, tmp_path):
        d = tmp_path / "jobs"
        manifest = write_feff_jobs(ensemble, "Si", str(d), n_sites=1,
                                   radius=4.0)
        assert manifest["n_jobs"] == 3
        for job in manifest["jobs"]:
            assert (d / job["dir"] / "feff.inp").exists()

    def test_all_sites_when_n_sites_none(self, ensemble, tmp_path):
        manifest = write_feff_jobs(ensemble, "Si", str(tmp_path / "j"),
                                   radius=4.0)
        assert manifest["n_jobs"] == 6          # 2 Si x 3 structures

    def test_manifest_round_trip(self, ensemble, tmp_path):
        d = tmp_path / "jobs"
        manifest = write_feff_jobs(ensemble, "Si", str(d), n_sites=1,
                                   radius=4.0, edge="L3")
        again = read_manifest(str(d))
        assert again == manifest
        assert again["absorber"] == "Si"
        assert again["edge"] == "L3"
        assert again["radius"] == 4.0

    def test_joblist_matches_manifest_order(self, ensemble, tmp_path):
        d = tmp_path / "jobs"
        manifest = write_feff_jobs(ensemble, "Si", str(d), n_sites=1,
                                   radius=4.0)
        listed = (d / "joblist.txt").read_text().split()
        assert listed == [j["dir"] for j in manifest["jobs"]]

    def test_runner_script_is_executable(self, ensemble, tmp_path):
        d = tmp_path / "jobs"
        write_feff_jobs(ensemble, "Si", str(d), n_sites=1, radius=4.0)
        assert os.access(d / "run_all.sh", os.X_OK)

    def test_sampling_is_deterministic(self, ensemble, tmp_path):
        a = write_feff_jobs(ensemble, "O", str(tmp_path / "a"), n_sites=2,
                            radius=4.0, seed=7)
        b = write_feff_jobs(ensemble, "O", str(tmp_path / "b"), n_sites=2,
                            radius=4.0, seed=7)
        assert [j["site"] for j in a["jobs"]] == [j["site"] for j in b["jobs"]]

    def test_absent_absorber_raises(self, ensemble, tmp_path):
        with pytest.raises(ValueError, match="no Hf atoms"):
            write_feff_jobs(ensemble, "Hf", str(tmp_path / "j"), radius=4.0)

    def test_read_manifest_on_plain_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="manifest.json"):
            read_manifest(str(tmp_path))


# ─── runner ───────────────────────────────────────────────────────────────

class TestRunner:
    def test_missing_binary_fails_fast_and_explains(self, tmp_path,
                                                    monkeypatch):
        """No binary must fail before the first job, with a message that
        says where FEFF comes from and how to run the tree elsewhere."""
        d = tmp_path / "jobs"
        write_feff_jobs([_sio2_cell()], "Si", str(d), n_sites=1, radius=4.0)
        monkeypatch.setattr("amorphgen.analysis.feff.shutil.which",
                            lambda _: None)
        with pytest.raises(FileNotFoundError) as excinfo:
            run_feff_jobs(str(d))
        message = str(excinfo.value)
        assert "not found on PATH" in message
        assert "does not ship FEFF" in message
        assert "collect_feff_results" in message


# ─── parsers ──────────────────────────────────────────────────────────────

class TestReadChi:
    def test_chi_dat_columns(self, tmp_path):
        path = tmp_path / "chi.dat"
        k_ref, chi_ref = _write_fake_chi(str(path))
        k, chi = read_chi(str(path))
        assert np.allclose(k, k_ref)
        assert np.allclose(chi, chi_ref, rtol=1e-5)

    def test_xmu_dat_columns(self, tmp_path):
        """xmu.dat puts k in column 2 and chi in column 5."""
        path = tmp_path / "xmu.dat"
        k_ref, chi_ref = _write_fake_chi(str(path), style="xmu")
        k, chi = read_chi(str(path))
        assert np.allclose(k, k_ref)
        assert np.allclose(chi, chi_ref, rtol=1e-5)

    def test_comments_and_blank_lines_skipped(self, tmp_path):
        path = tmp_path / "chi.dat"
        path.write_text("# header\n*  star comment\n\n"
                        "  1.0  0.10  0.10  0.0\n"
                        "  2.0 -0.20  0.20  0.0\n")
        k, chi = read_chi(str(path))
        assert list(k) == [1.0, 2.0]
        assert list(chi) == [0.10, -0.20]

    def test_explicit_columns_override(self, tmp_path):
        path = tmp_path / "chi.dat"
        path.write_text("  1.0  9.9  0.30\n  2.0  9.9  0.40\n")
        k, chi = read_chi(str(path), kcol=0, chicol=2)
        assert list(chi) == [0.30, 0.40]

    def test_unknown_filename_needs_columns(self, tmp_path):
        path = tmp_path / "something.dat"
        path.write_text("1.0 2.0\n")
        with pytest.raises(ValueError, match="cannot infer columns"):
            read_chi(str(path))

    def test_no_numeric_data_raises(self, tmp_path):
        path = tmp_path / "chi.dat"
        path.write_text("# only a header\n")
        with pytest.raises(ValueError, match="no numeric data"):
            read_chi(str(path))


# ─── configurational average ──────────────────────────────────────────────

class TestAverageChi:
    def test_mean_and_spread(self):
        k = np.arange(0.0, 20.01, 0.05)
        sources = [(k, np.full_like(k, v)) for v in (1.0, 2.0, 3.0)]
        avg = average_chi(sources, structure_ids=[0, 1, 2])
        assert avg["n_sites"] == 3
        assert np.allclose(avg["chi"], 2.0)
        assert np.allclose(avg["chi_std"], np.std([1.0, 2.0, 3.0]))
        assert np.allclose(avg["chi_sem"],
                           np.std([1.0, 2.0, 3.0], ddof=1) / np.sqrt(3))

    def test_regridded_onto_common_k(self):
        avg = average_chi([(np.array([0.0, 10.0]), np.array([0.0, 10.0]))],
                          kmax=10.0, kstep=0.5)
        assert np.allclose(avg["k"], np.arange(0.0, 10.01, 0.5))
        assert np.allclose(avg["chi"], avg["k"])          # linear ramp

    def test_grid_stops_at_source_range(self):
        """Uncomputed data must never dilute a mean with fabricated zeros."""
        avg = average_chi([(np.arange(0.0, 5.01, 0.05),
                            np.ones(101))], kmax=10.0, kstep=0.05)
        assert np.allclose(avg["chi"][avg["k"] <= 5.0], 1.0)
        assert avg["k"].max() == pytest.approx(5.0)

    def test_accepts_paths(self, tmp_path):
        paths = []
        for n in range(3):
            p = tmp_path / f"j{n}" / "chi.dat"
            _write_fake_chi(str(p), r0=1.60 + 0.02 * n)
            paths.append(str(p))
        avg = average_chi(paths, structure_ids=[0, 1, 2])
        assert avg["n_sites"] == 3
        assert np.any(avg["chi_sem"] > 0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="no chi"):
            average_chi([])


# ─── Fourier transform ────────────────────────────────────────────────────

class TestWindow:
    def test_plateau_and_tails(self):
        k = np.arange(0.0, 20.001, 0.05)
        w = ft_window(k, kmin=3.0, kmax=12.0, dk=1.0)
        assert np.all(w[k < 2.5] == 0.0)
        assert np.allclose(w[(k >= 3.5) & (k <= 11.5)], 1.0)
        assert np.all(w[k > 12.5] == 0.0)

    def test_rise_is_monotonic(self):
        k = np.arange(2.5, 3.5001, 0.01)
        w = ft_window(k, kmin=3.0, kmax=12.0, dk=1.0)
        assert np.all(np.diff(w) >= -1e-12)

    def test_bounded_between_zero_and_one(self):
        k = np.arange(0.0, 20.001, 0.05)
        w = ft_window(k, kmin=3.0, kmax=12.0, dk=1.0)
        assert w.min() >= 0.0 and w.max() <= 1.0


class TestFourierTransform:
    """chi(k) = sin(2 k R0) must transform to a |chi(R)| peak at R0.

    This is the whole convention — k-weighting, window, normalisation and
    the R grid — pinned by one closed-form case.
    """

    @pytest.mark.parametrize("r0", [1.85, 2.50, 3.20])
    def test_peak_sits_at_r0(self, r0):
        k = np.arange(0.0, 20.001, 0.05)
        chi = np.sin(2 * k * r0) / np.maximum(k, 1e-9) ** 2
        ft = chi_fourier_transform(k, chi, kweight=2, kmin=2.0, kmax=18.0)
        peak = ft["r"][np.argmax(ft["chir_mag"])]
        assert peak == pytest.approx(r0, abs=0.05)

    def test_r_grid_spacing(self):
        k = np.arange(0.0, 20.001, 0.05)
        ft = chi_fourier_transform(k, np.zeros_like(k), kstep=0.05, nfft=2048)
        assert ft["r"][1] - ft["r"][0] == pytest.approx(
            np.pi / (0.05 * 2048))
        assert len(ft["r"]) == 1024

    def test_magnitude_matches_components(self):
        k = np.arange(0.0, 20.001, 0.05)
        chi = np.sin(2 * k * 2.0) / np.maximum(k, 1e-9) ** 2
        ft = chi_fourier_transform(k, chi)
        assert np.allclose(ft["chir_mag"],
                           np.hypot(ft["chir_re"], ft["chir_im"]))

    def test_zero_signal_gives_zero_transform(self):
        k = np.arange(0.0, 20.001, 0.05)
        ft = chi_fourier_transform(k, np.zeros_like(k))
        assert np.allclose(ft["chir_mag"], 0.0)


class TestWindowStaysInsideTheData:
    """The taper is centred on kmax and runs dk/2 past it. With kmax at the
    last data point, half the taper fell off the end and the signal was
    cut mid-fall where k^2 weighting is largest. Small on a real a-Ga2O3
    average (~1% anywhere in R), but a window should reach zero inside
    the data whatever the signal."""

    K = np.arange(0.0, 20.001, 0.05)

    def test_default_kmax_is_pulled_in_by_half_a_taper(self):
        ft = chi_fourier_transform(self.K, np.ones_like(self.K), dk=1.0)
        assert ft["kmax"] == pytest.approx(19.5)

    def test_window_has_reached_zero_at_the_last_point(self):
        ft = chi_fourier_transform(self.K, np.ones_like(self.K), dk=1.0)
        assert ft["window"][-1] == pytest.approx(0.0, abs=1e-12)

    def test_kmax_past_the_data_is_clamped(self):
        ft = chi_fourier_transform(self.K, np.ones_like(self.K), kmax=25.0,
                                   dk=2.0)
        assert ft["kmax"] == pytest.approx(19.0)

    def test_kmax_inside_the_data_is_untouched(self):
        ft = chi_fourier_transform(self.K, np.ones_like(self.K), kmax=12.0)
        assert ft["kmax"] == 12.0
        assert (ft["kmin"], ft["dk"]) == (3.0, 1.0)

    def test_empty_window_raises(self):
        k = np.arange(0.0, 3.001, 0.05)
        with pytest.raises(ValueError, match="FT window is empty"):
            chi_fourier_transform(k, np.ones_like(k), kmin=3.0)


# ─── collection, report, CSV, plot ────────────────────────────────────────

class TestCollect:
    @pytest.fixture
    def finished_tree(self, tmp_path):
        d = tmp_path / "jobs"
        manifest = write_feff_jobs([_sio2_cell() for _ in range(2)], "Si",
                                   str(d), radius=4.0)
        rng = np.random.default_rng(0)
        for job in manifest["jobs"]:
            _write_fake_chi(str(d / job["dir"] / "chi.dat"),
                            r0=1.62 + rng.normal(0, 0.02))
        return d, manifest

    def test_collects_every_site(self, finished_tree):
        d, manifest = finished_tree
        res = collect_feff_results(str(d), trust_external=True)
        assert res["n_sites"] == manifest["n_jobs"]
        assert res["missing"] == []
        assert res["absorber"] == "Si"

    def test_peak_near_the_input_shell(self, finished_tree):
        d, _ = finished_tree
        res = collect_feff_results(str(d), trust_external=True)
        assert res["r"][np.argmax(res["chir_mag"])] == pytest.approx(
            1.62, abs=0.1)

    def test_partial_tree_reports_missing(self, finished_tree):
        d, manifest = finished_tree
        os.remove(d / manifest["jobs"][0]["dir"] / "chi.dat")
        res = collect_feff_results(str(d), trust_external=True)
        assert res["missing"] == [manifest["jobs"][0]["dir"]]
        assert res["n_sites"] == manifest["n_jobs"] - 1

    def test_unrun_tree_raises(self, tmp_path):
        d = tmp_path / "jobs"
        write_feff_jobs([_sio2_cell()], "Si", str(d), radius=4.0)
        with pytest.raises(FileNotFoundError, match="none of the"):
            collect_feff_results(str(d), trust_external=True)

    def test_rmax_truncates_r_grid(self, finished_tree):
        d, _ = finished_tree
        assert collect_feff_results(str(d), rmax=4.0, trust_external=True)["r"].max() <= 4.0


class TestReportAndOutputs:
    @pytest.fixture
    def result(self, tmp_path):
        d = tmp_path / "jobs"
        manifest = write_feff_jobs([_sio2_cell()], "Si", str(d), radius=4.0)
        for job in manifest["jobs"]:
            _write_fake_chi(str(d / job["dir"] / "chi.dat"))
        return collect_feff_results(str(d), trust_external=True)

    def test_report_names_absorber_and_counts(self, result):
        out = format_feff_report(result)
        assert "Si K edge" in out
        assert f"{result['n_sites']} of {result['n_jobs']}" in out

    def test_report_flags_phase_shift(self, result):
        """A reader must not mistake the |chi(R)| peak for a bond length."""
        assert "phase-uncorrected" in format_feff_report(result)

    def test_report_states_the_transform_window(self, result):
        """The k-range sets the peak widths in R; the RMS line just above
        quotes 3-12, so without this the figure reads as a 3-12 transform."""
        assert (result["ft_kmin"], result["ft_kmax"]) == (3.0, 19.5)
        assert "FT window      : k = 3.0-19.5 A^-1" in format_feff_report(result)

    def test_report_without_window_fields_still_renders(self, result):
        for key in ("ft_kmin", "ft_kmax", "ft_dk"):
            del result[key]
        assert "FT window" not in format_feff_report(result)

    def test_report_warns_about_missing_jobs(self, result):
        result["missing"] = ["s0000_a0001"]
        out = format_feff_report(result)
        assert "Not yet run" in out
        assert "biases it" in out

    def test_report_warns_when_unconverged(self, result):
        result["chi_sem"] = np.abs(result["chi"]) * 0.5
        assert "not converged" in format_feff_report(result)

    def test_csv_pair(self, result, tmp_path):
        written = write_feff_csv(result, str(tmp_path / "feff.csv"))
        assert [os.path.basename(f) for f in written] == ["feff.csv",
                                                          "feff_chir.csv"]
        k_lines = (tmp_path / "feff.csv").read_text().strip().split("\n")
        r_lines = (tmp_path / "feff_chir.csv").read_text().strip().split("\n")
        # parameter line first, as plot_tr writes it, then the header
        assert k_lines[0].startswith("# absorber=Si edge=K")
        assert "R=phase-uncorrected" in r_lines[0]
        assert k_lines[1] == ("k_invA,chi,k2chi_invA2,chi_site_std,"
                              "chi_structure_sem")
        assert r_lines[1] == "R_A,chir_mag_invA3,chir_re_invA3,chir_im_invA3"
        assert len(k_lines) == 2 + len(result["k"])
        assert len(r_lines) == 2 + len(result["r"])
        # one structure: the SEM is undefined and written as an empty cell
        assert k_lines[2].endswith(",")

    def test_plot_written(self, result, tmp_path):
        import matplotlib
        import matplotlib.pyplot as plt
        from amorphgen.analysis.feff import plot_feff
        backend, figures = matplotlib.get_backend(), plt.get_fignums()
        plot_feff(result, output_dir=str(tmp_path))
        for base in ("analysis_feff_chik", "analysis_feff_chir"):
            assert (tmp_path / f"{base}.png").exists()
            assert (tmp_path / f"{base}.csv").exists()
        # saved without pyplot, as the other analysis plots are
        assert matplotlib.get_backend() == backend
        assert plt.get_fignums() == figures
        # a single structure has no structure-to-structure uncertainty
        assert not (tmp_path / "analysis_feff_chik_uncertainty.csv").exists()


# ─── CLI ──────────────────────────────────────────────────────────────────

class TestCLI:
    @pytest.fixture
    def structure_dir(self, tmp_path):
        from ase.io import write
        d = tmp_path / "structs"
        d.mkdir()
        for n in range(2):
            write(str(d / f"s_{n}.xyz"), _sio2_cell(), format="extxyz")
        return d

    def test_feff_write_creates_tree(self, structure_dir, tmp_path, capsys):
        from unittest.mock import patch
        from amorphgen.cli import main
        jobs = tmp_path / "feffjobs"
        argv = ["amorphgen", "--analyse", "--input-dir", str(structure_dir),
                "--feff-write", str(jobs), "--feff-absorber", "Si",
                "--feff-sites", "1", "--feff-radius", "4.0"]
        with patch("sys.argv", argv):
            main()
        out = capsys.readouterr().out
        assert "FEFF job tree: 2 jobs" in out
        assert "--array=0-1" in out
        assert (jobs / "manifest.json").exists()
        assert (jobs / "joblist.txt").exists()

    def test_feff_write_without_absorber_explains(self, structure_dir,
                                                  tmp_path, capsys):
        from unittest.mock import patch
        from amorphgen.cli import main
        argv = ["amorphgen", "--analyse", "--input-dir", str(structure_dir),
                "--feff-write", str(tmp_path / "j")]
        with patch("sys.argv", argv), pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1
        assert "--feff-absorber" in capsys.readouterr().out

    def test_feff_collect_reports_and_saves(self, structure_dir, tmp_path,
                                            capsys):
        from unittest.mock import patch
        from amorphgen.cli import main
        jobs = tmp_path / "jobs"
        manifest = write_feff_jobs([_sio2_cell()], "Si", str(jobs),
                                   radius=4.0)
        for job in manifest["jobs"]:
            _write_fake_chi(str(jobs / job["dir"] / "chi.dat"))

        plots = tmp_path / "plots"
        argv = ["amorphgen", "--analyse", "--input-dir", str(structure_dir),
                "--feff-collect", str(jobs), "--feff-trust-external", "--save-plot", str(plots)]
        with patch("sys.argv", argv):
            main()
        out = capsys.readouterr().out
        assert "Configurationally averaged EXAFS" in out
        for base in ("analysis_Si_feff_chik", "analysis_Si_feff_chir"):
            assert (plots / f"{base}.png").exists()
            assert (plots / f"{base}.csv").exists()

    def test_feff_off_by_default(self, structure_dir, capsys):
        from unittest.mock import patch
        from amorphgen.cli import main
        argv = ["amorphgen", "--analyse", "--input-dir", str(structure_dir)]
        with patch("sys.argv", argv):
            main()
        assert "FEFF job tree" not in capsys.readouterr().out


class TestClusterSizeGuard:
    """FEFF rejects clusters over its compiled-in limit. Catching that at
    write time beats discovering it after a job array has been queued."""

    def test_oversized_cluster_warns(self):
        from amorphgen.analysis.feff import MAX_CLUSTER_ATOMS
        dense = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
        with pytest.warns(UserWarning, match="over FEFF's usual limit"):
            text = write_feff_input(dense, 0, radius=12.0)   # ~627 atoms
        assert len(_atoms_rows(text)) > MAX_CLUSTER_ATOMS

    def test_normal_cluster_does_not_warn(self, recwarn):
        write_feff_input(_sio2_cell(), 0, radius=5.0)
        assert not [w for w in recwarn
                    if "over FEFF's usual limit" in str(w.message)]


class TestPathRadiusCoupling:
    def test_rpath_tracks_cluster_radius(self):
        """Enumerating paths beyond the cluster finds nothing; an RPATH well
        under the radius throws away shells that were already paid for."""
        assert "RPATH     4.5" in write_feff_input(_sio2_cell(), 0,
                                                   radius=4.5)
        assert "RPATH     7.0" in write_feff_input(_sio2_cell(), 0,
                                                   radius=7.0)

    def test_explicit_rpath_still_wins(self):
        text = write_feff_input(_sio2_cell(), 0, radius=6.0,
                                cards={"RPATH": "3.0"})
        assert "RPATH     3.0" in text


class TestS02Comment:
    """The header comment must describe the S02 actually written.

    Found by aligning a generated input with FEFF's own Copper reference,
    which uses S02 = 0.950: the boilerplate claimed "S02 is 1.0 by design"
    regardless of what the caller passed.
    """

    def test_default_claims_unreduced(self):
        text = write_feff_input(_sio2_cell(), 0, radius=4.0)
        assert "S02       1.0" in text
        assert "S02 is 1.0 by design" in text

    def test_override_is_described_honestly(self):
        text = write_feff_input(_sio2_cell(), 0, radius=4.0,
                                cards={"S02": "0.950"})
        assert "S02       0.950" in text
        assert "S02 is 1.0 by design" not in text
        assert "overriding" in text


class TestFailureDiagnostics:
    """run_feff_jobs must report what FEFF said, not just that it failed.

    Every string below was taken from a real feff.log produced by
    FEFF8.5L on a structure with a too-close pair.
    """

    def _log(self, tmp_path, body):
        d = tmp_path / "job"
        d.mkdir()
        (d / "feff.log").write_text(body)
        return str(d)

    def test_close_atoms_message_is_surfaced(self, tmp_path):
        from amorphgen.analysis.feff import _feff_failure_reason
        job = self._log(tmp_path, "some preamble\n"
                        " WARNING:  TWO ATOMS VERY CLOSE TOGETHER.  "
                        "CHECK INPUT.\nmore noise\n")
        reason = _feff_failure_reason(job, "no chi.dat written")
        assert "TWO ATOMS VERY CLOSE TOGETHER" in reason

    def test_close_atoms_points_at_check_dimers(self, tmp_path):
        """The hint must name the AmorphGen command that finds the cause."""
        from amorphgen.analysis.feff import _feff_failure_reason
        job = self._log(tmp_path,
                        " WARNING:  TWO ATOMS VERY CLOSE TOGETHER.\n")
        assert "--check-dimers" in _feff_failure_reason(job, "fallback")

    def test_muffin_tin_overlap_message(self, tmp_path):
        from amorphgen.analysis.feff import _feff_failure_reason
        job = self._log(tmp_path, " Rmt >= distance to nearest neighbor.  "
                                  "Not physically, meaningful.\n")
        reason = _feff_failure_reason(job, "fallback")
        assert "Rmt >=" in reason and "--check-dimers" in reason

    def test_most_informative_pattern_wins(self, tmp_path):
        """A log usually holds several; the earliest, clearest one is the
        one worth showing."""
        from amorphgen.analysis.feff import _feff_failure_reason
        job = self._log(tmp_path,
                        " WARNING:  TWO ATOMS VERY CLOSE TOGETHER.\n"
                        " Fortran runtime error: Cannot open file 'pot.pad'\n"
                        " Error termination.\n")
        assert "TWO ATOMS" in _feff_failure_reason(job, "fallback")

    def test_unrecognised_log_falls_back(self, tmp_path):
        from amorphgen.analysis.feff import _feff_failure_reason
        job = self._log(tmp_path, "everything went fine, somehow\n")
        assert _feff_failure_reason(job, "exit code 7") == "exit code 7"

    def test_missing_log_falls_back(self, tmp_path):
        from amorphgen.analysis.feff import _feff_failure_reason
        d = tmp_path / "empty"
        d.mkdir()
        assert _feff_failure_reason(str(d), "exit code 1") == "exit code 1"
