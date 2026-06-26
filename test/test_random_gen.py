"""
tests/test_random_gen.py
------------------------
Tier 1 + 2 tests for amorphgen.pipeline.random_gen
All tests are CPU-only and require no MACE model.
"""

import os
import pytest
import numpy as np
from itertools import combinations

from amorphgen.pipeline.random_gen import (
    generate_random, batch_random,
    _default_minsep, _get_minsep, _estimate_cell_length,
    _classify_bond, _estimate_density, _get_ionic_radius,
    _get_metallic_radius, _auto_dmax,
    SHANNON_IONIC_RADII, METALLIC_RADII, ELEMENTAL_DENSITIES,
    NONMETALS, METALLOIDS,
)


def check_min_distances(atoms, minsep, tolerance=0.01):
    """Verify all pairwise minimum-image distances satisfy constraints."""
    symbols = atoms.get_chemical_symbols()
    L = atoms.cell[0, 0]
    violations = 0
    for i, j in combinations(range(len(atoms)), 2):
        d = atoms.positions[i] - atoms.positions[j]
        d -= L * np.round(d / L)
        dist = np.linalg.norm(d)
        req = _get_minsep(symbols[i], symbols[j], minsep)
        if dist < req - tolerance:
            violations += 1
    return violations


class TestDefaultMinsep:

    def test_returns_dict(self):
        result = _default_minsep(["In", "O"])
        assert isinstance(result, dict)
        assert len(result) == 3  # In-In, In-O, O-O

    def test_values_positive(self):
        result = _default_minsep(["Si", "O"])
        assert all(v > 0 for v in result.values())

    def test_scale_factor(self):
        # For elements without known radii, scale acts as fallback
        # For known elements, bonding-type scale factors are used
        r1 = _default_minsep(["Cu"])
        key = list(r1.keys())[0]
        # Cu-Cu should use metallic radii with SCALE_FACTORS["metallic"] = 0.85
        assert r1[key] > 0
        # Verify the fallback scale works for unknown elements
        # by checking that known elements produce consistent results
        r2 = _default_minsep(["Cu"], scale=0.5)
        # Both should give the same result (scale is only a fallback)
        assert abs(r2[key] - r1[key]) < 0.01


class TestEstimateCellLength:

    def test_returns_positive(self):
        L = _estimate_cell_length({"In": 32, "O": 48})
        assert L > 0

    def test_density_gives_smaller_cell(self):
        L_default = _estimate_cell_length({"In": 32, "O": 48})
        L_dense = _estimate_cell_length({"In": 32, "O": 48}, target_density=10.0)
        assert L_dense < L_default

    def test_more_atoms_bigger_cell(self):
        L_small = _estimate_cell_length({"O": 10})
        L_large = _estimate_cell_length({"O": 100})
        assert L_large > L_small

    def test_density_scale_shrinks_cell(self):
        """density_scale > 1 should yield a smaller (denser) cell."""
        L_default = _estimate_cell_length({"Si": 64})
        L_scaled  = _estimate_cell_length({"Si": 64}, density_scale=1.2)
        assert L_scaled < L_default
        # density ~ 1/V = 1/L^3, so L_scaled^3 / L_default^3 ~ 1 / 1.2
        ratio = (L_scaled / L_default) ** 3
        assert abs(ratio - (1 / 1.2)) < 0.05

    def test_density_scale_ignored_when_target_density_set(self):
        """Explicit target_density wins over density_scale."""
        L1 = _estimate_cell_length({"Si": 64}, target_density=2.30)
        L2 = _estimate_cell_length({"Si": 64}, target_density=2.30,
                                   density_scale=2.0)
        assert abs(L1 - L2) < 1e-9

    def test_density_scale_invalid_raises(self):
        import pytest
        with pytest.raises(ValueError):
            _estimate_cell_length({"Si": 64}, density_scale=0)
        with pytest.raises(ValueError):
            _estimate_cell_length({"Si": 64}, density_scale=-1)


class TestGenerateRandom:

    def test_basic_generation(self):
        atoms = generate_random(
            composition={"Cu": 10},
            cell_length_ang=8.0,
            seed=42,
        )
        assert len(atoms) == 10
        assert atoms.get_chemical_formula() == "Cu10"

    def test_binary_composition(self):
        atoms = generate_random(
            composition={"In": 4, "O": 6},
            cell_length_ang=8.0,
            seed=42,
        )
        assert len(atoms) == 10
        symbols = atoms.get_chemical_symbols()
        assert symbols.count("In") == 4
        assert symbols.count("O") == 6

    def test_pbc_set(self):
        atoms = generate_random({"Si": 5}, cell_length_ang=7.0, seed=1)
        assert all(atoms.pbc)

    def test_reproducible_with_seed(self):
        a1 = generate_random({"O": 8}, cell_length_ang=6.0, seed=123)
        a2 = generate_random({"O": 8}, cell_length_ang=6.0, seed=123)
        assert np.allclose(a1.positions, a2.positions)

    def test_different_seeds_differ(self):
        a1 = generate_random({"O": 8}, cell_length_ang=6.0, seed=1)
        a2 = generate_random({"O": 8}, cell_length_ang=6.0, seed=2)
        assert not np.allclose(a1.positions, a2.positions)

    def test_minsep_respected(self):
        minsep = {"Cu-Cu": 2.0}
        atoms = generate_random(
            composition={"Cu": 15},
            cell_length_ang=10.0,
            minsep=minsep,
            seed=42,
        )
        violations = check_min_distances(atoms, minsep)
        assert violations == 0

    def test_custom_minsep_binary(self):
        minsep = {"In-In": 2.8, "In-O": 1.9, "O-O": 2.5}
        atoms = generate_random(
            composition={"In": 8, "O": 12},
            cell_length_ang=10.0,
            minsep=minsep,
            seed=42,
        )
        violations = check_min_distances(atoms, minsep)
        assert violations == 0

    def test_too_dense_raises(self):
        """Packing too many atoms in a small box should fail."""
        with pytest.raises(RuntimeError, match="Could not place"):
            generate_random(
                composition={"Cu": 100},
                cell_length_ang=3.0,
                seed=42,
                max_attempts_per_atom=100,
            )

    def test_cell_length_auto_from_density(self):
        atoms = generate_random(
            composition={"In": 32, "O": 48},
            target_density=4.5,
            seed=42,
        )
        assert len(atoms) == 80

    def test_auto_expand_dense_covalent_network(self):
        """BeO (covalent_network_oxide) is too dense to place at its full
        equilibrium density with physical minseps; generate_random should
        auto-expand the cell and succeed for every seed, WITHOUT shrinking the
        (physical) O-O minimum separation."""
        from amorphgen.utils.radii import default_minsep, estimate_cell_length
        # O-O minsep must stay physical (Shannon ionic ~2.24 A), not covalent
        oo = default_minsep(["Be"] * 16 + ["O"] * 16)["O-O"]
        assert oo > 2.0, f"O-O minsep should stay physical, got {oo:.2f}"
        base_L = estimate_cell_length({"Be": 16, "O": 16})
        for seed in range(2):
            atoms = generate_random({"Be": 16, "O": 16}, seed=seed)
            assert len(atoms) == 32
            # cell was expanded beyond the (too-dense) base density estimate
            assert atoms.cell[0, 0] >= base_L - 1e-6


class TestBatchRandom:

    def test_generates_correct_count(self, tmp_work_dir):
        paths = batch_random(
            composition={"Cu": 5},
            n_structures=3,
            output_dir=str(tmp_work_dir / "batch"),
            cell_length_ang=6.0,
        )
        assert len(paths) == 3
        for p in paths:
            assert os.path.isfile(p)

    def test_output_files_readable(self, tmp_work_dir):
        from ase.io import read
        paths = batch_random(
            composition={"Cu": 5},
            n_structures=2,
            output_dir=str(tmp_work_dir / "batch2"),
            cell_length_ang=6.0,
        )
        for p in paths:
            atoms = read(p)
            assert len(atoms) == 5

    def test_output_format_vasp(self, tmp_work_dir):
        paths = batch_random(
            composition={"Cu": 5},
            n_structures=1,
            output_dir=str(tmp_work_dir / "vasp_out"),
            output_format="vasp",
            cell_length_ang=6.0,
        )
        assert paths[0].endswith(".vasp")
        assert os.path.isfile(paths[0])

    def test_output_format_cif(self, tmp_work_dir):
        paths = batch_random(
            composition={"Cu": 5},
            n_structures=1,
            output_dir=str(tmp_work_dir / "cif_out"),
            output_format="cif",
            cell_length_ang=6.0,
        )
        assert paths[0].endswith(".cif")

    def test_invalid_format_raises(self, tmp_work_dir):
        with pytest.raises(ValueError, match="Unknown output format"):
            batch_random(
                composition={"Cu": 5},
                n_structures=1,
                output_dir=str(tmp_work_dir / "bad"),
                output_format="invalid",
                cell_length_ang=6.0,
            )

    def test_log_file_created(self, tmp_work_dir):
        out = str(tmp_work_dir / "log_test")
        batch_random(composition={"Cu": 5}, n_structures=1,
                     output_dir=out, cell_length_ang=6.0)
        assert os.path.isfile(os.path.join(out, "random_gen.log"))


class TestClassifyBond:

    def test_metal_nonmetal_ionic(self):
        assert _classify_bond("Ti", "O") == "ionic"
        assert _classify_bond("Na", "Cl") == "ionic"
        assert _classify_bond("Li", "F") == "ionic"

    def test_metalloid_nonmetal_ionic(self):
        assert _classify_bond("Si", "O") == "ionic"
        assert _classify_bond("Ge", "O") == "ionic"

    def test_metalloid_metalloid_covalent(self):
        assert _classify_bond("Si", "Si") == "covalent"
        assert _classify_bond("Si", "Ge") == "covalent"

    def test_metal_metal_metallic(self):
        assert _classify_bond("Cu", "Cu") == "metallic"
        assert _classify_bond("Fe", "Ni") == "metallic"

    def test_nonmetal_nonmetal_covalent(self):
        assert _classify_bond("O", "O") == "covalent"
        assert _classify_bond("Cl", "Cl") == "covalent"
        assert _classify_bond("O", "N") == "covalent"

    def test_metalloid_metal_covalent_when_low_dchi(self):
        # Si-Ti has small Pauling Δχ (~0.36) → covalent (silicide chemistry,
        # not ionic).  Pre-2026-05-10 this returned "ionic" — that was a bug
        # because it gave unphysical Shannon-cation-only minsep distances
        # for III-V semiconductors and related covalent-metal pairs.
        assert _classify_bond("Si", "Ti") == "covalent"

    def test_metal_metalloid_high_dchi_still_ionic(self):
        # Negative control: large Δχ should still be ionic.
        assert _classify_bond("Li", "Cl") == "ionic"
        assert _classify_bond("Na", "F") == "ionic"

    def test_iiiv_pnictides_are_covalent(self):
        # Bug fix: III-V semiconductors (GaAs, AlAs, InAs) were classified
        # as ionic with Shannon cation+anion-cation radii summing to <1 Å,
        # producing atomic crashes during placement.  Should be covalent.
        assert _classify_bond("Ga", "As") == "covalent"
        assert _classify_bond("Al", "As") == "covalent"
        assert _classify_bond("In", "As") == "covalent"
        assert _classify_bond("In", "P")  == "covalent"


class TestEstimateDensity:

    def test_pure_element(self):
        d = _estimate_density({"Si": 40})
        assert d is not None
        assert 1.5 < d < 2.5  # ~1.86 g/cm3

    def test_pure_metal(self):
        d = _estimate_density({"Cu": 32})
        assert d is not None
        assert 6.0 < d < 9.0

    def test_compound_with_gas_returns_none(self):
        d = _estimate_density({"Si": 16, "O": 32})
        assert d is None  # O has no solid density

    def test_all_elements_in_table(self):
        for sym in ELEMENTAL_DENSITIES:
            d = _estimate_density({sym: 10})
            assert d is not None and d > 0


class TestRadiiTables:

    def test_shannon_has_common_elements(self):
        for sym in ["Li", "Na", "Ca", "Ti", "Zr", "O", "Cl", "Si"]:
            assert sym in SHANNON_IONIC_RADII

    def test_metallic_has_common_elements(self):
        for sym in ["Li", "Cu", "Fe", "Ti", "Al", "Si"]:
            assert sym in METALLIC_RADII

    def test_ionic_radius_returns_float(self):
        r = _get_ionic_radius("O")
        assert r is not None
        assert isinstance(r, float)
        assert r > 0

    def test_ionic_radius_unknown_returns_none(self):
        assert _get_ionic_radius("Xx") is None

    def test_metallic_radius_returns_float(self):
        r = _get_metallic_radius("Cu")
        assert r is not None
        assert isinstance(r, float)

    def test_metallic_radius_unknown_returns_none(self):
        assert _get_metallic_radius("O") is None


class TestSCPlacement:

    def test_sc_basic(self):
        """SC placement should produce a valid structure."""
        atoms = generate_random(
            composition={"Si": 8, "O": 16},
            cell_length_ang=8.0,
            seed=42,
            target_cn={"Si": 4, "O": 2},
        )
        assert len(atoms) == 24

    def test_sc_report_in_info(self):
        """SC should store coordination report in atoms.info."""
        atoms = generate_random(
            composition={"Si": 8, "O": 16},
            cell_length_ang=8.0,
            seed=42,
            target_cn={"Si": 4, "O": 2},
        )
        assert "sc_report" in atoms.info
        assert "Si" in atoms.info["sc_report"]
        assert atoms.info["sc_report"]["Si"]["target"] == 4

    def test_sc_caps_coordination_at_target(self):
        """SC placement never over-coordinates beyond the target — its core
        constraint — which unconstrained random placement does not guarantee.

        Checked via the SC report (``atoms.info["sc_report"]``), which uses the
        placement's own bonding shell, so the result is independent of any
        external analysis cutoff and of platform / RNG differences.

        (The previous form compared the *mean* Si CN of two single random
        placements with a tight tolerance. That is not a reliable signal: SC
        *regulates* coordination toward the target rather than maximising it,
        so its mean CN can sit at or below an unconstrained placement and the
        comparison flips with the RNG / numpy version — which made the test
        flaky across CI runners.)
        """
        for seed in (0, 7, 42):
            atoms = generate_random(
                composition={"Si": 8, "O": 16},
                cell_length_ang=8.0, seed=seed,
                target_cn={"Si": 4, "O": 2})
            report = atoms.info["sc_report"]
            assert report["Si"]["max"] <= 4, (seed, report["Si"])
            assert report["O"]["max"] <= 2, (seed, report["O"])

    def test_sc_unconstrained_element_reported(self):
        """Unconstrained elements should appear with target='auto'."""
        atoms = generate_random(
            composition={"Li": 4, "Zr": 2, "Cl": 12},
            cell_length_ang=10.0, seed=42,
            target_cn={"Zr": 6, "Li": 6})
        report = atoms.info.get("sc_report", {})
        assert "Cl" in report
        assert report["Cl"]["target"] == "auto"

    def test_auto_dmax(self):
        minsep = {"Si-O": 1.6, "O-O": 2.2, "Si-Si": 2.0}
        target_cn = {"Si": 4, "O": 2}
        dmax = _auto_dmax(minsep, target_cn)
        # Should include Si-O but not O-O or Si-Si
        assert "Si-O" in dmax or "O-Si" in dmax
        assert "O-O" not in dmax
        assert "Si-Si" not in dmax

    def test_auto_dmax_factor(self):
        minsep = {"Si-O": 1.6}
        target_cn = {"Si": 4}
        dmax = _auto_dmax(minsep, target_cn, factor=1.5)
        key = list(dmax.keys())[0]
        assert abs(dmax[key] - 1.6 * 1.5) < 0.01

    def test_auto_dmax_suppresses_cation_cation_covalent(self):
        """A covalent pair between two cations is NOT a bond when an anion is
        present (Si-Al in SiAlON): the cations coordinate the anions, not each
        other. Mirrors the metallic M-M suppression."""
        minsep = {"Si-O": 1.6, "Al-O": 1.6, "Si-N": 1.4, "Al-N": 1.6,
                  "Al-Si": 2.1, "O-O": 2.2, "N-N": 2.3}
        target_cn = {"Si": 4, "Al": 5}
        dmax = _auto_dmax(minsep, target_cn)
        assert "Al-Si" not in dmax and "Si-Al" not in dmax  # suppressed
        assert "Si-O" in dmax or "O-Si" in dmax             # ionic bond kept
        assert "Al-N" in dmax or "N-Al" in dmax

    def test_auto_dmax_keeps_covalent_when_no_anion(self):
        """Si-Ge in a-SiGe (no anion) is the primary covalent bond -> kept."""
        minsep = {"Si-Ge": 2.3, "Si-Si": 2.3, "Ge-Ge": 2.4}
        target_cn = {"Si": 4, "Ge": 4}
        dmax = _auto_dmax(minsep, target_cn)
        assert "Si-Ge" in dmax or "Ge-Si" in dmax


class TestMinsepBondTypes:

    def test_oxide_minsep_values(self):
        """Check that oxide minsep uses Shannon ionic radii."""
        m = _default_minsep(["Si", "O"])
        # Si-O: (Si4+=0.400 + O2-=1.400) * 0.80 = 1.44
        assert abs(m["O-Si"] - 1.44) < 0.05

    def test_chloride_large_anion_scale(self):
        """Cl-Cl should use large anion scale (0.70)."""
        m = _default_minsep(["Li", "Cl"])
        # Cl-Cl: (1.81 + 1.81) * 0.70 = 2.534
        assert abs(m["Cl-Cl"] - 2.534) < 0.05

    def test_oxide_small_anion_scale(self):
        """O-O should use small anion scale (0.80)."""
        m = _default_minsep(["Ti", "O"])
        # O-O: (1.40 + 1.40) * 0.80 = 2.24
        assert abs(m["O-O"] - 2.24) < 0.05

    def test_metallic_in_ionic_context(self):
        """M-M in oxide should use max(metallic, geometric)."""
        m = _default_minsep(["In", "O"])
        assert m["In-In"] > 2.0

    def test_ionic_radii_for_bonding_pairs(self):
        """M-O minsep should use Shannon ionic radii with scale 0.80.
        Default cation oxidation state is the highest positive one
        listed in the Shannon table (Ti(IV) for Ti, V(V) for V, ...).
        """
        m = _default_minsep(["Ti", "O"])
        # Ti-O: (Ti4+=0.605 + O2-=1.400) * 0.80 = 1.604
        assert abs(m["O-Ti"] - 1.604) < 0.05

    def test_unknown_pair_falls_back(self):
        """Unknown pairs should fall back to covalent radii."""
        m = _default_minsep(["Rb", "O"])
        assert "O-Rb" in m
        assert m["O-Rb"] > 1.5
