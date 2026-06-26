"""Advanced tests for random_gen.py — fills gaps left by test_random_gen.py.

Targets:
  - The experimental ``--repair-iters`` post-placement repair loop.
  - Oxidation-state inference paths (TiO2, Fe2O3, SnO2, ZrN).
  - Auto-target-CN detection across material classes.
  - Density estimation across material classes.
  - Single-element auto-density paths (covalent_oxide vs group_iv).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from amorphgen import generate_random
from amorphgen.utils.radii import (
    auto_target_cn,
    default_minsep,
    estimate_cell_length,
    _classify_compound,
    infer_oxidation_state,
    get_ionic_radius,
)


# ─── Oxidation-state inference ────────────────────────────────────────────

class TestInferOxidationState:
    def test_tio2_returns_4(self):
        # 2 O atoms × −2 = −4 → Ti must be +4
        assert infer_oxidation_state("Ti", {"Ti": 1, "O": 2}) == 4

    def test_fe2o3_returns_3(self):
        # 3 O × −2 = −6 → 2 Fe × +3 = +6
        assert infer_oxidation_state("Fe", {"Fe": 2, "O": 3}) == 3

    def test_sno2_returns_4(self):
        assert infer_oxidation_state("Sn", {"Sn": 1, "O": 2}) == 4

    def test_in2o3_returns_3(self):
        assert infer_oxidation_state("In", {"In": 2, "O": 3}) == 3

    def test_unknown_anion_returns_none(self):
        # Pure Si — no anion, no inference possible
        assert infer_oxidation_state("Si", {"Si": 16}) is None


# ─── get_ionic_radius with explicit OS ────────────────────────────────────

class TestGetIonicRadius:
    def test_default_picks_highest_positive(self):
        # Ti has multiple OS in Shannon (3+, 4+); default should pick 4+
        r4 = get_ionic_radius("Ti")
        r4_explicit = get_ionic_radius("Ti", oxidation_state=4)
        assert r4 == r4_explicit

    def test_explicit_os_used(self):
        # Fe with explicit 2+ vs 3+ should differ
        r2 = get_ionic_radius("Fe", oxidation_state=2)
        r3 = get_ionic_radius("Fe", oxidation_state=3)
        assert r2 is not None and r3 is not None
        # Higher oxidation state → smaller ionic radius (general trend)
        assert r3 < r2

    def test_unavailable_os_falls_back(self):
        # Ask for an OS that isn't tabulated — should fall back to nearest available.
        # Zr in Shannon has 4+ but maybe not 3+; falling back should return a value.
        r = get_ionic_radius("Zr", oxidation_state=3)
        assert r is not None and r > 0


# ─── _classify_compound coverage ──────────────────────────────────────────

class TestClassifyCompound:
    @pytest.mark.parametrize("comp,expected_class", [
        ({"Si": 16}, "group_iv"),
        ({"Ge": 16}, "group_iv"),
        ({"Cu": 32}, "alloy"),
        ({"In": 16, "O": 24}, "metal_oxide"),
        ({"Si": 16, "O": 32}, "covalent_oxide"),
        ({"Ga": 16, "As": 16}, "pnictide"),
        ({"Si": 16, "C": 16}, "covalent_carbide"),
        ({"Zn": 16, "S": 16}, "chalcogenide"),
        ({"Al": 16, "N": 16}, "small_cation_nitride"),
        ({"Li": 16, "Cl": 16}, "halide"),
    ])
    def test_classification(self, comp, expected_class):
        assert _classify_compound(comp) == expected_class


# ─── Density estimation across classes ────────────────────────────────────

class TestEstimateCellLengthClasses:
    """Confirm cell length is sensible (positive, scales correctly) across
    every material class."""

    @pytest.mark.parametrize("comp", [
        {"Si": 64},
        {"Ge": 64},
        {"Cu": 32},
        {"In": 32, "O": 48},
        {"Si": 32, "O": 64},
        {"Ga": 16, "As": 16},
        {"Zn": 16, "S": 16},
        {"Li": 16, "Cl": 16},
    ])
    def test_positive_and_finite(self, comp):
        L = estimate_cell_length(comp)
        assert L > 0
        assert np.isfinite(L)
        # Not absurd: should be between 4 and 30 Å for these system sizes.
        assert 4.0 < L < 30.0


# ─── auto_target_cn coverage ──────────────────────────────────────────────

class TestAutoTargetCn:
    def test_silicon_is_cn4(self):
        cn, tol = auto_target_cn({"Si": 16})
        assert cn["Si"] == 4

    def test_in2o3_oxide_cn(self):
        cn, tol = auto_target_cn({"In": 32, "O": 48})
        # Oxygen target is typically anion-coordination (~3-4)
        assert "In" in cn or "O" in cn

    def test_pure_metal_alloy_cn(self):
        cn, tol = auto_target_cn({"Cu": 32, "Zn": 32})
        # Alloys default to CN=8 with broader tolerance
        assert any(v >= 6 for v in cn.values())
        assert tol >= 1

    def test_iivi_chalcogenides_are_tetrahedral(self):
        # Bug fix: Group-12 cations (Zn, Cd, Hg) in chalcogenides form
        # zincblende/wurtzite (CN=4), not rocksalt.  Pre-2026-05-10 they
        # defaulted to CN=6 because the chalcogenide branch lumped all
        # cations together.
        for comp in [{"Zn": 16, "S": 16}, {"Zn": 16, "Te": 16},
                     {"Cd": 16, "S": 16}, {"Cd": 16, "Te": 16},
                     {"Hg": 16, "Te": 16}]:
            cn, _ = auto_target_cn(comp)
            cation = next(iter({"Zn", "Cd", "Hg"} & set(comp)))
            assert cn[cation] == 4, f"Expected CN=4 for {cation} in {comp}, got {cn[cation]}"

    def test_transition_metal_chalcogenide_still_octahedral(self):
        # Negative control: Mn / Fe sulfides are rocksalt (CN=6) and
        # should remain so.
        for comp in [{"Mn": 16, "S": 16}, {"Fe": 16, "S": 16}]:
            cn, _ = auto_target_cn(comp)
            cation = next(iter({"Mn", "Fe"} & set(comp)))
            assert cn[cation] == 6, f"Expected CN=6 for {cation} in {comp}, got {cn[cation]}"


# ─── Single-element auto-density paths ────────────────────────────────────

class TestSingleElementDensities:
    """Single-element compositions exercise different code paths than
    binary compounds — covered separately."""

    def test_silicon_64_atoms_reasonable_cell(self):
        atoms = generate_random(composition={"Si": 64}, seed=0)
        L = atoms.cell.lengths()[0]
        # a-Si density ~2.0-2.5 g/cm³ → for 64 Si atoms, L ≈ 10.5-11.5 Å
        assert 9.0 < L < 13.0

    def test_germanium_64_atoms(self):
        atoms = generate_random(composition={"Ge": 64}, seed=0)
        L = atoms.cell.lengths()[0]
        # Ge is denser than Si, so cell should be similar or smaller
        assert 9.0 < L < 13.0

    def test_copper_alloy_path(self):
        atoms = generate_random(composition={"Cu": 32}, cell_length_ang=10.0, seed=0)
        assert len(atoms) == 32

    @pytest.mark.parametrize("comp", [
        {"Cu": 32},
        {"Au": 32},
        {"Ni": 32},
        {"Ti": 32},
        {"Cu": 16, "Zn": 16},   # brass
        {"Ni": 16, "Ti": 16},   # NiTi shape-memory alloy
    ])
    def test_metal_auto_only_places(self, comp):
        # Regression: alloy packing factor was 0.70 (too tight — RCP cap is
        # ~0.64), which made auto-only placement (no cell_length_ang) fail
        # for every pure metal and alloy.  Lowered to 0.60 on 2026-05-09.
        n_total = sum(comp.values())
        atoms = generate_random(composition=comp, seed=0)
        assert len(atoms) == n_total
        L = atoms.cell.lengths()[0]
        assert 5.0 < L < 15.0


# ─── Repair-iters loop (the experimental --repair-iters feature) ─────────

class TestRepairIters:
    def test_zero_is_noop(self):
        """repair_iters=0 should give identical result to no repair."""
        a1 = generate_random(composition={"Si": 32}, cell_length_ang=10.0,
                              target_cn={"Si": 4}, seed=42, repair_iters=0)
        a2 = generate_random(composition={"Si": 32}, cell_length_ang=10.0,
                              target_cn={"Si": 4}, seed=42)
        # Same seed + repair_iters=0 should give bit-identical positions.
        np.testing.assert_array_almost_equal(
            a1.get_positions(), a2.get_positions()
        )

    def test_positive_iters_runs(self):
        """repair_iters>0 should run the repair loop without error."""
        atoms = generate_random(
            composition={"Si": 32}, cell_length_ang=10.0,
            target_cn={"Si": 4}, seed=0, repair_iters=50,
        )
        assert len(atoms) == 32

    def test_repair_records_in_sc_report(self):
        """When repair runs, the sc_report should record the post-repair CN."""
        atoms = generate_random(
            composition={"Si": 32}, cell_length_ang=10.0,
            target_cn={"Si": 4}, seed=0, repair_iters=100,
        )
        sc = atoms.info.get("sc_report")
        # If SC was active, sc_report should exist and have Si entry.
        if sc and "Si" in sc:
            assert "mean" in sc["Si"]
            assert sc["Si"]["target"] == 4


# ─── Composition-based path coverage ──────────────────────────────────────

class TestCompositionPaths:
    """End-to-end placement for compositions that exercise oxidation-state
    inference and various density paths."""

    @pytest.mark.parametrize("comp", [
        {"Ti": 16, "O": 32},   # Ti⁴⁺ via charge balance
        {"Fe": 16, "O": 24},   # Fe³⁺ via charge balance
        {"Sn": 16, "O": 32},   # Sn⁴⁺ via charge balance
        {"Zr": 16, "N": 16},   # nitride path
    ])
    def test_compound_places_all_atoms(self, comp):
        n_total = sum(comp.values())
        atoms = generate_random(composition=comp, seed=0)
        assert len(atoms) == n_total
        # Symbols match composition counts.
        from collections import Counter
        observed = Counter(atoms.get_chemical_symbols())
        assert observed == comp


# ─── Auto-derive log-line summary (single-line transparency record) ──────────

class TestFormatAutoDeriveSummary:
    """Unit tests for the one-line auto-derivation summary that appears at
    the top of random_gen.log.  The format is grep-friendly and exposes the
    chemistry-informed decisions AmorphGen makes from a bare composition.
    """

    def _build(self, comp, target_cn=None):
        from amorphgen.utils.radii import (
            format_auto_derive_summary,
            default_minsep,
            estimate_cell_length,
            auto_target_cn,
        )
        from ase.data import atomic_masses, atomic_numbers
        if target_cn is None:
            target_cn, _ = auto_target_cn(comp)
        syms = [s for s in comp for _ in range(comp[s])]
        ms = default_minsep(syms, target_cn=target_cn or {})
        L = estimate_cell_length(comp)
        mass = sum(atomic_masses[atomic_numbers[s]] * n for s, n in comp.items())
        rho = (mass / 6.022e23) / ((L ** 3) * 1e-24)
        return format_auto_derive_summary(comp, target_cn, ms, rho, L)

    def test_ga2o3_line_has_all_fields(self):
        line = self._build({"Ga": 16, "O": 24})
        # Header tag for grep-ability
        assert line.startswith("[auto-derive]")
        # Composition and material class
        assert "Ga16O24" in line
        assert "metal_oxide" in line
        # Oxidation state inferred
        assert "Ga:+3" in line
        # Pauling Δχ shown for the ionic Ga-O pair
        assert "Ga-O" in line
        assert "ionic" in line
        assert "Δχ=" in line
        # Density and cell length present
        assert "ρ=" in line
        assert "g/cm³" in line
        assert " L=" in line and " Å" in line

    def test_iiiv_pair_classified_covalent_not_ionic(self):
        """III-V pnictides should appear with the covalent label, not ionic
        — they go through the Pauling Δχ < 1.0 refinement branch.
        Note: pair keys are sorted alphabetically, so Ga-As → 'As-Ga'."""
        line = self._build({"Ga": 16, "As": 16})
        assert "As-Ga" in line
        # The As-Ga block specifically must say covalent, not ionic
        as_ga_block = line.split("As-Ga:")[1].split("|")[0]
        assert "covalent" in as_ga_block
        assert "ionic" not in as_ga_block

    def test_o_o_pair_labelled_anion_pack(self):
        """O-O same-element pair should be labelled 'anion-pack',
        not 'covalent' — even though the type rule would say covalent."""
        line = self._build({"Si": 16, "O": 32})
        assert "O-O" in line
        # The O-O block specifically should say anion-pack
        oo_block = line.split("O-O:")[1].split("|")[0]
        assert "anion-pack" in oo_block

    def test_metallic_pair_no_dchi_shown(self):
        """Metallic pairs (Cu-Cu in pure Cu) should NOT show a Δχ value;
        the type rule alone classifies them."""
        line = self._build({"Cu": 32})
        assert "Cu-Cu" in line
        cu_block = line.split("Cu-Cu:")[1].split("|")[0].split(",")[0]
        assert "metallic" in cu_block
        assert "Δχ" not in cu_block

    def test_line_is_single_line(self):
        """Critical: the summary must be one line so it's grep-friendly."""
        line = self._build({"In": 32, "O": 48})
        assert "\n" not in line

    def test_line_length_reasonable_for_simple_systems(self):
        """For typical 2-3 element systems the line should fit in ~300 chars."""
        for comp in [{"Si": 16}, {"In": 16, "O": 24}, {"Ga": 16, "O": 24},
                     {"Cd": 16, "Te": 16}, {"Cu": 32}]:
            line = self._build(comp)
            assert len(line) < 350, f"line too long ({len(line)} chars) for {comp}"


# ─── Auto-derive line actually appears in random_gen.log ─────────────────────

class TestAutoDeriveInLogFile:
    """Integration: run batch_random end-to-end on a tiny system and verify
    the [auto-derive] line is in the resulting random_gen.log."""

    def test_line_appears_in_log(self, tmp_path):
        from amorphgen.pipeline.random_gen import batch_random
        out_dir = tmp_path / "rg_out"
        batch_random(
            composition={"Si": 16, "O": 32},
            n_structures=1,
            output_dir=str(out_dir),
            seed=0,
        )
        log = (out_dir / "random_gen.log").read_text()
        # The auto-derive summary line must be present
        assert "[auto-derive]" in log
        # And contain the expected fields
        auto_line = next(l for l in log.splitlines() if l.startswith("[auto-derive]"))
        assert "Si16O32" in auto_line
        assert "covalent_oxide" in auto_line
        assert "Si:+4" in auto_line
        # Pair keys are sorted alphabetically: Si-O → 'O-Si'
        assert "O-Si:" in auto_line and "ionic" in auto_line
        assert "O-O:" in auto_line and "anion-pack" in auto_line
