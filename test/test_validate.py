"""Tests for amorphgen.analysis.validate (--reference YAML feature)."""
from __future__ import annotations

import pytest

from amorphgen.analysis.validate import (
    _verdict,
    format_validation_report,
    validate_against_reference,
)


# ─── _verdict() ────────────────────────────────────────────────────────────

class TestVerdict:
    def test_inside_range_is_match(self):
        assert _verdict(2.5, 2.0, 3.0) == "match"

    def test_at_lower_bound_is_match(self):
        assert _verdict(2.0, 2.0, 3.0) == "match"

    def test_at_upper_bound_is_match(self):
        assert _verdict(3.0, 2.0, 3.0) == "match"

    def test_just_below_range_is_concern(self):
        # range [2.0, 3.0], width=1.0; value 1.96 → margin 0.04, below 5% of
        # max(|2.0|,|3.0|,1.0)=3.0 → 0.15 cushion → "concern"
        assert _verdict(1.96, 2.0, 3.0) == "concern"

    def test_far_below_range_is_fail(self):
        assert _verdict(1.0, 2.0, 3.0) == "fail"

    def test_far_above_range_is_fail(self):
        assert _verdict(5.0, 2.0, 3.0) == "fail"

    def test_none_value_is_na(self):
        assert _verdict(None, 2.0, 3.0) == "n/a"

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
    def test_nonfinite_value_is_na(self, value):
        assert _verdict(value, 2.0, 3.0) == "n/a"

    @pytest.mark.parametrize("low, high", [(3.0, 2.0), (float("nan"), 3.0),
                                         (2.0, float("inf"))])
    def test_invalid_reference_is_inconclusive(self, low, high):
        assert _verdict(2.5, low, high) == "inconclusive"

    def test_custom_tolerance(self):
        # With 1% tol, 1.96 (margin 0.04) > 0.01*max(2,3,1)=0.03 → fail
        assert _verdict(1.96, 2.0, 3.0, tol_frac=0.01) == "fail"


# ─── stub analyser used by validate_against_reference ──────────────────────

class _StubAnalyser:
    """Minimal stand-in matching the methods validate_against_reference uses."""

    def density(self):
        return {"mean": 4.85}

    def bond_distances(self):
        return {"Ga-O": {"mean": 1.88}, "O-O": {"mean": 2.85}}

    def coordination(self):
        return {"Ga-O": {"mean": 4.2}}

    def bond_angles(self):
        return {"O-Ga-O": {"mean": 109.5}}


# ─── validate_against_reference() ─────────────────────────────────────────

class TestValidateAgainstReference:
    @pytest.fixture
    def reference(self):
        return {
            "system": "a-Ga2O3",
            "references": ["Kaewmeechai PRB 2025"],
            "density": {"expected": [4.70, 5.10], "units": "g/cm^3"},
            "bond_distances": {
                "Ga-O": {"expected": [1.85, 1.95], "units": "A"},
                "O-O": {"expected": [2.50, 2.70]},  # will be "fail"
            },
            "coordination": {
                "Ga-O": {"mean_expected": [4.0, 4.5]},
            },
            "bond_angles": {
                "O-Ga-O": {"expected": [105.0, 115.0]},
            },
        }

    def test_returns_system_and_sources(self, reference):
        result = validate_against_reference(_StubAnalyser(), reference)
        assert result["system"] == "a-Ga2O3"
        assert result["sources"] == ["Kaewmeechai PRB 2025"]

    def test_density_row_present(self, reference):
        result = validate_against_reference(_StubAnalyser(), reference)
        density_rows = [r for r in result["rows"] if r[0] == "Density"]
        assert len(density_rows) == 1
        descriptor, value, lo, hi, units, verdict = density_rows[0]
        assert value == 4.85
        assert (lo, hi) == (4.70, 5.10)
        assert units == "g/cm^3"
        assert verdict == "match"

    def test_bond_distance_match(self, reference):
        result = validate_against_reference(_StubAnalyser(), reference)
        ga_o = [r for r in result["rows"] if r[0] == "Bond Ga-O"][0]
        assert ga_o[5] == "match"
        assert ga_o[1] == 1.88

    def test_bond_distance_fail(self, reference):
        # O-O computed 2.85, expected [2.50, 2.70] → outside 5% margin → fail
        result = validate_against_reference(_StubAnalyser(), reference)
        o_o = [r for r in result["rows"] if r[0] == "Bond O-O"][0]
        assert o_o[5] == "fail"

    def test_coordination_row(self, reference):
        result = validate_against_reference(_StubAnalyser(), reference)
        cn = [r for r in result["rows"] if r[0] == "CN Ga-O"][0]
        assert cn[1] == 4.2
        assert cn[5] == "match"

    def test_bond_angle_row(self, reference):
        result = validate_against_reference(_StubAnalyser(), reference)
        ang = [r for r in result["rows"] if r[0] == "Angle O-Ga-O"][0]
        assert ang[1] == 109.5
        assert ang[5] == "match"

    def test_missing_section_skipped(self):
        # Reference with only density should produce only one row.
        ref = {"density": {"expected": [4.7, 5.1]}}
        result = validate_against_reference(_StubAnalyser(), ref)
        assert len(result["rows"]) == 1
        assert result["rows"][0][0] == "Density"

    def test_unknown_pair_reported_na(self):
        # Asking for In-O when the analyser only has Ga-O keeps the row, as
        # n/a, so a check that did not run shows in the table.
        ref = {"bond_distances": {"In-O": {"expected": [2.0, 2.2]}}}
        result = validate_against_reference(_StubAnalyser(), ref)
        assert result["rows"] == [("Bond In-O", None, 2.0, 2.2, "Å", "n/a")]

    def test_bond_and_angle_match_in_either_order(self):
        # the analyser names a pair in alphabetical order ("O-Si") and an
        # angle with its end atoms in that order; the reference may not
        class _SiO:
            def bond_distances(self):
                return {"O-Si": {"mean": 1.61}}

            def bond_angles(self):
                return {"N-Si-O": {"mean": 109.0}}

        ref = {"bond_distances": {"Si-O": {"expected": [1.58, 1.65]}},
               "bond_angles": {"O-Si-N": {"expected": [105.0, 113.0]}}}
        rows = validate_against_reference(_SiO(), ref)["rows"]
        assert [(r[0], r[1], r[5]) for r in rows] == [
            ("Bond Si-O", 1.61, "match"), ("Angle O-Si-N", 109.0, "match")]

    def test_coordination_is_directional(self):
        # CN O-Ga counts Ga around O: it must not take the Ga-O value
        ref = {"coordination": {"O-Ga": {"mean_expected": [2.7, 3.0]}}}
        rows = validate_against_reference(_StubAnalyser(), ref)["rows"]
        assert rows == [("CN O-Ga", None, 2.7, 3.0, "", "n/a")]

    def test_si_o_bond_checked_on_real_structure(self):
        # regression: --reference with examples/reference_a_SiO2.yaml left out
        # the Si-O bond, which StructureAnalyser keys "O-Si"
        from ase.spacegroup import crystal
        from amorphgen.analysis import StructureAnalyser
        crist = crystal(["Si", "O"], basis=[(0, 0, 0), (0.125, 0.125, 0.125)],
                        spacegroup=227, cellpar=[7.16, 7.16, 7.16, 90, 90, 90])
        ref = {"bond_distances": {"Si-O": {"expected": [1.50, 1.60]}},
               "coordination": {"Si-O": {"mean_expected": [3.9, 4.1]},
                                "O-Si": {"mean_expected": [1.95, 2.05]}},
               "bond_angles": {"Si-O-Si": {"expected": [175.0, 180.0]},
                               "O-Si-O": {"expected": [105.0, 113.0]}}}
        rows = validate_against_reference(StructureAnalyser([crist], cutoff=2.0),
                                          ref)["rows"]
        assert [r[0] for r in rows] == ["Bond Si-O", "CN Si-O", "CN O-Si",
                                        "Angle Si-O-Si", "Angle O-Si-O"]
        assert [r[5] for r in rows] == ["inconclusive"] * 5

    def test_no_system_falls_back_to_unspecified(self, reference):
        ref_no_system = {k: v for k, v in reference.items() if k != "system"}
        result = validate_against_reference(_StubAnalyser(), ref_no_system)
        assert result["system"] == "(unspecified)"

    @pytest.mark.parametrize("mean, interval, verdict", [
        (2.5, (2.2, 2.8), "match"),
        (2.5, (2.0, 3.0), "match"),
        (2.0, (2.0, 2.0), "match"),
        (2.5, (1.9, 2.8), "inconclusive"),
        (2.5, (2.2, 3.1), "inconclusive"),
        (2.5, (1.0, 4.0), "inconclusive"),
        (1.5, (1.0, 2.1), "inconclusive"),
        (3.5, (2.9, 4.0), "inconclusive"),
        (1.5, (1.0, 2.0), "inconclusive"),
        (3.5, (3.0, 4.0), "inconclusive"),
        (1.5, (1.1, 1.9), "fail"),
        (1.95, (1.92, 1.98), "concern"),
        (2.5, (None, None), "inconclusive"),
        (2.5, (float("nan"), float("nan")), "inconclusive"),
        (2.5, (2.0, float("inf")), "inconclusive"),
        (2.5, (2.8, 2.2), "inconclusive"),
    ])
    def test_structure_mean_interval_decides_verdict(self, mean, interval, verdict):
        uncertainty = {"mean": mean, "n_structures": 4, "sem": 0.1,
                       "ci_low": interval[0], "ci_high": interval[1],
                       "confidence": 0.95, "sampling_unit": "structure"}

        class Analyser:
            def density(self):
                # A different pooled mean catches accidental validation of
                # pooled atoms instead of the equal-weight structure mean.
                return {"mean": 20.0, "uncertainty": uncertainty}

        result = validate_against_reference(
            Analyser(), {"density": {"expected": [2.0, 3.0]}})
        assert result["rows"] == [("Density", mean, 2.0, 3.0, "g/cm³", verdict)]
        assert result["intervals"]["Density"] == uncertainty

    def test_all_metric_types_use_intervals_with_reversed_keys(self):
        entry = {"mean": 99.0, "uncertainty": {
            "mean": 2.5, "ci_low": 1.5, "ci_high": 3.5,
            "n_structures": 3, "confidence": 0.95}}

        class Analyser:
            def bond_distances(self):
                return {"Ga-O": entry}

            def coordination(self):
                return {"Ga-O": entry}

            def bond_angles(self):
                return {"N-Ga-O": entry}

        reference = {
            "bond_distances": {"O-Ga": {"expected": [2.0, 3.0]}},
            "coordination": {"Ga-O": {"mean_expected": [2.0, 3.0]}},
            "bond_angles": {"O-Ga-N": {"expected": [2.0, 3.0]}}}
        result = validate_against_reference(Analyser(), reference)
        assert [row[1] for row in result["rows"]] == [2.5] * 3
        assert [row[5] for row in result["rows"]] == ["inconclusive"] * 3
        assert set(result["intervals"]) == {"Bond O-Ga", "CN Ga-O", "Angle O-Ga-N"}

    def test_nonfinite_uncertainty_mean_is_na(self):
        class Analyser:
            def density(self):
                return {"mean": 2.5, "uncertainty": {
                    "mean": float("nan"), "ci_low": None, "ci_high": None}}

        result = validate_against_reference(
            Analyser(), {"density": {"expected": [2.0, 3.0]}})
        assert result["rows"][0][1] is None
        assert result["rows"][0][5] == "n/a"


# ─── format_validation_report() ───────────────────────────────────────────

class TestFormatValidationReport:
    def test_empty_rows_message(self):
        result = {"system": "X", "sources": [], "rows": []}
        out = format_validation_report(result)
        assert "No validation rows" in out

    def test_renders_header_and_rows(self):
        result = {
            "system": "a-Ga2O3",
            "sources": ["Some Reference"],
            "rows": [
                ("Density", 4.85, 4.70, 5.10, "g/cm³", "match"),
                ("Bond Ga-O", 1.88, 1.85, 1.95, "Å", "match"),
            ],
        }
        out = format_validation_report(result)
        assert "Validation: a-Ga2O3" in out
        assert "Some Reference" in out
        assert "Density" in out
        assert "Bond Ga-O" in out
        assert "match" in out
        assert "Summary: 2 match" in out

    def test_summary_counts(self):
        result = {
            "system": "X",
            "sources": [],
            "rows": [
                ("A", 1.0, 0.0, 2.0, "", "match"),
                ("B", 5.0, 0.0, 2.0, "", "fail"),
                ("C", 2.05, 0.0, 2.0, "", "concern"),
            ],
        }
        out = format_validation_report(result)
        assert "Summary: 1 match, 1 concern, 1 fail (out of 3 metrics)" in out
        assert "n/a" not in out

    def test_summary_counts_na(self):
        result = {
            "system": "X",
            "sources": [],
            "rows": [("A", 1.0, 0.0, 2.0, "", "match"),
                     ("Bond In-O", None, 2.0, 2.2, "Å", "n/a")],
        }
        out = format_validation_report(result)
        assert "Summary: 1 match, 0 concern, 0 fail, 1 n/a (out of 2 metrics)" in out

    def test_handles_none_value(self):
        result = {
            "system": "X",
            "sources": [],
            "rows": [("Density", None, 4.7, 5.1, "g/cm³", "n/a")],
        }
        out = format_validation_report(result)
        assert "n/a" in out

    def test_large_value_formatted_with_one_decimal(self):
        # value=123.4 should render with .1f, not .3f
        result = {
            "system": "X", "sources": [],
            "rows": [("Big", 123.456, 100.0, 200.0, "u", "match")],
        }
        out = format_validation_report(result)
        assert "123.5" in out

    def test_reports_confidence_interval_and_inconclusive_total(self):
        result = {
            "system": "X", "sources": [],
            "rows": [("Density", 2.5, 2.0, 3.0, "g/cm³", "inconclusive")],
            "intervals": {"Density": {
                "mean": 2.5, "ci_low": 1.8, "ci_high": 3.2,
                "confidence": 0.95, "n_structures": 4}},
        }
        out = format_validation_report(result)
        assert "95% [1.800, 3.200]" in out
        assert "uncertainty of that mean" in out
        assert "independent sampling units" in out
        assert "0 fail, 1 inconclusive (out of 1 metrics)" in out
        assert "interval crosses a reference bound" in out

    def test_reports_unavailable_interval(self):
        result = {
            "system": "X", "sources": [],
            "rows": [("Density", 2.5, 2.0, 3.0, "g/cm³", "inconclusive")],
            "intervals": {"Density": {
                "mean": 2.5, "ci_low": None, "ci_high": None,
                "confidence": 0.95, "n_structures": 1}},
        }
        out = format_validation_report(result)
        assert "unavailable" in out
        assert "At least two independent structures" in out
