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

    def test_unknown_pair_skipped(self):
        # Asking for In-O when the analyser only has Ga-O → no row.
        ref = {"bond_distances": {"In-O": {"expected": [2.0, 2.2]}}}
        result = validate_against_reference(_StubAnalyser(), ref)
        assert result["rows"] == []

    def test_no_system_falls_back_to_unspecified(self, reference):
        ref_no_system = {k: v for k, v in reference.items() if k != "system"}
        result = validate_against_reference(_StubAnalyser(), ref_no_system)
        assert result["system"] == "(unspecified)"


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
        assert "Summary: 1 match, 1 concern, 1 fail" in out

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
