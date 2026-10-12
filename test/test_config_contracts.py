"""Contracts shared by CLI parsing and runtime configuration consumers."""

import math

import pytest

from amorphgen.cli import _parse_dmax, _parse_minsep
from amorphgen.configs.default_config import DEFAULT_CONFIG
from amorphgen.configs.yaml_config import _STAGE_SCHEMA
from amorphgen.utils.repulsion import validate_repulsive_core_config
from amorphgen.utils.safety import validate_safety_config


@pytest.mark.parametrize("key,validate", [
    ("safety", validate_safety_config),
    ("repulsive_core", validate_repulsive_core_config),
])
def test_runtime_defaults_match_config_without_sharing_mutable_results(key, validate):
    first, second = validate(), validate()
    assert first == second == DEFAULT_CONFIG[key]
    first[next(iter(first))] = "changed"
    assert second == DEFAULT_CONFIG[key]
    assert first != second


def test_stage_schemas_preserve_specific_keys_and_independent_mappings():
    for stage in ("eq_premelt", "eq_high", "eq_low", "melt"):
        assert "make_cubic" in _STAGE_SCHEMA[stage]
    assert "make_cubic" not in _STAGE_SCHEMA["quench"]
    assert "T" not in _STAGE_SCHEMA["melt"]
    assert "T_start" not in _STAGE_SCHEMA["eq_high"]
    assert _STAGE_SCHEMA["eq_high"] is not _STAGE_SCHEMA["eq_low"]


@pytest.mark.parametrize("parse,label,example", [
    (_parse_minsep, "minsep", "1.6"), (_parse_dmax, "dmax", "2.0"),
])
def test_pair_distance_parser_preserves_messages_and_legacy_syntax(parse, label, example):
    assert parse(" ,Si-O = 2, Si-O=3,") == {"Si-O": 3.0}
    with pytest.raises(ValueError) as exc:
        parse("Si-O")
    assert str(exc.value) == (
        f"Invalid {label} entry: 'Si-O'. "
        f"Expected 'A-B=distance' (e.g. 'Si-O={example}')."
    )
    with pytest.raises(ValueError, match=f"Invalid {label} value"):
        parse("Si-O=bad")
    with pytest.raises(ValueError, match=f"{label.capitalize()} for 'Si-O' must be positive"):
        parse("Si-O=0")
    # Scientific validators downstream own finiteness; parsing remains unchanged.
    assert math.isnan(parse("Si-O=nan")["Si-O"])
