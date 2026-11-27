"""
test/test_placement_kernel.py
-----------------------------
The compiled candidate loop of random placement (optional Numba) must place
the same structure as the Python loop for the same seed: it makes the same
random draws in the same order and applies the same tests.
"""

import numpy as np
import pytest

from amorphgen.pipeline import _placement_kernel as kernel_module
from amorphgen.pipeline.random_gen import generate_random

pytest.importorskip("numba")

CASES = [
    ("coordination-aware oxide", {"Ga": 16, "O": 24}, {}),
    ("dense dioxide", {"Ti": 20, "O": 40}, {}),
    ("network former", {"Si": 16, "O": 32}, {}),
    ("three cations", {"In": 8, "Ga": 8, "Zn": 8, "O": 32}, {}),
    ("alloy, soft pack", {"Cu": 30, "Zr": 30}, {}),
    ("element", {"Si": 48}, {}),
    ("no coordination targets", {"Ti": 16, "O": 32}, {"target_cn": {}}),
    ("fixed density, no retry", {"Ga": 16, "O": 24},
     {"target_density": 4.8, "retry_mode": "none"}),
    ("not periodic", {"Ga": 8, "O": 12}, {"pbc": False}),
]


def place(composition, seed, options, monkeypatch, compiled):
    if compiled:
        monkeypatch.delenv("AMORPHGEN_NO_NUMBA", raising=False)
        assert kernel_module.placement_kernel() is not None
    else:
        monkeypatch.setenv("AMORPHGEN_NO_NUMBA", "1")
        assert kernel_module.placement_kernel() is None
    return generate_random(composition, seed=seed, **options)


@pytest.mark.parametrize("label, composition, options", CASES, ids=[c[0] for c in CASES])
def test_same_structure_as_the_python_loop(label, composition, options, monkeypatch):
    for seed in (3, 4):
        slow = place(composition, seed, options, monkeypatch, compiled=False)
        fast = place(composition, seed, options, monkeypatch, compiled=True)
        assert fast.get_chemical_symbols() == slow.get_chemical_symbols()
        assert np.allclose(fast.cell.array, slow.cell.array, atol=1e-12)
        assert np.allclose(fast.positions, slow.positions, atol=1e-9)
        assert fast.info.get("soft_pack") == slow.info.get("soft_pack")


def test_random_stream_is_left_in_the_same_state(monkeypatch):
    """Whatever follows placement draws the same numbers either way."""
    states = []
    for compiled in (False, True):
        if compiled:
            monkeypatch.delenv("AMORPHGEN_NO_NUMBA", raising=False)
        else:
            monkeypatch.setenv("AMORPHGEN_NO_NUMBA", "1")
        import amorphgen.pipeline.random_gen as rg
        captured = {}
        original = np.random.default_rng

        def spy(seed=None):
            captured["rng"] = original(seed)
            return captured["rng"]
        monkeypatch.setattr(rg.np.random, "default_rng", spy)
        generate_random({"Ti": 12, "O": 24}, seed=9, repair_floor=False)
        monkeypatch.setattr(rg.np.random, "default_rng", original)
        states.append(captured["rng"].random())
    assert states[0] == states[1]


def test_switch_and_fallback(monkeypatch):
    monkeypatch.setenv("AMORPHGEN_NO_NUMBA", "1")
    assert kernel_module.placement_kernel() is None
    monkeypatch.setenv("AMORPHGEN_NO_NUMBA", "0")
    assert kernel_module.placement_kernel() is not None
    # a Numba that cannot compile falls back to the Python loop
    monkeypatch.setattr(kernel_module, "_KERNEL", {})
    monkeypatch.setattr(kernel_module, "_build", lambda: 1 / 0)
    assert kernel_module.placement_kernel() is None
    assert len(generate_random({"Ga": 8, "O": 12}, seed=1)) == 20
