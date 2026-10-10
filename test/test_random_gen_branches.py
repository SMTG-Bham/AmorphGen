"""Random placement helpers, batch retry ladder and radii fallbacks against independent oracles."""

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.calculators.emt import EMT
from ase.data import atomic_masses, atomic_numbers, covalent_radii
from ase.io import read

from amorphgen_test_helpers import minsep_floors, shortest_pair_distances
from amorphgen.pipeline import random_gen
from amorphgen.pipeline.random_gen import (
    _auto_dmax, _derive_structure_seed, _push_apart, _random_calculator_settings,
    _repair_min_cn, _repair_undercoordination, _resume_value, _update_cn_array,
    batch_random, generate_random,
)
from amorphgen.utils import radii


def _density(atoms):
    """Mass over volume in g/cm^3 (1 amu/A^3 = 1.66053907 g/cm^3)."""
    mass = sum(atomic_masses[atomic_numbers[s]] for s in atoms.get_chemical_symbols())
    return mass * 1.66053907 / atoms.get_volume()


def _mic_distances(positions, L):
    d = positions[:, None, :] - positions[None, :, :]
    d -= L * np.round(d / L)
    return np.linalg.norm(d, axis=-1)


# -- placement helpers -------------------------------------------------------

def test_push_apart_reports_unfinished_overlap_removal():
    atoms = Atoms("Cu2", positions=[[5, 5, 5], [5.5, 5, 5]], cell=[10] * 3, pbc=True)
    work, converged, ratio = _push_apart(atoms, {"Cu-Cu": 2.0}, max_iter=1)
    assert converged is False
    # One pass moves each atom by a quarter of the overlap: 0.5 -> 1.25 A.
    assert ratio == pytest.approx(0.25)
    assert work.get_distance(0, 1, mic=True) == pytest.approx(1.25)
    assert atoms.get_distance(0, 1) == pytest.approx(0.5)
    done, converged, ratio = _push_apart(atoms, {"Cu-Cu": 2.0})
    assert converged is True
    assert done.get_distance(0, 1, mic=True) >= 0.985 * 2.0


def test_auto_dmax_ignores_pairs_without_a_coordinated_element():
    minsep = {"Na-O": 2.0, "Si-O": 1.6, "Na-Na": 3.0, "O-O": 2.4}
    dmax = _auto_dmax(minsep, {"Si": 4}, composition={"Na": 2, "Si": 1, "O": 3})
    assert dmax == {"Si-O": pytest.approx(2.4)}


def test_cn_update_uses_minimum_image_and_counts_both_partners():
    L = 10.0
    positions = np.array([[0.5, 5, 5], [5.0, 5, 5], [9.5, 5, 5]])
    table = np.array([[2.0 ** 2]])
    types = np.zeros(3, dtype=int)
    cn = np.array([7, 0, 0])
    _update_cn_array(cn, 0, positions, 1, types, 0, table, L, True)
    assert cn[0] == 0
    _update_cn_array(cn, 1, positions, 2, types, 0, table, L, True)
    _update_cn_array(cn, 2, positions, 3, types, 0, table, L, True)
    # Atom 2 is 1.0 A from atom 0 through the boundary and 4.5 A from atom 1.
    np.testing.assert_array_equal(cn, [1, 0, 1])
    open_cn = np.zeros(3, dtype=int)
    _update_cn_array(open_cn, 2, positions, 3, types, 0, table, L, False)
    np.testing.assert_array_equal(open_cn, [0, 0, 0])


def _repair_inputs(positions, target, minsep=1.0, dmax=2.0, L=10.0):
    n = len(positions)
    positions = np.array(positions, dtype=float)
    distances = _mic_distances(positions, L)
    cn = ((distances <= dmax) & ~np.eye(n, dtype=bool)).sum(axis=1)
    return dict(positions=positions, n_placed=n, placed_type_idx=np.zeros(n, dtype=int),
                cn_array=cn, target_cn_arr=np.array([target]), cn_tolerance=0,
                minsep_sq_table=np.array([[minsep ** 2]]),
                dmax_sq_table=np.array([[dmax ** 2]]), L=L, pbc=True)


def test_repair_leaves_satisfied_structures_untouched():
    inputs = _repair_inputs([[2, 2, 2], [3.5, 2, 2]], target=1)
    before = inputs["positions"].copy()
    rng = np.random.default_rng(4)
    assert _repair_undercoordination(**inputs, max_iter=50, rng=rng) == 0
    floors = np.array([1])
    assert _repair_min_cn(**inputs, floor_arr=floors, max_iter=5, rng=rng) == 0
    np.testing.assert_array_equal(inputs["positions"], before)
    # No proposal was drawn: the generator state is that of a fresh one.
    assert rng.random() == np.random.default_rng(4).random()


def test_repair_bonds_two_isolated_atoms_at_the_shell_midpoint():
    inputs = _repair_inputs([[2, 2, 2], [7, 7, 7]], target=1)
    left = _repair_undercoordination(**inputs, max_iter=50, rng=np.random.default_rng(0))
    assert left == 0
    np.testing.assert_array_equal(inputs["cn_array"], [1, 1])
    # The moved atom lands at (minsep + dmax) / 2 from its partner.
    assert _mic_distances(inputs["positions"], 10.0)[0, 1] == pytest.approx(1.5)


def test_repair_rejects_moves_that_do_not_reduce_undercoordination():
    # Atoms 0-1 are a saturated pair; atom 2 has no under-coordinated partner,
    # and every perturbation leaves it unbonded or would over-coordinate the pair.
    inputs = _repair_inputs([[2, 2, 2], [3.5, 2, 2], [7, 7, 7]], target=1)
    before = inputs["positions"].copy()
    left = _repair_undercoordination(**inputs, max_iter=40, rng=np.random.default_rng(6))
    assert left == 1
    np.testing.assert_array_equal(inputs["positions"], before)
    np.testing.assert_array_equal(inputs["cn_array"], [1, 1, 0])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_repair_keeps_cn_bookkeeping_minsep_and_caps_exact(seed):
    rng = np.random.default_rng(seed)
    L, minsep, dmax, target = 5.0, 1.0, 1.8, 2
    positions = []
    while len(positions) < 12:
        trial = rng.random(3) * L
        candidate = np.array(positions + [trial])
        distances = _mic_distances(candidate, L)
        np.fill_diagonal(distances, np.inf)
        if distances.min() < minsep:
            continue
        if ((distances <= dmax).sum(axis=1) > target).any():
            continue
        positions.append(trial)
    inputs = _repair_inputs(positions, target, minsep, dmax, L)
    initial = int((inputs["cn_array"] < target).sum())
    left = _repair_undercoordination(**inputs, max_iter=300, rng=rng)
    distances = _mic_distances(inputs["positions"], L)
    np.fill_diagonal(distances, np.inf)
    recount = (distances <= dmax).sum(axis=1)
    np.testing.assert_array_equal(inputs["cn_array"], recount)
    assert distances.min() >= minsep
    assert recount.max() <= target
    assert left == int((recount < target).sum()) <= initial
    assert np.all((inputs["positions"] >= 0) & (inputs["positions"] < L))


@pytest.mark.parametrize("min_cn, floors", [
    (5, {"O": 5, "Si": 4}),         # capped at Si's target; O has none in SiO2
    (-2, {"O": 0, "Si": 0}),
    ({"Si": 3}, {"O": 0, "Si": 3}),
    (None, {"O": 2, "Si": 3}),      # auto: anions 2, cations 3
])
def test_minimum_cn_floor_resolution(caplog, min_cn, floors):
    composition = {"Si": 4, "O": 8}
    with caplog.at_level(logging.INFO, logger="amorphgen.pipeline.random_gen"):
        atoms = generate_random(composition, seed=11, min_cn=min_cn)
    messages = [r.getMessage() for r in caplog.records if "Min-CN floor" in r.getMessage()]
    assert messages[0] == f"Min-CN floor per element: {floors}"
    assert atoms.get_chemical_formula() == "O8Si4"
    shortest = shortest_pair_distances(atoms, 4.0)
    for pair, floor in minsep_floors(composition).items():
        assert shortest.get("-".join(sorted(pair.split("-"))), np.inf) >= 0.985 * floor


@pytest.mark.parametrize("min_cn", [0, {"Si": 0, "O": 0}, {}])
def test_zero_floor_matches_disabled_floor_repair(min_cn):
    composition = {"Si": 4, "O": 8}
    reference = generate_random(composition, seed=5, repair_floor=False)
    atoms = generate_random(composition, seed=5, min_cn=min_cn)
    np.testing.assert_array_equal(atoms.positions, reference.positions)


# -- resume identity ---------------------------------------------------------

def test_resume_values_are_json_types_and_unknown_types_fail():
    assert _resume_value({"path": Path("/tmp/model.pt"), 3: np.float32(1.5),
                          "row": (np.int64(2), [np.bool_(True)])}) == {
        "path": "/tmp/model.pt", "3": 1.5, "row": [2, [True]]}
    with pytest.raises(ValueError, match="Cannot record run setting of type object"):
        _resume_value({"calc": object()})


def test_calculator_identity_skips_unserialisable_loaded_models():
    calc = SimpleNamespace(model=object(), model_path=Path("/models/a.model"),
                           head="omat", parameters={"cutoff": 5.0})
    settings = _random_calculator_settings(calc, {"model": "mace-mpa-0"})
    assert settings["attributes"] == {"model_path": "/models/a.model", "head": "omat"}
    assert settings["parameters"] == {"cutoff": 5.0}
    calc.model = "mace-mpa-0"
    assert _random_calculator_settings(calc, {"model": "mace-mpa-0"})["attributes"]["model"] == "mace-mpa-0"


# -- batch generation --------------------------------------------------------

def test_unknown_generation_setting_fails_before_any_output(tmp_path):
    out = tmp_path / "out"
    with pytest.raises(TypeError, match=r"Unknown generation setting\(s\): \['temperature'\]"):
        batch_random({"Si": 2, "O": 4}, n_structures=1, output_dir=str(out), temperature=300)
    assert os.listdir(out) == [".amorphgen.lock"]


def test_target_density_is_logged_and_realised(tmp_path):
    out = tmp_path / "out"
    batch_random({"Si": 4, "O": 8}, n_structures=1, output_dir=str(out), seed=3,
                 target_density=2.2)
    log = (out / "random_gen.log").read_text()
    assert "Target density: 2.20 g/cm3" in log
    assert "placed at 2.20 g/cm3" in log and "WARNING" not in log
    atoms = read(out / "random_initial" / "random_0000.xyz")
    assert _density(atoms) == pytest.approx(2.2, rel=1e-4)


def test_resume_regenerates_unreadable_nonempty_file_identically(tmp_path):
    out = tmp_path / "out"
    composition = {"Si": 4, "O": 8}
    batch_random(composition, n_structures=2, output_dir=str(out), seed=42)
    first, second = (out / "random_initial" / f"random_000{i}.xyz" for i in (0, 1))
    reference, untouched = read(second), first.read_bytes()
    second.write_text("not an extxyz file\n")
    batch_random(composition, n_structures=2, output_dir=str(out), seed=42, resume=True)
    regenerated = read(second)
    assert regenerated.get_chemical_symbols() == reference.get_chemical_symbols()
    np.testing.assert_allclose(regenerated.positions, reference.positions)
    assert first.read_bytes() == untouched


@pytest.fixture
def stalling(monkeypatch):
    """generate_random that stalls for the first ``failures`` calls."""
    state = SimpleNamespace(failures=0, calls=[])

    def generate(composition, seed=None, **kwargs):
        state.calls.append({"seed": seed, "density_scale": kwargs.get("density_scale", 1.0),
                            "minsep": kwargs.get("minsep")})
        if len(state.calls) <= state.failures:
            raise RuntimeError("placement stalled")
        return Atoms("Mg2O2", scaled_positions=[[0, 0, 0], [.5, .5, 0], [.5, 0, 0], [0, .5, 0]],
                     cell=[6.0] * 3, pbc=True)

    monkeypatch.setattr(random_gen, "generate_random", generate)
    return state


USER_MINSEP = {"Mg-Mg": 2.8, "Mg-O": 1.9, "O-O": 2.6}


@pytest.mark.parametrize("settings, scales, reductions", [
    ({}, [1, .92, .85, .78, .78, .78, .78], [0, 0, 0, 0, .05, .10, .15]),
    ({"target_density": 3.0}, [1, 1, 1, 1], [0, .05, .10, .15]),
    ({"retry_mode": "reduce-minsep", "minsep": USER_MINSEP}, [1, 1, 1, 1], [0, .05, .10, .15]),
    ({"retry_mode": "none", "minsep": USER_MINSEP}, [1], [0]),
])
def test_batch_retry_ladder_then_skip(stalling, tmp_path, settings, scales, reductions):
    composition = {"Mg": 2, "O": 2}
    stalling.failures = len(scales)
    out = tmp_path / "out"
    paths = batch_random(composition, n_structures=2, output_dir=str(out), seed=9,
                         max_retries=1, **settings)
    calls = stalling.calls
    assert len(calls) == len(scales) + 1
    assert [c["seed"] for c in calls] == (
        [_derive_structure_seed(9, 0, attempt) for attempt in range(len(scales))]
        + [_derive_structure_seed(9, 1, 0)])
    assert [c["density_scale"] for c in calls] == pytest.approx(scales + [1])
    base = settings.get("minsep") or minsep_floors(composition)
    for call, reduction in zip(calls, reductions):
        if reduction == 0:
            assert call["minsep"] == settings.get("minsep")
            continue
        # Only the metal-metal floor is softened; anion packing and bonds keep theirs.
        assert call["minsep"]["Mg-Mg"] == pytest.approx(base["Mg-Mg"] * (1 - reduction))
        assert call["minsep"]["O-O"] == pytest.approx(base["O-O"])
        assert call["minsep"]["Mg-O"] == pytest.approx(base["Mg-O"])
    # After the skip the next structure starts from the original settings.
    assert calls[-1]["minsep"] == settings.get("minsep")
    assert paths == [str(out / "random_initial" / "random_0001.xyz")]
    assert sorted(os.listdir(out / "random_initial")) == ["random_0001.xyz"]
    log = (out / "random_gen.log").read_text()
    assert f"Skipping structure 1 after 1 attempts x {len(scales)} retries" in log
    assert "only 1/2 structures were successfully generated" in log
    assert ("skipping the cell-expansion rungs" in log) == ("target_density" in settings)


class _FlakyEMT(EMT):
    """EMT whose first evaluation fails, as a model still loading might."""

    def __init__(self):
        super().__init__()
        self.evaluations = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        self.evaluations += 1
        if self.evaluations == 1:
            raise RuntimeError("weights not loaded")
        super().calculate(atoms, properties, system_changes)


def test_failed_warmup_is_logged_and_relaxation_reaches_fmax(tmp_path):
    out = tmp_path / "out"
    paths = batch_random({"Cu": 4}, n_structures=1, output_dir=str(out), seed=2,
                         relax=True, calc=_FlakyEMT(), fmax=0.3, max_relax_steps=300,
                         optimizer="FIRE", cell_filter="none", target_cn={})
    log = (out / "random_gen.log").read_text()
    assert "Calculator warmup skipped (RuntimeError: weights not loaded)" in log
    assert "Converged after" in log
    relaxed = read(paths[0])
    assert relaxed.info["relaxation_converged"]
    relaxed.calc = EMT()
    assert np.linalg.norm(relaxed.get_forces(), axis=1).max() < 0.3


def test_cell_relaxation_requires_a_stress_capable_calculator(tmp_path):
    class EnergyForcesEMT(EMT):
        implemented_properties = ["energy", "forces"]

    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="--relax with cell_filter='FrechetCellFilter' requires a stress"):
        batch_random({"Cu": 4}, n_structures=1, output_dir=str(out), seed=2, relax=True,
                     calc=EnergyForcesEMT(), cell_filter="FrechetCellFilter", target_cn={})
    assert (out / "random_initial" / "random_0000.xyz").is_file()
    assert not list((out / "random_opt").iterdir())


# -- radii fallbacks ---------------------------------------------------------

def _solve(composition):
    return radii._solve_oxidation_states(tuple(sorted(composition.items())))


@pytest.mark.parametrize("composition", [
    {"Cu": 1, "Zn": 1},                       # no anion
    {"O": 2},                                  # no cation
    {"Tc": 1, "Po": 1, "O": 4},                # two cations without Shannon states
    {"Sb": 2, "Li": 1, "O": 3},                # Sb would need +2.5
    {"Os": 1, "V": 1, "Cr": 1, "Ir": 1, "Ru": 1, "Ag": 1, "Re": 1, "Mn": 1,
     "Tl": 1, "Sn": 1, "O": 20},               # > 20000 assignments: not enumerated
])
def test_undetermined_oxidation_states_return_none(composition):
    assert _solve(composition) is None


def test_untabulated_cation_takes_the_remaining_charge():
    # Li+ fixes Sb at (6 - 2) / 2 = +2 once the remainder divides evenly.
    assert _solve({"Sb": 2, "Li": 2, "O": 3}) == {"Li": 1, "Sb": 2}
    assert radii.infer_oxidation_state("Fe", {"Si": 1, "O": 2}) is None


@pytest.mark.parametrize("composition, expected", [
    ({"Ti": 1, "B": 2}, ({"Ti": 6}, 0)),       # boride: metal octahedral
    ({"W": 1, "O": 3}, ({"W": 6}, 1)),         # high-valent oxide
    ({"Si": 1, "Cl": 4}, ({"Si": 4}, 0)),      # metalloid halide
    ({"Si": 1, "O": 1, "Cl": 2}, ({"Si": 4}, 1)),  # metalloid in a mixed-anion compound
])
def test_class_target_cn(composition, expected):
    assert radii.auto_target_cn(composition) == expected


def test_bond_class_without_electronegativity_stays_ionic():
    assert "Xe" not in radii.PAULING_EN
    assert radii.classify_bond("Si", "Xe") == "ionic"
    assert radii.classify_bond("Si", "C") == "covalent"


def test_radius_lookups_fall_back_in_documented_order():
    assert radii.get_ionic_radius("O", oxidation_state=2) is None
    assert radii.get_effective_radius("Cu") == radii.METALLIC_RADII["Cu"]
    assert radii.get_effective_radius("Cl") == radii.get_ionic_radius("Cl")
    for symbol in ("Xe", "Tc"):                # noble gas; metal with no metallic radius
        assert radii.get_effective_radius(symbol) == covalent_radii[atomic_numbers[symbol]]


def test_noble_gas_pair_minsep_uses_capped_covalent_fallback():
    minsep = radii.default_minsep(["Xe", "Kr"])
    expected = (covalent_radii[atomic_numbers["Xe"]] + covalent_radii[atomic_numbers["Kr"]]) * 0.85
    assert minsep["Kr-Xe"] == pytest.approx(min(expected, 3.0))


def test_density_and_packing_edge_cases():
    assert radii.estimate_density({}) is None
    assert radii.estimate_density({"Cu": 0}) is None
    assert radii._halogen_fraction({"Si": 1}) == 0.0
    assert radii._halogen_fraction({"Bi": 1, "O": 1, "Cl": 1}) == 0.5
    assert radii._oxyhalide_packing_factor({"Ti": 1, "O": 2}) == radii.PACKING_FACTORS["oxyhalide"]


@pytest.mark.parametrize("symbol, cls, composition, expected", [
    ("Sb", "high_valent_oxide", {"Sb": 2, "O": 5}, 0.60),
    ("Sb", "covalent_oxide", {"Sb": 2, "O": 3}, covalent_radii[atomic_numbers["Sb"]]),
    ("Tc", "metal_oxide", {"Tc": 1, "O": 2}, covalent_radii[atomic_numbers["Tc"]]),
    ("Tc", "boride", {"Tc": 1, "B": 2}, covalent_radii[atomic_numbers["Tc"]]),
    ("Tc", "default", {"Tc": 1, "Cl": 4}, covalent_radii[atomic_numbers["Tc"]]),
    # Ambiguous charge balance still promotes an oxoanion centre to its top state.
    ("S", "covalent_oxide", {"N": 1, "S": 1, "O": 5}, radii.SHANNON_IONIC_RADII["S"][6][6]),
])
def test_density_radius_selection(symbol, cls, composition, expected):
    assert radii._radius_for_density(symbol, cls, composition) == pytest.approx(expected)
