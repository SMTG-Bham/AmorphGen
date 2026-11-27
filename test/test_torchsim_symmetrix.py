"""
test/test_torchsim_symmetrix.py
-------------------------------
--engine torchsim with an accelerator: a batch of structures is evaluated in
one joined Symmetrix call and handed to torch-sim as its model. EMT stands in
for a calculator (one structure per call); stand-in evaluators with the
calling conventions of the two Symmetrix packages exercise the joined
evaluation, which must reproduce single-structure evaluations.
"""

import dataclasses
import types

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

import amorphgen.utils.accelerator as acc

class TestTorchSimModel:
    """--engine torchsim --accelerator symmetrix-xl: Symmetrix as torch-sim's model.
    EMT stands in for Symmetrix-XL."""

    @pytest.fixture
    def cells(self):
        out = []
        for i, repeat in enumerate([(2, 2, 2), (2, 2, 1), (2, 2, 2)]):
            atoms = bulk("Cu", "fcc", a=3.6 + 0.05 * i, cubic=True) * repeat
            atoms.rattle(0.05, seed=i)
            out.append(atoms)
        return out

    @pytest.fixture
    def model(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("torch_sim")
        from amorphgen.utils.torchsim_symmetrix import symmetrix_torchsim_model
        return symmetrix_torchsim_model(EMT(), torch.device("cpu"), torch.float64, batch=4)

    def test_forward_matches_the_calculator(self, model, cells):
        import torch_sim as ts
        from ase.stress import voigt_6_to_full_3x3_stress
        out = model(ts.initialize_state(cells, model.device, model.dtype))
        assert out["energy"].shape == (3,) and out["stress"].shape == (3, 3, 3)
        assert out["forces"].shape == (sum(len(a) for a in cells), 3)
        start = 0
        for i, atoms in enumerate(cells):
            atoms.calc = EMT()
            assert out["energy"][i].item() == pytest.approx(atoms.get_potential_energy())
            assert np.allclose(out["forces"][start:start + len(atoms)].numpy(),
                               atoms.get_forces(), atol=1e-10)
            assert np.allclose(out["stress"][i].numpy(),
                               voigt_6_to_full_3x3_stress(atoms.get_stress()), atol=1e-10)
            start += len(atoms)
        assert model.calls == [(3, 80)]

    def test_atom_limit_splits_the_call(self, cells):
        torch = pytest.importorskip("torch")
        ts = pytest.importorskip("torch_sim")
        from amorphgen.utils.torchsim_symmetrix import symmetrix_torchsim_model
        seen = []

        class Counting(EMT):
            def calculate(self, atoms=None, *args, **kwargs):
                seen.append(len(atoms))
                super().calculate(atoms, *args, **kwargs)
        limited = symmetrix_torchsim_model(Counting(), torch.device("cpu"), torch.float64,
                                           batch=4, batch_atoms=40)
        out = limited(ts.initialize_state(cells, limited.device, limited.dtype))
        assert seen == [32, 16, 32] and out["energy"].shape == (3,)

    def test_batched_relaxation(self, model, cells):
        """torch-sim's optimiser driven by the wrapped calculator reaches a minimum
        of that calculator."""
        from amorphgen.utils.torchsim_engine import batch_relax
        relaxed = batch_relax(cells, model, fmax=0.02, max_steps=300, cell_filter="cubic",
                              optimizer="fire", log=lambda *a: None)
        assert len(relaxed) == 3
        for atoms in relaxed:
            check = atoms.copy()
            check.calc = EMT()
            assert np.abs(check.get_forces()).max() < 0.03
            assert abs(check.get_stress()[:3].mean()) * 160.2 < 0.1      # GPa
        assert max(n for n, _ in model.calls) == 3            # all three in one call

    def test_batched_md(self, model, cells):
        from amorphgen.utils.torchsim_md import batch_nvt
        same = [cells[0].copy(), cells[2].copy()]
        out = batch_nvt(same, model, 300.0, n_steps=20, timestep_fs=1.0, seed=3,
                        log=lambda *a: None)
        frames = out[0] if isinstance(out, tuple) else out
        assert len(frames) == 2
        assert all(n == 2 for n, _ in model.calls)

    def test_engine_takes_batch_size_from_the_model(self, model, cells):
        from amorphgen.utils.torchsim_engine import estimate_batch_size
        assert estimate_batch_size(model, cells, log=lambda *a: None) == 4

    def test_build_model_routes_to_the_accelerator(self, monkeypatch):
        pytest.importorskip("torch_sim")
        from amorphgen.utils import torchsim_engine
        seen = {}

        def fake(accelerator, **kwargs):
            seen.update(accelerator=accelerator, **kwargs)
            return "accelerated"
        monkeypatch.setattr(acc, "torchsim_model", fake)
        assert torchsim_engine.build_model("mace-mpa-0", device="cpu", dtype="float32",
                                           accelerator={"engine": "symmetrix-xl"}) == "accelerated"
        assert seen["model"] == "mace-mpa-0" and seen["dtype"] == "float32"

    def test_torchsim_model_falls_back_without_a_package(self, monkeypatch, capsys):
        monkeypatch.setattr(acc, "installed_flavours", lambda: {})
        assert acc.torchsim_model("auto", model="mace-mpa-0", model_path=None,
                                  dtype="float32", device="cpu", torch_dtype=None) is None
        assert acc.torchsim_model(None, model="mace-mpa-0", model_path=None,
                                  dtype="float32", device="cpu", torch_dtype=None) is None


class FakeEvaluator:
    """Stand-in for a Symmetrix evaluator: a pair potential with the same call
    and the same outputs (node energies, per-pair forces)."""

    atomic_numbers = [8, 29, 14]          # model order differs from Z order
    r_cut = 4.0

    def __init__(self):
        self.calls = 0

    def compute_node_energies_forces(self, num_nodes, node_types, num_neigh, j_list,
                                     neigh_types, xyz, r):
        self.calls += 1
        assert isinstance(node_types, list) and isinstance(neigh_types, list)
        xyz = np.asarray(xyz).reshape(-1, 3)
        r = np.asarray(r)
        strength = 1.0 + 0.3 * np.asarray(neigh_types)          # species-dependent
        centre = np.repeat(np.arange(num_nodes), np.asarray(num_neigh))
        strength = strength * (1.0 + 0.3 * np.asarray(node_types)[centre])
        phi = strength * (self.r_cut - r) ** 2
        dphi = -2.0 * strength * (self.r_cut - r)
        self.node_energies = np.bincount(centre, weights=0.5 * phi, minlength=num_nodes)
        pair = -0.5 * dphi[:, None] * xyz / r[:, None]
        # like the real container, larger than the number of pairs
        self.node_forces = np.concatenate([pair.ravel(), np.zeros(30)])


class FakeSymmetrix:
    """A Symmetrix calculator's own single-structure evaluation
    (Symmetrix.calculate), for comparison with the joined one."""

    def __init__(self):
        self.evaluator = FakeEvaluator()
        self.cutoff = self.evaluator.r_cut
        self.results = {}

    def calculate(self, atoms=None, properties=("energy",), system_changes=None):
        from ase.neighborlist import neighbor_list
        from ase.stress import full_3x3_to_voigt_6_stress
        numbers = atoms.get_atomic_numbers().tolist()
        model = self.evaluator.atomic_numbers
        i_list, j_list, r, xyz = neighbor_list("ijdD", atoms, self.cutoff)
        n = len(atoms)
        self.evaluator.compute_node_energies_forces(
            n, [model.index(z) for z in numbers], np.bincount(j_list, minlength=n), j_list,
            [model.index(numbers[j]) for j in j_list], xyz.flatten(), r)
        energies = np.asarray(self.evaluator.node_energies)[:n]
        pair = np.asarray(self.evaluator.node_forces).reshape((-1, 3))[:len(i_list)]
        forces = np.zeros((n, 3))
        for c in range(3):
            forces[:, c] = (np.bincount(j_list, weights=pair[:, c], minlength=n)
                            - np.bincount(i_list, weights=pair[:, c], minlength=n))
        self.results = {"energies": energies, "energy": energies.sum(),
                        "free_energy": energies.sum(), "forces": forces,
                        "stress": full_3x3_to_voigt_6_stress((-pair.T @ xyz) / atoms.get_volume())}


def _split(symbols):
    import re
    return re.findall("[A-Z][a-z]?", symbols)


class FakeXL(FakeSymmetrix):
    """Stand-in with Symmetrix-XL's host-graph calls: ``_compute_mace`` on an
    input tuple, then ``_collect_mace_results`` (energies, reduced forces) and
    the evaluator's pair forces for the stress."""

    def __init__(self, break_forces=False):
        super().__init__()
        self.tuned = []
        self.break_forces = break_forces

    def _apply_kernel_launch_tuning(self, num_nodes, num_edges, properties):
        self.tuned.append((num_nodes, num_edges, tuple(properties)))

    def _compute_mace(self, mace_inputs, atoms=None):
        num_nodes, node_types, num_neigh, j_list, neigh_types, xyz, r, i_list = mace_inputs
        assert np.asarray(xyz).shape == (len(j_list), 3)
        assert np.array_equal(num_neigh, np.bincount(i_list, minlength=num_nodes))
        self.evaluator.compute_node_energies_forces(
            num_nodes, node_types, num_neigh, j_list, neigh_types, xyz, r)
        return 7

    def _collect_mace_results(self, atoms, mace_inputs, properties=None,
                              execution_graph_generation=0):
        assert execution_graph_generation == 7 and "stress" not in properties
        num_nodes, _, _, j_list, _, _, _, i_list = mace_inputs
        energies = np.asarray(self.evaluator.node_energies)[:num_nodes]
        pair = np.asarray(self.evaluator.node_forces).reshape((-1, 3))[:len(i_list)]
        forces = np.zeros((num_nodes, 3))
        for c in range(3):
            forces[:, c] = (np.bincount(j_list, weights=pair[:, c], minlength=num_nodes)
                            - np.bincount(i_list, weights=pair[:, c], minlength=num_nodes))
        if self.break_forces and num_nodes > 30:
            forces = forces * 1.05                  # a version whose internals differ
        return {"energy": energies.sum(), "energies": energies, "forces": forces}


class TestSymmetrixXLJoined:

    @pytest.fixture
    def mixed(self):
        from ase import Atoms
        rng = np.random.default_rng(5)
        out = []
        for n, box, symbols in ((12, 6.0, "OSi"), (20, 7.5, "CuO"), (9, 5.5, "Si"),
                                (16, 6.5, "OCuSi")):
            species = [symbols_i for symbols_i in
                       np.array(["O", "Cu", "Si"])[[("O", "Cu", "Si").index(x) for x in
                        _split(symbols)]][rng.integers(0, len(_split(symbols)), n)]]
            out.append(Atoms(species, positions=rng.uniform(0, box, (n, 3)),
                             cell=[box, box * 1.1, box * 0.9], pbc=True))
        return out

    def test_joined_equals_one_at_a_time(self, mixed, capsys):
        from amorphgen.utils.torchsim_symmetrix import batch_evaluator
        calc = FakeXL()
        evaluate, how = batch_evaluator(calc)
        assert how == "joined calls"
        joined = evaluate(mixed, ("energy", "forces", "stress"))
        assert "joined evaluation checked against a single call" in capsys.readouterr().out
        assert calc.tuned[0][0] == sum(len(a) for a in mixed)
        for atoms, result in zip(mixed, joined):
            single = FakeSymmetrix()
            single.calculate(atoms)
            assert result["energy"] == pytest.approx(single.results["energy"], rel=1e-12)
            assert np.allclose(result["forces"], single.results["forces"], atol=1e-10)
            assert np.allclose(result["stress"], single.results["stress"], atol=1e-12)

    def test_a_mismatch_with_single_calls_is_refused(self, mixed):
        """The one-off check catches a package whose internals do not match."""
        from amorphgen.utils.torchsim_symmetrix import batch_evaluator
        evaluate, _ = batch_evaluator(FakeXL(break_forces=True))
        with pytest.raises(RuntimeError, match="does not reproduce a single-structure"):
            evaluate(mixed, ("energy", "forces", "stress"))

    @pytest.mark.parametrize("attribute, value, reason", [
        ("_is_mace_les", True, "long-range"),
        ("_correction_pipeline", ["d3"], "dispersion"),
        ("alchemical", object(), "alchemical"),
    ])
    def test_models_that_cannot_be_joined_run_one_at_a_time(self, mixed, capsys, attribute,
                                                            value, reason):
        from amorphgen.utils.torchsim_symmetrix import batch_evaluator
        calc = FakeXL()
        setattr(calc, attribute, value)
        evaluate, how = batch_evaluator(calc)
        assert how == "one structure per call" and reason in capsys.readouterr().out
        assert len(evaluate(mixed[:2])) == 2 and calc.evaluator.calls == 2


@dataclasses.dataclass
class _PendingExplicitSegment:
    """What Symmetrix-XL calls one structure of a prepared group."""
    positions: np.ndarray
    cell: np.ndarray
    node_types: np.ndarray
    receivers: np.ndarray
    sources: np.ndarray
    shifts: np.ndarray
    cache_generation: int
    source_key: tuple = None
    geometry_deferred: bool = False


class FakeXLGroup(FakeXL):
    """Stand-in with Symmetrix-XL's prepared route for a group: the graph comes
    from the pending segments (candidate pairs, cut off here), forces and
    stresses from the evaluator's reductions."""

    def __init__(self, takes_group=True):
        super().__init__()
        self.takes_group = takes_group
        self.prepared = 0                     # times the graph was prepared
        self._explicit_segment_group_cache = None
        self._seen = None
        self.evaluator._prepare_factorized_batch = lambda *a: None
        self.evaluator._reduce_atom_forces = self._reduce_atom_forces
        self.evaluator._reduce_segmented_stress = self._reduce_segmented_stress

    def _compute_explicit_segment_group(self, *args):       # looked up, not called
        raise AssertionError

    def _compute_mace(self, mace_inputs, atoms=None):
        segments = getattr(self, "_pending_explicit_segments", [])
        self._pending_explicit_segments = []
        if not segments or not self.takes_group:
            if segments:
                return 7            # evaluated something else: the caller must notice
            return super()._compute_mace(mace_inputs, atoms)
        num_nodes, node_types, _, _, _, xyz, r, _ = mace_inputs
        assert len(r) == sum(len(s.receivers) for s in segments) and len(xyz) == len(r)
        seen = [(s.source_key, s.cache_generation) for s in segments]
        if seen != self._seen:
            self._seen = seen
            self.prepared += 1
            self._explicit_segment_group_cache = types.SimpleNamespace(
                generation=11, edge_identities=[
                    np.column_stack((s.receivers, s.sources, s.shifts)).astype(np.int32)
                    for s in segments])
        energies, pairs, self._pair_blocks = [], [], []
        for s, ids in zip(segments, self._explicit_segment_group_cache.edge_identities):
            vectors = s.positions[ids[:, 1]] - s.positions[ids[:, 0]] + ids[:, 2:] @ s.cell
            radii = np.linalg.norm(vectors, axis=1)
            on = radii < self.cutoff
            n = len(s.positions)
            self.evaluator.compute_node_energies_forces(
                n, s.node_types.tolist(), np.bincount(ids[on, 0], minlength=n), ids[on, 1],
                s.node_types[ids[on, 1]].tolist(), vectors[on], radii[on])
            energies.append(np.asarray(self.evaluator.node_energies)[:n])
            pair = np.zeros((len(ids), 3))
            pair[on] = np.asarray(self.evaluator.node_forces).reshape(-1, 3)[:on.sum()]
            pairs.append(pair)
        self.evaluator.node_energies = np.concatenate(energies + [np.zeros(5)])
        self._pair = np.concatenate(pairs)
        return 11

    def _reduce_atom_forces(self, num_nodes, receivers, senders, generation=0):
        assert generation == 11 and len(receivers) == len(self._pair)
        forces = np.zeros((num_nodes, 3))
        for c in range(3):
            forces[:, c] = (np.bincount(senders, weights=self._pair[:, c], minlength=num_nodes)
                            - np.bincount(receivers, weights=self._pair[:, c],
                                          minlength=num_nodes))
        return forces.ravel()

    def _reduce_segmented_stress(self, volumes, edge_offsets, xyz):
        xyz = np.asarray(xyz).reshape(-1, 3)
        return np.array([(-self._pair[a:b].T @ xyz[a:b]) / volume for a, b, volume
                         in zip(edge_offsets[:-1], edge_offsets[1:], volumes)])


class TestPreparedGroup:
    """With trajectory keys the joined graph is kept by Symmetrix-XL between
    calls and only positions and cells are handed over."""

    @pytest.fixture
    def cells(self):
        from ase import Atoms
        rng = np.random.default_rng(11)
        out = []
        for n, box, symbols in ((14, 9.0, ("O", "Si")), (22, 9.5, ("Cu", "O")),
                                (18, 10.0, ("O", "Cu", "Si"))):
            out.append(Atoms(rng.choice(symbols, n), positions=rng.uniform(0, box, (n, 3)),
                             cell=[box, box * 1.1, box * 0.95], pbc=True))
        return out

    @staticmethod
    def same(structures, results, stress=True):
        for atoms, result in zip(structures, results):
            single = FakeSymmetrix()
            single.calculate(atoms)
            assert result["energy"] == pytest.approx(single.results["energy"], rel=1e-12)
            assert np.allclose(result["forces"], single.results["forces"], atol=1e-10)
            if stress:
                assert np.allclose(result["stress"], single.results["stress"], atol=1e-12)

    def test_equals_single_calls_and_keeps_the_graph(self, cells, capsys):
        from amorphgen.utils.torchsim_symmetrix import batch_evaluator
        calc = FakeXLGroup()
        evaluate, how = batch_evaluator(calc)
        keys = list(range(len(cells)))
        properties = ("energy", "forces", "stress")
        self.same(cells, evaluate(cells, properties, keys=keys))
        assert "checked against a single call" in capsys.readouterr().out
        assert calc.tuned[-1][2] == properties
        rng = np.random.default_rng(3)
        for _ in range(5):                               # small moves: the graph is kept
            for atoms in cells:
                atoms.positions += rng.normal(0, 0.01, atoms.positions.shape)
            self.same(cells, evaluate(cells, properties, keys=keys))
        assert calc.prepared == 1
        for atoms in cells:                              # a cell change keeps it too
            atoms.set_cell(atoms.cell.array * 1.002, scale_atoms=True)
        self.same(cells, evaluate(cells, properties, keys=keys))
        assert calc.prepared == 1
        cells[1].positions += rng.normal(0, 0.6, cells[1].positions.shape)   # beyond the skin
        self.same(cells, evaluate(cells, properties, keys=keys))
        assert calc.prepared == 2
        self.same(cells, evaluate(cells, ("energy", "forces"), keys=keys), stress=False)
        assert evaluate.routes == {"prepared group": 9, "host graph": 0}

    def test_without_keys_the_graph_is_built_per_call(self, cells):
        from amorphgen.utils.torchsim_symmetrix import batch_evaluator
        calc = FakeXLGroup()
        evaluate, _ = batch_evaluator(calc)
        self.same(cells, evaluate(cells, ("energy", "forces", "stress")))
        assert evaluate.routes == {"prepared group": 0, "host graph": 1} and calc.prepared == 0

    def test_one_structure_or_an_open_cell_uses_the_host_graph(self, cells):
        from amorphgen.utils.torchsim_symmetrix import batch_evaluator
        evaluate, _ = batch_evaluator(FakeXLGroup())
        self.same(cells[:1], evaluate(cells[:1], ("energy", "forces", "stress"), keys=[0]))
        cells[0].pbc = [True, True, False]
        results = evaluate(cells, ("energy", "forces"), keys=[0, 1, 2])
        self.same(cells, results, stress=False)
        assert evaluate.routes == {"prepared group": 0, "host graph": 2}

    def test_a_calculator_that_does_not_take_the_route_falls_back(self, cells, capsys):
        from amorphgen.utils.torchsim_symmetrix import batch_evaluator
        calc = FakeXLGroup(takes_group=False)
        evaluate, _ = batch_evaluator(calc)
        for _ in range(2):
            self.same(cells, evaluate(cells, ("energy", "forces", "stress"), keys=[0, 1, 2]))
        assert capsys.readouterr().out.count("did not take its prepared route") == 1
        assert evaluate.routes == {"prepared group": 0, "host graph": 2}

    def test_a_changed_private_function_falls_back_on_first_use(self, cells, capsys):
        """A package version whose private functions take other arguments."""
        from amorphgen.utils.torchsim_symmetrix import batch_evaluator
        calc = FakeXLGroup()
        calc.evaluator._reduce_atom_forces = lambda num_nodes: None      # new signature
        evaluate, _ = batch_evaluator(calc)
        for _ in range(2):
            self.same(cells, evaluate(cells, ("energy", "forces", "stress"), keys=[0, 1, 2]))
        out = capsys.readouterr().out
        assert out.count("failed on first use (TypeError") == 1
        assert evaluate.routes == {"prepared group": 0, "host graph": 2}
        assert calc._pending_explicit_segments == []

    def test_an_error_after_the_route_has_worked_is_raised(self, cells):
        from amorphgen.utils.torchsim_symmetrix import batch_evaluator
        calc = FakeXLGroup()
        evaluate, _ = batch_evaluator(calc)
        evaluate(cells, ("energy", "forces"), keys=[0, 1, 2])

        def broken(*args, **kwargs):
            raise RuntimeError("device lost")
        calc._compute_mace = broken
        with pytest.raises(RuntimeError, match="device lost"):
            evaluate(cells, ("energy", "forces", "stress"), keys=[0, 1, 2])

    def test_a_version_without_the_route_is_as_before(self, cells):
        from amorphgen.utils.torchsim_symmetrix import batch_evaluator
        evaluate, _ = batch_evaluator(FakeXL())
        self.same(cells, evaluate(cells, ("energy", "forces", "stress"), keys=[0, 1, 2]))
        assert evaluate.routes == {"prepared group": 0, "host graph": 1}


class TestSettings:

    def test_joined_call_settings(self):
        from amorphgen.utils.accelerator import AcceleratorError, AcceleratorSettings
        settings = AcceleratorSettings.from_config(
            {"engine": "auto", "batch_max_atoms": 8000, "batch_skin": 0.8})
        assert (settings.batch_max_atoms, settings.batch_skin) == (8000, 0.8)
        assert AcceleratorSettings.from_config("auto").batch_skin == 0.5
        for bad in (-0.1, "wide", True):
            with pytest.raises(AcceleratorError, match="batch_skin"):
                AcceleratorSettings.from_config({"engine": "auto", "batch_skin": bad})

    def test_there_is_no_thread_batching_option(self):
        """Batching is torch-sim's (--engine torchsim), not a separate flag."""
        from amorphgen.cli import _get_parser
        from amorphgen.utils.accelerator import AcceleratorError, AcceleratorSettings
        with pytest.raises(AcceleratorError, match="Unknown accelerator key"):
            AcceleratorSettings.from_config({"engine": "auto", "batch": 16})
        with pytest.raises(SystemExit):
            _get_parser().parse_args(["--batch-opt", "--accelerator-batch", "16"])


class TestXLOnly:
    """The joined evaluation is for Symmetrix-XL calculators."""

    def test_other_calculators_are_not_joined(self):
        from amorphgen.utils.torchsim_symmetrix import batch_evaluator
        evaluate, how = batch_evaluator(FakeSymmetrix())      # no _compute_mace
        assert how == "one structure per call"


class TestJitCacheHint:

    def test_message_names_the_fix(self, monkeypatch):
        import amorphgen.utils.calculators as calc_mod
        monkeypatch.setenv("SYMMETRIX_JIT_CACHE", "/shared/project/jit")
        error = RuntimeError("Execution JIT is unavailable: the JIT cache filesystem does "
                             "not support atomic no-replace rename; use a node-local cache")
        hint = calc_mod._symmetrix_jit_cache_hint(error)
        assert "/shared/project/jit" in hint and "SYMMETRIX_JIT_CACHE=" in hint
        assert "lock-fallback" in hint
        assert calc_mod._symmetrix_jit_cache_hint(ValueError("something else")) is None

    def test_loader_raises_the_hint(self, monkeypatch, tmp_path):
        import sys
        import types
        import amorphgen.utils.calculators as calc_mod

        class Symmetrix:
            def __init__(self, path, dtype="float32"):
                raise RuntimeError("JitPublicationUnsupported: the JIT cache filesystem "
                                   "does not support atomic no-replace rename")
        monkeypatch.setitem(sys.modules, "symmetrix", types.SimpleNamespace(Symmetrix=Symmetrix))
        model = tmp_path / "m.json"
        model.write_text("{}")
        with pytest.raises(RuntimeError, match="node-local disk"):
            calc_mod._load_symmetrix(str(model), default_dtype="float32")
