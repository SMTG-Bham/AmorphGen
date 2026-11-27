"""
amorphgen.utils.torchsim_symmetrix
----------------------------------
Symmetrix-XL as a torch-sim model, so that

    amorphgen ... --engine torchsim --accelerator symmetrix-xl

runs torch-sim's batched optimisers and integrators with the forces of the
whole batch coming from one Symmetrix-XL call instead of from PyTorch MACE.

Joined evaluation (:func:`batch_evaluator`): the structures of a batch are
concatenated into one disconnected neighbour graph. Message passing only runs
along edges, so the structures do not interact, and one Symmetrix-XL call
returns all of them (:func:`joined_symmetrix_evaluator`). The original
Symmetrix (wcwitt/symmetrix) is not supported under torch-sim; it runs with
the ASE engine.

torch-sim asks a model for the energies, forces and stresses of a batched
state. This module answers that request: it turns the state into ASE
structures, evaluates them together, and returns tensors on the state's
device. Everything else (chunking, checkpoints, safety checks, the repulsive
core) is the torch-sim engine's and is unchanged.

The neighbour lists are kept on the host between calls and the joined graph
on the device (rebuilt only when atoms have moved further than the skin
allows); the positions still make a GPU -> CPU -> GPU round trip per call.
"""

from __future__ import annotations


def run_pending_check(calc, atoms) -> None:
    """Run a check against MACE that is still waiting for its first structure.

    Joined evaluations do not go through ``calc.calculate``, where the check
    is hooked (``accelerator._validate_on_first_use``), so trigger it here.
    """
    if "calculate" in vars(calc):
        probe = atoms.copy()
        probe.calc = calc
        probe.get_potential_energy()


def _drive_xl(calc):
    """One evaluator call of Symmetrix-XL on a host-built graph, with the same
    calls its own ``Symmetrix.calculate`` makes on that path."""
    import numpy as np
    evaluator = calc.evaluator

    def drive(num_nodes, node_types, i_list, j_list, xyz, r, want_stress):
        properties = ["energy", "forces"] + (["stress"] if want_stress else [])
        tune = getattr(calc, "_apply_kernel_launch_tuning", None)
        if callable(tune):
            tune(num_nodes, len(i_list), properties)
        inputs = (num_nodes, node_types.tolist(),
                  np.bincount(i_list, minlength=num_nodes), j_list,
                  node_types[j_list].tolist(), xyz, r, i_list)
        generation = calc._compute_mace(inputs, atoms=None)
        collected = calc._collect_mace_results(None, inputs, ["energy", "forces"],
                                               generation)
        pair = None
        if want_stress:
            pair = np.array(evaluator.node_forces, dtype=float).reshape((-1, 3))
            pair = pair[:len(i_list)]
        return (np.asarray(collected["energies"], dtype=float)[:num_nodes],
                np.asarray(collected["forces"], dtype=float), pair)

    return drive


def _xl_group_route(calc, neighbors):
    """Symmetrix-XL's prepared route for a group of structures, or None when
    this package version does not have it.

    Symmetrix-XL can keep the joined graph of a group of structures on the
    device and take only new positions (and cells) at each call; the graph is
    prepared again only when a neighbour list of the group has been rebuilt.
    Handing it a new host-built graph at every call instead makes it prepare
    the graph every time, which costs about ten times the evaluation itself
    (A100, 16 x 80 atoms: 80 ms against 8 ms per joined call).

    The route is reached through the calculator's pending segments, one per
    structure, built here from AmorphGen's own skin neighbour lists
    (candidate pairs within cutoff + skin; the device applies the cutoff).
    Returns ``route(structures, keys, node_types, offsets, want_stress)``,
    which gives the result dicts, or None when a structure cannot take part
    or the calculator did not take the route (the caller then uses the
    host-built graph).
    """
    import importlib

    import numpy as np
    from ase.stress import full_3x3_to_voigt_6_stress

    try:
        module = importlib.import_module(type(calc).__module__)
    except ImportError:
        return None
    segment_class = getattr(module, "_PendingExplicitSegment", None)
    evaluator = calc.evaluator
    reduce_forces = getattr(evaluator, "_reduce_atom_forces", None)
    reduce_stress = getattr(evaluator, "_reduce_segmented_stress", None)
    if (segment_class is None or neighbors.skin <= 0
            or not hasattr(calc, "_compute_explicit_segment_group")
            or not callable(reduce_forces) or not callable(reduce_stress)
            or not callable(getattr(evaluator, "_prepare_factorized_batch", None))):
        return None
    tune = getattr(calc, "_apply_kernel_launch_tuning", None)
    state = {"identities": None, "arrays": None, "off": False, "proven": False}

    def edge_arrays(identities, offsets):
        """Receivers, senders and edge offsets of the prepared graph; they
        change only when Symmetrix-XL has rebuilt its graph."""
        kept = state["identities"]
        if (kept is None or len(kept) != len(identities)
                or any(a is not b for a, b in zip(kept, identities))):
            state["identities"] = list(identities)
            state["arrays"] = (
                np.ascontiguousarray(np.concatenate(
                    [ids[:, 0] + o for ids, o in zip(identities, offsets)]), dtype=np.int32),
                np.ascontiguousarray(np.concatenate(
                    [ids[:, 1] + o for ids, o in zip(identities, offsets)]), dtype=np.int32),
                np.concatenate(([0], np.cumsum([len(ids) for ids in identities]))
                               ).astype(np.int64))
        return state["arrays"]

    def route(structures, keys, node_types, offsets, want_stress):
        if state["off"] or len(structures) < 2:
            return None
        if state["proven"]:
            return prepared(structures, keys, node_types, offsets, want_stress)
        # Until the route has worked once, an error in it is taken as a package
        # version whose internals differ (these are private functions), and the
        # run continues with the host-built graph. Later errors are real ones.
        try:
            results = prepared(structures, keys, node_types, offsets, want_stress)
        except Exception as error:      # noqa: BLE001 - any failure of a private API
            state["off"] = True
            calc._pending_explicit_segments = []
            print("[Accelerator] Symmetrix-XL's prepared route for the batch failed "
                  f"on first use ({type(error).__name__}: {error}); this package "
                  "version is probably newer than the ones AmorphGen was tested "
                  "with. Continuing with a host-built graph per call (slower)")
            return None
        state["proven"] = results is not None
        return results

    def prepared(structures, keys, node_types, offsets, want_stress):
        segments = []
        for atoms, key, start, stop in zip(structures, keys, offsets[:-1], offsets[1:]):
            topology = neighbors.topology(key, atoms)
            if topology is None:
                return None
            first, second, shifts, build = topology
            segments.append(segment_class(
                positions=np.ascontiguousarray(atoms.positions, dtype=float),
                cell=np.ascontiguousarray(atoms.cell.array, dtype=float),
                node_types=np.ascontiguousarray(node_types[start:stop], dtype=np.int32),
                receivers=first, sources=second, shifts=shifts,
                cache_generation=int(build), source_key=("amorphgen", key),
                geometry_deferred=False))
        num_nodes = int(offsets[-1])
        n_edges = sum(len(segment.receivers) for segment in segments)
        properties = ["energy", "forces"] + (["stress"] if want_stress else [])
        if callable(tune):
            tune(num_nodes, n_edges, properties)
        # the graph and its geometry come from the segments: the edge arrays
        # of the plain inputs are only held to their length
        calc._pending_explicit_segments = segments
        types = np.ascontiguousarray(node_types, dtype=np.int32)
        generation = calc._compute_mace(
            (num_nodes, types, None, None, None, np.empty((n_edges, 3)),
             np.empty(n_edges), None), atoms=None)
        group = getattr(calc, "_explicit_segment_group_cache", None)
        if group is None or group.generation != generation:
            state["off"] = True
            print("[Accelerator] Symmetrix-XL did not take its prepared route for the "
                  "batch; continuing with a host-built graph per call (slower)")
            return None
        identities = group.edge_identities
        receivers, senders, edge_offsets = edge_arrays(identities, offsets)
        node_energies = np.array(evaluator.node_energies, dtype=np.float64)[:num_nodes]
        forces = np.asarray(reduce_forces(num_nodes, receivers, senders, generation),
                            dtype=float).reshape((num_nodes, 3))
        stress = None
        if want_stress:
            vectors = np.ascontiguousarray(np.concatenate([
                segment.positions[ids[:, 1]] - segment.positions[ids[:, 0]]
                + ids[:, 2:] @ segment.cell
                for segment, ids in zip(segments, identities)]), dtype=np.float64)
            volumes = np.array([abs(atoms.get_volume()) for atoms in structures],
                               dtype=np.float64)
            stress = np.asarray(reduce_stress(volumes, edge_offsets, vectors.reshape(-1)),
                                dtype=float).reshape((len(structures), 3, 3))
        results = []
        for k in range(len(structures)):
            nodes = slice(offsets[k], offsets[k + 1])
            energies = node_energies[nodes].copy()
            result = {"energies": energies, "energy": float(energies.sum()),
                      "free_energy": float(energies.sum()),
                      "forces": forces[nodes].copy()}
            if want_stress:
                result["stress"] = full_3x3_to_voigt_6_stress(stress[k])
            results.append(result)
        return results

    return route


def _xl_needs_single_calls(calc) -> str | None:
    """Why this Symmetrix-XL calculator cannot be joined (long-range and
    field-coupled models, corrections), or None when it can."""
    if getattr(calc, "alchemical", None) is not None:
        return "alchemical model"
    if getattr(calc, "_is_mace_les", False):
        return "long-range (LES) model"
    if getattr(calc, "_family_frontend_adapter", None) is not None:
        return "model family with its own front end"
    field = getattr(calc, "_has_native_field_coupling", None)
    if callable(field) and field():
        return "field-coupled model"
    if getattr(calc, "_correction_pipeline", None):
        return "dispersion correction"
    return None


def joined_symmetrix_evaluator(calc, drive=None, verify: bool = True,
                               skin: float | None = None):
    """``evaluate(structures, properties, keys=None)`` in one Symmetrix-XL call.

    Symmetrix-XL has no batched call, but it evaluates a neighbour graph, so
    the graphs of all *structures* are concatenated (atom indices offset per
    structure, hence no edge between two structures) and evaluated once;
    energies are summed per structure, forces reduced per atom and the stress
    per structure from its own pair forces. These are the steps of
    ``Symmetrix.calculate`` applied to the joined graph.

    With ``verify`` the first joined call of two or more structures is checked
    against a single-structure evaluation by the calculator itself, so a
    package version whose internals differ is caught at once.

    ``keys`` name the trajectory each structure belongs to; with them the
    neighbour lists are kept between calls and rebuilt only when atoms have
    moved more than the ``skin`` allows (:mod:`amorphgen.utils.neighbor_cache`;
    ``skin=0`` builds a fresh list every time). With keys, a skin and fully
    periodic cells, the joined graph is also kept on the device between calls
    (:func:`_xl_group_route`); ``evaluate.routes`` counts the calls of each kind.
    """
    import numpy as np
    from ase.calculators.calculator import all_changes
    from ase.stress import full_3x3_to_voigt_6_stress

    from .neighbor_cache import DEFAULT_SKIN, NeighborCache

    own_driver = drive is None
    if own_driver:
        drive = _drive_xl(calc)
    evaluator, cutoff = calc.evaluator, calc.cutoff
    model_numbers = [int(z) for z in evaluator.atomic_numbers]
    type_of = np.full(max(model_numbers) + 1, -1, dtype=np.int64)
    type_of[model_numbers] = np.arange(len(model_numbers))
    checked = [not verify]
    neighbors = NeighborCache(cutoff, DEFAULT_SKIN if skin is None else skin)
    # with the default driver only: a caller's own driver evaluates host graphs
    group_route = _xl_group_route(calc, neighbors) if own_driver else None
    routes = {"prepared group": 0, "host graph": 0}

    def joined(structures, want_stress, keys):
        counts = np.array([len(atoms) for atoms in structures])
        offsets = np.concatenate(([0], np.cumsum(counts)))
        num_nodes = int(offsets[-1])
        numbers = np.concatenate([atoms.numbers for atoms in structures])
        known = numbers < len(type_of)
        node_types = np.where(known, type_of[np.where(known, numbers, 0)], -1)
        if (node_types < 0).any():
            from ase.data import chemical_symbols
            missing = sorted({chemical_symbols[z] for z in numbers[node_types < 0]})
            raise ValueError(
                f"The converted Symmetrix model has no {', '.join(missing)}; it was "
                f"converted for {', '.join(chemical_symbols[z] for z in model_numbers)}. "
                f"Convert it again with every element of the run (accelerator.elements).")

        if group_route is not None:
            if want_stress and not all(atoms.cell.rank == 3 for atoms in structures):
                raise ValueError("Stress requires three-dimensional cells.")
            results = group_route(structures, keys, node_types, offsets, want_stress)
            if results is not None:
                routes["prepared group"] += 1
                return results
        routes["host graph"] += 1

        i_parts, j_parts, r_parts, xyz_parts, edge_counts, full = [], [], [], [], [], []
        for atoms, offset, key in zip(structures, offsets, keys):
            full_cell = atoms.cell.rank == 3          # the stress needs it
            i_list, j_list, r, xyz = neighbors.pairs(key, atoms)
            i_parts.append(np.asarray(i_list) + offset)
            j_parts.append(np.asarray(j_list) + offset)
            r_parts.append(r)
            xyz_parts.append(xyz)
            edge_counts.append(len(i_list))
            full.append(full_cell)
        if want_stress and not all(full):
            raise ValueError("Stress requires three-dimensional cells.")
        i_list = np.ascontiguousarray(np.concatenate(i_parts), dtype=np.int32)
        j_list = np.ascontiguousarray(np.concatenate(j_parts), dtype=np.int32)
        r = np.ascontiguousarray(np.concatenate(r_parts))
        xyz = np.ascontiguousarray(np.concatenate(xyz_parts))
        edges = np.concatenate(([0], np.cumsum(edge_counts)))

        node_energies, forces, pair = drive(num_nodes, node_types, i_list, j_list,
                                            xyz, r, want_stress)
        if forces is None:
            forces = np.empty((num_nodes, 3))
            for axis in range(3):
                forces[:, axis] = (
                    np.bincount(j_list, weights=pair[:, axis], minlength=num_nodes)
                    - np.bincount(i_list, weights=pair[:, axis], minlength=num_nodes))

        results = []
        for k, atoms in enumerate(structures):
            nodes = slice(offsets[k], offsets[k + 1])
            pairs = slice(edges[k], edges[k + 1])
            energies = node_energies[nodes].copy()
            result = {"energies": energies, "energy": float(energies.sum()),
                      "free_energy": float(energies.sum()),
                      "forces": forces[nodes].copy()}
            if want_stress:
                result["stress"] = full_3x3_to_voigt_6_stress(
                    (-pair[pairs].T @ xyz[pairs]) / atoms.get_volume())
            results.append(result)
        return results

    def verify_once(structures, results, want_stress):
        """Hold the joined result of the last structure to the calculator's own."""
        checked[0] = True
        atoms = structures[-1]
        names = ["energy", "forces"] + (["stress"] if want_stress else [])
        calc.calculate(atoms.copy(), names, all_changes)
        own = calc.results
        d_energy = abs(float(own["energy"]) - results[-1]["energy"]) / len(atoms)
        d_force = float(np.abs(np.asarray(own["forces"]) - results[-1]["forces"]).max())
        d_stress = (float(np.abs(np.asarray(own["stress"]) - results[-1]["stress"]).max())
                    if want_stress else 0.0)
        print(f"[Accelerator] joined evaluation checked against a single call "
              f"({len(structures)} structures): dE {d_energy * 1e3:.4f} meV/atom, "
              f"max dF {d_force * 1e3:.4f} meV/A, max dS {d_stress * 160.21766:.5f} GPa")
        if d_energy > 1e-4 or d_force > 2e-3 or d_stress > 2e-4:
            raise RuntimeError(
                "The joined Symmetrix evaluation does not reproduce a single-structure "
                f"evaluation (dE {d_energy * 1e3:.3f} meV/atom, max dF {d_force * 1e3:.3f} "
                f"meV/A, max dS {d_stress * 160.21766:.4f} GPa). This Symmetrix version is "
                "not supported for joined evaluation: run with the ASE engine (drop "
                "--engine torchsim), and report the package version.")

    def evaluate(structures, properties=("energy", "forces", "stress"), keys=None):
        structures = list(structures)
        if not structures:
            return []
        want_stress = "stress" in properties
        results = joined(structures, want_stress,
                         [None] * len(structures) if keys is None else keys)
        if not checked[0] and len(structures) > 1:
            verify_once(structures, results, want_stress)
        return results

    evaluate.neighbors = neighbors
    evaluate.routes = routes
    return evaluate


def batch_evaluator(calc):
    """``evaluate(structures, properties, keys=None)`` -> one result dict per structure.

    * Symmetrix-XL: one joined call (:func:`joined_symmetrix_evaluator`);
    * any other ASE calculator, and Symmetrix-XL models that cannot be joined
      (long-range, field-coupled, with a dispersion correction): one structure
      per call.

    Returns the function and a short description of how it evaluates.
    """
    info = getattr(calc, "amorphgen_accelerator", None)
    skin = info.get("batch_skin") if isinstance(info, dict) else None
    if all(hasattr(calc, name) for name in ("evaluator", "cutoff", "_compute_mace")):
        reason = _xl_needs_single_calls(calc)
        if reason is None and hasattr(calc, "_collect_mace_results"):
            return joined_symmetrix_evaluator(calc, skin=skin), "joined calls"
        print(f"[Accelerator] Symmetrix-XL evaluates one structure per call here "
              f"({reason or 'unsupported package version'})")
    import copy

    import numpy as np
    from ase.calculators.calculator import all_changes

    def one_at_a_time(structures, properties=("energy", "forces", "stress"), keys=None):
        results = []
        for atoms in structures:
            calc.calculate(atoms.copy(), list(properties), all_changes)
            results.append({key: np.array(value, copy=True)
                            if isinstance(value, np.ndarray) else copy.copy(value)
                            for key, value in calc.results.items()})
        return results
    return one_at_a_time, "one structure per call"


DEFAULT_BATCH = 32      # structures per chunk when the torch-sim batch size is "auto"


def symmetrix_torchsim_model(calc, device, dtype, *, batch: int = DEFAULT_BATCH,
                             batch_atoms: int | None = None):
    """Wrap an ASE calculator (Symmetrix) as a torch-sim ``ModelInterface``.

    Parameters
    ----------
    calc : ASE calculator
        A Symmetrix-XL calculator is evaluated in joined calls
        (:func:`batch_evaluator`); any other ASE calculator works too, one
        structure per call (used by the tests).
    device, dtype : torch.device, torch.dtype
        Where torch-sim keeps the state, and the precision it runs in.
    batch : int
        Structures per chunk when the batch size is ``auto``.
    batch_atoms : int, optional
        Upper bound on the atoms in one joined call.
    """
    import numpy as np
    import torch
    import torch_sim as ts
    from ase.stress import voigt_6_to_full_3x3_stress
    from torch_sim.models.interface import ModelInterface

    evaluate_batch, how = batch_evaluator(calc)
    properties = ("energy", "forces", "stress")

    def evaluate(structures):
        # The place in the batch names the trajectory for the neighbour-list
        # cache: stable through an MD chunk; when torch-sim regroups the batch
        # a list is rebuilt, which is its only cost.
        keys = [("torch-sim", index) for index in range(len(structures))]
        if not batch_atoms:
            return evaluate_batch(structures, properties, keys=keys)
        results, start, size = [], 0, 0
        for index, atoms in enumerate(structures):
            if index > start and size + len(atoms) > batch_atoms:
                results += evaluate_batch(structures[start:index], properties,
                                          keys=keys[start:index])
                start, size = index, 0
            size += len(atoms)
        return results + evaluate_batch(structures[start:], properties, keys=keys[start:])

    class SymmetrixModel(ModelInterface):
        """Energies, forces and stresses of a batched state from one joined call."""

        # read by the torch-sim engine instead of probing PyTorch's GPU memory,
        # which says nothing about what Symmetrix needs
        amorphgen_batch = int(batch)
        amorphgen_batch_atoms = batch_atoms
        evaluation = how
        neighbors = getattr(evaluate_batch, "neighbors", None)
        routes = getattr(evaluate_batch, "routes", None)

        def __init__(self):
            super().__init__()
            self.calculator = calc
            self._device, self._dtype = device, dtype
            self._compute_forces = True
            self._compute_stress = True
            self._memory_scales_with = "n_atoms"
            self.calls = []                      # (structures, atoms) per call

        def forward(self, state, **kwargs):
            structures = ts.io.state_to_atoms(state)
            run_pending_check(calc, structures[0])
            results = evaluate(structures)
            self.calls.append((len(structures), sum(len(a) for a in structures)))

            def tensor(values):
                return torch.as_tensor(np.asarray(values), device=self._device,
                                       dtype=self._dtype)
            return {
                "energy": tensor([float(r["energy"]) for r in results]),
                "forces": tensor(np.concatenate([r["forces"] for r in results])),
                "stress": tensor([voigt_6_to_full_3x3_stress(r["stress"])
                                  for r in results]),
            }

    return SymmetrixModel()
