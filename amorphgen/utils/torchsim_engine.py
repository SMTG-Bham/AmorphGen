"""Optional batched relaxation engine built on torch-sim.

torch-sim (https://github.com/torchsim/torch-sim) relaxes many structures in
one batched MLIP call with automatic GPU memory management. AmorphGen uses it
for the ensemble modes only (``--batch-opt`` and ``--random-gen --relax``):
all structures are relaxed together instead of one after another through ASE.

Install with ``pip install "amorphgen[torchsim]"`` (needs Python >= 3.12 and a
CUDA GPU or CPU; Apple MPS is not supported by torch-sim). Supported models:
MACE foundation models and files, SevenNet checkpoints, Lennard-Jones
(single sigma/epsilon, used for tests). CHGNet and the Buckingham potential have
no torch-sim implementation; use the default ASE engine for those.

Inputs and outputs are plain ASE ``Atoms`` so every AmorphGen writer, log and
analysis path is unchanged.
"""
from __future__ import annotations

import os
import time

import numpy as np

_INSTALL = ("torch-sim is not installed. Install the optional extra with\n"
            "    pip install \"amorphgen[torchsim]\"\n"
            "(Python >= 3.12), or drop --engine torchsim to use the ASE engine.")


def _require():
    try:
        import torch_sim as ts  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_INSTALL) from exc


def resolve_torch_device(device: str):
    """AmorphGen device string -> torch.device usable by torch-sim."""
    import torch
    dev = (device or "auto").lower()
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "mps":
        raise ValueError("torch-sim does not support Apple MPS; use device cpu "
                         "or cuda, or the ASE engine.")
    return torch.device(dev)


def build_model(model: str, device: str = "auto", model_path: str | None = None,
                classical_params: dict | None = None, dtype: str = "float64"):
    """Build a torch-sim ModelInterface for an AmorphGen model name."""
    if model_path is not None and not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Custom MACE model file not found: {model_path}\n"
            "Please provide a valid path to a .model file.")
    _require()
    import torch
    from .calculators import MACE_FOUNDATION_MODELS, SEVENNET_MODELS, _ci_get
    dev = resolve_torch_device(device)
    tdtype = torch.float64 if str(dtype) == "float64" else torch.float32
    name = (model or "").lower()

    if name in ("lennard-jones", "lj"):
        from torch_sim.models.lennard_jones import LennardJonesModel
        cp = classical_params or {}
        params = cp.get("params", {})
        if not params:
            raise ValueError("Lennard-Jones needs classical_params with one pair "
                             "(torch-sim's LJ takes a single sigma/epsilon).")
        p = next(iter(params.values()))
        return LennardJonesModel(sigma=float(p["sigma"]), epsilon=float(p["epsilon"]),
                                 cutoff=float(cp.get("cutoff", 10.0)), device=dev,
                                 dtype=tdtype, compute_stress=True)

    if name in ("buckingham", "buck", "chgnet"):
        raise ValueError(f"'{model}' has no torch-sim implementation; use the ASE "
                         f"engine (drop --engine torchsim) for this model.")

    if model_path or name.startswith("mace") or _ci_get(MACE_FOUNDATION_MODELS, model) != model:
        from torch_sim.models.mace import MaceModel
        if model_path is not None:
            raw = model_path
        else:
            from mace.calculators.foundations_models import mace_mp
            resolved = _ci_get(MACE_FOUNDATION_MODELS, model)
            raw = mace_mp(model=resolved, return_raw_model=True, default_dtype=str(dtype),
                          device=str(dev))
        return MaceModel(model=raw, device=dev, dtype=tdtype, compute_stress=True)

    if name.startswith("7net") or name.startswith("sevennet") or _ci_get(SEVENNET_MODELS, model) != model:
        from torch_sim.models.sevennet import SevenNetModel
        ckpt = _ci_get(SEVENNET_MODELS, model)
        # SevenNet's torch-sim wrapper (sevenn.torchsim) runs in float32 only,
        # and multi-fidelity checkpoints (7net-mf-*) need a modal, as on the
        # ASE path (calculators.py: modal="mpa" = MPtrj + Alexandria, PBE).
        if tdtype == torch.float64:
            print("  [torch-sim] SevenNet runs in float32 (its wrapper accepts no other dtype)")
        modal = "mpa" if "mf" in str(ckpt) else None
        kw = {"modal": modal} if modal else {}
        return SevenNetModel(ckpt, device=dev, dtype=torch.float32, **kw)

    raise ValueError(f"Unknown model '{model}' for the torch-sim engine.")


_CELL_FILTERS = {
    "none": None,
    "cubic": ("unit", True),            # unit-cell filter, hydrostatic strain only
    "unitcellfilter": ("unit", False),
    "unit": ("unit", False),
    "frechetcellfilter": ("frechet", False),
    "frechet": ("frechet", False),
    "expcellfilter": ("frechet", False),
}


_OPTIMIZERS = {"fire": "fire", "lbfgs": "lbfgs", "bfgs": "bfgs",
               "gradient_descent": "gradient_descent", "gd": "gradient_descent"}


class _TorchSafetyBridge:
    """Carry per-input safety histories through torch-sim's dynamic batches.

    Local system indices are renumbered when converged systems are removed.
    A system extra is sliced, cloned and concatenated with the state, so the
    monitor and step number always belong to the original input structure.
    """

    _ID = "amorphgen_safety_id"

    def __init__(self, atoms_list, config=None, context="torch-sim"):
        from .safety import SafetyMonitor
        self.inputs = list(atoms_list)
        self.monitors = [SafetyMonitor(config, context=f"{context} system {i}")
                         for i in range(len(self.inputs))]
        self.steps = [0] * len(self.inputs)
        self.reference_calc = None
        for monitor, atoms in zip(self.monitors, self.inputs):
            monitor.check_geometry(atoms, step=0)

    def attach(self, state):
        import torch
        extras = getattr(state, "system_extras", None)
        if not isinstance(extras, dict):
            raise RuntimeError("torch-sim safety requires SimState.system_extras to "
                               "preserve structure identities; upgrade torch-sim-atomistic.")
        extras[self._ID] = torch.arange(len(self.inputs), device=state.device,
                                       dtype=torch.int64)
        return state

    def ids(self, state):
        extras = getattr(state, "system_extras", {})
        if self._ID not in extras:
            raise RuntimeError("torch-sim lost safety system identifiers during batching; "
                               "cannot safely continue. Upgrade torch-sim-atomistic.")
        ids = extras[self._ID].detach().cpu().numpy().reshape(-1)
        if len(ids) != state.n_systems or any(i < 0 or i >= len(self.inputs) for i in ids):
            raise RuntimeError("torch-sim returned invalid safety system identifiers")
        return [int(i) for i in ids]

    def atoms(self, state, results=None):
        import torch_sim as ts
        from ase.calculators.singlepoint import SinglePointCalculator
        ids = self.ids(state)
        atoms_list = ts.io.state_to_atoms(state)
        system_idx = state.system_idx.detach().cpu().numpy()
        masses = state.masses.detach().cpu().numpy()
        momenta = getattr(state, "momenta", None)
        momenta = None if momenta is None else momenta.detach().cpu().numpy()
        if results is not None:
            energies = results["energy"].detach().cpu().numpy().reshape(-1)
            forces = results["forces"].detach().cpu().numpy()
            stress = results.get("stress")
            stress = None if stress is None else stress.detach().cpu().numpy()
        for k, (atoms, original) in enumerate(zip(atoms_list, ids)):
            atoms.info.update(self.inputs[original].info)
            atoms.set_masses(masses[system_idx == k])
            if momenta is not None:
                atoms.set_momenta(momenta[system_idx == k], apply_constraint=False)
            if results is not None:
                props = {"energy": float(energies[k]), "forces": forces[system_idx == k]}
                if results.get("free_energy") is not None:
                    props["free_energy"] = float(
                        results["free_energy"].detach().cpu().numpy().reshape(-1)[k])
                if stress is not None:
                    props["stress"] = stress[k]
                atoms.calc = SinglePointCalculator(atoms, **props)
                base = results.get("_amorphgen_base_results")
                if base is not None:
                    base_props = {
                        "energy": float(base["energy"].detach().cpu().numpy().reshape(-1)[k]),
                        "forces": base["forces"].detach().cpu().numpy()[system_idx == k],
                    }
                    if base.get("stress") is not None:
                        base_props["stress"] = base["stress"].detach().cpu().numpy()[k]
                    if base.get("free_energy") is not None:
                        base_props["free_energy"] = float(
                            base["free_energy"].detach().cpu().numpy().reshape(-1)[k])
                    atoms.calc.base_calculator = SinglePointCalculator(atoms, **base_props)
        return atoms_list

    def check(self, state, results=None, geometry_only=False):
        if results is None and not geometry_only:
            results = {key: getattr(state, key) for key in ("energy", "forces")}
            if getattr(state, "stress", None) is not None:
                results["stress"] = state.stress
        frames = self.atoms(state, results)
        for original, atoms in zip(self.ids(state), frames):
            monitor = self.monitors[original]
            if geometry_only:
                monitor.check_geometry(atoms, step=self.steps[original])
            else:
                # The structures have independent histories, but one reference
                # model suffices for their sequential spot checks.
                if self.reference_calc is not None:
                    monitor.reference_calc = self.reference_calc
                monitor.check(atoms, step=self.steps[original])
                if monitor.reference_calc is not None:
                    self.reference_calc = monitor.reference_calc
        return frames

    def advance(self, state):
        for original in self.ids(state):
            self.steps[original] += 1

    def wrap_model(self, model):
        from torch_sim.models.interface import ModelInterface
        bridge = self

        class GuardedModel(ModelInterface):
            def __init__(self):
                super().__init__()
                self.inner = model
                self._device, self._dtype = model.device, model.dtype
                self._compute_forces = model.compute_forces
                self._compute_stress = model.compute_stress
                self._memory_scales_with = model.memory_scales_with

            def forward(self, state, **kwargs):
                bridge.check(state, geometry_only=True)
                results = self.inner(state, **kwargs)
                checked = dict(results)
                base = getattr(self.inner, "last_base_results", None)
                if base is not None:
                    checked["_amorphgen_base_results"] = base
                bridge.check(state, checked)
                return results

        return GuardedModel()


def batch_relax(atoms_list, model, fmax: float = 0.01, max_steps: int = 1000,
                cell_filter: str = "cubic", optimizer: str = "lbfgs",
                pressure_tol_gpa: float = 0.02,
                autobatch: bool | None = None, log=print,
                safety=None, repulsive_core=None):
    """Relax a list of ASE Atoms together with torch-sim.

    ``optimizer`` is ``lbfgs`` (default, matches AmorphGen's ASE default),
    ``fire``, ``bfgs`` or ``gradient_descent``. With a cell filter the
    convergence test also requires |pressure| < ``pressure_tol_gpa`` (default
    0.02 GPa), so the returned cells are at zero pressure like the ASE path.
    torch-sim's own cell-force criterion (stress x volume / N atoms) allows
    ~0.5 GPa residuals for hundreds of atoms.

    Returns a list of relaxed ASE Atoms (same order) carrying a
    SinglePointCalculator with the final energy and forces.
    """
    _require()
    import torch
    import torch_sim as ts
    from torch_sim.optimizers import OPTIM_REGISTRY
    from .repulsion import wrap_torch_model

    atoms_list = list(atoms_list)
    if not atoms_list:
        return []
    guard = _TorchSafetyBridge(atoms_list, safety, "torch-sim relaxation")
    model = guard.wrap_model(wrap_torch_model(model, repulsive_core))

    key = str(cell_filter or "none").lower()
    if key not in _CELL_FILTERS:
        raise ValueError(f"cell_filter '{cell_filter}' not supported by the torch-sim "
                         f"engine; choose one of {sorted(_CELL_FILTERS)}")
    cf = _CELL_FILTERS[key]
    init_kwargs = {}
    if cf is not None:
        init_kwargs = {"cell_filter": ts.CellFilter(cf[0]), "hydrostatic_strain": cf[1]}

    device = getattr(model, "device", torch.device("cpu"))
    if autobatch is None:
        autobatch = str(device).startswith("cuda")
    p_tol = float(pressure_tol_gpa) / 160.21766208          # GPa -> eV/A^3

    def conv(state, last_energy=None):
        ok = ts.system_wise_max_force(state) < float(fmax)
        if cf is not None:
            stress = getattr(state, "stress", None)
            if stress is None:
                raise ValueError("cell relaxation requested but the state has no stress")
            pressure = -torch.diagonal(stress, dim1=1, dim2=2).mean(dim=1)
            ok = ok & (pressure.abs() < p_tol)
        return ok
    opt_key = _OPTIMIZERS.get(str(optimizer or "lbfgs").lower())
    if opt_key is None:
        raise ValueError(f"optimizer '{optimizer}' not available in the torch-sim engine; "
                         f"choose one of {sorted(set(_OPTIMIZERS))}")
    ts_opt = ts.Optimizer(opt_key)
    init_fn, step_fn = OPTIM_REGISTRY[ts_opt]

    def checked_init(state, model, **kwargs):
        state = init_fn(state=state, model=model, **kwargs)
        guard.check(state)
        return state

    def checked_step(state, model, **kwargs):
        guard.advance(state)
        state = step_fn(state=state, model=model, **kwargs)
        guard.check(state)
        return state

    n = len(atoms_list)
    log(f"[torch-sim] {opt_key.upper()} relaxation of {n} structure(s) in one batch on "
        f"{device}; cell filter: {key}"
        + (f" (|P| < {pressure_tol_gpa} GPa required)" if cf is not None else "")
        + f"; fmax = {fmax} eV/A; max_steps = {max_steps}")
    t0 = time.time()
    state = guard.attach(ts.initialize_state(atoms_list, model.device, model.dtype))
    state = ts.optimize(system=state, model=model,
                        optimizer=(checked_init, checked_step), convergence_fn=conv,
                        max_steps=int(max_steps), autobatcher=bool(autobatch),
                        init_kwargs=init_kwargs)
    # final energies / forces for the log and the written files
    res = model(state)
    out = guard.check(state, res)
    if guard.ids(state) != list(range(len(atoms_list))):
        raise RuntimeError("torch-sim failed to restore the original structure order")
    for a in out:
        a.info["max_force"] = float(np.abs(a.get_forces(apply_constraint=False)).max())
    dt = time.time() - t0
    log(f"[torch-sim] done in {dt:.1f} s ({dt / max(n, 1):.2f} s per structure); "
        f"max|F| range {min(a.info['max_force'] for a in out):.3f} - "
        f"{max(a.info['max_force'] for a in out):.3f} eV/A")
    return out


def estimate_batch_size(model, atoms_list, fraction: float = 0.5, md: bool = False,
                        fallback: int = 16, log=print, safety=None,
                        repulsive_core=None) -> int:
    """Number of structures per chunk that fits the GPU, from a one-structure probe.

    Runs one forward pass (or one short MD step when ``md``) on the largest
    structure, reads the peak CUDA memory it needed, and returns
    ``fraction * total_memory / peak`` (the rest is headroom for optimiser
    state, neighbour lists and fragmentation). On CPU returns ``fallback``.
    """
    _require()
    import torch
    import torch_sim as ts
    dev = getattr(model, "device", torch.device("cpu"))
    atoms_list = list(atoms_list)
    if not atoms_list:
        return fallback
    big = max(atoms_list, key=len)
    guard = _TorchSafetyBridge([big], safety, "torch-sim memory probe")
    if not str(dev).startswith("cuda"):
        return fallback
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    base = torch.cuda.memory_allocated(dev)
    if md:
        from .torchsim_md import batch_nvt
        batch_nvt([big], model, 300.0, n_steps=2, safety=safety,
                  repulsive_core=repulsive_core, log=lambda *args: None)
    else:
        from .repulsion import wrap_torch_model
        guarded_model = guard.wrap_model(wrap_torch_model(model, repulsive_core))
        state = guard.attach(ts.io.atoms_to_state([big], device=dev, dtype=model.dtype))
        guarded_model(state)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(dev) - base
    total = torch.cuda.get_device_properties(dev).total_memory
    n = int(max(1, (fraction * total) // max(peak, 1)))
    n = min(n, max(1, len(atoms_list)))
    log(f"[torch-sim] memory probe: {len(big)} atoms need {peak / 2**30:.2f} GiB "
        f"({'MD' if md else 'forces'}); GPU has {total / 2**30:.0f} GiB -> chunks of {n}")
    torch.cuda.empty_cache()
    return n
