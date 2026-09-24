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
        if model_path and os.path.isfile(model_path):
            raw = model_path
        else:
            from mace.calculators.foundations_models import mace_mp
            resolved = _ci_get(MACE_FOUNDATION_MODELS, model)
            raw = mace_mp(model=resolved, return_raw_model=True, default_dtype=str(dtype),
                          device=str(dev))
        return MaceModel(model=raw, device=dev, dtype=tdtype, compute_stress=True)

    if name.startswith("7net") or name.startswith("sevennet") or _ci_get(SEVENNET_MODELS, model) != model:
        from torch_sim.models.sevennet import SevenNetModel
        from sevenn.calculator import SevenNetCalculator
        ckpt = _ci_get(SEVENNET_MODELS, model)
        calc = SevenNetCalculator(model=ckpt, device=str(dev))
        return SevenNetModel(model=calc.model, device=dev, dtype=tdtype)

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


def batch_relax(atoms_list, model, fmax: float = 0.01, max_steps: int = 1000,
                cell_filter: str = "cubic", optimizer: str = "lbfgs",
                pressure_tol_gpa: float = 0.02,
                autobatch: bool | None = None, log=print):
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
    from ase.calculators.singlepoint import SinglePointCalculator

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

    n = len(atoms_list)
    log(f"[torch-sim] {opt_key.upper()} relaxation of {n} structure(s) in one batch on "
        f"{device}; cell filter: {key}"
        + (f" (|P| < {pressure_tol_gpa} GPa required)" if cf is not None else "")
        + f"; fmax = {fmax} eV/A; max_steps = {max_steps}")
    t0 = time.time()
    state = ts.optimize(system=list(atoms_list), model=model,
                        optimizer=ts_opt, convergence_fn=conv,
                        max_steps=int(max_steps), autobatcher=bool(autobatch),
                        init_kwargs=init_kwargs)
    out = ts.io.state_to_atoms(state)
    # final energies / forces for the log and the written files
    res = model(ts.io.atoms_to_state(out, device=device, dtype=model.dtype))
    energies = res["energy"].detach().cpu().numpy().reshape(-1)
    forces_all = res["forces"].detach().cpu().numpy()
    start = 0
    for k, a in enumerate(out):
        f = forces_all[start:start + len(a)]; start += len(a)
        a.calc = SinglePointCalculator(a, energy=float(energies[k]), forces=f)
        a.info["max_force"] = float(np.abs(f).max())
    dt = time.time() - t0
    log(f"[torch-sim] done in {dt:.1f} s ({dt / max(n, 1):.2f} s per structure); "
        f"max|F| range {min(a.info['max_force'] for a in out):.3f} - "
        f"{max(a.info['max_force'] for a in out):.3f} eV/A")
    return out


def estimate_batch_size(model, atoms_list, fraction: float = 0.5, md: bool = False,
                        fallback: int = 16, log=print) -> int:
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
    if not str(dev).startswith("cuda"):
        return fallback
    big = max(atoms_list, key=len)
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    base = torch.cuda.memory_allocated(dev)
    state = ts.io.atoms_to_state([big], device=dev, dtype=model.dtype)
    if md:
        from torch_sim.integrators.nvt import nvt_langevin_init
        st = nvt_langevin_init(state, model, kT=300 * 8.617330337217213e-05)
        ts.integrate(system=st, model=model, integrator=ts.Integrator.nvt_langevin,
                     n_steps=2, temperature=300.0, timestep=0.0005, gamma=10.0)
    else:
        model(state)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(dev) - base
    total = torch.cuda.get_device_properties(dev).total_memory
    n = int(max(1, (fraction * total) // max(peak, 1)))
    n = min(n, max(1, len(atoms_list)))
    log(f"[torch-sim] memory probe: {len(big)} atoms need {peak / 2**30:.2f} GiB "
        f"({'MD' if md else 'forces'}); GPU has {total / 2**30:.0f} GiB -> chunks of {n}")
    torch.cuda.empty_cache()
    return n
