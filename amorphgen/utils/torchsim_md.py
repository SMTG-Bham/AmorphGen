"""Batched NVT molecular dynamics through torch-sim (optional engine, phase 2).

Runs many structures at once through torch-sim's NVT-Langevin integrator and
returns plain ASE ``Atoms`` (with momenta), writing per-run logs and extxyz
trajectories in the same format and file names as the ASE stages, so
``--analyse``, ``--resume`` (run level) and the ensemble collector are
unchanged.

Units: torch-sim's *internal* units are ASE's (eV, Å, amu, ASE time unit), so
momenta pass through without conversion; its public API takes the timestep in
ps, temperatures in K and the Langevin friction in 1/ps.

Not supported here: NPT stages (torch-sim has only Langevin NPT, which is not
what the ASE path uses) and frame-level resume inside a stage.
"""
from __future__ import annotations

import os
import time

import numpy as np

from .torchsim_engine import _require, resolve_torch_device  # noqa: F401
from .common import TRAJ_LOG_INTERVAL

_LOG_HEADER = (f"{'Step':>8}  {'Time_ps':>10}  {'T_K':>8}  {'Epot_eV':>12}  "
               f"{'Ekin_eV':>12}  {'Etot_eV':>12}  {'Vol_A3':>10}\n" + "-" * 84 + "\n")


def _seed_torch(seed, stage: int, tag: int = 0):
    if seed is None:
        return
    import torch
    s = int(np.random.SeedSequence([int(seed), int(stage), int(tag)]).generate_state(1)[0])
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _state_to_atoms_with_momenta(state, model):
    """torch-sim MDState -> list of ASE Atoms carrying momenta, energy, forces."""
    import torch_sim as ts
    from ase.calculators.singlepoint import SinglePointCalculator
    atoms_list = ts.io.state_to_atoms(state)
    sys_idx = state.system_idx.detach().cpu().numpy()
    mom = state.momenta.detach().cpu().numpy()
    forces = state.forces.detach().cpu().numpy() if getattr(state, "forces", None) is not None else None
    energy = state.energy.detach().cpu().numpy().reshape(-1) if getattr(state, "energy", None) is not None else None
    for k, a in enumerate(atoms_list):
        m = sys_idx == k
        a.set_momenta(mom[m])
        if forces is not None or energy is not None:
            a.calc = SinglePointCalculator(
                a, energy=float(energy[k]) if energy is not None else None,
                forces=forces[m] if forces is not None else None)
    return atoms_list


class _RunWriter:
    """Per-run MD log + extxyz trajectory in the ASE-stage format."""

    def __init__(self, run_dir, logname, trajname, append=False, step_offset=0):
        os.makedirs(run_dir, exist_ok=True)
        self.log = os.path.join(run_dir, logname)
        self.traj = os.path.join(run_dir, trajname)
        self.step_offset = int(step_offset)
        if not append:
            with open(self.log, "w") as fh:
                fh.write(_LOG_HEADER)
            if os.path.exists(self.traj):
                os.remove(self.traj)

    def write(self, atoms, step, timestep_fs):
        from ase.io import write
        step = step + self.step_offset
        epot = atoms.get_potential_energy() if atoms.calc is not None else float("nan")
        ekin = atoms.get_kinetic_energy()
        with open(self.log, "a") as fh:
            fh.write(f"{step:8d}  {step * timestep_fs / 1000.0:10.4f}  {atoms.get_temperature():8.1f}  "
                     f"{epot:12.4f}  {ekin:12.4f}  {epot + ekin:12.4f}  {atoms.get_volume():10.2f}\n")
        img = atoms.copy(); img.wrap()
        if atoms.calc is not None:
            from ase.calculators.singlepoint import SinglePointCalculator
            img.calc = SinglePointCalculator(img, **{k: v for k, v in atoms.calc.results.items()
                                                    if k in ("energy", "forces")})
        write(self.traj, img, format="extxyz", append=True)


def batch_nvt(atoms_list, model, temperatures, n_steps: int, timestep_fs: float = 0.5,
              friction: float = 0.01, seed=None, stage: int = 4,
              interval: int = TRAJ_LOG_INTERVAL, writers=None, log=print):
    """Batched NVT-Langevin MD of *atoms_list* for *n_steps*.

    Parameters
    ----------
    temperatures : float or 1-D sequence of length n_steps
        Constant temperature, or a per-step schedule (K) for a ramp.
    friction : float
        Langevin friction in 1/fs (AmorphGen convention; 0.01 = ASE default).
    writers : list[_RunWriter], optional
        One per structure; each receives a frame every *interval* steps.

    Returns the final structures as ASE Atoms with momenta.
    """
    _require()
    import torch
    import torch_sim as ts
    from torch_sim.integrators.nvt import nvt_langevin_init

    n = len(atoms_list)
    T_sched = np.asarray(temperatures, dtype=float)
    if T_sched.ndim == 0:
        T_sched = np.full(int(n_steps), float(T_sched))
    assert len(T_sched) == n_steps, "temperature schedule must have n_steps entries"
    _seed_torch(seed, stage)
    dt_ps = float(timestep_fs) / 1000.0
    gamma_ps = float(friction) * 1000.0

    # initial state; momenta carried in from the Atoms if present, else sampled
    state = ts.initialize_state(list(atoms_list), model.device, model.dtype)
    kT0 = float(T_sched[0]) * 8.617330337217213e-05
    md = nvt_langevin_init(state, model, kT=kT0)
    have_p = all(np.abs(a.get_momenta()).sum() > 0 for a in atoms_list)
    if have_p:
        md.momenta = torch.as_tensor(np.concatenate([a.get_momenta() for a in atoms_list]),
                                     dtype=model.dtype, device=model.device)

    log(f"[torch-sim] NVT-Langevin: {n} structure(s) in one batch on {model.device}, "
        f"{n_steps} steps x {timestep_fs} fs, T {T_sched[0]:.0f} -> {T_sched[-1]:.0f} K, "
        f"friction {friction}/fs, momenta {'carried over' if have_p else 'sampled'}")
    t0 = time.time(); done = 0
    while done < n_steps:
        block = min(int(interval), n_steps - done)
        T_block = T_sched[done:done + block]
        temp = float(T_block[0]) if np.allclose(T_block, T_block[0]) else T_block.tolist()
        md = ts.integrate(system=md, model=model, integrator=ts.Integrator.nvt_langevin,
                          n_steps=block, temperature=temp, timestep=dt_ps, gamma=gamma_ps)
        done += block
        if writers is not None:
            frames = _state_to_atoms_with_momenta(md, model)
            for w, a in zip(writers, frames):
                w.write(a, done, timestep_fs)
    out = _state_to_atoms_with_momenta(md, model)
    dt = time.time() - t0
    log(f"[torch-sim] done in {dt:.1f} s ({1000 * dt / n_steps / max(n, 1):.2f} ms per step per structure); "
        f"T = {np.mean([a.get_temperature() for a in out]):.0f} K")
    return out
