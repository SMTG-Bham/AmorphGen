"""
amorphgen.utils.classical
--------------------------
Classical pair potential calculators: Lennard-Jones and Buckingham+Coulomb.

Vectorized NumPy implementation with optional PyTorch GPU acceleration.
Uses ASE's Calculator interface. No external dependencies (no LAMMPS, no GULP).

Note: these are rigid-ion potentials (no core-shell model).
For shell model potentials, use LAMMPS or GULP via ASE.

Usage:
    from amorphgen.utils.classical import LennardJonesCalculator, BuckinghamCalculator

    # LJ (CPU)
    calc = LennardJonesCalculator(
        params={("Ar", "Ar"): {"epsilon": 0.0104, "sigma": 3.40}},
        cutoff=10.0,
    )

    # Buckingham + Coulomb (GPU)
    calc = BuckinghamCalculator(
        params={
            ("Si", "O"): {"A": 13702.905, "rho": 0.193817, "C": 54.681},
            ("O", "O"):  {"A": 2029.2204, "rho": 0.343645, "C": 192.58},
        },
        charges={"Si": 2.4, "O": -1.2},
        cutoff=10.0,
        device="cuda",   # "cpu", "cuda", or "mps"
    )
"""

from __future__ import annotations

from math import erfc as _erfc, pi as _pi, sqrt as _sqrt

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list

# Lazy torch import — only when GPU is requested
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _use_gpu(device: str) -> bool:
    """Check if device requests GPU acceleration."""
    return device in ("cuda", "mps")


class LennardJonesCalculator(Calculator):
    """
    Lennard-Jones pair potential calculator (vectorized).

    V(r) = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]

    Parameters
    ----------
    params : dict
        Per-pair LJ parameters, e.g.
        {("Ar", "Ar"): {"epsilon": 0.0104, "sigma": 3.40}}
        Pairs are symmetric: ("A","B") == ("B","A").
    cutoff : float
        Cutoff distance in Angstrom.
    device : str
        "cpu" (default, vectorized NumPy) or "cuda"/"mps" (PyTorch GPU).
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, params: dict, cutoff: float = 10.0,
                 device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        self.pair_params = {}
        for (s1, s2), p in params.items():
            self.pair_params[(s1, s2)] = p
            self.pair_params[(s2, s1)] = p
        self.cutoff = cutoff
        self.device = device

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        symbols = self.atoms.get_chemical_symbols()
        n = len(self.atoms)

        # Neighbour list (always on CPU via ASE)
        ii, jj, dd, DD = neighbor_list("ijdD", self.atoms, cutoff=self.cutoff)

        # Filter to unique pairs (i < j)
        mask = ii < jj
        ii, jj, dd, DD = ii[mask], jj[mask], dd[mask], DD[mask]

        if len(ii) == 0:
            self.results["energy"] = 0.0
            self.results["forces"] = np.zeros((n, 3))
            return

        # Vectorized parameter lookup via integer element indices
        unique_syms = sorted(set(symbols))
        sym_to_idx = {s: i for i, s in enumerate(unique_syms)}
        n_types = len(unique_syms)
        atom_type = np.array([sym_to_idx[s] for s in symbols])

        eps_table = np.zeros((n_types, n_types))
        sig_table = np.zeros((n_types, n_types))
        valid_table = np.zeros((n_types, n_types), dtype=bool)
        for (s1, s2), p in self.pair_params.items():
            if s1 in sym_to_idx and s2 in sym_to_idx:
                i1, i2 = sym_to_idx[s1], sym_to_idx[s2]
                eps_table[i1, i2] = eps_table[i2, i1] = p["epsilon"]
                sig_table[i1, i2] = sig_table[i2, i1] = p["sigma"]
                valid_table[i1, i2] = valid_table[i2, i1] = True

        ti, tj = atom_type[ii], atom_type[jj]
        eps_arr = eps_table[ti, tj]
        sig_arr = sig_table[ti, tj]
        valid = valid_table[ti, tj]

        if not np.any(valid):
            self.results["energy"] = 0.0
            self.results["forces"] = np.zeros((n, 3))
            return

        # Slice to valid pairs only
        ii_v, jj_v = ii[valid], jj[valid]
        dd_v, DD_v = dd[valid], DD[valid]
        eps_v, sig_v = eps_arr[valid], sig_arr[valid]

        if _use_gpu(self.device):
            energy, forces = self._calc_gpu(n, ii_v, jj_v, dd_v, DD_v,
                                            eps_v, sig_v)
        else:
            energy, forces = self._calc_cpu(n, ii_v, jj_v, dd_v, DD_v,
                                            eps_v, sig_v)

        self.results["energy"] = energy
        self.results["forces"] = forces

    @staticmethod
    def _calc_cpu(n, ii, jj, dd, DD, eps, sig):
        sr6 = (sig / dd) ** 6
        sr12 = sr6 ** 2

        # Energy
        e_pairs = 4.0 * eps * (sr12 - sr6)
        energy = float(np.sum(e_pairs))

        # Forces: dV/dr * (r_vec / r)
        dVdr = 4.0 * eps * (-12.0 * sr12 / dd + 6.0 * sr6 / dd)
        f_vecs = dVdr[:, None] * DD / dd[:, None]

        forces = np.zeros((n, 3))
        np.add.at(forces, ii, f_vecs)
        np.add.at(forces, jj, -f_vecs)
        return energy, forces

    def _calc_gpu(self, n, ii, jj, dd, DD, eps, sig):
        torch = _get_torch()
        dev = self.device
        dtype = torch.float32 if dev == "mps" else torch.float64

        dd_t = torch.tensor(dd, dtype=dtype, device=dev)
        DD_t = torch.tensor(DD, dtype=dtype, device=dev)
        eps_t = torch.tensor(eps, dtype=dtype, device=dev)
        sig_t = torch.tensor(sig, dtype=dtype, device=dev)
        ii_t = torch.tensor(ii, dtype=torch.long, device=dev)
        jj_t = torch.tensor(jj, dtype=torch.long, device=dev)

        sr6 = (sig_t / dd_t) ** 6
        sr12 = sr6 ** 2

        energy = float(torch.sum(4.0 * eps_t * (sr12 - sr6)).cpu())

        dVdr = 4.0 * eps_t * (-12.0 * sr12 / dd_t + 6.0 * sr6 / dd_t)
        f_vecs = dVdr.unsqueeze(1) * DD_t / dd_t.unsqueeze(1)

        forces = torch.zeros(n, 3, dtype=dtype, device=dev)
        forces.index_add_(0, ii_t, f_vecs)
        forces.index_add_(0, jj_t, -f_vecs)
        return energy, forces.cpu().numpy().astype(np.float64)


class BuckinghamCalculator(Calculator):
    """
    Buckingham + Coulomb pair potential calculator (rigid-ion, vectorized).

    V(r) = A * exp(-r/rho) - C/r^6 + q_i * q_j / (4*pi*eps0*r)

    The Coulomb term uses a Wolf summation for convergence under PBC.
    This is an approximation to full Ewald -- accurate for amorphous
    structures but not for crystalline long-range order.

    Parameters
    ----------
    params : dict
        Per-pair Buckingham parameters, e.g.
        {("Si", "O"): {"A": 13702.905, "rho": 0.193817, "C": 54.681}}
    charges : dict
        Per-element charges in electron units, e.g.
        {"Si": 2.4, "O": -1.2}
    cutoff : float
        Cutoff distance in Angstrom.
    alpha : float
        Wolf summation damping parameter (1/Angstrom).
        Default 0.2 works well for cutoff >= 8 A.
    coulomb : bool
        If True, include Coulomb interactions. Default True.
    device : str
        "cpu" (default, vectorized NumPy) or "cuda"/"mps" (PyTorch GPU).
    """

    implemented_properties = ["energy", "forces"]

    # Coulomb constant: e^2 / (4*pi*eps0) in eV*A
    _KE = 14.3996

    def __init__(self, params: dict, charges: dict | None = None,
                 cutoff: float = 10.0, alpha: float = 0.2,
                 coulomb: bool = True, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        self.pair_params = {}
        for (s1, s2), p in params.items():
            self.pair_params[(s1, s2)] = p
            self.pair_params[(s2, s1)] = p
        self.charges = charges or {}
        self.cutoff = cutoff
        self.alpha = alpha
        self.coulomb = coulomb
        self.device = device

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        symbols = self.atoms.get_chemical_symbols()
        n = len(self.atoms)
        alpha = self.alpha
        rc = self.cutoff

        # Neighbour list (CPU)
        ii, jj, dd, DD = neighbor_list("ijdD", self.atoms, cutoff=rc)

        # Filter to unique pairs (i < j)
        mask = ii < jj
        ii, jj, dd, DD = ii[mask], jj[mask], dd[mask], DD[mask]
        n_pairs = len(ii)

        # Vectorized parameter lookup via integer element indices
        unique_syms = sorted(set(symbols))
        sym_to_idx = {s: i for i, s in enumerate(unique_syms)}
        n_types = len(unique_syms)
        atom_type = np.array([sym_to_idx[s] for s in symbols])

        # Build pair-type lookup tables
        A_table = np.zeros((n_types, n_types))
        rho_table = np.zeros((n_types, n_types))
        C_table = np.zeros((n_types, n_types))
        valid_table = np.zeros((n_types, n_types), dtype=bool)
        q_table = np.zeros(n_types)

        for (s1, s2), p in self.pair_params.items():
            if s1 in sym_to_idx and s2 in sym_to_idx:
                i1, i2 = sym_to_idx[s1], sym_to_idx[s2]
                A_table[i1, i2] = A_table[i2, i1] = p["A"]
                rho_table[i1, i2] = rho_table[i2, i1] = p["rho"]
                C_table[i1, i2] = C_table[i2, i1] = p.get("C", 0.0)
                valid_table[i1, i2] = valid_table[i2, i1] = True

        if self.coulomb and self.charges:
            for s, q in self.charges.items():
                if s in sym_to_idx:
                    q_table[sym_to_idx[s]] = q

        # Lookup per-pair parameters via integer indexing (fully vectorized)
        ti = atom_type[ii]
        tj = atom_type[jj]
        A_arr = A_table[ti, tj]
        rho_arr = rho_table[ti, tj]
        C_arr = C_table[ti, tj]
        buck_valid = valid_table[ti, tj]
        qi_arr = q_table[ti]
        qj_arr = q_table[tj]

        # Wolf self-energy correction
        wolf_self = 0.0
        if self.coulomb and self.charges:
            q_atoms = np.array([self.charges.get(s, 0.0) for s in symbols])
            wolf_self = -self._KE * np.sum(q_atoms ** 2) * (
                _erfc(alpha * rc) / (2.0 * rc) + alpha / _sqrt(_pi)
            )

        if _use_gpu(self.device):
            energy, forces = self._calc_gpu(
                n, ii, jj, dd, DD,
                A_arr, rho_arr, C_arr, buck_valid,
                qi_arr, qj_arr, alpha, rc, wolf_self,
            )
        else:
            energy, forces = self._calc_cpu(
                n, ii, jj, dd, DD,
                A_arr, rho_arr, C_arr, buck_valid,
                qi_arr, qj_arr, alpha, rc, wolf_self,
            )

        self.results["energy"] = energy
        self.results["forces"] = forces

    @staticmethod
    def _calc_cpu(n, ii, jj, dd, DD,
                  A, rho, C, buck_valid, qi, qj,
                  alpha, rc, wolf_self):
        KE = BuckinghamCalculator._KE
        energy = wolf_self
        forces = np.zeros((n, 3))

        # --- Buckingham ---
        if np.any(buck_valid):
            bm = buck_valid
            r_b = dd[bm]
            exp_term = A[bm] * np.exp(-r_b / rho[bm])
            r6_term = C[bm] / r_b ** 6
            energy += float(np.sum(exp_term - r6_term))

            dVdr_buck = -A[bm] / rho[bm] * np.exp(-r_b / rho[bm]) + 6.0 * C[bm] / r_b ** 7
            f_buck = dVdr_buck[:, None] * DD[bm] / r_b[:, None]
            np.add.at(forces, ii[bm], f_buck)
            np.add.at(forces, jj[bm], -f_buck)

        # --- Coulomb (Wolf) ---
        coul_mask = (qi != 0.0) & (qj != 0.0)
        if np.any(coul_mask):
            cm = coul_mask
            r_c = dd[cm]
            qiqj = KE * qi[cm] * qj[cm]

            # erfc via scipy for array operation
            from scipy.special import erfc as _erfc_arr
            erfc_ar = _erfc_arr(alpha * r_c)
            erfc_arc = _erfc_arr(alpha * rc)

            e_coul = qiqj * (erfc_ar / r_c - erfc_arc / rc)
            energy += float(np.sum(e_coul))

            dVdr_coul = qiqj * (
                -erfc_ar / r_c ** 2
                - 2.0 * alpha / _sqrt(_pi) * np.exp(-(alpha * r_c) ** 2) / r_c
            )
            f_coul = dVdr_coul[:, None] * DD[cm] / r_c[:, None]
            np.add.at(forces, ii[cm], f_coul)
            np.add.at(forces, jj[cm], -f_coul)

        return energy, forces

    def _calc_gpu(self, n, ii, jj, dd, DD,
                  A, rho, C, buck_valid, qi, qj,
                  alpha, rc, wolf_self):
        torch = _get_torch()
        dev = self.device
        KE = self._KE
        dtype = torch.float32 if dev == "mps" else torch.float64

        dd_t = torch.tensor(dd, dtype=dtype, device=dev)
        DD_t = torch.tensor(DD, dtype=dtype, device=dev)
        ii_t = torch.tensor(ii, dtype=torch.long, device=dev)
        jj_t = torch.tensor(jj, dtype=torch.long, device=dev)

        energy = wolf_self
        forces = torch.zeros(n, 3, dtype=dtype, device=dev)

        # --- Buckingham ---
        bm_t = torch.tensor(buck_valid, dtype=torch.bool, device=dev)
        if torch.any(bm_t):
            A_t = torch.tensor(A, dtype=dtype, device=dev)[bm_t]
            rho_t = torch.tensor(rho, dtype=dtype, device=dev)[bm_t]
            C_t = torch.tensor(C, dtype=dtype, device=dev)[bm_t]
            r_b = dd_t[bm_t]
            D_b = DD_t[bm_t]
            ii_b = ii_t[bm_t]
            jj_b = jj_t[bm_t]

            exp_term = A_t * torch.exp(-r_b / rho_t)
            r6_term = C_t / r_b ** 6
            energy += float(torch.sum(exp_term - r6_term).cpu())

            dVdr = -A_t / rho_t * torch.exp(-r_b / rho_t) + 6.0 * C_t / r_b ** 7
            f_b = dVdr.unsqueeze(1) * D_b / r_b.unsqueeze(1)
            forces.index_add_(0, ii_b, f_b)
            forces.index_add_(0, jj_b, -f_b)

        # --- Coulomb (Wolf) ---
        qi_t = torch.tensor(qi, dtype=dtype, device=dev)
        qj_t = torch.tensor(qj, dtype=dtype, device=dev)
        cm_t = (qi_t != 0.0) & (qj_t != 0.0)
        if torch.any(cm_t):
            r_c = dd_t[cm_t]
            D_c = DD_t[cm_t]
            ii_c = ii_t[cm_t]
            jj_c = jj_t[cm_t]
            qiqj = KE * qi_t[cm_t] * qj_t[cm_t]

            erfc_ar = torch.erfc(alpha * r_c)
            erfc_arc = float(_erfc(alpha * rc))

            e_coul = qiqj * (erfc_ar / r_c - erfc_arc / rc)
            energy += float(torch.sum(e_coul).cpu())

            dVdr_c = qiqj * (
                -erfc_ar / r_c ** 2
                - 2.0 * alpha / _sqrt(_pi) * torch.exp(-(alpha * r_c) ** 2) / r_c
            )
            f_c = dVdr_c.unsqueeze(1) * D_c / r_c.unsqueeze(1)
            forces.index_add_(0, ii_c, f_c)
            forces.index_add_(0, jj_c, -f_c)

        return energy, forces.cpu().numpy().astype(np.float64)
