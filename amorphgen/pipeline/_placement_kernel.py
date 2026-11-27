"""
amorphgen.pipeline._placement_kernel
------------------------------------
The candidate loop of random placement, compiled with Numba (optional).

Placing one atom means drawing candidate positions until one passes the
minimum-separation and over-coordination tests. In a dense cell the last
atoms need thousands of candidates each, and in pure Python every candidate
costs about 14 microseconds of interpreter and NumPy-call overhead; that loop
is nearly all of the run time (a 270-atom TiO2 cell tries ~1.5 million).

This module is that loop, statement for statement, compiled. It takes the
same ``numpy.random.Generator`` and makes the same draws in the same order
(``standard_normal(3)`` and ``uniform`` for a seeded candidate, ``random(3)``
for a free one), so a seed gives the same structure as the Python loop, up to
floating-point rounding in the last place. Without Numba, or with
``AMORPHGEN_NO_NUMBA=1``, :func:`placement_kernel` returns None and
``generate_random`` runs its Python loop.
"""

from __future__ import annotations

import os

_KERNEL = {}


def _build():
    import numba
    import numpy as np

    def place_atom(rng, positions, n_placed, placed_type_idx, sym_idx,
                   minsep_sq_table, dmax_table, dmax_sq_table, target_cn_arr,
                   cn_array, cn_tolerance, L, pbc, use_sc, bondable_idx,
                   attempts_per_seed, max_attempts_per_atom):
        """Try to place one atom of type ``sym_idx``; write it to
        ``positions[n_placed]`` and return True, or return False."""
        pos = np.empty(3)

        # -- Coordination-aware path: in the bonding shell of a seed --------
        for b in range(bondable_idx.shape[0]):
            seed_idx = bondable_idx[b]
            seed_type = placed_type_idx[seed_idx]
            ms = minsep_sq_table[seed_type, sym_idx] ** 0.5
            dm = dmax_table[seed_type, sym_idx]
            for _ in range(attempts_per_seed):
                direction = rng.standard_normal(3)
                norm = np.sqrt(direction[0] * direction[0] + direction[1] * direction[1]
                               + direction[2] * direction[2])
                distance = rng.uniform(ms, dm)
                for c in range(3):
                    pos[c] = positions[seed_idx, c] + direction[c] / norm * distance
                    if pbc:
                        pos[c] = pos[c] % L
                accept = True
                for j in range(n_placed):
                    d_sq = 0.0
                    for c in range(3):
                        d = pos[c] - positions[j, c]
                        if pbc:
                            d -= L * np.round(d / L)
                        d_sq += d * d
                    type_j = placed_type_idx[j]
                    if not d_sq >= minsep_sq_table[sym_idx, type_j]:
                        accept = False
                        break
                    # a neighbour within its bonding shell that is already
                    # fully coordinated (the seed itself excepted)
                    dm_sq = dmax_sq_table[type_j, sym_idx]
                    if (j != seed_idx and dm_sq > 0 and d_sq <= dm_sq
                            and cn_array[j] >= target_cn_arr[type_j] + cn_tolerance):
                        accept = False
                        break
                if accept:
                    positions[n_placed, 0] = pos[0]
                    positions[n_placed, 1] = pos[1]
                    positions[n_placed, 2] = pos[2]
                    return True

        # -- Standard path: anywhere in the cell -----------------------------
        for _ in range(max_attempts_per_atom):
            free = rng.random(3)
            for c in range(3):
                pos[c] = free[c] * L
            accept = True
            for j in range(n_placed):
                d_sq = 0.0
                for c in range(3):
                    d = pos[c] - positions[j, c]
                    if pbc:
                        d -= L * np.round(d / L)
                    d_sq += d * d
                type_j = placed_type_idx[j]
                if not d_sq >= minsep_sq_table[sym_idx, type_j]:
                    accept = False
                    break
                if use_sc:
                    dm_sq = dmax_sq_table[type_j, sym_idx]
                    if (dm_sq > 0 and d_sq <= dm_sq
                            and cn_array[j] >= target_cn_arr[type_j] + cn_tolerance):
                        accept = False
                        break
            if accept:
                positions[n_placed, 0] = pos[0]
                positions[n_placed, 1] = pos[1]
                positions[n_placed, 2] = pos[2]
                return True
        return False

    try:        # keep the compiled code between processes when the cache is writable
        kernel = numba.njit(cache=True, nogil=True)(place_atom)
        _probe(kernel, np)
    except Exception:
        kernel = numba.njit(cache=False, nogil=True)(place_atom)
        _probe(kernel, np)
    return kernel


def _probe(kernel, np):
    """Compile now (and fail here, not in the middle of a run)."""
    positions = np.zeros((2, 3))
    table = np.ones((1, 1))
    ok = kernel(np.random.default_rng(0), positions, 1, np.zeros(2, dtype=np.int64), 0,
                table, table * 2.0, table * 4.0, np.full(1, 999, dtype=np.int64),
                np.zeros(2, dtype=np.int64), 0, 10.0, True, True,
                np.zeros(1, dtype=np.int64), 5, 5)
    if not ok:
        raise RuntimeError("placement kernel probe failed")


def placement_kernel():
    """The compiled candidate loop, or None when Numba is not to be used."""
    if os.environ.get("AMORPHGEN_NO_NUMBA", "").strip() not in ("", "0"):
        return None
    if "kernel" not in _KERNEL:
        try:
            _KERNEL["kernel"] = _build()
        except Exception:
            _KERNEL["kernel"] = None
    return _KERNEL["kernel"]
