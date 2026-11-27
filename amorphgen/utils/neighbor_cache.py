"""
amorphgen.utils.neighbor_cache
------------------------------
Neighbour lists that are reused between steps (a Verlet skin list).

Building a neighbour list is the most expensive thing the host does for a
joined Symmetrix evaluation, and in MD or a fixed-cell relaxation the list
hardly changes from one step to the next. So the list is built once with a
larger cutoff (``cutoff + skin``) and kept: each later step only recomputes
the vectors of the kept pairs and drops those beyond ``cutoff``.

The kept list stays complete as long as no pair that was outside
``cutoff + skin`` can have come inside ``cutoff``. With the deformation of the
cell since the list was built, ``F = cell_ref^-1 . cell``, and the
displacements ``u`` of the atoms on top of that deformation, every pair vector
is exactly ``d = d_ref . F + u_j - u_i``, so the list is valid while

    2 max|u|  <=  sigma_min(F) (cutoff + skin) - cutoff

and is rebuilt otherwise. This covers a changing cell (cell relaxation, NPT):
a uniform strain alone costs skin only through ``sigma_min``. Atoms wrapped
back into the cell show up as a large ``u`` and simply trigger a rebuild.

The per-step update is a single loop over the kept pairs. It is compiled with
Numba when Numba is installed (optional), and otherwise runs in NumPy.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

DEFAULT_SKIN = 0.5      # Angstrom


def _pairs_numpy(positions, cell, first, second, shifts, cutoff):
    vectors = positions[second] - positions[first] + shifts @ cell
    squared = np.einsum("ij,ij->i", vectors, vectors)
    keep = squared < cutoff * cutoff
    return first[keep], second[keep], np.sqrt(squared[keep]), vectors[keep]


def _compile_numba():
    """The pair update compiled by Numba, or None when Numba is not usable."""
    try:
        import numba
    except Exception:
        return None

    @numba.njit(cache=False, nogil=True)
    def kernel(positions, cell, first, second, shifts, cutoff, out_first, out_second,
               out_distance, out_vector):
        limit = cutoff * cutoff
        count = 0
        for edge in range(first.shape[0]):
            a, b = first[edge], second[edge]
            s0, s1, s2 = shifts[edge, 0], shifts[edge, 1], shifts[edge, 2]
            x = positions[b, 0] - positions[a, 0] + s0 * cell[0, 0] + s1 * cell[1, 0] + s2 * cell[2, 0]
            y = positions[b, 1] - positions[a, 1] + s0 * cell[0, 1] + s1 * cell[1, 1] + s2 * cell[2, 1]
            z = positions[b, 2] - positions[a, 2] + s0 * cell[0, 2] + s1 * cell[1, 2] + s2 * cell[2, 2]
            squared = x * x + y * y + z * z
            if squared < limit:
                out_first[count] = a
                out_second[count] = b
                out_distance[count] = np.sqrt(squared)
                out_vector[count, 0] = x
                out_vector[count, 1] = y
                out_vector[count, 2] = z
                count += 1
        return count

    def pairs(positions, cell, first, second, shifts, cutoff):
        n = first.shape[0]
        out_first = np.empty(n, dtype=np.int32)
        out_second = np.empty(n, dtype=np.int32)
        out_distance = np.empty(n)
        out_vector = np.empty((n, 3))
        count = kernel(positions, cell, first, second, shifts, cutoff, out_first,
                       out_second, out_distance, out_vector)
        return (out_first[:count], out_second[:count], out_distance[:count],
                out_vector[:count])

    try:        # compile now, so a broken Numba install falls back to NumPy
        pairs(np.zeros((2, 3)), np.eye(3), np.zeros(1, np.int32), np.ones(1, np.int32),
              np.zeros((1, 3)), 1.0)
    except Exception:
        return None
    return pairs


_KERNEL = {}


def pair_kernel(use_numba: bool | None = None):
    """The function that updates kept pairs, and its name (``numba`` / ``numpy``).

    ``use_numba=None`` takes Numba when it is installed.
    """
    if use_numba is False:
        return _pairs_numpy, "numpy"
    if "numba" not in _KERNEL:
        _KERNEL["numba"] = _compile_numba()
    if _KERNEL["numba"] is not None:
        return _KERNEL["numba"], "numba"
    if use_numba:
        raise ImportError("Numba is not installed (pip install numba)")
    return _pairs_numpy, "numpy"


def _neighbor_list(quantities, atoms, cutoff):
    try:
        from matscipy.neighbours import neighbour_list
    except ImportError:
        from ase.neighborlist import neighbor_list as neighbour_list
    return neighbour_list(quantities, atoms, cutoff)


class SkinNeighborList:
    """The neighbour list of one trajectory, rebuilt only when it has to be."""

    def __init__(self, cutoff: float, skin: float = DEFAULT_SKIN, use_numba=None):
        self.cutoff = float(cutoff)
        self.skin = float(skin)
        self._update, self.kernel = pair_kernel(use_numba)
        self._reference = None
        self.builds = 0
        self.reuses = 0

    def _build(self, atoms):
        first, second, shifts = _neighbor_list("ijS", atoms, self.cutoff + self.skin)
        cell = np.array(atoms.cell.array, dtype=float)
        self._reference = {
            "positions": np.array(atoms.positions, dtype=float),
            "cell": cell,
            "inverse": np.linalg.inv(cell),
            "first": np.ascontiguousarray(first, dtype=np.int32),
            "second": np.ascontiguousarray(second, dtype=np.int32),
            "shifts": np.ascontiguousarray(shifts, dtype=float),
            "cell_shifts": np.ascontiguousarray(shifts, dtype=np.int32),
        }
        self.builds += 1

    def _still_complete(self, positions, cell) -> bool:
        ref = self._reference
        if ref is None or positions.shape != ref["positions"].shape:
            return False
        deformation = ref["inverse"] @ cell
        smallest = np.linalg.svd(deformation, compute_uv=False)[-1]
        moved = positions - ref["positions"] @ deformation
        largest = np.sqrt(np.einsum("ij,ij->i", moved, moved).max())
        return 2.0 * largest <= smallest * (self.cutoff + self.skin) - self.cutoff

    def topology(self, atoms):
        """``(i, j, S, build)``: the kept candidate pairs (everything within
        cutoff + skin when the list was built, with integer cell shifts ``S``)
        and the number of the build they come from. For a caller that keeps
        the graph itself and only needs to know when it has changed."""
        positions = np.ascontiguousarray(atoms.positions, dtype=float)
        cell = np.ascontiguousarray(atoms.cell.array, dtype=float)
        if self._still_complete(positions, cell):
            self.reuses += 1
        else:
            self._build(atoms)
        ref = self._reference
        return ref["first"], ref["second"], ref["cell_shifts"], self.builds

    def pairs(self, atoms):
        """``(i, j, r, D)`` of all pairs within the cutoff, as
        ``neighbor_list("ijdD", atoms, cutoff)`` gives them (grouped by ``i``)."""
        positions = np.ascontiguousarray(atoms.positions, dtype=float)
        cell = np.ascontiguousarray(atoms.cell.array, dtype=float)
        if self._still_complete(positions, cell):
            self.reuses += 1
        else:
            self._build(atoms)
        ref = self._reference
        return self._update(positions, cell, ref["first"], ref["second"], ref["shifts"],
                            self.cutoff)


class NeighborCache:
    """One :class:`SkinNeighborList` per trajectory key.

    A key only has to be stable for a trajectory to profit: a list is checked
    against the structure it is asked for every time, so a key that comes to
    stand for another structure costs one rebuild and nothing else.
    Structures without a full three-dimensional periodic cell are not kept.
    """

    def __init__(self, cutoff: float, skin: float = DEFAULT_SKIN, max_lists: int = 2048,
                 use_numba=None):
        self.cutoff, self.skin, self.use_numba = float(cutoff), float(skin), use_numba
        self.max_lists = int(max_lists)
        self._lists = OrderedDict()
        self._retired = [0, 0]              # builds, reuses of dropped lists
        self.kernel = pair_kernel(use_numba)[1] if self.skin > 0 else "none"

    def _kept(self, key):
        kept = self._lists.get(key)
        if kept is None:
            kept = self._lists[key] = SkinNeighborList(self.cutoff, self.skin,
                                                       self.use_numba)
            while len(self._lists) > self.max_lists:
                _, old = self._lists.popitem(last=False)
                self._retired[0] += old.builds
                self._retired[1] += old.reuses
        else:
            self._lists.move_to_end(key)
        return kept

    def topology(self, key, atoms):
        """:meth:`SkinNeighborList.topology` of the trajectory *key*, or None
        for a structure whose list is not kept (no key, no skin, not fully
        periodic)."""
        if (key is None or self.skin <= 0 or atoms.cell.rank != 3
                or not atoms.pbc.all()):
            return None
        return self._kept(key).topology(atoms)

    def pairs(self, key, atoms):
        if (key is None or self.skin <= 0 or atoms.cell.rank != 3
                or not atoms.pbc.all()):
            from ase.neighborlist import neighbor_list as ase_neighbor_list
            full = atoms.cell.rank == 3
            nl = _neighbor_list if full else ase_neighbor_list
            self._retired[0] += 1
            return nl("ijdD", atoms, self.cutoff)
        return self._kept(key).pairs(atoms)

    @property
    def builds(self) -> int:
        return self._retired[0] + sum(kept.builds for kept in self._lists.values())

    @property
    def reuses(self) -> int:
        return self._retired[1] + sum(kept.reuses for kept in self._lists.values())

    def summary(self) -> str:
        total = self.builds + self.reuses
        if not total:
            return "neighbour lists: none built"
        return (f"neighbour lists: {self.builds} built, {self.reuses} reused "
                f"({100.0 * self.reuses / total:.0f} %; skin {self.skin} A, "
                f"{self.kernel} update)")
