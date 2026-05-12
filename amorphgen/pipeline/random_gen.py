"""
amorphgen.pipeline.random_gen
------------------------------
Generate amorphous structures by random atom placement with
minimum-separation constraints, then optionally relax with a
foundation model.

This provides an alternative to the melt-and-quench route: instead of
melting a crystal and cooling it, we place atoms randomly inside a box
subject to pairwise distance constraints, then optimise the structure.

Data tables (Shannon ionic radii, metallic radii, electronegativities,
element classification) and helper functions (bond classification,
minsep calculation, density estimation) are in amorphgen.utils.radii.
"""

from __future__ import annotations

import logging
import os
import importlib
import time
import numpy as np
from ase import Atoms
from ase.io import write
from ase.data import atomic_masses, atomic_numbers

from ..utils.radii import (
    # Data tables (re-exported for backward compatibility)
    SHANNON_IONIC_RADII, METALLIC_RADII, PAULING_EN,
    NONMETALS, METALLOIDS, ELEMENTAL_DENSITIES, SCALE_FACTORS,
    # Functions (aliased with underscore for internal use)
    classify_bond as _classify_bond,
    get_ionic_radius as _get_ionic_radius,
    get_metallic_radius as _get_metallic_radius,
    get_effective_radius as _get_effective_radius,
    default_minsep as _default_minsep,
    estimate_density as _estimate_density,
    estimate_cell_length as _estimate_cell_length,
    auto_target_cn as _auto_target_cn,
    format_auto_derive_summary as _format_auto_derive_summary,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Internal helpers
# ==============================================================================

def _get_minsep(s1: str, s2: str, minsep: dict) -> float:
    """Look up the minimum separation for a pair of species."""
    key1 = f"{s1}-{s2}"
    key2 = f"{s2}-{s1}"
    return minsep.get(key1, minsep.get(key2, 1.5))


# ==============================================================================
# Optimizer and cell filter helpers
# ==============================================================================

def _get_optimizer_class(name: str):
    """Import and return an ASE optimizer class by name."""
    optimizers = {
        "LBFGS":          ("ase.optimize", "LBFGS"),
        "FIRE":           ("ase.optimize", "FIRE"),
        "BFGSLineSearch": ("ase.optimize", "BFGSLineSearch"),
        "BFGS":           ("ase.optimize", "BFGS"),
        "MDMin":          ("ase.optimize", "MDMin"),
    }
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from: {', '.join(optimizers)}")
    module_path, cls_name = optimizers[name]
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def _build_cell_filter(atoms, cell_filter: str):
    """Wrap atoms in the requested cell filter for optimisation."""
    if cell_filter == "none" or cell_filter is None:
        return atoms
    elif cell_filter == "cubic":
        from ase.filters import ExpCellFilter
        return ExpCellFilter(atoms, hydrostatic_strain=True)
    elif cell_filter == "ExpCellFilter":
        from ase.filters import ExpCellFilter
        return ExpCellFilter(atoms)
    elif cell_filter == "StrainFilter":
        from ase.filters import StrainFilter
        return StrainFilter(atoms)
    elif cell_filter == "UnitCellFilter":
        from ase.filters import UnitCellFilter
        return UnitCellFilter(atoms)
    else:
        # Default: FrechetCellFilter (better convergence for non-cubic)
        from ase.filters import FrechetCellFilter
        return FrechetCellFilter(atoms)


# ==============================================================================
# Coordination-aware placement helpers
# ==============================================================================

def _get_dmax(s1: str, s2: str, dmax: dict) -> float:
    """Look up the maximum bond distance for a pair of species."""
    key1 = f"{s1}-{s2}"
    key2 = f"{s2}-{s1}"
    return dmax.get(key1, dmax.get(key2, 0.0))


def _auto_dmax(minsep: dict, target_cn: dict, factor: float = 1.5) -> dict:
    """
    Auto-generate dmax from minsep for bonding pairs.

    A bond is defined as minsep <= d <= dmax, so dmax = minsep * factor
    gives a reasonable bonding shell width.

    Coverage:
    - Ionic bonds (M-O, M-Cl): dmax = minsep * 1.5 (primary bonding)
    - Covalent cross-element (Si-Ge): dmax = minsep * 1.5
    - Metallic bonds (M-M): dmax = minsep * 1.3 (only when M-M is
      the primary bond, i.e. alloys/pure metals with no anions)
    - Pure elements (Si, Ge, Sn, etc.): dmax = minsep * 1.2
    """
    dmax = {}
    cn_elements = set(target_cn.keys())

    # Check if pure element or alloy (no anion present)
    all_elements = set()
    for pair in minsep:
        s1, s2 = pair.split("-")
        all_elements.add(s1)
        all_elements.add(s2)
    has_anion = any(s in NONMETALS for s in all_elements)
    is_pure = all(p.split("-")[0] == p.split("-")[1] for p in minsep)

    for pair, dist in minsep.items():
        s1, s2 = pair.split("-")

        if s1 == s2:
            # Pure elements: tight dmax so atoms are placed close enough
            # for MLIP to relax into proper bonds
            if is_pure and s1 in cn_elements:
                dmax[pair] = dist * 1.2
            continue

        # At least one element must have a target CN
        if s1 not in cn_elements and s2 not in cn_elements:
            continue

        bond_type = _classify_bond(s1, s2)
        if bond_type == "ionic" or bond_type == "covalent":
            # Primary bonding (M-O, M-Cl, Si-Ge)
            dmax[pair] = dist * factor
        elif bond_type == "metallic" and not has_anion:
            # M-M bonds in alloys/pure metals (no anion context)
            # Don't add M-M dmax in ionic compounds (In-In in In2O3)
            # because it creates conflicting CN constraints
            dmax[pair] = dist * factor

    return dmax


def _update_cn_array(cn_array: np.ndarray, new_idx: int,
                     positions: np.ndarray, n_placed: int,
                     placed_type_idx: np.ndarray, sym_idx: int,
                     dmax_table: np.ndarray, L: float, pbc: bool):
    """Incrementally update the CN array after placing a new atom (vectorized)."""
    pos_new = positions[new_idx]

    # Build indices of all OTHER placed atoms
    others = np.arange(n_placed)
    others = others[others != new_idx]
    if len(others) == 0:
        cn_array[new_idx] = 0
        return

    # Vectorized distance computation (squared — avoid sqrt)
    d_vec = pos_new - positions[others]
    if pbc:
        d_vec -= L * np.round(d_vec / L)
    d_sq = np.sum(d_vec * d_vec, axis=1)

    # Lookup dmax^2 from precomputed table
    dm_sq = dmax_table[placed_type_idx[others], sym_idx]

    # Find bonded neighbours: dmax > 0 and dist^2 <= dmax^2
    bonded = (dm_sq > 0) & (d_sq <= dm_sq)

    cn_array[new_idx] = int(np.sum(bonded))
    cn_array[others[bonded]] += 1


def _repair_undercoordination(
    positions: np.ndarray,
    n_placed: int,
    placed_type_idx: np.ndarray,
    cn_array: np.ndarray,
    target_cn_arr: np.ndarray,
    cn_tolerance: int,
    minsep_sq_table: np.ndarray,
    dmax_sq_table: np.ndarray,
    L: float,
    pbc: bool,
    max_iter: int,
    rng: np.random.Generator,
) -> int:
    """**Experimental** post-placement repair pass for under-coordinated atoms.

    Greedy single-atom relocation: each iteration picks one
    under-coordinated atom and proposes a new position inside the bonding
    shell of another under-coordinated atom; the move is accepted if it
    reduces the total under-coord count without violating minsep or
    over-coordinating any neighbour.  Iterates until all atoms reach
    target CN, or ``max_iter`` proposals are exhausted.

    This addresses a fundamental limitation of one-pass greedy
    placement: once an atom is placed it stays put, so atoms placed
    "too late" can end up under-coordinated even when geometrically a
    better arrangement exists.  Repair imitates (poorly) what an MD
    anneal would do continuously, but at the cost of a sequential
    Monte-Carlo-style loop.  Default off because:
      * doesn't beat the hybrid workflow in quality
      * adds 5-30 s of placement time per structure
      * not a substitute for MD on tetrahedral semiconductors

    Returns the final number of under-coordinated atoms.
    """
    def _under_count(cn):
        return int(np.sum(cn[:n_placed] < target_cn_arr[placed_type_idx[:n_placed]]))

    # Bond-length sweet spot per type-pair: midpoint of (sqrt(minsep), sqrt(dmax)).
    minsep_table = np.sqrt(minsep_sq_table)
    dmax_table = np.sqrt(dmax_sq_table)
    target_dist_table = 0.5 * (minsep_table + dmax_table)

    initial_under = _under_count(cn_array)
    if initial_under == 0:
        return 0

    accepted = 0
    rejected = 0
    for it in range(max_iter):
        # Identify under-coordinated atoms.
        under_mask = (cn_array[:n_placed] < target_cn_arr[placed_type_idx[:n_placed]])
        under_idx = np.where(under_mask)[0]
        if len(under_idx) == 0:
            break

        # Pick one to move; sample partner from remaining under-coord atoms.
        idx = int(rng.choice(under_idx))
        partners = under_idx[under_idx != idx]
        if len(partners) > 0:
            partner = int(rng.choice(partners))
            t_target = float(target_dist_table[
                placed_type_idx[idx], placed_type_idx[partner]
            ])
            direction = rng.standard_normal(3)
            direction /= np.linalg.norm(direction)
            new_pos = positions[partner] + direction * t_target
        else:
            # No other under-coord atom; perturb existing position.
            new_pos = positions[idx] + rng.standard_normal(3) * 0.5

        # Wrap into cell.
        new_pos = new_pos - L * np.floor(new_pos / L) if pbc else new_pos

        # Minsep check against all other placed atoms.
        others_idx = np.arange(n_placed)
        others_idx = others_idx[others_idx != idx]
        d_vec = new_pos - positions[others_idx]
        if pbc:
            d_vec -= L * np.round(d_vec / L)
        d_sq = np.sum(d_vec * d_vec, axis=1)
        min_sq = minsep_sq_table[placed_type_idx[idx], placed_type_idx[others_idx]]
        if np.any(d_sq < min_sq):
            rejected += 1
            continue

        # Compute proposed neighbour list (bonded set) for the new position.
        dm_sq = dmax_sq_table[placed_type_idx[idx], placed_type_idx[others_idx]]
        new_bonded = (dm_sq > 0) & (d_sq <= dm_sq)
        new_neighbour_count = int(np.sum(new_bonded))

        # Old bonded set (so we know whose CN to decrement after move).
        old_pos = positions[idx].copy()
        d_vec_old = old_pos - positions[others_idx]
        if pbc:
            d_vec_old -= L * np.round(d_vec_old / L)
        d_sq_old = np.sum(d_vec_old * d_vec_old, axis=1)
        old_bonded = (dm_sq > 0) & (d_sq_old <= dm_sq)

        # Tentative CN array after move.
        cn_trial = cn_array.copy()
        cn_trial[others_idx[old_bonded]] -= 1
        cn_trial[others_idx[new_bonded]] += 1
        cn_trial[idx] = new_neighbour_count

        # Reject if any neighbour is now over-coordinated beyond tolerance.
        target_neighbours = target_cn_arr[placed_type_idx[others_idx[new_bonded]]]
        new_neighbour_cn = cn_trial[others_idx[new_bonded]]
        if np.any(new_neighbour_cn > target_neighbours + cn_tolerance):
            rejected += 1
            continue
        if cn_trial[idx] > target_cn_arr[placed_type_idx[idx]] + cn_tolerance:
            rejected += 1
            continue

        # Accept iff total under-coord count strictly drops.
        new_under = _under_count(cn_trial)
        cur_under = _under_count(cn_array)
        if new_under < cur_under:
            positions[idx] = new_pos
            cn_array[:] = cn_trial
            accepted += 1
        else:
            rejected += 1

    final_under = _under_count(cn_array)
    logger.info(
        "  Repair pass: %d/%d proposals accepted; under-coord %d -> %d",
        accepted, accepted + rejected, initial_under, final_under,
    )
    return final_under


# ==============================================================================
# Structure generation
# ==============================================================================

def generate_random(
    composition: dict[str, int],
    cell_length_ang: float | None = None,
    target_density: float | None = None,
    density_scale: float = 1.0,
    minsep: dict[str, float] | None = None,
    minsep_scale: float = 0.85,
    seed: int | None = None,
    max_attempts_per_atom: int = 700000,
    pbc: bool = True,
    target_cn: dict[str, int] | None = None,
    dmax: dict[str, float] | None = None,
    cn_tolerance: int | None = None,
    dmax_factor: float = 1.5,
    repair_iters: int = 0,
) -> Atoms:
    """
    Generate a single random structure.

    Parameters
    ----------
    composition : dict
        Atom counts per element, e.g. {"In": 32, "O": 48} for 80-atom In2O3.
        The CLI also accepts formula format (``In2O3*16``) which is
        converted to this dict form automatically.
    cell_length_ang : float, optional
        Cubic cell edge length. If None, estimated from target_density.
    target_density : float, optional
        Target density in g/cm3 for cell size estimation. If supplied,
        ``density_scale`` is ignored.
    density_scale : float, default 1.0
        Multiplier applied to the *auto-estimated* density (sphere-packing
        or elemental-mixing path) before cell sizing. Useful for tight-network
        compositions (covalent semiconductors, metallic glasses) where the
        sphere-packing model underestimates the equilibrium amorphous
        density by 15-25%; setting ``density_scale=1.2`` boosts the auto
        density 20%.  Has no effect when ``target_density`` is supplied
        explicitly.
    minsep : dict, optional
        Minimum pair separations, e.g. {"In-In": 2.8, "In-O": 1.9}.
        If None, auto-generated from Shannon ionic / metallic / covalent
        radii with bonding-type-aware scale factors.
    minsep_scale : float
        Fallback scale factor for default minsep (default 0.85).
    seed : int, optional
        Random seed for reproducibility.
    max_attempts_per_atom : int
        Max placement attempts per atom before raising an error.
    pbc : bool
        Periodic boundary conditions.
    target_cn : dict, optional
        Target coordination numbers, e.g. {"Si": 4, "O": 2}.
        Enables coordination-aware placement (atoms biased toward
        existing under-coordinated sites within the bonding shell).
        Also uses CN-specific Shannon radii for tighter minsep.
    dmax : dict, optional
        Maximum bond distances (defines "bonded"), e.g.
        {"Si-O": 2.0}. Auto-generated from minsep * 1.5 if not provided.
    cn_tolerance : int
        Over-coordination tolerance for coordination-aware placement.
        Default 0 (strict:
        reject if any neighbour is at target CN). Set to 1 to allow
        temporary +1 over-coordination for tighter CN matching.
    dmax_factor : float
        Multiplier for auto dmax: dmax = minsep * dmax_factor (default 1.5).
        Controls the width of the bonding shell. Ignored if dmax is
        provided explicitly.
    repair_iters : int, default 0
        **Experimental.** If > 0, run a post-placement repair loop that
        relocates under-coordinated atoms within the bonding shell of
        other under-coordinated atoms, accepting moves that reduce the
        total under-coord count.  Useful for tetrahedral covalent
        networks (a-Si, a-Ge) where greedy placement leaves many atoms
        below target CN.  Default 0 (off).  See
        :func:`_repair_undercoordination` for caveats.

    Returns
    -------
    ase.Atoms
    """
    rng = np.random.default_rng(seed)

    symbols = []
    for species, count in composition.items():
        symbols.extend([species] * count)
    n_atoms = len(symbols)

    rng.shuffle(symbols)

    # Auto-detect target CN and tolerance if not provided
    # Empty dict {} means user explicitly disabled coordination-aware mode (--no-sc)
    if target_cn is None:
        target_cn, auto_tol = _auto_target_cn(composition)
        if cn_tolerance is None:
            cn_tolerance = auto_tol
    elif target_cn == {}:
        target_cn = None  # disable coordination-aware placement
    # If cn_tolerance still None, default 0
    if cn_tolerance is None:
        cn_tolerance = 0

    if cell_length_ang is None:
        cell_length_ang = _estimate_cell_length(
            composition, target_density,
            density_scale=density_scale,
        )
    L = cell_length_ang

    if minsep is None:
        minsep = _default_minsep(symbols, scale=minsep_scale,
                                target_cn=target_cn)

    # Coordination-aware mode: auto-generate dmax from minsep if not provided
    use_sc = target_cn is not None
    if use_sc and dmax is None:
        dmax = _auto_dmax(minsep, target_cn, factor=dmax_factor)
        logger.info("Auto-generated dmax from minsep * 1.5: %s", dmax)

    # CN tracking array (only used in coordination-aware mode)
    cn_array = np.zeros(n_atoms, dtype=int) if use_sc else None

    positions = np.empty((n_atoms, 3))
    placed_symbols = []
    n_placed = 0

    # Precompute lookup tables for O(1) access (indexed by element type)
    unique_syms = sorted(set(symbols))
    _sym_to_idx = {s: i for i, s in enumerate(unique_syms)}
    _n_types = len(unique_syms)

    # Minsep tables
    _minsep_sq_table = np.zeros((_n_types, _n_types))
    for s1 in unique_syms:
        for s2 in unique_syms:
            d = _get_minsep(s1, s2, minsep)
            _minsep_sq_table[_sym_to_idx[s1], _sym_to_idx[s2]] = d * d

    # Dmax table (for coordination-aware mode)
    _dmax_table = np.zeros((_n_types, _n_types))
    _dmax_sq_table = np.zeros((_n_types, _n_types))
    if use_sc and dmax:
        for s1 in unique_syms:
            for s2 in unique_syms:
                d = _get_dmax(s1, s2, dmax)
                _dmax_table[_sym_to_idx[s1], _sym_to_idx[s2]] = d
                _dmax_sq_table[_sym_to_idx[s1], _sym_to_idx[s2]] = d * d

    # Target CN + tolerance table (for coordination-aware mode)
    _target_cn_arr = np.full(_n_types, 999, dtype=int)
    if use_sc and target_cn:
        for s, cn_val in target_cn.items():
            if s in _sym_to_idx:
                _target_cn_arr[_sym_to_idx[s]] = cn_val

    # Track integer type index for each placed atom
    placed_type_idx = np.empty(n_atoms, dtype=int)

    import time as _time
    _t_start = _time.time()
    _last_progress = _t_start

    for i, sym in enumerate(symbols):
        sym_idx = _sym_to_idx[sym]
        if n_placed > 0:
            min_dists_sq = _minsep_sq_table[sym_idx, placed_type_idx[:n_placed]]

        placed = False

        # -- Coordination-aware path --
        # Find seeds that need more bonds (vectorized).
        if use_sc and n_placed > 0:
            types_placed = placed_type_idx[:n_placed]
            dm_to_new = _dmax_table[types_placed, sym_idx]
            tgt_placed = _target_cn_arr[types_placed]
            cn_placed = cn_array[:n_placed]

            # Bondable: has dmax > 0, target > 0, cn < target + tolerance
            bondable_mask = (
                (dm_to_new > 0)
                & (tgt_placed < 999)
                & (cn_placed < tgt_placed + cn_tolerance)
            )
            bondable_idx = np.where(bondable_mask)[0]

            if len(bondable_idx) > 0:
                # Sort by deficit (most under-coordinated first)
                deficits = tgt_placed[bondable_idx] - cn_placed[bondable_idx]
                order = np.argsort(-deficits)
                bondable_idx = bondable_idx[order]

                attempts_per_seed = max(
                    1000, max_attempts_per_atom // len(bondable_idx)
                )

                # Precompute over-coordination check arrays for this atom type
                dm_check = _dmax_sq_table[types_placed, sym_idx]
                tgt_check = _target_cn_arr[types_placed]

                for seed_idx in bondable_idx:
                    seed_pos = positions[seed_idx]
                    ms_sq = _minsep_sq_table[
                        placed_type_idx[seed_idx], sym_idx]
                    ms = ms_sq ** 0.5
                    dm = dm_to_new[seed_idx]

                    for attempt in range(attempts_per_seed):
                        direction = rng.standard_normal(3)
                        direction /= np.linalg.norm(direction)
                        distance = rng.uniform(ms, dm)
                        pos = seed_pos + direction * distance
                        if pbc:
                            pos = pos % L

                        d_vec = pos - positions[:n_placed]
                        if pbc:
                            d_vec -= L * np.round(d_vec / L)
                        d_sq = np.sum(d_vec * d_vec, axis=1)

                        if not np.all(d_sq >= min_dists_sq):
                            continue

                        # Vectorized over-coordination check:
                        # Find neighbours within dmax that are already
                        # at or above target + tolerance (excluding seed)
                        within_dmax = (dm_check > 0) & (d_sq <= dm_check)
                        within_dmax[seed_idx] = False
                        if np.any(within_dmax):
                            over = cn_placed[within_dmax] >= (
                                tgt_check[within_dmax] + cn_tolerance)
                            if np.any(over):
                                continue

                        # Accept
                        positions[n_placed] = pos
                        placed_symbols.append(sym)
                        placed_type_idx[n_placed] = sym_idx
                        n_placed += 1
                        _update_cn_array(cn_array, n_placed - 1,
                                         positions, n_placed,
                                         placed_type_idx, sym_idx,
                                         _dmax_sq_table, L, pbc)
                        placed = True
                        break

                    if placed:
                        break

        # -- Standard path: pure rejection sampling --
        if not placed:
            for attempt in range(max_attempts_per_atom):
                pos = rng.random(3) * L

                if n_placed == 0:
                    positions[0] = pos
                    placed_symbols.append(sym)
                    placed_type_idx[0] = sym_idx
                    n_placed = 1
                    if use_sc:
                        cn_array[0] = 0
                    placed = True
                    break

                d_vec = pos - positions[:n_placed]
                if pbc:
                    d_vec -= L * np.round(d_vec / L)
                d_sq = np.sum(d_vec * d_vec, axis=1)

                if not np.all(d_sq >= min_dists_sq):
                    continue

                # Vectorized over-coordination check
                if use_sc:
                    types_p = placed_type_idx[:n_placed]
                    dm_sq_check = _dmax_sq_table[types_p, sym_idx]
                    within = (dm_sq_check > 0) & (d_sq <= dm_sq_check)
                    if np.any(within):
                        tgt_w = _target_cn_arr[types_p[within]]
                        cn_w = cn_array[:n_placed][within]
                        if np.any(cn_w >= tgt_w + cn_tolerance):
                            continue

                positions[n_placed] = pos
                placed_symbols.append(sym)
                placed_type_idx[n_placed] = sym_idx
                n_placed += 1
                if use_sc:
                    _update_cn_array(cn_array, n_placed - 1,
                                     positions, n_placed,
                                     placed_type_idx, sym_idx,
                                     _dmax_sq_table, L, pbc)
                placed = True
                break

        if not placed:
            raise RuntimeError(
                f"Could not place atom {i+1}/{n_atoms} ({sym}) after "
                f"{max_attempts_per_atom} attempts.\n"
                f"  Cell: {L:.2f} A, placed {n_placed}/{n_atoms} atoms so far.\n"
                f"  Suggestions:\n"
                f"    1. Use --target-density with a lower value\n"
                f"    2. Reduce minsep via --minsep flag\n"
                f"    3. Try --no-sc to disable coordination-aware placement\n"
                f"    4. Use batch_random() which auto-retries with reduced M-M minsep"
            )

        # Progress reporting (every 5 seconds for large systems)
        _now = _time.time()
        if _now - _last_progress > 5.0:
            _last_progress = _now
            elapsed = _now - _t_start
            logger.info("  Placed %d/%d atoms (%.1f s)", n_placed, n_atoms, elapsed)

    # Optional repair pass for under-coordinated atoms (experimental).
    if use_sc and repair_iters > 0:
        _repair_undercoordination(
            positions=positions,
            n_placed=n_placed,
            placed_type_idx=placed_type_idx,
            cn_array=cn_array,
            target_cn_arr=_target_cn_arr,
            cn_tolerance=cn_tolerance,
            minsep_sq_table=_minsep_sq_table,
            dmax_sq_table=_dmax_sq_table,
            L=L,
            pbc=pbc,
            max_iter=repair_iters,
            rng=rng,
        )

    atoms = Atoms(
        symbols=placed_symbols,
        positions=positions[:n_placed],
        cell=[L, L, L],
        pbc=pbc,
    )
    atoms.wrap()

    # Post-placement CN report (coordination-aware mode)
    sc_report = {}
    if use_sc:
        for elem, tgt in target_cn.items():
            indices = [k for k, s in enumerate(placed_symbols) if s == elem]
            cns = [int(cn_array[k]) for k in indices]
            if cns:
                sc_report[elem] = {
                    "target": tgt,
                    "mean": float(np.mean(cns)),
                    "min": min(cns),
                    "max": max(cns),
                }
        for elem in sorted(set(placed_symbols)):
            if elem not in target_cn:
                indices = [k for k, s in enumerate(placed_symbols) if s == elem]
                cns = [int(cn_array[k]) for k in indices]
                if cns:
                    sc_report[elem] = {
                        "target": "auto",
                        "mean": float(np.mean(cns)),
                        "min": min(cns),
                        "max": max(cns),
                    }
        atoms.info["sc_report"] = sc_report

    return atoms


# ==============================================================================
# Batch generation
# ==============================================================================

# Map output format names to ASE format strings and file extensions
_FORMAT_MAP = {
    "xyz":    ("extxyz", ".xyz"),     # default: extxyz format, .xyz extension
    "extxyz": ("extxyz", ".xyz"),     # alias
    "vasp":   ("vasp",   ".vasp"),
    "cif":    ("cif",    ".cif"),
}


def batch_random(
    composition: dict[str, int],
    n_structures: int = 1,
    output_dir: str = "random_structures",
    output_format: str = "xyz",
    relax: bool = False,
    calc=None,
    fmax: float = 0.05,
    max_relax_steps: int = 200,
    optimizer: str = "FIRE",
    cell_filter: str = "FrechetCellFilter",
    max_retries: int = 10,
    resume: bool = False,
    **kwargs,
) -> list[str]:
    """
    Generate multiple random structures, optionally relaxing each.

    Parameters
    ----------
    composition : dict
    n_structures : int
    output_dir : str
    output_format : str
        Output file format: "xyz" (default, extxyz format), "vasp", "cif".
    relax : bool
        If True and calc is provided, optimise each structure.
    calc : ASE calculator, optional
    fmax : float
    max_relax_steps : int
    optimizer : str
    cell_filter : str
    max_retries : int
    **kwargs
        Forwarded to generate_random().

    Returns
    -------
    list of str
        Paths to generated structure files. Returns file paths (not Atoms
        objects) because batch generation writes each structure to disk
        as it is created, enabling recovery from interruptions and
        keeping memory usage constant regardless of batch size.

    Notes
    -----
    For a single structure as an Atoms object, use ``generate_random()``
    directly. Keyword arguments (``target_cn``, ``minsep``, ``dmax``,
    ``minsep_scale``, ``cn_tolerance``, ``target_density``,
    ``density_scale``, ``cell_length_ang``, ``max_attempts_per_atom``)
    are forwarded to ``generate_random()``.
    """
    if output_format not in _FORMAT_MAP:
        raise ValueError(
            f"Unknown output format '{output_format}'. "
            f"Choose from: {', '.join(_FORMAT_MAP)}"
        )
    ase_format, ext = _FORMAT_MAP[output_format]

    os.makedirs(output_dir, exist_ok=True)
    paths = []

    # ── Resume support: scan for existing completed structures ──
    existing_indices = set()
    if resume:
        import json as _json

        # Write or check run metadata for consistency
        meta_path = os.path.join(output_dir, "run_metadata.json")
        current_meta = {
            "composition": composition,
            "output_format": output_format,
            "relax": relax,
        }
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as mf:
                    prev_meta = _json.load(mf)
                if prev_meta.get("composition") != composition:
                    import warnings
                    warnings.warn(
                        f"Resume: composition changed from "
                        f"{prev_meta['composition']} to {composition}. "
                        f"Structures may be inconsistent."
                    )
            except Exception:
                pass  # corrupted metadata, proceed anyway

        # Scan for completed files
        for idx in range(n_structures):
            if relax:
                check_file = os.path.join(
                    output_dir, f"random_{idx:04d}_opt{ext}")
            else:
                check_file = os.path.join(
                    output_dir, f"random_{idx:04d}{ext}")
            if os.path.isfile(check_file) and os.path.getsize(check_file) > 0:
                # Validate ASE can read it
                try:
                    from ase.io import read as _read
                    _read(check_file)
                    existing_indices.add(idx)
                    paths.append(check_file)
                except Exception:
                    pass  # corrupted file, will regenerate

        if existing_indices:
            missing = [i for i in range(n_structures)
                       if i not in existing_indices]
            print(f"  Resume: found {len(existing_indices)} existing "
                  f"structures, generating {len(missing)} more "
                  f"(indices {missing[0]}-{missing[-1]})"
                  if missing else
                  f"  Resume: all {n_structures} structures already exist.")
        else:
            print(f"  Resume: no existing structures found, "
                  f"starting fresh.")

    base_seed = kwargs.pop("seed", None)

    target_cn = kwargs.get("target_cn")
    # If target_cn not specified, auto-detect for logging
    if target_cn is None:
        target_cn, _ = _auto_target_cn(composition)
        if target_cn is not None:
            kwargs["target_cn"] = target_cn
    dmax_user = kwargs.get("dmax")

    # Pre-compute minsep and cell for logging
    symbols_all = []
    for species, count in composition.items():
        symbols_all.extend([species] * count)
    n_atoms = len(symbols_all)
    formula = Atoms(symbols_all).get_chemical_formula(mode="hill")

    minsep_log = kwargs.get("minsep")
    if minsep_log is None:
        minsep_log = _default_minsep(symbols_all,
                                     scale=kwargs.get("minsep_scale", 0.85))

    td = kwargs.get("target_density")
    ds = kwargs.get("density_scale", 1.0)
    cell_L = kwargs.get("cell_length_ang")
    if cell_L is None:
        cell_L = _estimate_cell_length(composition, td, density_scale=ds)
    vol = cell_L ** 3
    total_mass = sum(atomic_masses[atomic_numbers[s]] * n
                     for s, n in composition.items())
    est_density = (total_mass / 6.022e23) / (vol * 1e-24)

    use_sc = target_cn is not None and target_cn != {}
    dmax_fac = kwargs.get("dmax_factor", 1.5)
    if use_sc and dmax_user is None:
        dmax_log = _auto_dmax(minsep_log, target_cn, factor=dmax_fac)
    else:
        dmax_log = dmax_user

    # -- Write log file --
    logfile = os.path.join(output_dir, "random_gen.log")
    bar = "=" * 60

    def _log(msg, lf):
        print(msg)
        lf.write(msg + "\n")
        lf.flush()

    lf = open(logfile, "a" if resume else "w")
    try:
        _log(f"\n{bar}", lf)
        _log(f"  AmorphGen - Random Structure Generation", lf)
        _log(f"  Composition: {formula} ({n_atoms} atoms)", lf)
        _log(f"  Cell: {cell_L:.2f} A (cubic), Volume: {vol:.1f} A3", lf)
        _log(f"  Estimated density: {est_density:.2f} g/cm3", lf)
        if td is not None:
            _log(f"  Target density: {td:.2f} g/cm3", lf)
        else:
            has_gas_element = any(
                ELEMENTAL_DENSITIES.get(s) is None
                for s in composition
            )
            if has_gas_element:
                _log(f"  NOTE: Auto density is approximate for this composition.", lf)
                _log(f"        Use --target-density for more accurate results.", lf)
        _log(f"  Mode: {'Coordination-aware' if use_sc else 'Standard rejection'}", lf)
        _log(f"  N structures: {n_structures}", lf)
        _log(f"  Output format: {output_format}", lf)
        _log(f"  Output dir: {output_dir}/", lf)
        if use_sc:
            _log(f"  Target CN: {target_cn}", lf)
        _log(f"{bar}", lf)

        # One-line auto-derivation summary — captures the chemistry-informed
        # decisions (class, OS, CN, bond types, Pauling Δχ, minsep, density)
        # in a single grep-friendly log line.
        _log(_format_auto_derive_summary(
            composition=composition,
            target_cn=target_cn if use_sc else None,
            minsep=minsep_log,
            est_density=est_density,
            cell_length=cell_L,
        ), lf)

        _log(f"\n  Minsep values:", lf)
        for pair in sorted(minsep_log):
            _log(f"    {pair}: {minsep_log[pair]:.3f} A", lf)

        if use_sc and dmax_log:
            _log(f"\n  Dmax values{' (auto)' if dmax_user is None else ''}:", lf)
            for pair in sorted(dmax_log):
                _log(f"    {pair}: {dmax_log[pair]:.3f} A", lf)

        _log("", lf)

        # -- Generate structures --
        # Auto-retry: reduce M-M minsep by 5/10/15/20%
        generated = 0
        seed_offset = 0
        failures = 0
        retry_level = 0
        current_minsep = dict(kwargs.get("minsep") or minsep_log)

        def _reduce_mm_minsep(ms, reduction):
            """Reduce same-element minsep values by a factor."""
            reduced = dict(ms)
            for pair in reduced:
                s1, s2 = pair.split("-")
                if s1 == s2 and s1 not in NONMETALS:
                    reduced[pair] = reduced[pair] * (1.0 - reduction)
            return reduced

        # Save run metadata for resume consistency checks
        if resume or True:  # always write metadata
            import json as _json
            meta_path = os.path.join(output_dir, "run_metadata.json")
            meta = {
                "composition": composition,
                "output_format": output_format,
                "relax": relax,
                "n_structures": n_structures,
            }
            with open(meta_path, "w") as mf:
                _json.dump(meta, mf, indent=2)

        # Pre-warm calculator: model load + MPS graph compile happen once,
        # outside the per-structure timing. Otherwise the first structure
        # (or first structure after --resume) absorbs ~minutes of init cost.
        if relax and calc is not None:
            try:
                syms = list(composition.keys())
                positions = [[2.5 * i, 0.0, 0.0] for i in range(len(syms))]
                warmup_atoms = Atoms(
                    symbols=syms,
                    positions=positions,
                    cell=[10.0, 10.0, 10.0],
                    pbc=True,
                )
                warmup_atoms.calc = calc
                t_warm = time.perf_counter()
                warmup_atoms.get_potential_energy()
                t_warm = time.perf_counter() - t_warm
                _log(f"  Calculator warmup: {t_warm:.2f} s "
                     f"(model load + first inference)\n", lf)
            except Exception as e:
                _log(f"  Calculator warmup skipped ({type(e).__name__}: {e})\n",
                     lf)

        while generated < n_structures:
            # Skip if resume and this index already completed
            if generated in existing_indices:
                _log(f"  [{generated+1}/{n_structures}] Already exists "
                     f"-- skipping.", lf)
                generated += 1
                continue

            seed_i = base_seed + seed_offset if base_seed is not None else None
            seed_offset += 1

            try:
                atoms = generate_random(composition, seed=seed_i, **kwargs)
            except RuntimeError:
                failures += 1
                if failures >= max_retries:
                    if retry_level < 3:
                        retry_level += 1
                        reduction = 0.05 * retry_level
                        new_minsep = _reduce_mm_minsep(current_minsep, reduction)
                        kwargs["minsep"] = new_minsep
                        changed = []
                        for pair in sorted(new_minsep):
                            s1, s2 = pair.split("-")
                            if s1 == s2 and s1 not in NONMETALS:
                                changed.append(f"{pair}={new_minsep[pair]:.2f}")
                        _log(f"  [Auto-retry] Reducing M-M minsep by "
                             f"{reduction*100:.0f}%: {', '.join(changed)}", lf)
                        failures = 0
                        continue
                    elif retry_level < 4:
                        retry_level += 1
                        reduction = 0.05 * retry_level
                        new_minsep = _reduce_mm_minsep(current_minsep, reduction)
                        kwargs["minsep"] = new_minsep
                        changed = []
                        for pair in sorted(new_minsep):
                            s1, s2 = pair.split("-")
                            if s1 == s2 and s1 not in NONMETALS:
                                changed.append(f"{pair}={new_minsep[pair]:.2f}")
                        _log(f"  [Auto-retry] Reducing M-M minsep by "
                             f"{reduction*100:.0f}%: {', '.join(changed)}", lf)
                        failures = 0
                        continue
                    else:
                        _log(f"  [Warning] Skipping structure {generated + 1} "
                             f"after {max_retries} attempts x {retry_level+1} "
                             f"retries. Consider using --target-density.", lf)
                        failures = 0
                        retry_level = 0
                        generated += 1
                continue

            failures = 0

            # Save unrelaxed structure
            fname = os.path.join(output_dir, f"random_{generated:04d}{ext}")
            if ase_format == "vasp":
                sorted_atoms_ur = atoms[atoms.numbers.argsort()]
                write(fname, sorted_atoms_ur, format=ase_format, sort=True)
            else:
                write(fname, atoms, format=ase_format)

            if relax and calc is not None:
                from ..utils.common import compute_density_gcm3
                from ase.geometry import cell_to_cellpar
                atoms.calc = calc
                OptimizerClass = _get_optimizer_class(optimizer)
                target = _build_cell_filter(atoms, cell_filter)
                opt_logfile = os.path.join(output_dir,
                                           f"random_{generated:04d}_opt.log")

                # Detailed logging (consistent with batch-opt)
                formula = atoms.get_chemical_formula(mode="hill")
                cp = cell_to_cellpar(atoms.cell)
                vol = atoms.get_volume()
                density = compute_density_gcm3(atoms)
                _log(f"\n    Composition: {formula} ({len(atoms)} atoms)", lf)
                _log(f"    Initial cell: a={cp[0]:.4f}  b={cp[1]:.4f}  "
                     f"c={cp[2]:.4f}", lf)
                _log(f"    Volume: {vol:.2f} A^3  Density: {density:.2f} g/cm3", lf)
                _log(f"    Optimizer: {optimizer}  fmax={fmax}  "
                     f"max_steps={max_relax_steps}", lf)
                _log(f"    Cell filter: {cell_filter}", lf)

                header = (f"\n    {'Step':>5}  {'Energy(eV)':>14}  "
                          f"{'Fmax(eV/A)':>11}  {'a(A)':>10}  "
                          f"{'b(A)':>10}  {'c(A)':>10}  {'Vol(A3)':>10}")
                sep = "    " + "-" * 85
                _log(header, lf)
                _log(sep, lf)

                opt = OptimizerClass(target, logfile=None)
                t_relax_start = time.perf_counter()
                steps_done = 0
                for step in range(max_relax_steps):
                    opt.step()
                    energy = atoms.get_potential_energy()
                    forces = target.get_forces()
                    max_f = float((forces ** 2).sum(axis=1).max() ** 0.5)
                    cp = cell_to_cellpar(atoms.cell)
                    vol = atoms.get_volume()
                    line = (f"    {step+1:5d}  {energy:14.6f}  {max_f:11.6f}  "
                            f"{cp[0]:10.4f}  {cp[1]:10.4f}  {cp[2]:10.4f}  "
                            f"{vol:10.1f}")
                    _log(line, lf)
                    steps_done = step + 1
                    if max_f < fmax:
                        _log(sep, lf)
                        _log(f"\n    Converged after {step+1} steps!  "
                             f"Fmax = {max_f:.6f} eV/A", lf)
                        break
                else:
                    _log(sep, lf)
                    _log(f"\n    WARNING: did not converge in "
                         f"{max_relax_steps} steps.", lf)
                t_relax = time.perf_counter() - t_relax_start
                per_step = t_relax / steps_done if steps_done else float("nan")
                _log(f"    Wall time: {t_relax:.2f} s  "
                     f"({steps_done} steps, {per_step:.3f} s/step)", lf)

                density_f = compute_density_gcm3(atoms)
                _log(f"    Final density: {density_f:.2f} g/cm3", lf)

                # Save relaxed structure as _opt
                fname_opt = os.path.join(
                    output_dir, f"random_{generated:04d}_opt{ext}")
                if ase_format == "vasp":
                    sorted_atoms_r = atoms[atoms.numbers.argsort()]
                    write(fname_opt, sorted_atoms_r, format=ase_format,
                          sort=True)
                else:
                    write(fname_opt, atoms, format=ase_format)
                fname = fname_opt  # return path to relaxed version

            paths.append(fname)
            out_formula = atoms.get_chemical_formula(mode="hill")
            seed_str = f" (seed={seed_i})" if seed_i is not None else ""
            _log(f"  [{generated+1}/{n_structures}] {out_formula} -> "
                 f"{fname}{seed_str}", lf)

            sc_report = atoms.info.get("sc_report")
            if sc_report:
                for elem, data in sorted(sc_report.items()):
                    _log(f"    CN: {elem} target={data['target']}, "
                         f"mean={data['mean']:.1f}, "
                         f"range=[{data['min']},{data['max']}]", lf)
            generated += 1

        _log(f"\n  Generated {len(paths)} structures in {output_dir}/", lf)
        if len(paths) < n_structures:
            _log(f"  Warning: only {len(paths)}/{n_structures} structures "
                 f"were successfully generated.", lf)
        _log(f"  Log saved: {logfile}", lf)
    finally:
        lf.close()

    return paths
