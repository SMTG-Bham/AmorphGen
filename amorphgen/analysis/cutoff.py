"""Auto-cutoff detection for neighbour analysis."""

from __future__ import annotations

import warnings
import numpy as np
from ase.neighborlist import neighbor_list


def auto_cutoff_minsep(atoms_list: list) -> dict[str, float]:
    """Estimate pair-specific cutoffs from bonding radii x 1.3."""
    try:
        from ..pipeline.random_gen import (
            _default_minsep, _classify_bond, NONMETALS, METALLOIDS
        )
    except ImportError:
        from amorphgen.pipeline.random_gen import (
            _default_minsep, _classify_bond, NONMETALS, METALLOIDS
        )

    symbols = atoms_list[0].get_chemical_symbols()
    minsep = _default_minsep(symbols)

    cutoffs = {}
    for pair, dist in minsep.items():
        s1, s2 = pair.split("-")
        bond_type = _classify_bond(s1, s2)
        if bond_type == "ionic" or (bond_type == "covalent" and s1 != s2):
            cutoffs[pair] = dist * 1.3
        elif s1 == s2 and s1 not in NONMETALS and s1 not in METALLOIDS:
            cutoffs[pair] = dist * 1.15
        else:
            cutoffs[pair] = dist * 1.2

    return cutoffs


def auto_cutoff_rdf(atoms_list: list, rmax: float = 6.0,
                    nbins: int = 300) -> dict[str, float]:
    """Determine pair-specific cutoffs from the first minimum of g(r)."""
    unique = sorted(set(atoms_list[0].get_chemical_symbols()))
    dr = rmax / nbins
    r_centres = np.linspace(dr / 2, rmax - dr / 2, nbins)

    pairs = []
    for i, s1 in enumerate(unique):
        for s2 in unique[i:]:
            pairs.append(f"{s1}-{s2}")

    g_r_accum = {p: np.zeros(nbins) for p in pairs}
    n_frames = len(atoms_list)

    for atoms in atoms_list:
        idx_i, idx_j, dists = neighbor_list('ijd', atoms, cutoff=rmax)
        syms = np.array(atoms.get_chemical_symbols())
        vol = atoms.get_volume()

        for pair_key in pairs:
            p1, p2 = pair_key.split("-")
            n_source = int(np.sum(syms == p1))
            n_target = int(np.sum(syms == p2))
            if n_source == 0 or n_target == 0:
                continue

            if p1 == p2:
                rho_target = (n_target - 1) / vol
            else:
                rho_target = n_target / vol

            mask = (syms[idx_i] == p1) & (syms[idx_j] == p2)
            pair_dists = dists[mask]
            if len(pair_dists) == 0:
                continue

            in_range = (pair_dists > 0) & (pair_dists < rmax)
            if not np.any(in_range):
                continue
            hist, _ = np.histogram(pair_dists[in_range], bins=nbins,
                                   range=(0, rmax))

            for i_bin in range(nbins):
                r = r_centres[i_bin]
                shell_vol = 4 * np.pi * r**2 * dr
                if shell_vol > 0 and rho_target > 0:
                    g_r_accum[pair_key][i_bin] += (
                        hist[i_bin] / (n_source * rho_target * shell_vol))

    for key in g_r_accum:
        g_r_accum[key] /= n_frames

    cutoffs = {}
    fallback = auto_cutoff_minsep(atoms_list)

    for key, g_r in g_r_accum.items():
        cutoff_found = False
        g_smooth = (np.convolve(g_r, np.ones(5) / 5, mode='same')
                    if len(g_r) > 5 else g_r)

        # First peak = the first local maximum that is at least half the
        # strongest feature of g(r).  Taking the first bump above a fixed
        # threshold (old behaviour) latched onto placement noise on the rising
        # edge of unrelaxed random structures and returned a cutoff *below*
        # the bond length (CN ~ 0).
        g_max = np.max(g_smooth) if len(g_smooth) > 0 else 0
        peak_threshold = max(1.5, 0.5 * g_max)   # a bonded shell has g >> 1

        peak_idx = None
        for i in range(1, len(g_smooth) - 1):
            if (g_smooth[i] >= peak_threshold
                    and g_smooth[i] >= g_smooth[i - 1]
                    and g_smooth[i] > g_smooth[i + 1]):
                peak_idx = i
                break

        # First minimum after the peak: a local minimum that is a genuine
        # depletion (g <= half the peak height), so a small dip on the
        # descending flank is skipped.
        if peak_idx is not None:
            g_peak = g_smooth[peak_idx]
            for i in range(peak_idx + 1, len(g_smooth) - 1):
                if (g_smooth[i] <= 0.5 * g_peak
                        and g_smooth[i] <= g_smooth[i - 1]
                        and g_smooth[i] <= g_smooth[i + 1]):
                    cutoffs[key] = float(r_centres[i])
                    cutoff_found = True
                    break

        if not cutoff_found:
            cutoffs[key] = fallback.get(key, 3.5)
            warnings.warn(
                f"auto-rdf cutoff: no clear first minimum in g(r) for {key}; "
                f"using the radii-table cutoff {cutoffs[key]:.2f} A instead "
                f"(structures may be unrelaxed).", stacklevel=2)

    return cutoffs


def check_rmax(atoms_list: list, rmax: float) -> None:
    """Warn if rmax exceeds half the minimum cell length."""
    for atoms in atoms_list:
        half_cell = min(atoms.cell.lengths()) / 2
        if rmax > half_cell:
            warnings.warn(
                f"rmax={rmax:.1f} A exceeds half the shortest cell "
                f"vector ({half_cell:.1f} A).",
                stacklevel=4,
            )
            return


_BASE_MODES = ("auto-rdf", "auto")


def parse_cutoff_spec(spec):
    """Parse a cutoff specification from the CLI or YAML.

    Accepted forms::

        "auto-rdf" | "auto" | 2.4                 -> one rule for every pair
        "In-O=2.6,Zn-O=2.3"                       -> per-pair overrides on top of auto-rdf
        "auto,In-O=2.6"  /  "2.4,In-O=2.6"        -> overrides on top of that base
        {"In-O": 2.6}  /  {"default": 2.4, "In-O": 2.6}   (YAML dict)

    Returns a str, a float, or a dict whose optional ``"default"`` entry is
    the base rule for pairs not listed (``"auto-rdf"`` when absent).
    """
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, dict):
        out = {}
        for k, v in spec.items():
            if k == "default":
                out["default"] = parse_cutoff_spec(v) if isinstance(v, str) else float(v)
            else:
                out[str(k)] = float(v)
        return out
    s = str(spec).strip()
    if s in _BASE_MODES:
        return s
    try:
        return float(s)
    except ValueError:
        pass
    out = {}
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" in tok:
            pair, val = tok.split("=", 1)
            pair = pair.strip()
            if pair.count("-") != 1:
                raise ValueError(f"cutoff override '{tok}': expected A-B=value")
            out[pair] = float(val)
        elif tok in _BASE_MODES:
            out["default"] = tok
        else:
            out["default"] = float(tok)      # raises ValueError with a clear message
    if not out:
        raise ValueError(f"invalid cutoff '{spec}'")
    return out


def resolve_cutoffs(atoms_list, spec):
    """Turn a parsed cutoff spec into ``(cutoffs, mode_label)``.

    ``cutoffs`` is a float (one value for every pair) or a dict with an entry
    for every element pair present in ``atoms_list``: the base rule filled in
    first (``auto-rdf`` first minima, ``auto`` radii minseps, or a number),
    then the per-pair overrides applied on top.
    """
    spec = parse_cutoff_spec(spec)
    if not isinstance(spec, dict):
        if spec == "auto":
            return auto_cutoff_minsep(atoms_list), "auto (minsep)"
        if spec == "auto-rdf":
            return auto_cutoff_rdf(atoms_list), "auto (RDF)"
        return float(spec), "fixed"
    overrides = {k: v for k, v in spec.items() if k != "default"}
    base = spec.get("default", "auto-rdf")
    unique = sorted(set(atoms_list[0].get_chemical_symbols()))
    pairs = [f"{a}-{b}" for i, a in enumerate(unique) for b in unique[i:]]
    if base == "auto":
        table, label = auto_cutoff_minsep(atoms_list), "auto (minsep)"
    elif base == "auto-rdf":
        table, label = auto_cutoff_rdf(atoms_list), "auto (RDF)"
    else:
        table, label = {p: float(base) for p in pairs}, f"fixed {float(base):.2f} A"
    cutoffs = {}
    for p in pairs:
        a, b = p.split("-")
        rev = f"{b}-{a}"
        if p in overrides:
            cutoffs[p] = overrides[p]
        elif rev in overrides:
            cutoffs[p] = overrides[rev]
        else:
            cutoffs[p] = table.get(p, table.get(rev, 3.5))
    unknown = [k for k in overrides if k not in pairs and
               "-".join(reversed(k.split("-"))) not in pairs]
    if unknown:
        import warnings
        warnings.warn(f"cutoff override(s) for pair(s) not in the structure: "
                      f"{', '.join(unknown)}", stacklevel=2)
    if overrides:
        label += " + overrides " + ", ".join(f"{k}={v:.2f}" for k, v in overrides.items())
    return cutoffs, label
