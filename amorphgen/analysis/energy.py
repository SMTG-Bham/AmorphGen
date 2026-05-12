"""Energy ranking for multiple structures."""

from __future__ import annotations

import re


def compute_energy_ranking(atoms_list):
    """
    Rank structures by potential energy.

    Reads energy from atoms.info or calculator.
    """
    energies = []
    for atoms in atoms_list:
        e = None
        for key in ['energy', 'Energy', 'potential_energy']:
            if key in atoms.info:
                e = atoms.info[key]
                break
        if e is None:
            try:
                e = atoms.get_potential_energy()
            except Exception:
                e = None
        energies.append(e)

    valid = [(i, e) for i, e in enumerate(energies) if e is not None]
    if not valid:
        return {
            "energies_per_atom": {},
            "ranking": [],
            "best": None,
            "worst": None,
            "best_energy": None,
            "worst_energy": None,
            "spread": 0.0,
            "warning": "No energy data found in structures",
        }

    e_per_atom = [(i, e / len(atoms_list[i])) for i, e in valid]
    e_per_atom.sort(key=lambda x: x[1])

    ranking = [i for i, _ in e_per_atom]
    energies_sorted = [e for _, e in e_per_atom]

    return {
        "energies_per_atom": {i: e for i, e in e_per_atom},
        "ranking": ranking,
        "best": ranking[0],
        "worst": ranking[-1],
        "best_energy": energies_sorted[0],
        "worst_energy": energies_sorted[-1],
        "spread": energies_sorted[-1] - energies_sorted[0],
    }


# ── Log-based ranking (for --random-gen --relax outputs) ──────────────────

_HDR_RE  = re.compile(r"\[\s*\d+/\d+\]\s+\S+\s+->\s+\S*random_(\d+)_opt\.\w+")
_STEP_RE = re.compile(r"^\s*\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
_CONV_RE = re.compile(r"Converged after (\d+) steps!")
_FAIL_RE = re.compile(r"WARNING: did not converge")
_COMP_RE = re.compile(r"^\s+Composition:\s+\S+\s+\((\d+)\s+atoms\)")


def rank_from_log(logfile):
    """Parse a random-gen log file and rank structures by total energy.

    The relax loop in batch_random() prints final energy on the last
    optimizer step row. This function reads those rows directly, so
    energy ranking works for VASP/CIF outputs that don't store energy.

    Parameters
    ----------
    logfile : str
        Path to ``random_gen.log``.

    Returns
    -------
    dict
        ``{"rows": [(idx, energy, e_per_atom, fmax, n_steps, status), ...]
        sorted by e_per_atom ascending, "n_atoms": int, "best": idx,
        "worst": idx, "spread_meV_per_atom": float}``.
    """
    rows_by_idx = {}
    n_atoms = None
    in_block = False
    last_e = last_fmax = n_steps = status = None

    with open(logfile) as f:
        for line in f:
            cm = _COMP_RE.match(line)
            if cm:
                if n_atoms is None:
                    n_atoms = int(cm.group(1))
                in_block = True
                last_e = last_fmax = n_steps = status = None
                continue
            if not in_block:
                continue
            sm = _STEP_RE.match(line)
            if sm:
                last_e = float(sm.group(1))
                last_fmax = float(sm.group(2))
                continue
            cv = _CONV_RE.search(line)
            if cv:
                n_steps = int(cv.group(1))
                status = "converged"
                continue
            if _FAIL_RE.search(line):
                status = "not converged"
                continue
            hm = _HDR_RE.search(line)
            if hm and last_e is not None:
                idx = int(hm.group(1))
                rows_by_idx[idx] = (last_e, last_fmax, n_steps, status)
                in_block = False

    if n_atoms is None or not rows_by_idx:
        return {"rows": [], "n_atoms": n_atoms, "best": None, "worst": None,
                "spread_meV_per_atom": 0.0}

    rows = [(idx, e, e / n_atoms, fmax, n, st)
            for idx, (e, fmax, n, st) in rows_by_idx.items()]
    rows.sort(key=lambda r: r[2])
    return {
        "rows": rows,
        "n_atoms": n_atoms,
        "best": rows[0][0],
        "worst": rows[-1][0],
        "spread_meV_per_atom": (rows[-1][2] - rows[0][2]) * 1000.0,
    }


def format_log_ranking(result, logfile=None):
    """Render the dict from rank_from_log() as a printable table."""
    rows = result["rows"]
    if not rows:
        return "No converged structures found in log."
    n_atoms = result["n_atoms"]
    lines = []
    src = f" from {logfile}" if logfile else ""
    lines.append(f"\n  Energy ranking{src} "
                 f"({len(rows)} structures, {n_atoms} atoms each)")
    lines.append("  " + "-" * 72)
    lines.append(f"  {'Rank':>4}  {'Idx':>4}  {'E (eV)':>13}  "
                 f"{'E/atom (eV)':>13}  {'Fmax':>7}  {'Steps':>5}  Status")
    lines.append("  " + "-" * 72)
    for rank, (idx, e, e_pa, fmax, n, st) in enumerate(rows, 1):
        n_str = f"{n}" if n is not None else "N/A"
        lines.append(f"  {rank:>4}  {idx:>4}  {e:>13.4f}  {e_pa:>13.6f}  "
                     f"{fmax:>7.4f}  {n_str:>5}  {st}")
    lines.append("  " + "-" * 72)
    lines.append(f"  Best : random_{result['best']:04d}_opt  ->  "
                 f"{rows[0][2]:.6f} eV/atom")
    lines.append(f"  Worst: random_{result['worst']:04d}_opt  ->  "
                 f"{rows[-1][2]:.6f} eV/atom")
    lines.append(f"  Spread: {result['spread_meV_per_atom']:.1f} meV/atom")
    return "\n".join(lines)
