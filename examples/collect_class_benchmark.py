#!/usr/bin/env python
"""Summarise the class benchmark into one table.

Usage: collect_class_benchmark.py CLASSES_DIR SYSTEMS_FILE

For every system listed in SYSTEMS_FILE it reads classes/<name>/random_gen.log
(estimated density), the placed and relaxed structures (densities, energies)
and classes/<name>/analysis/report.txt (bonding coordination, dimers, edge
sharing) and writes classes/class_benchmark.tsv plus a short console table.
"""
import glob
import os
import re
import sys

import numpy as np
from ase.io import read


def density(atoms):
    from ase.units import _amu
    m = atoms.get_masses().sum() * _amu * 1e3          # g
    return m / (atoms.get_volume() * 1e-24)             # g/cm3


def stats(files):
    rho, e = [], []
    for f in files:
        a = read(f)
        rho.append(density(a))
        try:
            e.append(a.get_potential_energy() / len(a))
        except Exception:
            pass
    return (np.mean(rho) if rho else np.nan, np.std(rho) if rho else np.nan,
            np.mean(e) if e else np.nan)


def parse_report(path):
    out = {"cn": "", "dimers": "", "edge": ""}
    if not os.path.isfile(path):
        return out
    txt = open(path).read()
    m = re.search(r"Bonding coordination numbers:\n(.*?)(?:\n\n|\n  Non-bonded|\n  Total coordination)", txt, re.S)
    if m:
        pairs = re.findall(r"^\s+(\S+): mean=([\d.]+)", m.group(1), re.M)
        out["cn"] = " ".join(f"{p}={v}" for p, v in pairs[:4])
    m = re.search(r"(\d+) dimer\(s\) in (\d+)/(\d+) structure", txt)
    out["dimers"] = f"{m.group(2)}/{m.group(3)}" if m else ("0" if "DIMER-FREE" in txt else "")
    m = re.search(r"per-structure edge/face-sharing cations: ([\d.]+)", txt)
    out["edge"] = m.group(1) if m else ""
    return out


def main(root, systems_file):
    rows = []
    for line in open(systems_file):
        if not line.strip() or line.startswith("#"):
            continue
        cls, name, comp, rho_ref, kind = line.split()
        d = os.path.join(root, name)
        try:                                   # the class the code actually assigned
            from amorphgen.cli import _parse_composition
            from amorphgen.utils.radii import _classify_compound
            code_cls = _classify_compound(_parse_composition(comp))
        except Exception:
            code_cls = ""
        est = np.nan
        log = os.path.join(d, "random_gen.log")
        if os.path.isfile(log):
            m = re.search(r"Estimated density:\s*([\d.]+)", open(log).read())
            est = float(m.group(1)) if m else np.nan
        placed = sorted(glob.glob(os.path.join(d, "random_initial", "*.xyz")))
        relaxed = sorted(glob.glob(os.path.join(d, "random_opt", "*_opt.xyz")))
        rp, _, _ = stats(placed)
        rr, rs, e = stats(relaxed)
        rep = parse_report(os.path.join(d, "analysis", "report.txt"))
        ref = float(rho_ref)
        rows.append(dict(cls=cls, code_cls=code_cls, name=name, comp=comp, n_placed=len(placed), n_relaxed=len(relaxed),
                         rho_est=est, rho_placed=rp, rho_relaxed=rr, rho_std=rs, e_atom=e,
                         rho_ref=ref, kind=kind,
                         ratio=(rr / ref if rr == rr else np.nan), **rep))
    cols = ["cls", "code_cls", "name", "comp", "n_placed", "n_relaxed", "rho_est", "rho_placed", "rho_relaxed",
            "rho_std", "rho_ref", "kind", "ratio", "e_atom", "cn", "dimers", "edge"]
    out = os.path.join(root, "class_benchmark.tsv")
    with open(out, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    print(f"{'class':<24} {'system':<9} {'est':>5} {'placed':>6} {'relaxed':>7} {'ref':>6} {'ratio':>5}  dimers  edge%  CN")
    for r in rows:
        f = lambda v: "  -  " if v != v else f"{v:5.2f}"
        tag = "" if r["code_cls"] in ("", r["cls"]) else f" [{r['code_cls']}]"
        print(f"{r['cls']:<24} {r['name']:<9} {f(r['rho_est'])} {f(r['rho_placed']):>6} {f(r['rho_relaxed']):>7} "
              f"{r['rho_ref']:6.2f}{r['kind']} {f(r['ratio'])}  {r['dimers']:<6}  {r['edge']:<5}  {r['cn']}{tag}")
    print(f"\nwritten: {out}   (ratio = relaxed / reference; 'a' reference is an amorphous value, "
          f"'c' a crystal one where 0.80-0.90 is the expected amorphous range)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
