#!/usr/bin/env python
"""Total correlation function T(r) = 4 pi r rho g(r) in the diffraction convention.

T(r) is what high-energy X-ray diffraction papers plot beside S(Q) (for example
Kumara et al., J. Phys. Chem. C 125, 13619 (2021), Figure 1b), because the area
under a peak of r*T(r) gives the coordination number directly. Reproducing it
from a model needs the same three steps the experiment uses:

  1. the Faber-Ziman X-ray weighted total structure factor S(Q);
  2. g(r) from S(Q) by Fourier transform over the measured Q range,
         g(r) = 1 + 1/(2 pi^2 r rho) * Int_Qmin^Qmax Q [S(Q) - 1] sin(Qr) dQ
     with a Lorch window to damp the Qmax truncation ripple;
  3. T(r) = 4 pi r rho g(r), with rho the atomic number density of the model.

Using AmorphGen's unweighted g(r) instead would NOT be comparable: a diffraction
g(r) weights each pair by scattering power, which for IGZO means indium dominates.

Usage:
    python total_correlation_function.py STRUCTURE_DIR [-o OUT] [--qmin 0.3]
                                         [--qmax 20] [--rmax 10] [--label NAME]
"""
import argparse
import csv
import glob
import os

import numpy as np
from ase.io import read

# numpy 2 renamed trapz -> trapezoid and removed the old name
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def structure_factor_and_Tr(atoms_list, qmin=0.3, qmax=20.0, nq=400,
                            rmax=10.0, nr=600, sigma_q=0.05, lorch=True):
    """Return (Q, S_Q, r, g_r, T_r, rho) in the diffraction convention."""
    from amorphgen.analysis import StructureAnalyser

    sa = StructureAnalyser(list(atoms_list))
    sq = sa.structure_factor_direct(weighting="xray", qmax=qmax, nq=nq, sigma_q=sigma_q)
    Q = np.asarray(sq["q"], dtype=float)
    S = np.asarray(sq["s_q"], dtype=float)
    n = np.asarray(sq["n_per_bin"], dtype=float) if "n_per_bin" in sq else np.ones_like(Q)
    keep = (n > 0) & ~np.isnan(S) & (Q >= qmin)
    Q, S = Q[keep], S[keep]

    rho = float(np.mean([len(a) / a.get_volume() for a in atoms_list]))
    r = np.linspace(0.5, rmax, nr)
    window = np.sinc(Q / Q.max()) if lorch else np.ones_like(Q)
    integrand = (Q * (S - 1.0) * window)[None, :] * np.sin(np.outer(r, Q))
    g = 1.0 + _trapz(integrand, Q, axis=1) / (2 * np.pi ** 2 * r * rho)
    return Q, S, r, g, 4 * np.pi * r * rho * g, rho


def first_peak_coordination(r, T, rho, r_lo, r_hi):
    """Coordination number from the first T(r) peak: N = Int r*T(r) dr over the peak."""
    m = (r >= r_lo) & (r <= r_hi)
    return _trapz(r[m] * T[m], r[m])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_dir", help="directory of structure files")
    p.add_argument("-o", "--out", default="Tr", help="output basename")
    p.add_argument("--qmin", type=float, default=0.3)
    p.add_argument("--qmax", type=float, default=20.0)
    p.add_argument("--rmax", type=float, default=10.0)
    p.add_argument("--peak", nargs=2, type=float, default=(1.4, 2.6),
                   metavar=("R_LO", "R_HI"), help="first-peak window for the CN integral")
    p.add_argument("--label", default=None)
    a = p.parse_args()

    files = sorted(sum((glob.glob(os.path.join(a.input_dir, e))
                        for e in ("*.xyz", "*.extxyz", "*.cif", "*.vasp")), []))
    if not files:
        raise SystemExit(f"no structure files in {a.input_dir}")
    atoms = [read(f) for f in files]
    Q, S, r, g, T, rho = structure_factor_and_Tr(atoms, a.qmin, a.qmax, rmax=a.rmax)
    N = first_peak_coordination(r, T, rho, *a.peak)
    label = a.label or os.path.basename(os.path.normpath(a.input_dir))

    print(f"{label}: {len(files)} structures, rho = {rho:.4f} atoms/A^3")
    print(f"  S(Q) first peak : Q = {Q[np.argmax(np.where((Q > 1) & (Q < 4), S, -9))]:.2f} A^-1")
    print(f"  T(r) first peak : r = {r[np.argmax(np.where(r < a.peak[1], T, -9))]:.2f} A")
    print(f"  coordination from the first peak ({a.peak[0]}-{a.peak[1]} A): N = {N:.2f}")

    with open(f"{a.out}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["r_A", "g_r_xray_weighted", "T_r_invA2"])
        w.writerows(zip(r, g, T))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        ax.plot(r, T, lw=1.5, color="#0072B2", label=label)
        ax.set_xlabel(r"$r$ ($\mathrm{\AA}$)")
        ax.set_ylabel(r"$T(r)$ ($\mathrm{\AA}^{-2}$)")
        ax.set_xlim(0.5, a.rmax)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="in", top=False, right=False)
        ax.legend(frameon=False)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(f"{a.out}.{ext}", dpi=300, bbox_inches="tight")
        print(f"  written: {a.out}.{{png,pdf,csv}}")
    except ImportError:
        print(f"  written: {a.out}.csv (matplotlib not available for the figure)")


if __name__ == "__main__":
    main()
