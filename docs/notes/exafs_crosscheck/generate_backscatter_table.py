"""Generate amorphgen/analysis/data/backscatter_feff6l.json.

The tier-2 harmonic fit (amorphgen.analysis.harmonic) needs the *shape* of
each scattering element's effective backscattering amplitude |f(k)| to
weight its residual the way a real chi(k) fit does. This script computes
it once per element with FEFF6L, so AmorphGen itself never runs FEFF.

Protocol, identical for every element X: a two-atom cluster, Zn absorber
at the origin and X on the z axis at the sum of the two covalent radii
(Cordero et al. 2008, as tabulated by ASE), K edge, single scattering.
The tabulated quantity is FEFF's ``mag_feff`` for that one path. The
absorber-dependent reduction factor is left out: it moves a fitted N by
~1% (checked on a-Ga2O3) and would tie the table to one absorber.

Run from a Python environment with xraylarch and ase installed:

    python generate_backscatter_table.py OUT.json [WORKDIR]

Generated with FEFF6L, distributed with xraylarch:
"Based on or developed using Distribution: FEFF6L
 Copyright (c) [2002] University of Washington"
"""
import json
import os
import sys
import tempfile

import numpy as np
from ase.data import chemical_symbols, covalent_radii
from larch import __version__ as larch_version
from larch.xafs import feff6l, feffpath

ABSORBER_Z = 30          # Zn: mid-row, unremarkable, the same for every X
Z_RANGE = range(3, 84)   # Li .. Bi
K_GRID = np.round(np.arange(0.0, 20.0 + 1e-9, 0.1), 2)


def amplitude(z, workdir):
    """|f(k)| of element ``z`` as a scatterer, on K_GRID, plus the distance."""
    r = float(covalent_radii[ABSORBER_Z] + covalent_radii[z])
    d = os.path.join(workdir, chemical_symbols[z])
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "feff.inp"), "w") as fh:
        fh.write(f"TITLE Zn-{chemical_symbols[z]} backscattering amplitude\n"
                 "HOLE      1   1.0\nCONTROL   1 1 1 0\nPRINT     0 0 0 0\n"
                 f"RMAX      {r + 0.1:.3f}\nNLEG      2\n"
                 "POTENTIALS\n"
                 f"    0  {ABSORBER_Z}  Zn\n    1  {z}  {chemical_symbols[z]}\n"
                 "ATOMS\n"
                 "    0.00000   0.00000   0.00000   0\n"
                 f"    0.00000   0.00000   {r:.5f}   1\n"
                 "END\n")
    feff6l(feffinp="feff.inp", folder=d, verbose=False)
    fd = feffpath(os.path.join(d, "feff0001.dat"))._feffdat
    return np.interp(K_GRID, fd.k, fd.mag_feff), r


def main(out, workdir=None):
    workdir = workdir or tempfile.mkdtemp(prefix="feff6l_fk_")
    amp, dist = {}, {}
    for z in Z_RANGE:
        a, r = amplitude(z, workdir)
        amp[chemical_symbols[z]] = [float(f"{v:.4g}") for v in a]
        dist[chemical_symbols[z]] = round(r, 3)
        print(f"{chemical_symbols[z]:>3}  R {r:.2f} A  |f| at k=3,8,12: "
              f"{a[30]:.3f} {a[80]:.3f} {a[120]:.3f}", flush=True)
    json.dump({
        "description": "Effective backscattering amplitude |f(k)| per "
                       "scattering element, for weighting the tier-2 "
                       "harmonic-fit residual. Only its shape in k is used.",
        "generated_with": f"FEFF6L via xraylarch {larch_version} feff6l",
        "copyright": "Based on or developed using Distribution: FEFF6L. "
                     "Copyright (c) [2002] University of Washington",
        "protocol": "Zn absorber, scatterer at the sum of covalent radii "
                    "(Cordero 2008), K edge, single-scattering mag_feff; "
                    "see docs/notes/exafs_crosscheck/generate_backscatter_table.py",
        "units": {"k": "1/A", "amplitude": "A"},
        "k": [float(v) for v in K_GRID],
        "distance": dist,
        "amplitude": amp,
    }, open(out, "w"), separators=(",", ":"))
    print(f"wrote {out}: {len(amp)} elements")


if __name__ == "__main__":
    main(*sys.argv[1:])
