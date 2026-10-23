# FEFF8.5L copper reference

`copper_chi.dat` is the non-SCF copper `chi.dat` reference distributed in
[xraypy/feff85exafs](https://github.com/xraypy/feff85exafs/tree/master/tests/Copper/baseline/noSCF).
It was copied from the user's locally installed reference, originally obtained
on 26 September 2026. Its original header is retained.

The reference uses fcc Cu, a=3.61 Å, an 8 Å cluster, RPATH=5.2 Å,
EXAFS=20 and S02=0. The integration test generates the geometry independently
with ASE and compares chi(k) over k=2–18 Å⁻¹. AmorphGen retains NLEG=4;
the allowed relative RMS difference is 0.1%.

Run `python -m pytest test/test_feff_integration.py --run-feff`.
FEFF must already be installed; `FEFF_EXECUTABLE` can specify its path.
No executable is bundled or downloaded. This check targets FEFF8.5L, and does
not promise numerical equivalence across different FEFF releases.

## Additional oxide and self-consistent references

`nio_chi.dat`, `uo2_chi.dat` and `copper_scf_chi.dat` are copied without
modification from the pinned FEFF8.5L revision
`ec8dcb07ca8ee034d0fa7431782074f0f65357a5`, respectively
`tests/NiO/baseline/noSCF/chi.dat`, `tests/UO2/baseline/noSCF/chi.dat` and
`tests/Copper/baseline/withSCF/chi.dat`. Source URLs and SHA-256 hashes are
recorded in `extended_reference_provenance.json`; the upstream license is
retained in `LICENSE.feff85exafs`.

ASE independently generates rocksalt NiO (a=4.1684 Å), fluorite UO2
(a=5.4682 Å) and fcc Cu (a=3.61 Å). The tests use 8 Å clusters, explicit
parent-cell stoichiometry, S02=0 (FEFF's internally calculated reduction),
NLEG=8, and RPATH=5.9, 5.5 and 5.2 Å respectively. Cu uses SCF=4 Å.
They test Ni K, U L3 and Cu K edges. Comparison is unweighted chi(k),
k=2–18 Å⁻¹, with a 0.1% relative RMS tolerance and no fitted parameters.
These are numerical references, not experimental measurements.
