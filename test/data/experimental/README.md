# Experimental benchmark references

Run `pytest test/test_experimental_*.py` from the repository root. These tests
run offline on CPU with the core dependencies. They compare specific observables
with published measurements or explicitly identified derivatives of measurements;
they do not certify a potential or a melt-quench protocol.

## Structure fixtures

`sio2.xyz` and `si.xyz` are **simulated structures**, copied from
`examples/validation/sio2/example_final.xyz` (48 atoms) and
`examples/validation/si/example_final.xyz` (64 atoms) at repository commit
`9a9f3338cb72115bbd5a84c010ba42981c470f98`. Species, coordinates, cell and
periodic boundaries are preserved. Momenta, calculator results and other unused
metadata were removed. Neither structure was relaxed, rescaled or fitted to a
measurement for these tests. Copies are kept here so the sdist can run the tests
without the larger `examples/` tree.

The repository supplies one representative snapshot per material, without the
full ensemble or complete original run provenance. Accordingly, the tests
retain unavailable ensemble confidence intervals and `inconclusive` reference
verdicts. A single snapshot's agreement within a chosen regression tolerance
does not establish an ensemble's agreement with experiment.

## Amorphous silica and silicon

| Reference | Measured quantity | Test interpretation |
|---|---|---|
| [Mozzi & Warren, J. Appl. Cryst. 2, 164–172 (1969)](https://doi.org/10.1107/S0021889869006868), abstract | Vitreous silica: Si–O distance about 1.62 Å; Si–O coordination 4; O–Si coordination 2 | Silica snapshot: distance within 0.03 Å, coordination within 0.1 and 0.05, respectively. These are model regression allowances, not reported measurement uncertainties. |
| [Laaziri et al., Phys. Rev. B 60, 13520–13533 (1999)](https://doi.org/10.1103/PhysRevB.60.13520), abstract | Pure ion-implanted amorphous Si: coordination 3.79 before annealing and 3.88 after annealing at 600 °C | The snapshot has coordination 4.0. Tests explicitly retain its disagreement with both measured values. |

Silica neighbour tests use 2.0 Å, 2.2 Å and the default RDF-derived cutoff.
A 10% uniformly expanded negative control must leave the accepted Si–O distance
range while retaining fourfold coordination. Silicon uses a 2.8 Å first-shell
cutoff. Density is not tested as a prediction because the saved cell volume is
an input. The silica paper's 144° Si–O–Si value is a distribution maximum;
it is not used as a reference for the analyser's mean angle.

The silica scattering test compares the first peak of calculated X-ray- and
neutron-weighted T(r) with the same measured Si–O distance. The 0.05 Å allowance
covers the model and finite-Q transform. Qmax = 20 Å⁻¹ follows the experimental
range; Qmin = 0.3 Å⁻¹ and the Lorch window are calculation choices. There is no
fit to measured intensities. No measured S(Q) or T(r) curve is included, so this
test cannot establish whole-curve agreement or experimental Rw/chi-square.

## Silicon powder scattering standard

[Cline et al., NIST SP 260-245 (2024)](https://doi.org/10.6028/NIST.SP.260-245),
Table 2, gives line positions derived from the experimentally certified SRM
640g powder lattice parameter, 0.5431109 nm at 22.5 °C, using wavelength
0.15405929 nm. The first six tabulated 2θ positions are 28.441°, 47.301°,
56.120°, 69.127°, 76.373° and 88.026°.

The test supplies that lattice parameter to a diamond silicon cell and checks
all six reflections. This is a calibration of reciprocal scattering and angular
conversion using an experimental standard; it is not an independent lattice
prediction. A 0.03° numerical tolerance covers the 0.002 Å⁻¹ q-bin spacing
and table rounding, rather than representing the much smaller SRM uncertainty.
Peak intensities and instrumental line shapes are not compared.

## Copper elasticity

[Overton & Gaffney, Phys. Rev. 98, 969–977 (1955)](https://doi.org/10.1103/PhysRev.98.969),
Tables I–II (p. 975), extrapolate ultrasonic single-crystal measurements
over 4.2–300 K to 0 K. Those extrapolated values, converted from
10¹¹ dyne/cm² to GPa, are C11 = 176.20, C12 = 124.94, C44 = 81.77 and bulk
modulus = 142.03. They are not direct measurements at exactly 0 K.

The tests relax an FCC copper cell with ASE's empirical EMT calculator before
running AmorphGen's elastic analysis. Relative model tolerances are 5%, 10%,
12% and 8%, respectively. These bounds accommodate the approximation in EMT
and are distinct from experimental uncertainty. A separate strain-convergence
check requires agreement within 0.005 GPa, so the broad physical tolerance
does not hide an unconverged numerical derivative. This is a crystalline
benchmark of the elastic-analysis pipeline, not amorphous-metal validation.
