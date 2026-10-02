# S(q) and XRD: methodology notes

This page describes the two structure-factor implementations in
`amorphgen/analysis/rdf.py`. See {doc}`/guides/analysis` for runnable examples.

## Choosing a method

| Method | Python API | CLI | Finite-cell limitation |
|---|---|---|---|
| Direct reciprocal-space sum | `structure_factor_direct()` | `--sq-method direct` (default) | Discrete q vectors and sparse low-q shells |
| Fourier transform of g(r) | `structure_factor()` | `--sq-method ft` | Finite-r truncation and histogram resolution |

Use configurations with the same composition and a valid periodic cell.
The direct Python method defaults to X-ray weighting; the FT Python method
defaults to unweighted scattering. The CLI defaults to X-ray weighting for
both methods. Specify the weighting explicitly when comparing results.

## Normalization of the direct method

For each nonzero reciprocal vector, the code calculates

$$I(\vec q)=\left|\sum_i f_i(q)e^{i\vec q\cdot\vec r_i}\right|^2,$$

then applies the Faber–Ziman convention:

$$S(\vec q)=1+\frac{I(\vec q)/N-\langle f^2(q)\rangle}
{\langle f(q)\rangle^2}.$$

The composition averages are $\langle f\rangle=\sum_\alpha c_\alpha f_\alpha$
and $\langle f^2\rangle=\sum_\alpha c_\alpha f_\alpha^2$.
For uncorrelated positions this normalization tends to 1. The uncorrected
quantity $I/(N\langle f\rangle^2)$ instead tends to
$\langle f^2\rangle/\langle f\rangle^2$, which generally differs from 1 in a
mixture. These quantities should not be compared without converting them
to the same convention.

With real-space cell vectors as rows of $\mathbf A$, the q vectors are
$\vec G=2\pi\vec n\mathbf A^{-\top}$ for integer triplets $\vec n$, excluding
the origin. Values are averaged into shells by their magnitude. For a cubic
cell, $q_{\min}=2\pi/L$; increasing the number of bins does not improve that
limit. `n_per_bin` counts the sampled vectors, and empty shells return `NaN`.
Gaussian smoothing weights each shell by its vector count and retains the
unsmoothed total as `s_q_raw`.

## Fourier transform of g(r)

The isotropic relation used by the FT method is

$$S(q)-1=4\pi\rho\int_0^{r_{\max}}[g(r)-1]
\frac{\sin(qr)}{qr}\,r^2\,\mathrm{d}r.$$

The implementation transforms each structure's unsmoothed RDF with 500 radial
bins and its own $(N-1)/V$ finite-system density prefactor before averaging
the resulting curves. By default, the shared `rmax` is half the shortest
cell-vector length across the ensemble, rounded down to 0.1 Å. Truncation
can change peak heights and introduce ripples. Its size and direction depend
on the structure and chosen range; there is no general factor-of-two
correction, nor a guarantee that truncation errors cancel between ensembles.
The direct method avoids this integral's cutoff but retains its own cell-size
and sampling limits.

For a weighted total, the FT method first computes the partials and combines
them as

$$S(q)=\frac{\sum_{\alpha,\beta}c_\alpha c_\beta f_\alpha(q)f_\beta(q)
S_{\alpha\beta}(q)}{\langle f(q)\rangle^2}.$$

X-ray weights use q-dependent Waasmaier–Kirfel neutral-atom form factors;
neutron weights use tabulated coherent scattering lengths. Unweighted
scattering sets every factor to 1. Weighting changes the relative contribution
of each partial; it cannot repair finite-cell or truncation errors.

## Checks and experimental comparison

`TestSqNormalisation` in `test/test_analysis.py` checks the high-q limit, the
$r^2$ FT integrand, selected scattering-table entries, q-dependent X-ray
weights, smoothing and exact recombination of neutron partials. These tests
verify implementation properties. They do not establish a fixed experimental
accuracy or equivalence to another analysis package.

Before comparing with a measured curve, match its normalization, weights,
q range, temperature and resolution. A normalized `S(q)` must be converted
back to coherent intensity before applying any instrument-specific model;
see the simulated-XRD tab in {doc}`/guides/analysis`. Reference publications
and software citation guidance are in {doc}`sq_xrd_credits`.

`load_experiment()` accepts measured S(q) or T(r), and `compare_experiment()`
interpolates each calculated structure onto the measured grid before
summarizing residuals and pointwise confidence intervals. Extrapolation and
missing-bin gaps are excluded and counted. Measurement uncertainties, when
supplied, weight Rw and chi-square; ensemble SEM is not substituted for them.
No scaling, offset, resolution or background is fitted. Correlated points
limit interpretation of the diagonal chi-square statistic.

`xrd_pattern()` restores coherent intensity per atom using each structure's
own composition, then forms an equal-weight ensemble mean. The direct method
uses bin-centre form factors for this conversion, so finite shell width
introduces an approximation. q maps to 2θ using the supplied wavelength;
inaccessible requested q ranges are rejected. The returned profile has no
Lorentz–polarization, absorption, background or instrument-response correction.
