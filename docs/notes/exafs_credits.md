---
orphan: true
---

# How AmorphGen computes EXAFS descriptors, and where the method comes from

The same question that the {doc}`S(q) note <sq_xrd_credits>` answers,
for `--exafs`: is this a known method, or did AmorphGen invent
something that a reviewer will want justified?

> Interpretation: geometric shell cumulants and the harmonic approximation
> are not generally identical to experimentally fitted EXAFS parameters.
> The effective-distance correction uses a scalar mean free path and a
> truncated expansion. See the [current EXAFS guide](../guides/exafs.md)
> for the distinction, provenance requirements and uncertainty conventions.

## What's standard physics

| Ingredient | First published | Status |
|---|---|---|
| Cumulant expansion of the EXAFS signal ($C_1$, $\sigma^2$, $C_3$) | [Bunker, *Nucl. Instrum. Methods Phys. Res.* **207** (1983) 437](https://doi.org/10.1016/0167-5087(83)90655-5) | Standard; built into every EXAFS fitting package |
| $\sigma^2$ as the mean-square *relative* displacement along the bond | [Beni & Platzman, *Phys. Rev. B* **14** (1976) 1514](https://doi.org/10.1103/PhysRevB.14.1514) | Textbook |
| Static and thermal disorder adding in $\sigma^2$ | Same lineage | Universal convention |
| Effective vs real distribution, $C_1^{\rm eff} = C_1 - 2\sigma^2(1/C_1 + 1/\lambda)$ | [Dalba & Fornasini, *J. Synchrotron Rad.* **4** (1997) 243](https://doi.org/10.1107/S0909049597006900); [Filipponi, *J. Phys.: Condens. Matter* **13** (2001) R23](https://doi.org/10.1088/0953-8984/13/7/201) | Standard correction |
| Extracting these cumulants from an MD or DFT ensemble instead of fitting them | Established practice in the MD-EXAFS literature | Routine |

There is no equation in `analysis/exafs.py` whose first publication is
later than 2001, and the core of it is from the 1970s–80s.

## What's ours

The Python that wires the standard definitions together:

- ~330 lines in `amorphgen/analysis/exafs.py`, over half of it docstring
- the API surface (`exafs_descriptors()`, the return-dict format, the
  `exafs:` section of the reference-YAML schema)
- the report layout, including the `skew` flag and the
  configurational-$\sigma^2$ warning

That's all. AmorphGen contributes the integration and the ergonomics,
not the physics.

## What this is *not*

These are the quantities an EXAFS fit **reports**, computed from the
ensemble. This is **not** a simulated $\chi(k)$: no photoelectron
scattering amplitudes, no phase shifts, no mean-free-path calculation
beyond the single scalar $\lambda$ in the $R_{\rm eff}$ correction, and
no multiple scattering. Nothing here replaces FEFF, FDMNES or a fitting
package such as Artemis/Larch.

The honest statement of what it buys you: a published EXAFS fit table
(N, R, $\sigma^2$) can be inspected alongside an AmorphGen ensemble
provided the different definitions and approximations are accounted for, and `skew` tells you when the
harmonic assumption behind that published table is shaky.

## Validation evidence

Three checks, all in `test/test_exafs.py`, all against closed-form
values rather than recorded output:

| Check | Expected | AmorphGen |
|---|---|---|
| Perfect diamond-Si first shell | $N = 4$, $R = a\sqrt{3}/4$, $\sigma^2 = 0$, $C_3 = 0$ | exact to machine precision |
| Uncorrelated Gaussian rattle of width $s$ | $\sigma^2 \to 2s^2$ (MSRD limit, *not* $6s^2$) | within 2 % at $s = 0.05$ Å |
| $N$ against `coordination()` on the same ensemble | identical by construction | identical |
| Hand-placed shells $\{1.5, 1.5, 1.5, 1.9\}$ Å | $C_1 = 1.6$, $\sigma^2 = 0.03$, $C_3 = +0.006$ | exact |

The MSRD check is the one worth keeping: summing two independent
atomic mean-square displacements would give $6s^2$, three times too
large. Getting $2s^2$ confirms the variance is taken on the pair
distance, which is what EXAFS measures.

## How to cite

Cite the physics, not AmorphGen as a method:

> First-shell coordination numbers, mean distances and EXAFS
> Debye–Waller factors were obtained as the cumulants of the
> absorber–scatterer distance distribution of the generated ensemble
> [Bunker 1983; Beni & Platzman 1976], with distances corrected to the
> effective distribution sampled by EXAFS [Dalba & Fornasini 1997].
> All analysis was performed using the AmorphGen package [AmorphGen
> citation].

State the ensemble temperature. A $\sigma^2$ from relaxed structures is
the configurational term only and should be labelled as such.

## Ensemble-sampling error bars

Both the descriptors and the harmonic fit carry a bootstrap standard
error, obtained by resampling whole *structures* with replacement. The
distinction matters: the paths inside one structure are not independent
samples of the disorder, so resampling paths would understate the
uncertainty badly.

This was added after the fact, and for a good reason. Comparing two
a-SiO$_2$ ensembles quenched at 1016 and 242 K/ps, every descriptor
moved in the expected direction and **every difference sat between 0.4
and 1.2 standard errors** — nothing was resolved by three structures per
ensemble. The point estimates alone read as a clear result. Resolving a
$\sigma^2$ difference of that size at three standard errors needs about
24 structures per ensemble.

Note what this does *not* claim to be: a fit uncertainty. The input to
the fit is an exact sum over the ensemble, not a noisy measurement, so a
formal least-squares error bar would be meaningless. What is reported is
the uncertainty that actually dominates — how much the answer would move
with a different draw of structures.

## The harmonic-fit route

`amorphgen.analysis.harmonic` reports the N, R and $\sigma^2$ a
harmonic EXAFS fit would return for an ensemble, so the `skew` flag
becomes a measured bias. The physics is again standard: the cumulant
expansion above, the Gaussian pair-distribution model that every
fitting package uses, and the observation that $f(k)$, $\phi(k)$,
$S_0^2$ and $1/k$ cancel between model and data so no scattering
calculation is needed. AmorphGen contributes the least-squares wiring
and the side-by-side presentation.

Two checks are worth quoting if a reviewer asks whether the fit is
doing the right thing. On a Gaussian shell the fitted R lands within
$2\times10^{-5}$ Å of the $C_1^{\rm eff}$ that the cumulant correction
predicts independently, with an R-factor around $10^{-9}$. On a skewed
shell, N is under-counted whichever way the shell leans, R moves
against the tail, and the bias grows with $k_{\max}$ — the three
signatures the cumulant expansion predicts.

## The FEFF route, validated against FEFF's own reference

The generated input is not merely format-checked by unit tests: it has
been run through FEFF8.5L and compared to the reference calculation
shipped with [feff85exafs](https://github.com/xraypy/feff85exafs). On
fcc Cu (a = 3.61 Å) AmorphGen's 8 Å cluster is **atom-for-atom identical**
to the reference (177 atoms, 12 first neighbours at 2.55266 Å), and the
resulting $\chi(k)$ agrees with the reference to **0.04 % RMS** over
k = 2–18 Å⁻¹. For scale, re-running the reference's *own* input through
the same build reproduces it to 0.002 %, so the residual is the handful
of card differences, not the structure.

FEFF8L is the EXAFS-only version, free and BSD-licensed, unlike the
commercially licensed FEFF9/FEFF10 — and EXAFS-only is exactly what
AmorphGen writes.

## The FEFF route

`amorphgen.analysis.feff` forward-models $\chi(k)$ itself: one FEFF
calculation per absorber site, averaged over the ensemble, with nothing
fitted. That makes the harmonic-fit bias measurable rather than merely
flagged by `skew` — fit the simulated $\chi(k)$ the way the
experimentalist fitted theirs, and the difference between the fitted N
and the geometric N is the bias.

The same division of labour applies there. The physics is FEFF's; the
cumulant-free configurational average over sites is standard MD-EXAFS
practice; AmorphGen contributes the cluster extraction, the job tree,
the site average and the Athena/Larch-convention Fourier transform.
Cite FEFF for the scattering calculation.

## Open issues / future work

- Fourth cumulant $C_4$, occasionally reported for strongly disordered
  shells.
- Second and higher shells; the current cutoff-based shell definition
  is first-shell only.
- Second and higher shells in the harmonic fit; it currently fits one
  shell at a time, as defined by the pair cutoff.
- Per-shell error bars beyond the first coordination shell.
