---
orphan: true
---

# S(q) and XRD: methodology notes

This page archives the investigation behind AmorphGen's choice of
S(q) implementation. The user-facing recommendation in
{doc}`/guides/analysis` is the **direct q-vector method**
(``structure_factor_direct()``); this note keeps the underlying
comparison and physical reasoning for posterity.

## Two methods that were on the table

AmorphGen ships both implementations because they have genuinely
different properties:

| Method | API | Speed | Peak intensity |
|---|---|---|---|
| **FT-of-g(r)** | ``structure_factor()`` | Fast (seconds) | Damped ~2× by finite ``rmax`` |
| **Direct q-vector (Debye sum)** | ``structure_factor_direct()`` | Slower (~20s × 20 structs) | Quantitatively correct |

Both implement well-established physics; the difference is only how
the Fourier integral is handled in a finite simulation cell.

## Why peak intensities differ between the two

**FT-of-g(r).** Starts from the ensemble-averaged radial distribution
function:

$$S(q) - 1 = 4\pi\rho \int_0^{r_{\max}} [g(r)-1]\,\frac{\sin(qr)}{qr}\,r^2\,\mathrm{d}r$$

In a finite cell with side $L$, the integral has to be truncated at
$r_{\max} = L/2$ because beyond this the minimum-image convention
becomes ambiguous. For a typical 400-atom amorphous-oxide cell
$L \approx 16$ Å so $r_{\max} = 8$ Å — exactly where the medium-range
correlations responsible for the FSDP live. Cutting them off damps
the FSDP intensity by about 50 %.

**Direct q-vector.** Evaluates the Debye scattering equation at the
reciprocal-lattice vectors of the periodic cell:

$$S(\vec G) = \frac{1}{N\langle f\rangle^{2}}\,\Bigl|\sum_i f_i\,
\mathrm{e}^{\mathrm{i}\vec G\cdot\vec r_i}\Bigr|^{2}, \qquad
\vec G = 2\pi(n_1 \vec b_1 + n_2 \vec b_2 + n_3 \vec b_3)$$

No truncation, no minimum-image issues. Spherical averaging then
gives a clean S(q) curve.

## Validation that drove the decision

We benchmarked both methods on the published a-Ga₂O₃ DFT-PBE0 ensemble
([Kaewmeechai, Strand & Shluger, *Phys. Rev. B* **111** (2025) 035203](https://doi.org/10.1103/PhysRevB.111.035203)).
Comparing against the experimental X-ray S(Q) and the GAP_500 simulation
from the same reference (Fig. S2b):

| Method | FSDP intensity at q = 2.4 Å⁻¹ | Match to experiment (~1.8-2.0)? |
|---|---|---|
| FT-of-g(r), unweighted | 0.84 | ❌ ~2× low |
| FT-of-g(r), X-ray weighted | 0.84 | ❌ ~2× low |
| **Direct q-vector, X-ray weighted** | **2.00** | ✅ |
| GAP_500 (Csányi group) | ~1.8 | ✅ |
| Experiment (Fig. S2b) | ~1.8-2.0 | ✅ reference |

The direct method matches both the experimental S(Q) and the GAP
simulation from the same reference. The FT method positions peaks
correctly but consistently under-shoots their height.

## Why the FT method is still in the package

Even though the direct method is preferred for paper figures and
experimental comparison, ``structure_factor()`` (FT-of-g(r)) remains
useful for:

- **Ensemble-vs-ensemble comparisons** — the systematic damping
  cancels when comparing two ensembles computed the same way.
- **Quick sanity checks** — seconds rather than tens of seconds.
- **Backwards compatibility** — existing scripts and the JOSS paper
  validation figures use the FT method; preserving the API avoids
  silent behaviour changes.

The default ``weighting="unweighted"`` of ``structure_factor()`` is
fine for ensemble comparison and matches the historical AmorphGen
behaviour. The Faber–Ziman weighted total (``weighting="xray"``) is
available for users who want X-ray-like intensities from the fast
method, with the caveat that the FSDP height will still be damped.

## A worked example of the FSDP cancellation in unweighted sums

For a-Ga₂O₃ at q = 2.5 Å⁻¹ (from the PRB ensemble):

- $S_{\rm Ga-Ga}(2.5) \approx 1.23$ (the FSDP itself)
- $S_{\rm Ga-O}(2.5) \approx 0.25$ (an anti-peak — Ga–O correlations
  are anti-phase at this q)
- $S_{\rm O-O}(2.5) \approx 1.25$

**Unweighted sum** ($f_\alpha = 1$):

$$S^{(\rm unwt)}(2.5) = c_{\rm Ga}^{2}(1.23) + 2 c_{\rm Ga} c_{\rm O}(0.25) + c_{\rm O}^{2}(1.25) \approx 0.84$$

The Ga–O dip cancels the like-pair peaks — the FSDP disappears.

**X-ray weighted** ($Z_{\rm Ga} = 31, Z_{\rm O} = 8$):

$$S^{(\rm xray)}(2.5) = \frac{0.4^2 \cdot 961 \cdot 1.23 + 2\cdot 0.4\cdot 0.6\cdot 248\cdot 0.25 + 0.6^2\cdot 64\cdot 1.25}{17.2^{2}} \approx 2.0$$

Heavy Ga–Ga dominates ($Z^2 = 961 \gg 64$). The FSDP survives — and
matches experiment.

This is why X-ray diffraction sees the FSDP that an unweighted total
or a chemistry-blind "first-shell-only" analysis would miss.

## References

The methodology decision and the FT-vs-direct comparison are
documented for the JCTC methods paper (in preparation). Primary
references for the underlying physics are listed in the
{doc}`/guides/analysis` "Physical validity" tab.
