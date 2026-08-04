# Analysis

AmorphGen's `--analyse` mode computes structural descriptors for an
ensemble of amorphous structures and produces publication-quality
figures plus CSV data for every plot.

```bash
amorphgen --analyse --input-dir my_structures/ --save-plot plots/
```

That single command computes density, partial radial distribution
functions, coordination distributions, bond-angle distributions, and a
per-structure density violin — saved as four PNG figures plus CSV
companions.

## Recipes

Pick the scenario that matches what you want:

::::::{tab-set}

:::::{tab-item} Quick start

The minimum command, four standalone figures (no comparison, no
validation against literature):

```bash
amorphgen --analyse \
    --input-dir hybrid_ga2o3/random_opt/ \
    --save-plot plots/ \
    --save-pdf
```

Output:

```
plots/
├── analysis_rdf.{png,pdf,csv}        # partial RDFs (all pairs)
├── analysis_cn.{png,pdf,csv}         # coordination distribution
├── analysis_angles.{png,pdf,csv}     # bond-angle distributions
└── analysis_density.{png,pdf,csv}    # per-structure density violin
```

The CSV files contain the raw numbers (r, g(r), CN counts, angle
histograms, per-structure densities) so you can re-plot in any tool.

:::::

:::::{tab-item} Single ensemble + text report

Add a full text report alongside the plots:

```bash
amorphgen --analyse \
    --input-dir hybrid_ga2o3/random_opt/ \
    --save-report report.txt \
    --save-plot plots/ \
    --save-pdf
```

The report shows: density mean ± std, bond distances (mean / std / count
per pair), coordination numbers (with distribution histograms), bond
angles (mean / std / count per triplet).

To also see the breakdown **per structure** (one row per file with
density, energy, CN), add ``--per-structure``:

```bash
amorphgen --analyse \
    --input-dir hybrid_ga2o3/random_opt/ \
    --per-structure
```

If the structure files don't carry per-atom energies in their headers
(VASP, CIF), AmorphGen automatically reads ``random_gen.log`` in the
parent directory (when present) to fill in the E/atom column.

:::::

:::::{tab-item} Validate against a reference YAML

Compare your ensemble against literature ranges with automatic
match/concern/fail scoring:

```bash
amorphgen --analyse \
    --input-dir hybrid_ga2o3/random_opt/ \
    --reference examples/reference_a_Ga2O3.yaml \
    --save-report report.txt \
    --save-plot plots/ \
    --save-pdf
```

The report adds a section like:

```
Validation: a-Ga2O3
  Descriptor              Computed        Expected  Units    Verdict
  ----------------------------------------------------------------------
  Density                    4.37    [4.70, 5.10]  g/cm^3   fail
  Bond Ga-O                  1.91    [1.85, 1.95]  A        match
  CN Ga-O                    4.42    [4.00, 4.80]           match
  Angle Ga-O-Ga             116.8  [110.00, 130.0]  deg     match
  Angle O-Ga-O              108.2  [100.00, 115.0]  deg     match
  ----------------------------------------------------------------------
  Summary: 4 match, 0 concern, 1 fail (out of 5 metrics)
```

AmorphGen ships one reference YAML at
``examples/reference_a_Ga2O3.yaml``. Write your own for other systems
by following the same schema.

:::::

:::::{tab-item} Compare multiple ensembles

For overlaying Random vs Hybrid vs DFT-reference (or any combination),
use the Python API. There's no single CLI flag for this yet —
``compare_ensembles()`` is the entry point:

```python
from amorphgen.analysis import EnsembleSpec, compare_ensembles

compare_ensembles(
    ensembles=[
        EnsembleSpec("DFT-PBE0", "prb_ensemble/*.cif"),
        EnsembleSpec("Random",   "random_inputs/*.vasp"),
        EnsembleSpec("Hybrid",   "hybrid_run/*.xyz"),
    ],
    rdf_pairs=[("Ga-O", "-"), ("Ga-Ga", "--"), ("O-O", ":")],
    cn_top_key="Ga-O",
    cn_bot_key="O-Ga",
    angle_keys=[("O-Ga-O", "-"), ("Ga-O-Ga", "--")],
    exp_density=(4.78, 4.84),
    output_dir="comparison/",
    prefix="ga2o3",
)
```

Output: `comparison/ga2o3_rdf.{png,pdf,csv}`,
`comparison/ga2o3_coordination.{...}`,
`comparison/ga2o3_angles.{...}`,
`comparison/ga2o3_density.{...}` — same layout as `--analyse
--save-plot`, but each figure overlays all listed ensembles with
distinct colours from the Okabe-Ito palette.

See {doc}`/validation/index` for fully-worked examples of this on four
material systems.

:::::

::::::

## Cutoff

The cutoff defines what counts as a "first-shell" bond and affects
coordination, bond-length statistics, and bond-angle triplets. Default
in v1.0.0+ is `auto-rdf`, which finds the first minimum of each partial
RDF — the standard convention in neutron-diffraction analysis of
glasses.

| Cutoff mode | When to use |
|---|---|
| **`auto-rdf`** (default) | All structural analysis. Robust across systems with broad bond-length distributions (a-Si, a-HfO₂, chalcogenides). |
| `auto` | Legacy. Uses minsep from Shannon/Cordero/Goldschmidt radii. Fast but can under-count coordination for systems with long first-shell bonds. |
| Numeric, e.g. `--cutoff 2.5` | Single cutoff (in Å) for all pairs. Useful for tight-bonded covalent networks. |
| Per-pair dict via YAML | Custom per-pair cutoffs for unusual chemistries. |

Setting an explicit cutoff is rarely needed; `auto-rdf` handles
practically every amorphous system AmorphGen targets.

## Structure factor S(q) and simulated XRD

AmorphGen computes the structure factor S(q) directly from atomic
positions using the Debye scattering equation evaluated at the
reciprocal-lattice vectors of the simulation cell, and converts it
to a simulated X-ray diffraction pattern I(2θ) through Bragg's law
plus the standard Lorentz–polarization correction. Pick the tab that
matches your workflow.

```{note}
A legacy fast method ``structure_factor()`` based on the
Fourier transform of g(r) is also available. It is kept for
ensemble-vs-ensemble comparisons but **under-shoots peak intensities
by ~2×** in typical amorphous-MD cells due to the finite-``rmax``
truncation. For all paper figures and experimental comparisons use
``structure_factor_direct()`` as described below. The methodology
comparison and the reason the two methods differ is documented in
the {doc}`/notes/sq_xrd_methodology` note.
```

::::{tab-set}

:::{tab-item} Structure factor S(q)
:sync: directmethod

```python
sq = sa.structure_factor_direct(weighting="xray")
```

**What it does.** Evaluates the Debye scattering equation
([Debye, *Ann. Phys.* **351** (1915) 809](https://doi.org/10.1002/andp.19153510606))
at the **reciprocal-lattice vectors** of the simulation cell. For a
single q-vector,

$$S(\vec q) \;=\; \frac{1}{N\,\langle f\rangle^2}
\,\Bigl|\sum_{i=1}^{N} f_i\,\mathrm{e}^{\mathrm{i}\vec q\cdot\vec r_i}\Bigr|^{2}$$

where $f_i$ is the scattering factor of atom $i$ and
$\langle f\rangle = \sum_{\alpha} c_{\alpha} f_{\alpha}$ is the
composition-weighted average. Expanding the square and taking the
real part:

$$S(\vec q) \;=\; \frac{1}{N\langle f\rangle^{2}}
\Bigl[\sum_{i} f_{i}^{2}
\;+\; 2\sum_{i<j} f_{i}f_{j}\cos\bigl(\vec q\cdot(\vec r_{i}-\vec r_{j})\bigr)\Bigr].$$

For a periodic system the natural q-grid is the **reciprocal lattice**

$$\vec G_{n_1 n_2 n_3} \;=\; 2\pi (n_1\, \vec b_1 + n_2\, \vec b_2 + n_3\, \vec b_3),
\qquad \vec b_i = (\mathbf{A}^{-\top})_{i},$$

where $\mathbf A$ is the cell matrix. Each $\vec G$ exactly satisfies
the Born–von Kármán boundary conditions, so $S(\vec G)$ is the
**exact** discrete Fourier transform of the atomic distribution — no
truncation, no minimum-image issues. AmorphGen enumerates all
$\vec G$ with $|\vec G| \le q_{\max}$, computes $S(\vec G)$ for each,
then **spherically averages** within bins $|q| \in [q_k, q_{k+1})$:

$$S(q_k) \;=\; \langle S(\vec G) \rangle_{|\vec G|\in[q_k,q_{k+1})}.$$

The dictionary returned by ``structure_factor_direct`` contains
``"n_per_bin"`` — the number of reciprocal-lattice vectors in each
shell — so you can mask poorly-sampled low-q bins (typically those
with fewer than ~5 vectors).

This is the same reciprocal-lattice-sum approach used in the
established amorphous-MD analysis packages ISAACS
([Le Roux & Petkov, *J. Appl. Cryst.* **43** (2010) 181](https://doi.org/10.1107/S0021889809051929))
and LiquidLib
([Walter, Bian, Mendoza & Schweizer, *Comput. Phys. Commun.* **228**
(2018) 209](https://doi.org/10.1016/j.cpc.2018.03.005)).

:::

:::{tab-item} Simulated XRD I(2θ)
:sync: xrdrecipe

To produce a simulated XRD pattern that can be compared to a raw
diffractometer trace, take the direct-method S(q) and apply the
standard X-ray-scattering geometry corrections:

```python
import numpy as np
from scipy.ndimage import gaussian_filter1d

LAMBDA_CU_KA = 1.5406  # Å; switch to 0.7107 for Mo-Kα

# 1. S(q) at appropriate qmax
sq = sa.structure_factor_direct(weighting="xray", qmax=8.0, nq=400)
q   = np.array(sq["q"]); s = np.array(sq["s_q"]); nb = np.array(sq["n_per_bin"])
keep = nb >= 5

# 2. 2θ grid, map q ↔ 2θ via Bragg's law
two_theta = np.linspace(20.0, 90.0, 1000)
theta = np.deg2rad(two_theta / 2.0)
q_at_2t = 4.0 * np.pi * np.sin(theta) / LAMBDA_CU_KA
s_at_2t = np.interp(q_at_2t, q[keep], s[keep])

# 3. Lorentz-polarization correction
cos2t = np.cos(np.deg2rad(two_theta))
lp = (1.0 + cos2t**2) / (np.sin(theta)**2 * np.cos(theta))
intensity_raw = s_at_2t * lp

# 4. Instrument broadening (Gaussian σ ≈ 1° typical lab Cu-Kα)
d2t = two_theta[1] - two_theta[0]
intensity = gaussian_filter1d(intensity_raw, sigma=1.0 / d2t)
intensity = intensity / np.nanmax(intensity)
```

**Equations behind each step.**

(1) **Bragg's law.** Wavelength $\lambda$, scattering angle $2\theta$:

$$q = \frac{4\pi}{\lambda}\sin\theta, \qquad
2\theta = 2\arcsin\!\left(\frac{q\lambda}{4\pi}\right).$$

For Cu-Kα ($\lambda = 1.5406$ Å) the accessible q range up to
$2\theta = 90^\circ$ is $q \in [0, 5.77]$ Å⁻¹. Mo-Kα ($\lambda =
0.7107$ Å) extends to $q = 12.5$ Å⁻¹ over the same angular range.

(2) **Lorentz–polarization correction.** The scattered intensity
measured by a diffractometer is

$$I(2\theta) \;\propto\; S(q)\;\cdot\;
\underbrace{\frac{1 + \cos^{2}(2\theta)}{2}}_{\text{polarization}}
\;\cdot\;
\underbrace{\frac{1}{\sin^{2}\theta\,\cos\theta}}_{\text{Lorentz}}.$$

The polarization factor $(1+\cos^2 2\theta)/2$ comes from averaging
the Thomson differential cross-section over unpolarised incident
X-rays. The Lorentz factor $1/(\sin^2\theta\cos\theta)$ accounts for
the geometry of an ideally-aligned powder sample in a Bragg–Brentano
parallel-beam diffractometer (rotation time of each crystallite
through the diffraction condition); see Pecharsky & Zavalij,
*Fundamentals of Powder Diffraction and Structural Characterization
of Materials* (2nd ed., Springer 2009), §2.4 and §2.5. Their combined
form is the no-monochromator default used in essentially every
powder-diffraction analysis package (GSAS-II, FullProf, Topas):

$$\mathrm{LP}(\theta) \;=\; \frac{1+\cos^{2}(2\theta)}{\sin^{2}\theta\,\cos\theta}.$$

LP diverges as $\theta \to 0$. In real measurements the direct-beam
stop hides 2θ ≲ 5°; in simulation we simply truncate the plot at
2θ ≥ 20° (anything you'd see below that is dominated by the diverging
LP, not by structural features).

(3) **Instrument broadening.** A measured XRD profile is convolved
with the instrumental resolution function, which for a well-aligned
laboratory Cu-Kα system is approximately Gaussian with full-width
at half maximum ~0.05–0.1° in 2θ for crystalline samples and ~1–3°
for amorphous halos (see Klug & Alexander, *X-ray Diffraction
Procedures*, Wiley 1974, Ch. 9; Cullity & Stock, *Elements of X-ray
Diffraction*, 3rd ed., Prentice Hall 2001, §6.7). AmorphGen applies
a 1D Gaussian convolution with $\sigma_{2\theta}$ as a user-tunable
parameter. For amorphous halos σ ≈ 1° is a reasonable starting point;
adjust to match the specific instrument and sample.

The full simulated XRD intensity at $2\theta$ from a periodic
amorphous cell is therefore

$$I_{\text{sim}}(2\theta) \;=\; \bigl[S\bigl(q(2\theta)\bigr)\,\cdot\,
\mathrm{LP}(\theta)\bigr] \;*\; G_{\sigma_{2\theta}}$$

where $S(q(2\theta))$ is interpolated from the direct-method S(q),
$\mathrm{LP}(\theta)$ is the formula above, and $G_{\sigma}$ is a
Gaussian of width $\sigma_{2\theta}$.

**Important caveat.** The XRD peak position depends on whether LP is
applied:

| Plotted quantity | First peak for a-Ga₂O₃ | Comment |
|---|---|---|
| S(q) vs q | q = 2.5 Å⁻¹ | Real structural FSDP |
| S(q(2θ)) without LP, vs 2θ | 2θ = 35° | Same feature, plotted in angle |
| **S(q(2θ)) × LP vs 2θ** | **2θ = 23°** | What a diffractometer measures |

The "23° peak" in raw experimental XRD is **the same physics** as the
"35° peak" in non-LP-corrected plots; both are the FSDP, just viewed
through different geometric corrections.

:::

:::{tab-item} Weighting
:sync: weighting

Both methods accept the same ``weighting=`` argument. Behind it lies
the **Faber–Ziman partial-summation convention** for the total
structure factor
([Faber & Ziman, *Philos. Mag.* **11** (1965) 153](https://doi.org/10.1080/14786436508211931)):

$$S(q) \;=\; \frac{\displaystyle\sum_{\alpha,\beta}
c_\alpha c_\beta\, f_\alpha(q)\, f_\beta(q)\, S_{\alpha\beta}(q)}
{\displaystyle\Bigl(\sum_\alpha c_\alpha f_\alpha(q)\Bigr)^{2}}$$

where $c_\alpha = N_\alpha/N$ is the number fraction of element
$\alpha$ and $f_\alpha(q)$ is its scattering factor (X-ray atomic
form factor or neutron coherent scattering length). The Faber–Ziman
partials $S_{\alpha\beta}(q)$ are the ones defined in the previous
tab.

For multi-element systems with $A \neq B$ pairs, the sum
$\sum_{\alpha\beta}$ counts each ordered pair so the off-diagonal
$AB$ and $BA$ terms together give a factor of 2:

$$S(q) \;=\; \frac{1}{\langle f \rangle^{2}}
\left[\sum_{\alpha} c_\alpha^{2} f_\alpha^{2} S_{\alpha\alpha}
\;+\; 2\sum_{\alpha<\beta} c_\alpha c_\beta f_\alpha f_\beta
S_{\alpha\beta}\right].$$

The three ``weighting=`` options pick different per-element
scattering factors $f_\alpha$:

| ``weighting`` | Per-element factor $f_\alpha$ | When to use |
|---|---|---|
| ``"xray"``       | Atomic number $Z_\alpha$ (Z-approximation; q-independent) | Comparing to X-ray diffraction |
| ``"neutron"``    | Tabulated coherent neutron scattering length $b_\alpha$ | Comparing to neutron diffraction |
| ``"unweighted"`` | $f_\alpha = 1$ for all species | Pure structure comparison; same as a single-species sum |

**X-ray Z-approximation.** AmorphGen uses $f_\alpha = Z_\alpha$ —
the $q \to 0$ limit of the atomic X-ray form factor. The full
$q$-dependent X-ray form factors $f_\alpha(q)$ tabulated by
Cromer & Mann
([Cromer & Mann, *Acta Cryst. A* **24** (1968) 321](https://doi.org/10.1107/S0567739468000550))
fall off with $q$ approximately as

$$f_\alpha(q) \;\approx\; \sum_{i=1}^{4} a_i \exp\!\bigl(-b_i\,(q/4\pi)^{2}\bigr) + c$$

so the Z-approximation systematically over-estimates the heavy-atom
weighting at high $q$. For the first sharp diffraction peak
($q \lesssim 3$ Å⁻¹) the error from using $Z$ instead of $f(q)$ is
typically < 5 %; for the high-$q$ region it grows to ~20 %. Full
Cromer–Mann form factors are listed as future work in this guide.

**Neutron scattering lengths.** AmorphGen ships a built-in table of
~50 common-element coherent scattering lengths from Sears
([Sears, *Neutron News* **3** (1992) 26](https://doi.org/10.1080/10448639208218770)).
Unlike X-ray scattering, neutron $b$ values do **not** scale with
$Z$ — they vary irregularly (e.g. $b_{\rm H} = -3.74$ fm but
$b_{\rm D} = +6.67$ fm), which makes neutron diffraction sensitive to
contrasts hidden in X-ray patterns. Use ``weighting="neutron"`` when
comparing to neutron data.

**Why ``"unweighted"`` misses the FSDP in oxides.** In an oxide
like a-Ga₂O₃, the FSDP at $q \approx 2.5$ Å⁻¹ comes from medium-
range Ga–Ga (peaks at ~2.5 Å⁻¹) and O–O (peaks at ~2.6 Å⁻¹) partial
correlations. The Ga–O partial $S_{\rm GaO}(q)$ **dips** to ~0.25 at
the same $q$. With $f = 1$ for all elements, the negative Ga–O
contribution exactly cancels the positive like-pair contributions:

$$S^{(\rm unwt)}(q\!=\!2.5) = c_{\rm Ga}^{2} (1.23) + 2c_{\rm Ga}c_{\rm O}(0.25) + c_{\rm O}^{2}(1.25) \approx 0.84$$

With X-ray weighting ($Z_{\rm Ga} = 31 \gg Z_{\rm O} = 8$), the
Ga–Ga partial gets multiplied by $31^2 = 961$, the O–O by $64$, the
Ga–O by $248$ — the heavy Ga–Ga dominates and the FSDP survives:

$$S^{(\rm xray)}(q\!=\!2.5) \approx \frac{0.4^2\cdot 961\cdot 1.23 + 2\cdot 0.4\cdot 0.6\cdot 248\cdot 0.25 + 0.6^2\cdot 64\cdot 1.25}{17.2^{2}} \approx 2.0$$

which is why X-ray experiments **see** the FSDP and the unweighted
ensemble-comparison plots do not.

:::

:::{tab-item} Physical validity
:sync: physics

The direct S(q) method and the Bragg + LP simulated-XRD recipe both
implement standard, well-established physics. Below: the master
equations, who first wrote them down, and the spot-checks AmorphGen
passes against published data.

### Master equations and primary references

**Debye scattering equation** for an isotropic ensemble:

$$S(q) \;=\; \frac{1}{N\langle f\rangle^{2}}
\sum_{i=1}^{N}\sum_{j=1}^{N} f_i f_j\,\frac{\sin(q r_{ij})}{q r_{ij}}$$

([Debye, *Ann. Phys.* **351** (1915) 809](https://doi.org/10.1002/andp.19153510606)).
The discrete-Fourier-transform version
$S(\vec G) = N^{-1}\langle f\rangle^{-2}|\sum_i f_i \mathrm{e}^{\mathrm{i}\vec G\cdot\vec r_i}|^{2}$
follows by writing $\sin(qr)/(qr) = \langle \mathrm{e}^{\mathrm{i}\vec q\cdot\vec r}\rangle_{|\vec q|=q}$
and the convolution theorem; it is the form most numerical
implementations use for periodic systems.

**Fourier-Bessel link to g(r):**

$$S(q) - 1 \;=\; 4\pi\rho \int_0^\infty [g(r)-1]\,
\frac{\sin(qr)}{qr}\,r^2\,\mathrm{d}r$$

([Zernike & Prins, *Z. Phys.* **41** (1927) 184](https://doi.org/10.1007/BF01391926); textbook
treatments in Egami & Billinge, *Underneath the Bragg Peaks*,
Pergamon 2003, §3.2; Hansen & McDonald, *Theory of Simple Liquids*,
4th ed., Elsevier 2013, Ch. 2).

**Faber–Ziman partial-summation convention** for multi-element
totals:

$$S(q) \;=\; \frac{\sum_{\alpha\beta} c_\alpha c_\beta f_\alpha f_\beta
S_{\alpha\beta}(q)}{\bigl(\sum_\alpha c_\alpha f_\alpha\bigr)^{2}}$$

([Faber & Ziman, *Philos. Mag.* **11** (1965) 153](https://doi.org/10.1080/14786436508211931)).

**Bragg's law** linking momentum transfer to scattering angle:

$$q = \frac{4\pi}{\lambda}\sin\theta$$

([Bragg & Bragg, *Proc. R. Soc. Lond. A* **88** (1913) 428](https://doi.org/10.1098/rspa.1913.0040)).

**Lorentz–polarization correction** for parallel-beam
Bragg–Brentano powder diffraction with unpolarised X-rays:

$$\mathrm{LP}(\theta) \;=\; \frac{1+\cos^{2}(2\theta)}{\sin^{2}\theta\,\cos\theta}$$

(textbook derivation in Cullity & Stock, *Elements of X-ray
Diffraction*, 3rd ed., Prentice Hall 2001, §4.10 and §4.12;
Pecharsky & Zavalij, *Fundamentals of Powder Diffraction*, 2nd ed.,
Springer 2009, §2.4–2.5). It is the no-monochromator default in
essentially every powder-diffraction analysis program — GSAS-II
([Toby & Von Dreele, *J. Appl. Cryst.* **46** (2013) 544](https://doi.org/10.1107/S0021889813003531)),
FullProf
([Rodríguez-Carvajal, *Physica B* **192** (1993) 55](https://doi.org/10.1016/0921-4526(93)90108-I)),
and Topas
([Coelho, *J. Appl. Cryst.* **51** (2018) 210](https://doi.org/10.1107/S1600576718000183)).

**Reciprocal-lattice-sum approach** for amorphous MD has been
established and benchmarked in the dedicated analysis packages
ISAACS
([Le Roux & Petkov, *J. Appl. Cryst.* **43** (2010) 181](https://doi.org/10.1107/S0021889809051929))
and LiquidLib
([Walter, Bian, Mendoza & Schweizer, *Comput. Phys. Commun.* **228** (2018) 209](https://doi.org/10.1016/j.cpc.2018.03.005));
AmorphGen's ``structure_factor_direct`` is the same algorithm.

### Spot-checks the implementation passes

| Test | Expected (from literature) | AmorphGen | ✓ |
|---|---|---|---|
| Asymptotic S(q→∞), X-ray, a-Ga₂O₃ | $\langle f^{2}\rangle/\langle f\rangle^{2} = 1.43$ | S(q=7) = 1.44 | ✅ |
| FSDP intensity, a-Ga₂O₃ | 1.8–2.0 (X-ray expt + GAP); [Kaewmeechai et al. PRB **111** (2025) 035203, Fig. S2b](https://doi.org/10.1103/PhysRevB.111.035203) | 2.0 | ✅ |
| FSDP position, a-Ga₂O₃ | $q \approx 2.4$ Å⁻¹ (same reference) | 2.5 | ✅ |
| FSDP position, a-SiO₂ | $q \approx 1.5$ Å⁻¹; [Wright, *J. Non-Cryst. Solids* **179** (1994) 84](https://doi.org/10.1016/0022-3093(94)90683-1) | 1.38 | ⚠️ ~10% low (96-atom cell artifact) |
| XRD halo position, a-Ga₂O₃, Cu-Kα | $2\theta \approx 23°$ (Verlet *et al.* and various) | 22.7° | ✅ |

### Known limitations and recommended workarounds

| Limitation | Effect | Workaround |
|---|---|---|
| **Cromer–Mann $f(q)$ not implemented** — Z-approximation only ([Cromer & Mann, *Acta Cryst. A* **24** (1968) 321](https://doi.org/10.1107/S0567739468000550)) | < 5 % error for FSDP region (q < 3 Å⁻¹); ~20 % at high q (q > 5 Å⁻¹) | For high-q precision, post-process with external XRD analysis software |
| **No Debye–Waller factor** | Static-snapshot S(q); no thermal attenuation | MD ensemble already samples thermal motion; for low-T multiply by $\exp(-q^{2}\langle u^{2}\rangle/3)$ |
| **No sample-displacement / absorption corrections** | < 1 % effect for amorphous-halo positions | Use full Rietveld package (GSAS-II, FullProf) for crystalline-peak fitting |

### Bottom line on physical correctness

For glass-physics work where peak positions to ~0.1 Å⁻¹ and
intensities to ~10 % are sufficient, AmorphGen's S(q) and XRD
implementations are **physically correct and use standard textbook
conventions** referenced above. The output is directly comparable to
neutron / X-ray experimental data and to other MD-analysis packages
(ISAACS, LiquidLib) within these accuracy tolerances. For
milliÅngström-precision peak fitting and intensity ratios to better
than 1 %, you would need to add Cromer–Mann $f(q)$, Debye–Waller, and
instrument-specific corrections.

```{admonition} Is the implementation novel?
:class: note

No — every equation comes from primary literature published between
1913 and 1992, and the algorithm is the same one used in ISAACS,
LiquidLib, freud, OVITO and DL_POLY. AmorphGen contributes the
Python wiring, not the physics. For details on what's "ours" vs
what's standard, and how to cite the method in a paper, see
{doc}`/notes/sq_xrd_credits`.
```

:::

::::

### Example: a-Ga₂O₃

The settings below reproduce the validation result against the
DFT-PBE0 ensemble from
[Kaewmeechai, Strand & Shluger, *Phys. Rev. B* **111** (2025) 035203](https://doi.org/10.1103/PhysRevB.111.035203)
(FSDP at $q \approx 2.4$ Å⁻¹ with intensity ~2.0; XRD halo at
$2\theta \approx 23°$ Cu-Kα):

```python
import numpy as np
from scipy.ndimage import gaussian_filter1d
from amorphgen.analysis import StructureAnalyser

sa = StructureAnalyser("ga2o3_ensemble/*.cif", cutoff="auto-rdf")

# X-ray S(q)
sq = sa.structure_factor_direct(weighting="xray", qmax=12.0, nq=300)
q, s, nb = (np.array(sq[k]) for k in ("q", "s_q", "n_per_bin"))
mask = nb >= 5
s_safe = np.where(np.isnan(s), 1.0, s)   # NaN-safe before smoothing
s_smooth = gaussian_filter1d(s_safe, sigma=0.05 / (q[1] - q[0]))

# Simulated XRD I(2θ)
sq = sa.structure_factor_direct(weighting="xray", qmax=8.0, nq=400)
# ... Bragg + LP + smearing as shown in the "Simulated XRD" tab
```

The same settings apply to other oxide glasses (a-SiO₂, a-HfO₂)
and to a-Si; only the ``cutoff`` and the input file pattern need
adjusting.

### Open issues / future work

- **Cromer-Mann f(q) form factors** for q > 5 Å⁻¹ accuracy (currently
  Z² approximation)
- **Direct-method Faber-Ziman partials** — currently only the total
  S(q) is computed via the direct Debye sum; the partials
  $S_{\alpha\beta}(q)$ still come from the FT-of-g(r) path
- **Per-bin error bars** from the spread of S(q) across structures
- **`xrd_pattern()` convenience method** on ``StructureAnalyser``
  bundling the Bragg + LP + smearing recipe

## Full CLI flag reference

```bash
amorphgen --analyse \
    --input-dir DIR_OF_STRUCTURES \
    [--cutoff MODE_OR_NUMBER] \
    [--per-structure] \
    [--save-report FILE] \
    [--save-plot DIR] \
    [--save-pdf] \
    [--reference YAML] \
    [--smearing SIGMA] \
    [--total-rdf] \
    [--dpi N] \
    [--show-title]
```

| Flag | What it does |
|---|---|
| `--input-dir DIR` | Directory of structure files (``.xyz``, ``.extxyz``, ``.cif``, ``.vasp``). Globs everything matching these extensions. |
| `--cutoff MODE` | `auto-rdf` (default), `auto`, or a number in Å. |
| `--per-structure` | Print a per-structure table (one row per file: density, E/atom, CN). |
| `--save-report FILE` | Write the full text report (densities, bond distances, coordination, angles) to a file. |
| `--save-plot DIR` | Save the four standard figures (RDF, CN, angles, density) plus CSV data into ``DIR``. |
| `--save-pdf` | Also save vector PDF copies alongside the PNGs. |
| `--reference YAML` | Validate against the literature ranges in YAML, print a match/concern/fail table. |
| `--smearing SIGMA` | Gaussian smearing of the RDF in Å (typical: ``0.02–0.05`` to compare against experimental neutron data). |
| `--total-rdf` | Overlay the total g(r) on the partial-RDF plot. |
| `--dpi N` | PNG DPI (default 300). |
| `--show-title` | Add titles to each plot (default off — captions usually clearer in figures). |

## Outputs explained

For each ensemble, ``--analyse --save-plot DIR`` writes:

| File | What's in it |
|---|---|
| `analysis_rdf.png` / `.pdf` | Partial RDFs for all unique pairs in the system. One line per pair using the Okabe-Ito palette. |
| `analysis_rdf.csv` | Columns: ``r(A), g(r)_<pair1>, g(r)_<pair2>, …``. Re-plot in any tool. |
| `analysis_cn.png` / `.pdf` | Coordination distribution. For binary AB systems (e.g. SiO₂), shown as **mirrored bars**: A-B on top, B-A reflected below the zero line. For mono-element systems (a-Si), shown side-by-side. |
| `analysis_cn.csv` | Per-pair CN counts as percentages of the centred atom population. |
| `analysis_angles.png` / `.pdf` | Bond-angle histograms (normalised). One line per triplet. |
| `analysis_angles.csv` | Raw angle values, one row per triplet observation. |
| `analysis_density.png` / `.pdf` | Per-structure density violin with jittered scatter and mean ± std label. |
| `analysis_density.csv` | One row per structure: ``structure_index, density_g_per_cm3``. |

## Python API

For programmatic access — useful in scripts, notebooks, and the
comparison workflow:

```python
from amorphgen.analysis import StructureAnalyser

sa = StructureAnalyser("hybrid_ga2o3/random_opt/")   # accepts a dir OR list of files
sa.summary()                                         # print structural summary

# Individual descriptors
rho = sa.density()
print(rho["mean"], rho["std"], rho["values"])

rdf = sa.rdf(pair="Ga-O", sigma=0.05)
print(rdf["r"], rdf["g_r"])

cn = sa.coordination()
print(cn["Ga-O"]["mean"], cn["Ga-O"]["distribution"])

ang = sa.bond_angles()
print(ang["O-Ga-O"]["mean"])

# Multi-ensemble comparison
from amorphgen.analysis import EnsembleSpec, compare_ensembles
compare_ensembles(...)        # see "Compare multiple ensembles" tab above

# Validate against reference
from amorphgen.analysis import validate_against_reference, format_validation_report
import yaml
with open("examples/reference_a_Ga2O3.yaml") as f:
    ref = yaml.safe_load(f)
print(format_validation_report(validate_against_reference(sa, ref)))
```

See {doc}`/api/analysis` for the full Python API.

## Troubleshooting

### "Density is high but CN looks too low"

The default `auto-rdf` cutoff handles most systems, but if you're using
the legacy `--cutoff auto` (minsep-based), it may truncate the first
RDF peak for materials with broad bond distributions. Symptom: many
atoms appear under-coordinated. Fix: drop the `--cutoff auto` and let
the default `auto-rdf` resolve it.

### "Per-structure E/atom shows N/A"

If your structures are in VASP or CIF format, they don't carry energy
in their headers. AmorphGen falls back to reading ``random_gen.log``
in the parent directory (the file written by ``amorphgen --random-gen
--relax``). If you've moved the structures away from their original
``--random-gen`` output, copy the log alongside them, or use
``--format xyz`` / ``--format extxyz`` (which embed energy in the
comment line) when generating.

### "Si–Si appears as a bond in a-SiO₂"

That's an analysis artifact. Same-element pairs in multi-element ionic
compounds (Si–Si in SiO₂, Hf–Hf in HfO₂, Ga–Ga in Ga₂O₃) are
**second-shell contacts mediated through the anion**, not first-shell
bonds. From v1.0.0+ they're excluded from bond-angle triplets
automatically. For coordination, you can ignore the X–X mean — the
relevant CN for AB systems is A–B and B–A.

### "RDF goes to zero suddenly at large r"

That's the auto-detected ``rmax`` (half the smallest cell vector). RDFs
beyond half the cell can't be computed correctly under periodic
boundary conditions. To extend, generate with a larger cell.
