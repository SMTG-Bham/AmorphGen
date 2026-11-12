"""EXAFS-comparable descriptors: N, R, sigma^2 and the third cumulant.

Why this module exists
----------------------
AmorphGen already reports first-shell coordination numbers and bond
distances, and the reference YAMLs under ``examples/`` compare them to
values taken from published EXAFS fits (``examples/reference_a_Ga2O3.yaml``
quotes ~4.5 for Ga-O; ``docs/validation/index.md`` tabulates it against a
computed 4.42). Those two numbers are not the same quantity:

- AmorphGen's ``coordination()`` is a *geometric count* inside a cutoff.
- An EXAFS N is a *fit parameter*, extracted from chi(k) under a harmonic
  (Gaussian) model of the pair distribution, correlated with sigma^2 and
  with the amplitude reduction factor S0^2.

In a glass, where the first shell is genuinely asymmetric, those two
differ systematically. This module computes the quantities an EXAFS paper
actually reports — N, R, sigma^2, and the asymmetry that biases the fit —
directly from the ensemble. These geometric cumulants are not in general
identical to parameters fitted to experimental chi(k); R_eff is a leading-order
correction with a scalar mean free path, not an exact prediction of a fitted R.

What is computed
----------------
For an ordered pair (absorber A, scatterer B), every A-B distance inside
the first-shell cutoff is collected over every A atom of every structure.
The cumulants of that distribution are:

    C1 = <R>                  first-shell distance (A)
    C2 = <(R - C1)^2> = sigma^2   EXAFS Debye-Waller factor (A^2)
    C3 = <(R - C1)^3>         third cumulant (A^3), the leading asymmetry

and N is the mean number of B neighbours per A atom — identical by
construction to ``StructureAnalyser.coordination()[f"{A}-{B}"]["mean"]``.

sigma^2 here is the *mean-square relative displacement* (MSRD) along the
bond, which is what EXAFS measures — not the sum of two independent
atomic mean-square displacements. Correlated motion of the absorber and
its neighbour is already accounted for, because the variance is taken on
the pair distance itself.

Real vs effective distribution
------------------------------
EXAFS does not sample P(R). The photoelectron amplitude carries a 1/R^2
geometric factor and an exp(-2R/lambda) mean-free-path damping, so the
measurement samples the *effective* distribution

    P_eff(R) ~ P(R) * R^-2 * exp(-2 R / lambda)

Both factors weight short distances more heavily, so the R an EXAFS fit
reports sits below the true mean. To first order in the cumulant
expansion,

    C1_eff  =  C1 - 2 sigma^2 (1/C1 + 1/lambda)

(Dalba & Fornasini, J. Synchrotron Rad. 4 (1997) 243,
doi:10.1107/S0909049597006900; Filipponi, J. Phys.: Condens. Matter 13
(2001) R23, doi:10.1088/0953-8984/13/7/201.) For a typical oxide first
shell this is a shift of 0.005-0.02 A — small, but the same size as the
precision an EXAFS paper quotes on R. ``R_eff`` is reported alongside
``R`` so a user can compare against the published number without having
to apply the correction by hand.

The static-ensemble caveat
--------------------------
A set of relaxed (0 K) structures carries only the *configurational*
part of the disorder:

    sigma^2_experiment  =  sigma^2_static  +  sigma^2_thermal

so the sigma^2 computed from a relaxed ensemble is a **lower bound** on a
room-temperature measurement, and the gap is the vibrational term. Two
ways to close it:

1. Point the analyser at a finite-temperature trajectory (the stage-6
   low-T equilibration frames) instead of the relaxed structures; the
   thermal term is then sampled directly.
2. Compare against a measurement extrapolated to T -> 0, or against the
   static term separated out by a correlated-Debye/Einstein fit of
   temperature-dependent EXAFS.

This caveat is printed in the report rather than buried here, because
quoting a 0 K sigma^2 against a 300 K measurement is the easiest way to
misuse this module.
"""

from __future__ import annotations

import numpy as np

from .structure import build_neighbour_dict

#: Default photoelectron mean free path (A) used for the effective-
#: distribution correction. First-shell EXAFS in oxides sits in the
#: 4-6 A range; C1_eff depends only weakly on the choice (a 4 -> 6 A
#: change moves C1_eff by well under 0.005 A for sigma^2 ~ 0.01 A^2).
DEFAULT_MFP = 5.0

#: Bootstrap resamples used to put an ensemble-sampling error bar on the
#: descriptors. Resampling whole structures (not individual paths) is the
#: point: the paths within one structure are not independent samples of
#: the disorder, the structures are.
DEFAULT_BOOTSTRAP = 200

#: Below this many structures the bootstrap interval is itself poorly
#: determined -- there are only n^n distinct resamples, and the tails are
#: unreachable. The error bar is still reported, because "wide and
#: unreliable" is the honest answer, but the report says so.
MIN_BOOTSTRAP_STRUCTURES = 8

#: ``|skew|`` above which a shell is asymmetric enough to bias a harmonic
#: fit of N and R.
SKEW_THRESHOLD = 0.3

#: ``|kurt|`` (excess kurtosis) above which a *symmetric* shell is still
#: not Gaussian. The skew test alone passes a flat-topped shell with sharp
#: edges, and a harmonic fit of one can be meaningless: for a-CaZrCl6
#: Ca-Cl (skew -0.04, kurt -0.83) Larch fits of FEFF chi(k) reported N
#: 40-50% above the structure's, moving by tens of percent with the fitting
#: choices (docs/notes/exafs_crosscheck). Whether it matters also depends
#: on the width -- see :data:`amorphgen.analysis.harmonic.C4_TERM_FAILED`
#: -- so this is a warning, not a verdict. Ordinary sampling noise on a
#: Gaussian is ~sqrt(24/n), below 0.2 for any real shell.
KURT_THRESHOLD = 0.5


def collect_shell_distances(atoms_list, max_cutoff, get_cutoff_fn,
                            pair=None) -> dict:
    """Every first-shell absorber-scatterer distance, per ordered pair.

    The raw material behind both :func:`compute_exafs_descriptors` and
    :func:`~amorphgen.analysis.harmonic.fit_shells`: the former reduces
    it to cumulants, the latter builds the EXAFS amplitude from it
    directly, with no binning in between.

    Parameters
    ----------
    atoms_list : list of ase.Atoms
        The ensemble (or the frames of a trajectory).
    max_cutoff : float
        Largest pair cutoff; used to build the neighbour list once.
    get_cutoff_fn : callable
        ``(sym_i, sym_j) -> cutoff`` for the first shell of that pair.
    pair : str, optional
        Restrict to one ordered ``"<absorber>-<scatterer>"`` pair.

    Returns
    -------
    dict
        Keyed by ``"<absorber>-<scatterer>"``, each value a dict with
        ``absorber``, ``scatterer``, ``distances`` (every A-B distance,
        seen from the A side), ``counts`` (one entry per A atom, so
        ``counts.mean()`` is N), ``n_structures``, and
        ``distance_slices`` / ``count_slices`` -- one ``(start, stop)``
        per structure, so a bootstrap can resample whole structures.
    """
    want = None
    if pair is not None:
        parts = pair.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"pair must be '<absorber>-<scatterer>', got {pair!r}")
        want = (parts[0], parts[1])

    distances: dict[tuple[str, str], list[float]] = {}
    counts: dict[tuple[str, str], list[int]] = {}
    # Where each structure's entries start and stop, so that a bootstrap
    # can resample whole structures rather than individual paths. The
    # loop appends in structure order, so each structure owns one
    # contiguous block per pair.
    d_slices: dict[tuple[str, str], list[tuple[int, int]]] = {}
    c_slices: dict[tuple[str, str], list[tuple[int, int]]] = {}

    for s_idx, atoms in enumerate(atoms_list):
        d_start = {k: len(v) for k, v in distances.items()}
        c_start = {k: len(v) for k, v in counts.items()}
        nbr_dict, syms = build_neighbour_dict(atoms, max_cutoff, get_cutoff_fn)
        unique = sorted(set(syms))

        for i, si in enumerate(syms):
            if want is not None and si != want[0]:
                continue
            per_sym: dict[str, int] = {s: 0 for s in unique}
            for _, sj, d, _ in nbr_dict[i]:
                if want is not None and sj != want[1]:
                    continue
                per_sym[sj] = per_sym.get(sj, 0) + 1
                distances.setdefault((si, sj), []).append(d)
            for sj, n in per_sym.items():
                if want is not None and sj != want[1]:
                    continue
                counts.setdefault((si, sj), []).append(n)

        for key in set(distances) | set(counts):
            d_slices.setdefault(key, []).append(
                (d_start.get(key, 0), len(distances.get(key, []))))
            c_slices.setdefault(key, []).append(
                (c_start.get(key, 0), len(counts.get(key, []))))

    return {
        f"{a}-{b}": {
            "absorber": a,
            "scatterer": b,
            "distances": np.asarray(distances.get((a, b), []), dtype=float),
            "counts": np.asarray(counts[(a, b)], dtype=float),
            "distance_slices": d_slices.get((a, b), []),
            "count_slices": c_slices.get((a, b), []),
            "n_structures": len(atoms_list),
        }
        for a, b in sorted(counts)
    }


def _gather(values, slices, chosen):
    """Concatenate the blocks of ``values`` belonging to ``chosen`` structures."""
    parts = [values[slices[s][0]:slices[s][1]] for s in chosen]
    return np.concatenate(parts) if parts else np.empty(0)


def shell_cumulants(distances, counts):
    """N and the first four cumulants of one shell.

    The single place these are defined, so the point estimate and every
    bootstrap resample cannot drift apart.
    """
    c1 = float(distances.mean())
    delta = distances - c1
    c2 = float((delta ** 2).mean())
    c3 = float((delta ** 3).mean())
    c4 = float((delta ** 4).mean()) - 3.0 * c2 ** 2
    sigma = np.sqrt(c2)
    degenerate = sigma <= 1e-12
    return {
        "N": float(counts.mean()),
        "R": c1,
        "sigma2": c2,
        "C3": c3,
        "skew": 0.0 if degenerate else float(c3 / sigma ** 3),
        "C4": c4,
        "kurt": 0.0 if degenerate else float(c4 / c2 ** 2),
    }


def bootstrap_shell(shell, statistic, n_boot=DEFAULT_BOOTSTRAP, seed=0):
    """Standard errors from resampling whole structures with replacement.

    Parameters
    ----------
    shell : dict
        One entry of :func:`collect_shell_distances`.
    statistic : callable
        ``(distances, counts) -> dict of float``. Called once per
        resample; keys that appear in every resample get an error.
    n_boot : int
        Resamples. 0 disables and returns ``{}``.
    seed : int
        Makes the error bars reproducible, so a rerun does not appear to
        change the answer.

    Returns
    -------
    dict
        ``{key + "_err": standard deviation over resamples}``, plus
        ``n_boot``. Empty when the bootstrap is off or there is only one
        structure (nothing to resample).
    """
    n_struct = shell["n_structures"]
    if not n_boot or n_struct < 2:
        return {}

    rng = np.random.default_rng(seed)
    d, c = shell["distances"], shell["counts"]
    d_sl, c_sl = shell["distance_slices"], shell["count_slices"]

    samples: dict[str, list[float]] = {}
    for _ in range(int(n_boot)):
        chosen = rng.integers(0, n_struct, size=n_struct)
        dd = _gather(d, d_sl, chosen)
        cc = _gather(c, c_sl, chosen)
        if dd.size == 0 or cc.size == 0:
            continue
        for k, v in statistic(dd, cc).items():
            samples.setdefault(k, []).append(v)

    out = {f"{k}_err": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
           for k, v in samples.items()}
    out["n_boot"] = int(n_boot)
    return out


def compute_exafs_descriptors(atoms_list, max_cutoff, get_cutoff_fn,
                              pair=None, mfp=DEFAULT_MFP, nbins=120,
                              bootstrap=DEFAULT_BOOTSTRAP, seed=0) -> dict:
    """First-shell EXAFS descriptors for every ordered absorber-scatterer pair.

    Parameters
    ----------
    atoms_list : list of ase.Atoms
        The ensemble (or the frames of a trajectory).
    max_cutoff : float
        Largest pair cutoff; used to build the neighbour list once.
    get_cutoff_fn : callable
        ``(sym_i, sym_j) -> cutoff`` — the per-pair first-shell cutoff.
        Pass ``StructureAnalyser._get_cutoff`` to inherit ``auto-rdf``.
    pair : str, optional
        Restrict to one **ordered** pair, ``"Ga-O"`` meaning *Ga absorber,
        O scatterer* (the Ga K-edge Ga-O path). Unlike
        :func:`~amorphgen.analysis.structure.compute_bond_distances`, which
        keys pairs symmetrically, ``"Ga-O"`` and ``"O-Ga"`` are distinct
        entries here: they share R, sigma^2 and C3 (the same set of
        distances) but have different N.
    mfp : float
        Photoelectron mean free path lambda in A for the effective-
        distribution correction (default :data:`DEFAULT_MFP`).
    nbins : int
        Bins for the returned first-shell distance distribution P(R).
    bootstrap : int
        Resamples for the ensemble-sampling error bars, obtained by
        resampling whole structures with replacement
        (:data:`DEFAULT_BOOTSTRAP`). 0 disables them. Paths inside one
        structure are not independent samples of the disorder, so
        resampling paths would badly understate the uncertainty.
    seed : int
        Seed for the resampling, so the error bars are reproducible.

    Returns
    -------
    dict
        Keyed by ``"<absorber>-<scatterer>"``. Each value is a dict with:

        ``N``, ``N_std``
            Mean and standard deviation (over absorbers) of the number of
            scatterers in the shell. ``N`` equals the matching entry of
            ``coordination()`` by construction.
        ``R``
            C1, the mean first-shell distance (A).
        ``R_eff``
            C1 corrected to the effective distribution EXAFS samples (A) —
            what a fit would report. See the module docstring.
        ``sigma2``
            C2, the static EXAFS Debye-Waller factor (A^2). A lower bound
            on a finite-temperature measurement.
        ``C3``
            Third cumulant (A^3); positive means a tail to long distance.
        ``skew``
            C3 / sigma^3, dimensionless. ``|skew|`` below ~0.3 means the
            shell is symmetric; larger values mean the published N and R
            were probably biased by the Gaussian assumption.
        ``C4``, ``kurt``
            Fourth cumulant (A^4) and the excess kurtosis C4 / sigma^4.
            A Gaussian has zero; a flat-topped shell with sharp edges is
            negative. Symmetry is not enough for a harmonic fit to be
            safe -- a shell with small ``skew`` but large ``|kurt|`` is
            still not Gaussian, and the fit can fail outright.
        ``cutoff``
            First-shell cutoff used for this pair (A).
        ``n_paths``, ``n_absorbers``, ``n_structures``
            Sample sizes behind the statistics.
        ``P_R``
            ``{"r": [...], "p": [...]}`` — the normalised first-shell
            distance distribution, for plotting or CSV export.
        ``N_err``, ``R_err``, ``R_eff_err``, ``sigma2_err``, ``C3_err``,
        ``skew_err``, ``C4_err``, ``kurt_err``, ``n_boot``
            Standard errors over the ensemble, present when ``bootstrap``
            is non-zero and there is more than one structure. They are
            often larger than the difference between two similar
            ensembles — check them before reading anything into such a
            difference.

    Notes
    -----
    Pairs with no distances inside the cutoff are omitted from the result.
    """
    shells = collect_shell_distances(atoms_list, max_cutoff, get_cutoff_fn,
                                     pair=pair)

    result = {}
    for key, shell in shells.items():
        ds = shell["distances"]
        if ds.size == 0:
            continue                      # no first-shell B around any A
        ns = shell["counts"]
        key = (shell["absorber"], shell["scatterer"])

        cum = shell_cumulants(ds, ns)
        c1, sigma2, c3, skew = (cum["R"], cum["sigma2"], cum["C3"],
                                cum["skew"])

        # Effective-distribution shift; first order in the cumulant
        # expansion, see module docstring.
        r_eff = c1 - 2.0 * sigma2 * (1.0 / c1 + 1.0 / mfp)

        # Ensemble-sampling error bars. Cheap here (~0.1 ms a resample),
        # so they are on by default: without them a user comparing two
        # ensembles reads differences the structure-to-structure spread
        # does not support.
        errs = bootstrap_shell(shell, shell_cumulants, n_boot=bootstrap,
                               seed=seed)
        r_eff_err = (2.0 * errs["sigma2_err"] * (1.0 / c1 + 1.0 / mfp)
                     if "sigma2_err" in errs else None)

        # A crystal's symmetry-equivalent distances can differ by a few
        # floating-point rounding units. NumPy expands an exactly constant
        # range, but an almost-constant range can give repeated bin edges
        # (or fail outright with NumPy 2). Use the same 1 Å display span as
        # NumPy's constant-data fallback when bins cannot be resolved.
        # Cumulants above are still calculated from the original distances.
        lo, hi = float(ds.min()), float(ds.max())
        min_span = 4 * nbins * np.spacing(max(abs(lo), abs(hi), 1.0))
        hist_range = (lo - 0.5, hi + 0.5) if hi - lo < min_span else None
        counts_hist, edges = np.histogram(ds, bins=nbins, range=hist_range,
                                         density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])

        result[f"{key[0]}-{key[1]}"] = {
            "absorber": key[0],
            "scatterer": key[1],
            "N": float(ns.mean()),
            "N_std": float(ns.std()),
            "R": c1,
            "R_eff": float(r_eff),
            "sigma2": sigma2,
            "C3": c3,
            "skew": skew,
            "C4": cum["C4"],
            "kurt": cum["kurt"],
            "cutoff": float(get_cutoff_fn(key[0], key[1])),
            "n_paths": int(ds.size),
            "n_absorbers": int(ns.size),
            "n_structures": len(atoms_list),
            "mfp": float(mfp),
            "P_R": {"r": centres.tolist(), "p": counts_hist.tolist()},
            **errs,
        }
        if r_eff_err is not None:
            result[f"{key[0]}-{key[1]}"]["R_eff_err"] = float(r_eff_err)

    return result


def format_exafs_report(result: dict, bonded_only: bool = True) -> str:
    """Human-readable table for :func:`compute_exafs_descriptors` output.

    Parameters
    ----------
    result : dict
        Output of :func:`compute_exafs_descriptors`.
    bonded_only : bool
        Drop same-element and anion-anion entries, which are second-shell
        contacts rather than single-scattering first-shell paths and are
        not what an EXAFS first-shell fit reports. Set ``False`` to see
        every pair.
    """
    rows = _select_rows(result, bonded_only)
    if not rows:
        return ("\n  No EXAFS descriptors "
                "(no pairs inside the first-shell cutoff).")

    bar = "=" * 78
    n_struct = next(iter(result.values()))["n_structures"]
    lines = ["", bar,
             f"  EXAFS first-shell descriptors  ({n_struct} structure(s))",
             bar,
             f"  {'edge path':<12}{'N':>6}{'R (A)':>9}{'R_eff':>8}"
             f"{'sigma2 (A^2)':>14}{'C3 (A^3)':>11}{'skew':>7}",
             "  " + "-" * 72]

    for key, d in rows:
        lines.append(
            f"  {key:<12}{d['N']:>6.2f}{d['R']:>9.3f}{d['R_eff']:>8.3f}"
            f"{d['sigma2']:>14.5f}{d['C3']:>11.2e}{d['skew']:>7.2f}"
        )
        if "N_err" in d:
            lines.append(
                f"  {'+/-':<12}{d['N_err']:>6.2f}{d['R_err']:>9.3f}"
                f"{d.get('R_eff_err', 0.0):>8.3f}{d['sigma2_err']:>14.5f}"
                f"{d['C3_err']:>11.2e}{d['skew_err']:>7.2f}"
            )

    lines.append("  " + "-" * 72)
    lines.append("  N, R and sigma^2 are the quantities an EXAFS fit reports; "
                 "R_eff is R shifted")
    lines.append("  onto the effective distribution EXAFS samples "
                 "(1/R^2 and exp(-2R/lambda) weighting,")
    lines.append(f"  lambda = {next(iter(rows))[1]['mfp']:.1f} A) — compare "
                 "published R against R_eff.")

    first = rows[0][1]
    if "N_err" in first:
        lines.append(f"  The +/- row is the ensemble-sampling standard "
                     f"error, from {first['n_boot']} bootstrap resamples of")
        lines.append("  the structures (not of the paths: paths within one "
                     "structure are not")
        lines.append("  independent samples of the disorder). Compare two "
                     "ensembles against it,")
        lines.append("  not against the third decimal place.")
        n_struct = first["n_structures"]
        if n_struct < MIN_BOOTSTRAP_STRUCTURES:
            lines.append("")
            lines.append(f"  Only {n_struct} structures: the error bar is "
                         f"itself poorly determined and")
            lines.append("  is likely an UNDERESTIMATE -- a bootstrap over "
                         "so few items cannot reach")
            lines.append(f"  the tails. Treat it as a lower bound; "
                         f"{MIN_BOOTSTRAP_STRUCTURES}+ structures make it "
                         f"meaningful.")
        lines.append("")

    skewed = [k for k, d in rows if abs(d["skew"]) > SKEW_THRESHOLD]
    if skewed:
        lines.append("")
        lines.append(f"  Asymmetric shell(s): {', '.join(skewed)} "
                     f"(|skew| > {SKEW_THRESHOLD}). A harmonic")
        lines.append("  (Gaussian) EXAFS fit of these paths biases N and R; "
                     "a published N below the")
        lines.append("  value here may be a fitting artefact rather than a "
                     "structural disagreement.")

    # Symmetric is not Gaussian: these pass the skew test above, so say so.
    flat = [(k, d["kurt"]) for k, d in rows
            if abs(d["skew"]) <= SKEW_THRESHOLD
            and abs(d.get("kurt", 0.0)) > KURT_THRESHOLD]
    if flat:
        lines.append("")
        lines.append("  Symmetric but non-Gaussian shell(s): "
                     + ", ".join(f"{k} (kurtosis {v:+.2f})" for k, v in flat)
                     + ".")
        lines.append(f"  |skew| <= {SKEW_THRESHOLD} but |excess kurtosis| > "
                     f"{KURT_THRESHOLD}: a harmonic fit can")
        lines.append("  fail outright on such a shell, not merely bias N. "
                     "Check the R-factor")
        lines.append("  from --harmonic-fit before trusting a fitted N, R "
                     "or sigma^2 for it.")

    lines.append("")
    lines.append("  sigma^2 is CONFIGURATIONAL only for a relaxed (0 K) "
                 "ensemble; an experimental")
    lines.append("  sigma^2 also carries a thermal term. Analyse a "
                 "finite-T trajectory, or compare")
    lines.append("  against a T -> 0 extrapolation, before calling a "
                 "sigma^2 mismatch a disagreement.")
    lines.append(bar)
    return "\n".join(lines)


#: Pauling electronegativity at or above which a nonmetal is acting as an
#: anion rather than as a network former. It separates the true anions
#: (O 3.44, F 3.98, Cl 3.16, N 3.04, Br 2.96) from the formers that bond
#: to them (S 2.58, C 2.55, P 2.19, B 2.04, Si 1.90), with a wide margin
#: on both sides.
ANION_ELECTRONEGATIVITY = 2.8


def is_bonded_pair(a: str, b: str) -> bool:
    """Whether an A-B pair is a first-shell bond or a second-shell contact.

    The package's usual rule — ionic, or hetero-covalent — mistakes a
    *hetero-anion* pair for a bond. ``O-O`` and ``Cl-Cl`` are excluded
    because the two symbols match, but ``O-Cl`` in an oxychloride slips
    through, and then shows up as an EXAFS path that does not exist: in
    Na-Nb-O-Cl both are anions around Nb and Na, never bonded to each
    other.

    Two nonmetals that are both electronegative enough to be anions
    (:data:`ANION_ELECTRONEGATIVITY`) are therefore treated as a contact.
    A network former stays bonded to its anions, because its own
    electronegativity is far below the threshold — phosphate P-O,
    thiophosphate P-S, LiPON P-N, borate B-O and silicate Si-O are all
    unaffected.
    """
    try:
        from ..utils.radii import NONMETALS, PAULING_EN, classify_bond
    except ImportError:                                  # pragma: no cover
        from amorphgen.utils.radii import (NONMETALS, PAULING_EN,
                                           classify_bond)

    if a in NONMETALS and b in NONMETALS:
        en_a, en_b = PAULING_EN.get(a), PAULING_EN.get(b)
        if (en_a is not None and en_b is not None
                and min(en_a, en_b) >= ANION_ELECTRONEGATIVITY):
            return False

    bond_type = classify_bond(a, b)
    return bond_type == "ionic" or (bond_type == "covalent" and a != b)


def _select_rows(result: dict, bonded_only: bool) -> list[tuple[str, dict]]:
    """Ordered (key, data) rows for the report, optionally bonded pairs only.

    Reuses the package's bond classifier so the selection matches the
    "Bonding coordination numbers" section of ``summary()``.
    """
    items = sorted(result.items())
    if not bonded_only:
        return items

    kept = [(key, d) for key, d in items
            if is_bonded_pair(d["absorber"], d["scatterer"])]
    return kept or items


def write_exafs_csv(result: dict, filepath: str) -> None:
    """Write the descriptor table to CSV, one row per ordered pair."""
    cols = ["pair", "absorber", "scatterer", "N", "N_std", "R", "R_eff",
            "sigma2", "C3", "skew", "C4", "kurt", "cutoff", "n_paths",
            "n_absorbers", "n_structures"]
    # The error bars belong in the CSV too: a user re-plotting from it
    # should not silently lose them.
    err_cols = [c for c in ("N_err", "R_err", "R_eff_err", "sigma2_err",
                            "C3_err", "skew_err", "C4_err", "kurt_err",
                            "n_boot")
                if result and c in next(iter(result.values()))]
    cols += err_cols
    with open(filepath, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for key, d in sorted(result.items()):
            fh.write(",".join(
                [key] + [f"{d[c]}" if isinstance(d[c], str) else f"{d[c]:.6g}"
                         for c in cols[1:]]) + "\n")
