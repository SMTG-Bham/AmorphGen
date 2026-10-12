"""Uncertainty of ensemble means, with structures as the sampling units.

Atoms, bonds and bins within a structure are correlated observations.  Their
spread is descriptive; it is not a standard error for the ensemble mean.
These helpers first reduce each structure to a value (or a complete curve),
give each available structure equal weight, and resample entire structures.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t

from ._serialization import finite_native as _native


def summarize_structures(values, confidence=0.95, n_bootstrap=1000,
                         seed=0) -> dict:
    """Summarize independent per-structure scalars or aligned curves.

    ``values`` is a sequence of scalars, or a structure-by-bin matrix.  None
    and non-finite values are missing, not zero.  ``per_structure`` preserves
    input order, including missing structures.  Curve intervals are pointwise,
    not simultaneous bands.  The same bootstrap row draw is used in every bin
    to preserve within-structure covariance.  Missing bins use only available
    structures, and ``n_per_point`` records their effective sample sizes.

    ``std`` is the sample spread between structures (ddof=1); ``sem`` is
    std/sqrt(n).  The t interval uses n-1 degrees of freedom.  With fewer than
    two observations the spread, SEM and interval bounds are undefined (None),
    including for the bootstrap.  Set n_bootstrap=0 to skip the bootstrap.
    Inputs should be independent structures, not correlated trajectory frames.
    """
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if (not isinstance(n_bootstrap, (int, np.integer))
            or isinstance(n_bootstrap, (bool, np.bool_)) or n_bootstrap < 0):
        raise ValueError("n_bootstrap must be a non-negative integer")
    # Permit None in place of a whole missing curve without losing its shape.
    rows = list(values)
    shape = next((np.asarray(row).shape for row in rows
                  if row is not None and np.asarray(row).ndim), None)
    if shape is not None:
        if len(shape) != 1:
            raise ValueError("values must be scalars or a structure-by-bin matrix")
        rows = [np.full(shape, np.nan) if row is None else row for row in rows]
    array = np.asarray(rows, dtype=float)
    if not rows and isinstance(values, np.ndarray) and values.ndim == 2:
        array = np.empty(values.shape, dtype=float)
    if array.ndim not in (1, 2):
        raise ValueError("values must be scalars or a structure-by-bin matrix")
    scalar = array.ndim == 1
    matrix = array[:, None] if scalar else array
    finite = np.isfinite(matrix)
    matrix = np.where(finite, matrix, np.nan)
    counts = finite.sum(axis=0)
    total = np.where(finite, matrix, 0.0).sum(axis=0)
    mean = np.divide(total, counts, out=np.full(total.shape, np.nan),
                     where=counts > 0)
    residual = np.where(finite, matrix - mean, 0.0)
    variance = np.divide(np.sum(residual ** 2, axis=0), counts - 1,
                         out=np.full(total.shape, np.nan), where=counts > 1)
    std = np.sqrt(variance)
    sem = np.divide(std, np.sqrt(counts),
                    out=np.full(total.shape, np.nan), where=counts > 1)
    quantile = np.full(total.shape, np.nan)
    quantile[counts > 1] = t.ppf((1 + confidence) / 2, counts[counts > 1] - 1)
    ci_low, ci_high = mean - quantile * sem, mean + quantile * sem
    boot_low = np.full(total.shape, np.nan)
    boot_high = np.full(total.shape, np.nan)
    usable = finite.any(axis=1)
    n_valid = int(usable.sum())
    if n_bootstrap and n_valid > 1 and np.any(counts > 1):
        # Chunk the draws: avoid allocating bootstrap x structures x bins.
        available = np.where(finite[usable], matrix[usable], 0.0)
        present = finite[usable].astype(float)
        bootstrap = np.full((n_bootstrap, matrix.shape[1]), np.nan)
        rng = np.random.default_rng(seed)
        for start in range(0, n_bootstrap, 128):
            stop = min(start + 128, n_bootstrap)
            indices = rng.integers(0, n_valid, size=(stop - start, n_valid))
            weights = np.zeros((stop - start, n_valid), dtype=float)
            np.add.at(weights, (np.arange(stop - start)[:, None], indices), 1)
            denominator = weights @ present
            np.divide(weights @ available, denominator,
                      out=bootstrap[start:stop], where=denominator > 0)
        tail = (1 - confidence) / 2
        for point in np.flatnonzero(counts > 1):
            samples = bootstrap[:, point]
            samples = samples[np.isfinite(samples)]
            if samples.size:
                boot_low[point], boot_high[point] = np.quantile(
                    samples, [tail, 1 - tail])

    def native(value):
        return _native(value[0] if scalar else value)

    return {
        "per_structure": _native(matrix[:, 0] if scalar else matrix),
        "n_structures": n_valid,
        "n_total_structures": len(matrix),
        "n_per_point": int(counts[0]) if scalar else counts.tolist(),
        "mean": native(mean), "std": native(std), "sem": native(sem),
        "ci_low": native(ci_low), "ci_high": native(ci_high),
        "bootstrap_low": native(boot_low), "bootstrap_high": native(boot_high),
        "confidence": float(confidence), "sampling_unit": "structure",
        "estimator": "equal-weight mean of available structures",
        "interval_method": "Student t", "bootstrap_method": "percentile",
        "band_type": "pointwise", "n_bootstrap": int(n_bootstrap),
        "seed": int(seed) if seed is not None else None,
    }


def summarize_observations(values, per_structure) -> dict:
    """Describe pooled continuous observations and their structure means.

    ``values`` contains the nonempty pooled bond lengths or angles;
    ``per_structure`` contains one mean per structure, with absent observations
    represented by None. Keep the pooled population spread distinct from the
    uncertainty of the equal-weight structure mean.
    """
    values = np.asarray(values)
    mean, std = float(np.mean(values)), float(np.std(values))
    return {
        "mean": mean, "std": std,
        "pooled_mean": mean, "pooled_std": std,
        "per_structure": per_structure,
        "uncertainty": summarize_structures(per_structure),
        "min": float(np.min(values)), "max": float(np.max(values)),
        "count": len(values),
    }


def summarize_site_groups(groups) -> dict:
    """Separate pooled site statistics from uncertainty of structure means.

    ``groups`` holds the site values from each structure, using an empty
    sequence for structures without the descriptor's sites.  Integer values
    (coordination numbers) also receive explicitly named site fractions and
    structure prevalence, both on the 0..1 scale.  Structure prevalence counts
    *all* input structures, including those without the central species.
    """
    groups = [list(group) if group is not None else [] for group in groups]
    values = [value for group in groups for value in group]
    if not values:
        return {}
    data = np.asarray(values)
    per_structure = [float(np.mean(group)) if group else None for group in groups]
    uncertainty = summarize_structures(per_structure)
    result = {
        "mean": float(np.mean(data)), "std": float(np.std(data)),
        "pooled_mean": float(np.mean(data)), "pooled_std": float(np.std(data)),
        "per_structure": per_structure, "uncertainty": uncertainty,
        "min": int(np.min(data)), "max": int(np.max(data)),
        "total_atoms": len(values),
    }
    unique, counts = np.unique(data, return_counts=True)
    result["distribution"] = {int(cn): round(100.0 * count / len(values), 1)
                              for cn, count in zip(unique, counts)}
    result["fraction_of_sites"] = {int(cn): float(count / len(values))
                                   for cn, count in zip(unique, counts)}
    result["fraction_of_structures"] = {
        int(cn): float(sum(cn in group for group in groups) / len(groups))
        for cn in unique}
    result["fraction_of_structures_definition"] = (
        "fraction of all input structures with at least one site of this coordination")
    result["site_fraction_uncertainty"] = {
        int(cn): summarize_structures([
            group.count(cn) / len(group) if group else None for group in groups])
        for cn in unique}
    result["structure_fraction_uncertainty"] = {
        int(cn): summarize_structures([float(cn in group) for group in groups])
        for cn in unique}
    return result
