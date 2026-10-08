"""Plotting functions for structure analysis.

Visual style matches :mod:`amorphgen.analysis.comparison_plots`: Okabe-Ito
palette, inward ticks, no top/right spines, mirrored-bar coordination
layout for reciprocal A-B / B-A pairs, and a per-structure density
violin panel.

Each descriptor is saved as a standalone figure (PNG/PDF) plus a CSV
of the raw numbers — so each file is meant to be read on its own and
does not carry (a)-(d) sub-panel labels.
"""

from __future__ import annotations

import os
import numpy as np

from .rdf import DEFAULT_SMEARING

# Okabe-Ito colour-blind-safe palette (RGB hex)
_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
            "#F0E442", "#56B4E9", "#E69F00", "#000000"]

_EXP_COLOR = "#222222"


def _draw_uncertainty(ax, x, uncertainty, color):
    """Draw pointwise whole-structure bootstrap bounds, if estimable."""
    if not uncertainty:
        return False
    lo = np.asarray(uncertainty["bootstrap_low"], dtype=float)
    hi = np.asarray(uncertainty["bootstrap_high"], dtype=float)
    valid = np.isfinite(lo) & np.isfinite(hi)
    if not valid.any():
        return False
    label = f"{100 * uncertainty['confidence']:g}% pointwise bootstrap CI"
    if label in ax.get_legend_handles_labels()[1]:
        label = None
    ax.fill_between(x, lo, hi, where=valid, color=color, alpha=0.18,
                    linewidth=0, label=label)
    return True


def _save_curve_uncertainty(base, x, summaries):
    """Save intervals, contributing structures and reproducibility metadata."""
    import csv
    import json
    summaries = {k: v for k, v in summaries.items() if v}
    if not summaries:
        return
    with open(f"{base}_uncertainty.json", "w") as handle:
        json.dump({"grid": list(x), "descriptors": summaries}, handle,
                  indent=2, allow_nan=False)
    fields = ["mean", "std", "sem", "ci_low", "ci_high", "bootstrap_low", "bootstrap_high"]
    with open(f"{base}_uncertainty.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["descriptor", "coordinate", "n_structures", *fields])
        for name, u in summaries.items():
            for i, coordinate in enumerate(x):
                writer.writerow([name, coordinate, u["n_per_point"][i],
                                 *[u[field][i] for field in fields]])
    with open(f"{base}_per_structure.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["descriptor", "structure_index", "coordinate", "value"])
        for name, u in summaries.items():
            for index, curve in enumerate(u["per_structure"]):
                writer.writerows((name, index, coordinate, value)
                                 for coordinate, value in zip(x, curve))


def _apply_pub_style(ax, label_fs=11, tick_fs=10):
    """Apply publication-style cosmetics: hide top/right spines, inward ticks,
    minor ticks, consistent font sizing on tick labels."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_linewidth(1.0)
    ax.tick_params(axis='both', which='major', labelsize=tick_fs,
                   direction='in', length=4, width=0.9, top=False, right=False)
    ax.tick_params(axis='both', which='minor', direction='in', length=2.5,
                   width=0.7, top=False, right=False)
    ax.minorticks_on()
    ax.xaxis.label.set_size(label_fs)
    ax.yaxis.label.set_size(label_fs)


def _panel_letter(ax, letter):
    """Top-left bold panel letter, matching the comparison_plots style."""
    ax.text(0.03, 0.97, f"({letter})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")


def _find_reciprocal_pair(cn_data):
    """Return (top_key, bot_key) if exactly one A-B / B-A reciprocal pair
    exists in cn_data; else (None, None).

    Used to switch the coordination panel into the mirrored-bar layout
    (e.g. Si-O on top, O-Si mirrored below) — matches comparison_plots."""
    keys = list(cn_data.keys())
    pair_map = {k: tuple(k.split("-")) for k in keys}
    for a_key, (a, b) in pair_map.items():
        for b_key, (c, d) in pair_map.items():
            if a == d and b == c and a != b and a_key != b_key:
                # Prefer the canonical order: bonding (CN > 0.5) on top
                if cn_data[a_key]["mean"] >= cn_data[b_key]["mean"]:
                    return a_key, b_key
                return b_key, a_key
    return None, None


def _figure(nrows=1, ncols=1, *, figsize=None, dpi=None, **kwargs):
    """``plt.subplots`` without pyplot.

    These figures are only saved, never shown, so they stay out of pyplot:
    the caller's backend and open figures are left alone (a notebook keeps
    showing its own figures inline), no display is needed, and there is
    nothing to close.
    """
    from matplotlib.figure import Figure
    fig = Figure(figsize=figsize, dpi=dpi)
    return fig, fig.subplots(nrows, ncols, **kwargs)


def _save_fig(fig, base_path, dpi=300, save_pdf=False):
    """Save PNG (and optionally PDF) at the given base path (no extension)."""
    fig.savefig(f"{base_path}.png", dpi=dpi, bbox_inches='tight')
    print(f"  Saved: {base_path}.png")
    if save_pdf:
        fig.savefig(f"{base_path}.pdf", bbox_inches='tight')
        print(f"  Saved: {base_path}.pdf")


def plot_pair_panels(x, curves, xlabel, ylabel, base_path, dpi=300,
                     save_pdf=False, xlim=None, hline=1.0, title=None,
                     uncertainties=None):
    """One small panel per element pair (shared axes), for g(r) or S_ab(q).

    ``curves`` maps a pair label to its y-array on the common grid ``x``.
    Up to three panels per row; NaNs are skipped.
    """
    import math

    labels = list(curves)
    n = len(labels)
    ncols = 1 if n == 1 else (2 if n <= 4 else 3)
    nrows = math.ceil(n / ncols)
    fig, axes = _figure(nrows, ncols, figsize=(3.3 * ncols, 2.5 * nrows),
                        sharex=True, sharey=True, squeeze=False)
    x = np.asarray(x, dtype=float)
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.axis("off"); continue
        y = np.asarray(curves[labels[k]], dtype=float)
        m = ~np.isnan(y)
        if hline is not None:
            ax.axhline(hline, ls=":", color="grey", alpha=0.6, lw=0.8)
        ax.plot(x[m], y[m], lw=1.3, color=_PALETTE[k % len(_PALETTE)])
        if uncertainties and _draw_uncertainty(
                ax, x, uncertainties.get(labels[k]), _PALETTE[k % len(_PALETTE)]):
            ax.legend(fontsize=6, frameon=False, loc="upper left")
        ax.text(0.97, 0.92, labels[k], transform=ax.transAxes, ha="right",
                va="top", fontsize=9)
        _apply_pub_style(ax, label_fs=10, tick_fs=8)
        if xlim is not None:
            ax.set_xlim(*xlim)
    # the lowest occupied panel of each column carries the x labels (a
    # column may end above the last row when the grid is not full)
    for c in range(ncols):
        rows = [r for r in range(nrows) if r * ncols + c < n]
        if rows:
            ax = axes[rows[-1], c]
            ax.set_xlabel(xlabel)
            ax.tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    _save_fig(fig, base_path, dpi, save_pdf)


def plot_analysis(analyser, output_dir=".", prefix="analysis",
                  rdf_pairs=None, angle_triplets=None,
                  rmax=None, normalise=True, angle_style="line",
                  save_csv=True, show_total_rdf=False,
                  smearing=DEFAULT_SMEARING,
                  dpi=300, save_pdf=False, show_title=False,
                  pair_panels=False, total_cn=None, cutoff_window=0.1):
    """
    Generate and save analysis plots and raw data.

    Produces: RDF plot, CN distribution, bond angle distribution, CSV files.

    Publication-quality defaults: 300 DPI, no top/right spines, inward ticks,
    Okabe-Ito colour-blind palette, no titles (use figure captions instead).

    Parameters
    ----------
    dpi : int
        Image DPI for PNG output (default 300).
    save_pdf : bool
        Also save vector PDF copies of every plot (default False).
    show_title : bool
        Show title above each panel (default False — publications prefer
        figure captions).
    pair_panels : bool
        Also write ``{prefix}_rdf_panels.png``, one small panel per pair
        (default False).
    total_cn : list of str, optional
        Total-coordination requests (``"O"``, ``"O:In+Ga"``) plotted as
        ``{prefix}_cn_total.png`` with a CSV.
    cutoff_window : float
        Half-window in Angstrom for the cutoff-robustness JSON/CSV exports
        written with ``save_csv`` (default 0.10).
    """
    os.makedirs(output_dir, exist_ok=True)
    formula = analyser.atoms_list[0].get_chemical_formula(mode="hill")

    # Auto rmax
    if rmax is None:
        half_cells = [min(atoms.cell.lengths()) / 2
                      for atoms in analyser.atoms_list]
        rmax = float(np.floor(min(half_cells) * 10) / 10)
        print(f"  Auto rmax = {rmax:.1f} A (half cell)")

    # ── 1. RDF plot ──────────────────────────────────────────────────
    unique = sorted(set(analyser.atoms_list[0].get_chemical_symbols()))
    pairs = [f"{s1}-{s2}" for i, s1 in enumerate(unique)
             for s2 in unique[i:]]
    if rdf_pairs is not None:
        pairs = [p for p in pairs if p in rdf_pairs]

    fig, ax = _figure(figsize=(7, 4.5))
    rdf_csv_data = {}
    rdf_uncertainties = {}

    if normalise:
        ax.axhline(y=1, color='0.5', linestyle='--', linewidth=0.8, alpha=0.6,
                   zorder=0)

    # The total g(r) always goes into the CSV; it is drawn for single-element
    # systems or when requested (--total-rdf), so a multi-pair plot stays legible.
    rdf_total = analyser.rdf(pair=None, rmax=rmax, sigma=smearing)
    r = np.array(rdf_total["r"])
    g_r_total = np.array(rdf_total["g_r"])
    rdf_csv_data["Total"] = (r, g_r_total)
    rdf_uncertainties["Total"] = rdf_total.get("uncertainty")
    if len(unique) == 1 or show_total_rdf:
        ax.plot(r, g_r_total, label="Total", linewidth=2.0,
                color='black', linestyle='--' if len(unique) > 1 else '-')
        _draw_uncertainty(ax, r, rdf_total.get("uncertainty"), "black")

    for i, pair in enumerate(pairs):
        rdf_data = analyser.rdf(pair=pair, rmax=rmax, sigma=smearing)
        r = np.array(rdf_data["r"])
        g_r = np.array(rdf_data["g_r"])
        ax.plot(r, g_r, label=pair, linewidth=1.8,
                color=_PALETTE[i % len(_PALETTE)])
        rdf_csv_data[pair] = (r, g_r)
        rdf_uncertainties[pair] = rdf_data.get("uncertainty")
        _draw_uncertainty(ax, r, rdf_data.get("uncertainty"), _PALETTE[i % len(_PALETTE)])

    ax.set_xlabel(r"r (Å)")
    ax.set_ylabel("g(r)" if normalise else "Count")
    if show_title:
        ax.set_title(f"RDF — {formula}", fontsize=12)
    ax.legend(fontsize=9, frameon=False, loc='best')
    ax.set_xlim(0, rmax)
    _apply_pub_style(ax)
    fig.tight_layout()
    _save_fig(fig, os.path.join(output_dir, f"{prefix}_rdf"), dpi, save_pdf)

    if pair_panels and pairs:
        plot_pair_panels(rdf_csv_data[pairs[0]][0],
                         {p: rdf_csv_data[p][1] for p in pairs},
                         r"r (Å)", "g(r)" if normalise else "Count",
                         os.path.join(output_dir, f"{prefix}_rdf_panels"),
                         dpi, save_pdf, xlim=(0, rmax),
                         hline=1.0 if normalise else None,
                         title=f"Partial RDFs — {formula}" if show_title else None,
                         uncertainties=rdf_uncertainties)

    if save_csv:
        rdf_csv_path = os.path.join(output_dir, f"{prefix}_rdf.csv")
        with open(rdf_csv_path, "w") as f:
            headers = ["r(A)"] + [f"g(r)_{p}" for p in rdf_csv_data]
            f.write(",".join(headers) + "\n")
            r_vals = list(rdf_csv_data.values())[0][0]
            for i in range(len(r_vals)):
                row = [f"{r_vals[i]:.4f}"]
                for pair in rdf_csv_data:
                    row.append(f"{rdf_csv_data[pair][1][i]:.6f}")
                f.write(",".join(row) + "\n")
        print(f"  Saved: {rdf_csv_path}")
        _save_curve_uncertainty(os.path.join(output_dir, f"{prefix}_rdf"), r,
                                rdf_uncertainties)

    # ── 2. CN distribution ───────────────────────────────────────────
    # Only BONDED pairs are plotted (cation-anion, hetero covalent), the same
    # rule as the report's "Bonding coordination numbers". A binary oxide
    # (Si-O / O-Si) gets the mirrored-bars layout; a multi-cation compound
    # gets one panel per cation-centred pair plus the anion's total over all
    # its cations (O-(Ga+In+Zn)). Single-element and alloy systems, which
    # have no cation-anion pair, fall back to every pair with CN > 0.5.
    from collections import Counter
    from .structure import is_bonding_pair
    _elements = Counter(analyser.atoms_list[0].get_chemical_symbols())

    def _is_bond(pair):
        a, b = pair.split("-")
        return is_bonding_pair(a, b, _elements)

    cn_data = analyser.coordination()
    bonded = {k: v for k, v in cn_data.items() if _is_bond(k)}
    if bonded:
        cn_pairs = dict(bonded)          # a bonded pair is shown whatever its CN
    else:
        cn_pairs = {k: v for k, v in cn_data.items() if v["mean"] > 0.5}

    # anion totals: centre elements bonded to more than one partner type
    partners = {}
    for k in bonded:
        a, b = k.split("-")
        partners.setdefault(a, []).append(b)
    totals = {}
    for centre, ps in partners.items():
        if len(ps) > 1:
            t = analyser.total_coordination(centre=centre)
            if centre in t:
                totals[f"{centre}-({'+'.join(sorted(ps))})"] = t[centre]

    def _bar_panel(ax, label, data, color):
        cn_vals = sorted(data["distribution"].keys())
        pcts = [data["distribution"][cn] for cn in cn_vals]
        bars = ax.bar(cn_vals, pcts, color=color, edgecolor='black', linewidth=0.4)
        ax.set_xlabel(f"{label} CN")
        ax.set_ylabel("Fraction of sites (%)")
        ax.set_xticks(cn_vals)
        ax.set_ylim(0, max(pcts, default=1) * 1.22)     # room for the mean box
        ax.text(0.97, 0.95, f"mean = {data['mean']:.1f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7', alpha=0.85))
        for bar, pct in zip(bars, pcts):
            if pct > 2:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{pct:.0f}%", ha='center', va='bottom', fontsize=8)
        _apply_pub_style(ax)

    def _panels_figure(items, base):
        fig, axes = _figure(1, len(items), figsize=(3.8 * len(items), 3.8),
                            squeeze=False)
        for idx, (label, data) in enumerate(items.items()):
            _bar_panel(axes[0, idx], label, data, _PALETTE[idx % len(_PALETTE)])
        if show_title:
            fig.suptitle(f"CN Distribution — {formula}", fontsize=12, y=1.02)
        fig.tight_layout()
        _save_fig(fig, base, dpi, save_pdf)

    if cn_pairs:
        from matplotlib.ticker import FuncFormatter
        # cation-centred bonded pairs (Ga-O, In-O, Zn-O): the reciprocal
        # anion-centred ones (O-Ga, ...) are the mirrored half or the total
        centred = [k for k in cn_pairs if k.split("-")[0] in partners
                   and len(partners[k.split("-")[0]]) == 1] if bonded else []
        top_key, bot_key = (None, None)
        if bonded and not totals:
            top_key, bot_key = _find_reciprocal_pair(cn_pairs)

        if top_key and bot_key:
            # ── Mirrored layout (binary compound) ──
            top = cn_pairs[top_key]["distribution"]
            bot = cn_pairs[bot_key]["distribution"]
            all_cn = sorted(set(top) | set(bot))
            x = np.array(all_cn, dtype=float)

            fig, ax = _figure(figsize=(5.0, 4.0))
            ax.bar(x, [top.get(c, 0) for c in all_cn], 0.6,
                   color=_PALETTE[0], edgecolor="black", lw=0.4,
                   label=top_key)
            ax.bar(x, [-bot.get(c, 0) for c in all_cn], 0.6,
                   color=_PALETTE[1], edgecolor="black", lw=0.4,
                   alpha=0.85, label=bot_key)
            ax.axhline(0, color="black", lw=0.9, zorder=4)
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, p: f"{abs(v):.0f}"))
            ymax = max(max(top.values(), default=0),
                       max(bot.values(), default=0))
            ax.set_ylim(-1.20 * ymax, 1.20 * ymax)
            ax.text(0.97, 0.93, top_key, transform=ax.transAxes,
                    ha="right", va="top", fontsize=10,
                    fontweight="bold", color="0.25")
            ax.text(0.97, 0.07, bot_key, transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=10,
                    fontweight="bold", color="0.25")
            ax.set_xlabel("Coordination number")
            ax.set_ylabel("Fraction of sites (%)")
            ax.set_xticks(all_cn)
            ax.legend(frameon=False, fontsize=9, loc="upper left",
                      labelspacing=0.25)
            _apply_pub_style(ax)
            fig.tight_layout()
            _save_fig(fig, os.path.join(output_dir, f"{prefix}_cn"),
                      dpi, save_pdf)
        else:
            # ── Side-by-side panels: multi-cation (Ga-O, In-O, Zn-O, then
            #    O-(Ga+In+Zn)), or mono-element / alloy (every pair) ──
            items = {k: cn_pairs[k] for k in centred} if totals else dict(cn_pairs)
            if totals:
                items.update(totals)
            _panels_figure(items, os.path.join(output_dir, f"{prefix}_cn"))

        if save_csv:
            cn_csv_path = os.path.join(output_dir, f"{prefix}_cn.csv")
            with open(cn_csv_path, "w") as f:
                f.write("pair,CN,fraction_of_sites(%),fraction_of_structures(%),count\n")
                rows = dict(cn_pairs); rows.update(totals)
                for pair, data in rows.items():
                    total_atoms = data.get("total_atoms", 0)
                    for cn_val, pct in sorted(data["distribution"].items()):
                        count = int(round(data["fraction_of_sites"][cn_val] * total_atoms)) if total_atoms else ""
                        f.write(f"{pair},{cn_val},{pct:.1f},{100 * data['fraction_of_structures'][cn_val]:.3f},{count}\n")
            print(f"  Saved: {cn_csv_path}")

    # ── 2b. Requested totals (--total-cn / total_cn:) ────────────────
    if total_cn:
        from .analyser import parse_total_cn_spec
        items = {}
        for spec in total_cn:
            centre, ps = parse_total_cn_spec(spec)
            t = analyser.total_coordination(centre=centre, partners=ps)
            if centre in t:
                items[f"{centre}-({'+'.join(ps)})" if ps else f"{centre}-(all bonded)"] = t[centre]
        if items:
            _panels_figure(items, os.path.join(output_dir, f"{prefix}_cn_total"))
            if save_csv:
                path = os.path.join(output_dir, f"{prefix}_cn_total.csv")
                with open(path, "w") as f:
                    f.write("centre,CN,fraction_of_sites(%),fraction_of_structures(%)\n")
                    for label, data in items.items():
                        for cn_val, pct in sorted(data["distribution"].items()):
                            f.write(f"{label},{cn_val},{pct:.1f},{100 * data['fraction_of_structures'][cn_val]:.3f}\n")
                print(f"  Saved: {path}")

    # ── 3. Bond angle distribution ───────────────────────────────────
    all_angle_data = analyser._compute_all_angles()
    distributions = analyser.angle_distribution(normalise=normalise)
    selected = [k for k in distributions
                if angle_triplets is None or k in angle_triplets]

    if selected:
        fig, ax = _figure(figsize=(7, 4.5))
        for i, triplet in enumerate(selected):
            data = distributions[triplet]
            colour = _PALETTE[i % len(_PALETTE)]
            bin_centres = np.asarray(data["angle"])
            hist = np.asarray(data["distribution"], dtype=float)
            if angle_style in ("histogram", "both"):
                ax.bar(bin_centres, hist, width=np.diff(data["bin_edges"]), alpha=0.3,
                       label=triplet if angle_style == "histogram" else None,
                       color=colour, edgecolor="black", linewidth=0.3)
            if angle_style in ("line", "both"):
                ax.plot(bin_centres, hist, label=triplet, linewidth=2.0, color=colour)
            _draw_uncertainty(ax, bin_centres, data["uncertainty"], colour)

        ax.set_xlabel("Angle (°)")
        ax.set_ylabel("Probability density" if normalise else "Count")
        if show_title:
            ax.set_title(f"Bond Angle Distribution — {formula}", fontsize=12)
        ax.legend(fontsize=9, frameon=False, loc='best')
        ax.set_xlim(0, 180)
        _apply_pub_style(ax)
        fig.tight_layout()
        _save_fig(fig, os.path.join(output_dir, f"{prefix}_angles"),
                  dpi, save_pdf)

        if save_csv:
            angle_csv_path = os.path.join(output_dir, f"{prefix}_angles.csv")
            with open(angle_csv_path, "w") as f:
                f.write("triplet,angle(deg),structure_index\n")
                for index, frame in enumerate(all_angle_data.per_structure):
                    for triplet in selected:
                        for a in frame.get(triplet, []):
                            f.write(f"{triplet},{a:.2f},{index}\n")
            print(f"  Saved: {angle_csv_path}")
            _save_curve_uncertainty(os.path.join(output_dir, f"{prefix}_angles"),
                                    bin_centres,
                                    {k: distributions[k]["uncertainty"] for k in selected})

    # ── 4. Per-structure density violin ──────────────────────────────
    # New in v1.0.0: matches the density panel in
    # amorphgen.analysis.comparison_plots.plot_density.
    density_dict = analyser.density()
    rho_values = np.array(density_dict.get("values", []))
    if len(rho_values) >= 2:
        fig, ax = _figure(figsize=(5.0, 4.0))
        x_pos = 1
        vp = ax.violinplot([rho_values], positions=[x_pos], widths=0.65,
                           showmeans=False, showmedians=False,
                           showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(_PALETTE[0])
            body.set_alpha(0.32)
            body.set_edgecolor("black")
            body.set_linewidth(0.9)
        rng = np.random.default_rng(0)
        jx = x_pos + 0.06 * rng.standard_normal(len(rho_values))
        ax.scatter(jx, rho_values, color=_PALETTE[0], s=22, alpha=0.9,
                   edgecolor="black", lw=0.4, zorder=3)
        m = rho_values.mean()
        u = density_dict["uncertainty"]
        s = u["ci_high"] - m
        ax.hlines(m, x_pos - 0.22, x_pos + 0.22, color="black", lw=1.6,
                  zorder=4)
        ax.errorbar(x_pos, m, yerr=s, color="black", lw=1.0, capsize=4,
                    fmt="none", zorder=4)
        ax.text(x_pos, rho_values.max()
                + 0.03 * (rho_values.max() - rho_values.min() + 0.1) + 0.02,
                f"{m:.2f} ± {s:.2f} (95% t CI)",
                ha="center", va="bottom", fontsize=9,
                color=_PALETTE[0], fontweight="bold")
        ax.set_xticks([x_pos])
        # Label the category by what it is (composition, ensemble size),
        # not by the tool.
        ax.set_xticklabels([f"{formula} (n = {len(rho_values)})"])
        ax.set_xlim(0.4, 1.6)
        ymin = min(rho_values.min(), u["ci_low"]) - 0.10 * max(0.05,
                                              rho_values.max() - rho_values.min())
        ymax = max(rho_values.max(), u["ci_high"]) + 0.20 * max(0.05,
                                              rho_values.max() - rho_values.min())
        span = max(0.1, ymax - ymin)
        ax.set_ylim(ymin - 0.05 * span, ymax + 0.10 * span)
        ax.set_ylabel(r"Density (g cm$^{-3}$)")
        if show_title:
            ax.set_title(f"Density — {formula}", fontsize=12)
        _apply_pub_style(ax)
        fig.tight_layout()
        _save_fig(fig, os.path.join(output_dir, f"{prefix}_density"),
                  dpi, save_pdf)

        if save_csv:
            rho_csv_path = os.path.join(output_dir, f"{prefix}_density.csv")
            with open(rho_csv_path, "w") as f:
                f.write("structure_index,density_g_per_cm3\n")
                for i, rho in enumerate(rho_values):
                    f.write(f"{i},{rho:.4f}\n")
            print(f"  Saved: {rho_csv_path}")

    if save_csv:
        import csv
        import json
        from .robustness import save_cutoff_robustness
        from .structure import compute_bond_angle_stats
        paths = save_cutoff_robustness(
            analyser.cutoff_robustness(window=cutoff_window), output_dir, prefix)
        for path in paths.values():
            print(f"  Saved: {path}")
        # Scalar estimates retain raw structure identity as well as pooled
        # descriptors, so reports can be audited without rerunning geometry.
        statistics = {"density": density_dict, "coordination": cn_data,
                      "total_coordination": analyser.total_coordination(),
                      "bond_distances": analyser.bond_distances(),
                      "bond_angles": compute_bond_angle_stats(all_angle_data)}
        base = os.path.join(output_dir, f"{prefix}_statistics")
        with open(f"{base}.json", "w") as handle:
            json.dump(statistics, handle, indent=2, allow_nan=False)
        with open(f"{base}.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            fields = ["mean", "std", "sem", "ci_low", "ci_high", "n_structures"]
            writer.writerow(["descriptor", "pooled_mean", "pooled_std", *fields])
            entries = [("density", density_dict)]
            entries.extend((f"{section}.{key}", data)
                           for section in ("coordination", "total_coordination", "bond_distances", "bond_angles")
                           for key, data in statistics[section].items())
            for name, data in entries:
                writer.writerow([name, data.get("pooled_mean"), data.get("pooled_std"),
                                 *[data["uncertainty"][key] for key in fields]])


def plot_sq(sq_result, output_dir=".", prefix="analysis", dpi=300,
            save_pdf=False, weighting="xray", show_title=False,
            method="direct", pair_panels=False):
    """Plot the direct-method total structure factor S(q) + write a CSV.

    Direct (Debye) S(q) with Faber-Ziman normalisation (S(q→∞)=1). The FSDP
    region is resolvable down to q_min ≈ 2π/L, so small boxes leave the
    low-q part noisy — the CSV includes ``n_per_bin`` so shells built from
    only 1-3 reciprocal vectors can be identified.

    When ``sq_result`` carries ``"partials"`` (``partials=True`` on the
    direct method), the Faber-Ziman partials S_ab(q) are added to the CSV
    as ``s_<A-B>`` columns and drawn in ``{prefix}_sq_partials.png``; with
    ``pair_panels`` also one panel per pair in ``{prefix}_sq_partials_panels.png``.
    """
    import csv

    os.makedirs(output_dir, exist_ok=True)
    q = np.array(sq_result["q"], dtype=float)
    s = np.array(sq_result["s_q"], dtype=float)
    # n_per_bin exists only for the direct method (q-vectors per shell);
    # the FT method has no such count.
    n = (np.array(sq_result["n_per_bin"], dtype=int)
         if "n_per_bin" in sq_result else None)
    m = ~np.isnan(s) & ((n > 0) if n is not None else True)

    fig, ax = _figure(figsize=(5.4, 4.0))
    ax.plot(q[m], s[m], lw=1.4, color=_PALETTE[0])
    if _draw_uncertainty(ax, q, sq_result.get("uncertainty"), _PALETTE[0]):
        ax.legend(frameon=False, fontsize=8)
    ax.axhline(1.0, ls=":", color="grey", alpha=0.6)
    # Conventional S(q) presentation starts at 0. Note the Faber-Ziman total
    # can be negative at low q for multi-component x-ray weighting (down to
    # -(<f^2>-<f>^2)/<f>^2); that part is clipped from the plot only -- the
    # CSV keeps every value.
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel(r"$S(q)$")
    _apply_pub_style(ax)
    if show_title:
        ax.set_title(f"Total S(q) — {method} method, {weighting} weighting")
    base = os.path.join(output_dir, f"{prefix}_sq")
    _save_fig(fig, base, dpi, save_pdf)

    partials = sq_result.get("partials") or {}
    if partials:
        fig2, ax2 = _figure(figsize=(5.4, 4.0))
        for i, (pair, s_ab) in enumerate(partials.items()):
            s_ab = np.array(s_ab, dtype=float)
            mm = ~np.isnan(s_ab) & ((n > 0) if n is not None else True)
            ax2.plot(q[mm], s_ab[mm], lw=1.3, color=_PALETTE[i % len(_PALETTE)],
                     label=rf"$S_{{\mathrm{{{pair.replace('-', '')}}}}}(q)$")
            _draw_uncertainty(ax2, q, sq_result.get("partials_uncertainty", {}).get(pair),
                              _PALETTE[i % len(_PALETTE)])
        ax2.axhline(1.0, ls=":", color="grey", alpha=0.6)
        ax2.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
        ax2.set_ylabel(r"$S_{ab}(q)$")
        ax2.legend(frameon=False, fontsize=8)
        _apply_pub_style(ax2)
        if show_title:
            ax2.set_title("Faber-Ziman partial structure factors")
        _save_fig(fig2, f"{base}_partials", dpi, save_pdf)
        if pair_panels:
            good = (n > 0) if n is not None else np.ones(len(q), bool)
            plot_pair_panels(q, {p: np.where(good, np.array(v, dtype=float), np.nan)
                                 for p, v in partials.items()},
                             r"$q$ ($\mathrm{\AA}^{-1}$)", r"$S_{ab}(q)$",
                             f"{base}_partials_panels", dpi, save_pdf,
                             title="Faber-Ziman partials" if show_title else None,
                             uncertainties=sq_result.get("partials_uncertainty"))

    with open(f"{base}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        raw = (np.array(sq_result["s_q_raw"], dtype=float)
               if "s_q_raw" in sq_result else None)
        pcols = [(f"s_{p}", np.array(v, dtype=float)) for p, v in partials.items()]
        if n is not None:
            hdr = (["q_invA", "s_q", "n_per_bin"] + (["s_q_raw"] if raw is not None else [])
                   + [c for c, _ in pcols])
            w.writerow(hdr)
            for k, (qi, si, ni) in enumerate(zip(q, s, n)):
                row = [f"{qi:.5f}", "" if np.isnan(si) else f"{si:.6f}", ni]
                if raw is not None:
                    row.append("" if np.isnan(raw[k]) else f"{raw[k]:.6f}")
                for _, v in pcols:
                    row.append("" if np.isnan(v[k]) else f"{v[k]:.6f}")
                w.writerow(row)
        else:
            w.writerow(["q_invA", "s_q"])
            for qi, si in zip(q, s):
                w.writerow([f"{qi:.5f}", "" if np.isnan(si) else f"{si:.6f}"])
    print(f"  Saved: {base}.csv")
    _save_curve_uncertainty(base, q, {"s_q": sq_result.get("uncertainty"),
                                     **sq_result.get("partials_uncertainty", {})})


def plot_tr(tr_result, output_dir=".", prefix="analysis", dpi=300,
            save_pdf=False, show_title=False):
    """Plot T(r) and write r, the weighted g(r), T(r) and G(r) as CSV.

    The CSV is what you overlay on a digitised figure from a diffraction paper.
    """
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    r = np.asarray(tr_result["r"], dtype=float)
    T = np.asarray(tr_result["T_r"], dtype=float)
    g = np.asarray(tr_result["g_r"], dtype=float)
    G = np.asarray(tr_result["G_r"], dtype=float)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(r, T, lw=1.5, color=_PALETTE[0])
    if _draw_uncertainty(ax, r, tr_result.get("uncertainty"), _PALETTE[0]):
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel(r"$r$ ($\mathrm{\AA}$)")
    ax.set_ylabel(r"$T(r)$ ($\mathrm{\AA}^{-2}$)")
    ax.set_xlim(r.min(), r.max())
    _apply_pub_style(ax)
    if show_title:
        ax.set_title(f"T(r) — {tr_result['weighting']} weighting, "
                     f"Q = {tr_result['qmin']:.1f}-{tr_result['qmax']:.1f} " r"$\mathrm{\AA}^{-1}$")
    fig.tight_layout()
    base = os.path.join(output_dir, f"{prefix}_tr")
    _save_fig(fig, base, dpi, save_pdf)
    plt.close(fig)

    with open(f"{base}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"# weighting={tr_result['weighting']} qmin={tr_result['qmin']} "
                    f"qmax={tr_result['qmax']} window={tr_result['window']} "
                    f"rho={tr_result['rho']:.6f} at/A^3"])
        w.writerow(["r_A", "g_r_weighted", "T_r_invA2", "G_r_invA2"])
        for row in zip(r, g, T, G):
            w.writerow([f"{v:.6f}" for v in row])
    print(f"  Saved: {base}.csv")
    curves = tr_result.get("curve_uncertainty", {})
    _save_curve_uncertainty(base, r, {k: curves[k] for k in ("T_r", "g_r", "G_r") if k in curves})


def plot_rings(rings, output_dir, label="auto", dpi=300, save_pdf=False,
               show_title=False):
    """Bar chart of ring-size assignments (percent of resolved network edges)."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    sizes = list(rings["ring_sizes"])
    frac = list(rings["fractions"])
    fig, ax = _figure(figsize=(5.0, 3.4), dpi=dpi)
    if sizes:
        ax.bar(sizes, frac, color=_PALETTE[0], width=0.7)
    else:
        limit = rings.get("max_ring")
        message = (f"No ring closures resolved\nwith size ≤ {limit}" if limit is not None
                   else "No ring closures resolved")
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center")
        ax.set_ylim(0, 100)
    ax.set_xlabel("Ring size (network nodes)")
    ax.set_ylabel("Fraction of resolved edges (%)")
    ax.set_xticks(sizes)
    _apply_pub_style(ax)
    if show_title:
        ax.set_title(f"Ring statistics ({label})")
    fig.tight_layout()
    base = os.path.join(output_dir, "analysis_rings")
    _save_fig(fig, base, dpi=dpi, save_pdf=save_pdf)
    # _figure deliberately avoids pyplot and therefore never registers an
    # open figure. Release its artists after saving the potentially large plot.
    fig.clear()
    return base + ".png"
