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

# Okabe-Ito colour-blind-safe palette (RGB hex)
_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
            "#F0E442", "#56B4E9", "#E69F00", "#000000"]

_EXP_COLOR = "#222222"


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


def _save_fig(fig, base_path, dpi=300, save_pdf=False):
    """Save PNG (and optionally PDF) at the given base path (no extension)."""
    fig.savefig(f"{base_path}.png", dpi=dpi, bbox_inches='tight')
    print(f"  Saved: {base_path}.png")
    if save_pdf:
        fig.savefig(f"{base_path}.pdf", bbox_inches='tight')
        print(f"  Saved: {base_path}.pdf")


def plot_analysis(analyser, output_dir=".", prefix="analysis",
                  rdf_pairs=None, angle_triplets=None,
                  rmax=None, normalise=True, angle_style="line",
                  save_csv=True, show_total_rdf=False,
                  smearing=0.0,
                  dpi=300, save_pdf=False, show_title=False):
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
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    fig, ax = plt.subplots(figsize=(7, 4.5))
    rdf_csv_data = {}

    if normalise:
        ax.axhline(y=1, color='0.5', linestyle='--', linewidth=0.8, alpha=0.6,
                   zorder=0)

    # Show total RDF for single-element systems, or if explicitly requested
    if len(unique) == 1 or show_total_rdf:
        rdf_total = analyser.rdf(pair=None, rmax=rmax, sigma=smearing)
        r = np.array(rdf_total["r"])
        g_r_total = np.array(rdf_total["g_r"])
        ax.plot(r, g_r_total, label="Total", linewidth=2.0,
                color='black', linestyle='--' if len(unique) > 1 else '-')
        rdf_csv_data["Total"] = (r, g_r_total)

    for i, pair in enumerate(pairs):
        rdf_data = analyser.rdf(pair=pair, rmax=rmax, sigma=smearing)
        r = np.array(rdf_data["r"])
        g_r = np.array(rdf_data["g_r"])
        ax.plot(r, g_r, label=pair, linewidth=1.8,
                color=_PALETTE[i % len(_PALETTE)])
        rdf_csv_data[pair] = (r, g_r)

    ax.set_xlabel(r"r (Å)")
    ax.set_ylabel("g(r)" if normalise else "Count")
    if show_title:
        ax.set_title(f"RDF — {formula}", fontsize=12)
    ax.legend(fontsize=9, frameon=False, loc='best')
    ax.set_xlim(0, rmax)
    _apply_pub_style(ax)
    fig.tight_layout()
    _save_fig(fig, os.path.join(output_dir, f"{prefix}_rdf"), dpi, save_pdf)
    plt.close(fig)

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

    # ── 2. CN distribution ───────────────────────────────────────────
    # When an A-B / B-A reciprocal pair exists (e.g. Si-O and O-Si),
    # use the mirrored-bars layout: bonding CN on top, mirrored counterpart
    # below the zero line. Otherwise fall back to side-by-side panels.
    cn_data = analyser.coordination()
    cn_pairs = {k: v for k, v in cn_data.items() if v["mean"] > 0.5}

    if cn_pairs:
        from matplotlib.ticker import FuncFormatter
        top_key, bot_key = _find_reciprocal_pair(cn_pairs)

        if top_key and bot_key:
            # ── Mirrored layout ──
            top = cn_pairs[top_key]["distribution"]
            bot = cn_pairs[bot_key]["distribution"]
            all_cn = sorted(set(top) | set(bot))
            x = np.array(all_cn, dtype=float)

            fig, ax = plt.subplots(figsize=(5.0, 4.0))
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
            ax.set_ylabel("Fraction (%)")
            ax.set_xticks(all_cn)
            ax.legend(frameon=False, fontsize=9, loc="upper left",
                      labelspacing=0.25)
            _apply_pub_style(ax)
            fig.tight_layout()
            _save_fig(fig, os.path.join(output_dir, f"{prefix}_cn"),
                      dpi, save_pdf)
            plt.close(fig)

        else:
            # ── Side-by-side panels (mono-element or multi-cation) ──
            n_panels = len(cn_pairs)
            fig, axes = plt.subplots(1, n_panels,
                                     figsize=(3.8 * n_panels, 3.8),
                                     squeeze=False)
            for idx, (pair, data) in enumerate(cn_pairs.items()):
                ax = axes[0, idx]
                cn_vals = sorted(data["distribution"].keys())
                pcts = [data["distribution"][cn] for cn in cn_vals]
                bars = ax.bar(cn_vals, pcts,
                              color=_PALETTE[idx % len(_PALETTE)],
                              edgecolor='black', linewidth=0.4)
                ax.set_xlabel(f"{pair} CN")
                ax.set_ylabel("Fraction (%)")
                ax.set_xticks(cn_vals)
                ax.text(0.97, 0.95, f"mean = {data['mean']:.1f}",
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white',
                                  ec='0.7', alpha=0.85))
                for bar, pct in zip(bars, pcts):
                    if pct > 2:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 1,
                                f"{pct:.0f}%", ha='center', va='bottom',
                                fontsize=8)
                _apply_pub_style(ax)
            if show_title:
                fig.suptitle(f"CN Distribution — {formula}", fontsize=12,
                             y=1.02)
            fig.tight_layout()
            _save_fig(fig, os.path.join(output_dir, f"{prefix}_cn"),
                      dpi, save_pdf)
            plt.close(fig)

        if save_csv:
            cn_csv_path = os.path.join(output_dir, f"{prefix}_cn.csv")
            with open(cn_csv_path, "w") as f:
                f.write("pair,CN,fraction(%),count\n")
                for pair, data in cn_pairs.items():
                    total_atoms = data["total_atoms"]
                    for cn_val, pct in sorted(data["distribution"].items()):
                        count = int(round(pct * total_atoms / 100))
                        f.write(f"{pair},{cn_val},{pct:.1f},{count}\n")
            print(f"  Saved: {cn_csv_path}")

    # ── 3. Bond angle distribution ───────────────────────────────────
    all_angle_data = analyser._compute_all_angles()

    if angle_triplets is not None:
        all_angle_data = {k: v for k, v in all_angle_data.items()
                         if k in angle_triplets}
    all_angle_data = {k: v for k, v in all_angle_data.items() if len(v) > 10}

    if all_angle_data:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bins = np.arange(40, 180, 2)
        bin_centres = (bins[:-1] + bins[1:]) / 2

        for i, (triplet, angles) in enumerate(all_angle_data.items()):
            colour = _PALETTE[i % len(_PALETTE)]
            if angle_style in ("histogram", "both"):
                ax.hist(angles, bins=bins, alpha=0.3,
                        label=triplet if angle_style == "histogram" else None,
                        density=normalise, color=colour,
                        edgecolor='black', linewidth=0.3)
            if angle_style in ("line", "both"):
                hist, _ = np.histogram(angles, bins=bins, density=normalise)
                ax.plot(bin_centres, hist, label=triplet, linewidth=2.0,
                        color=colour)

        ax.set_xlabel("Angle (°)")
        ax.set_ylabel("Probability density" if normalise else "Count")
        if show_title:
            ax.set_title(f"Bond Angle Distribution — {formula}", fontsize=12)
        ax.legend(fontsize=9, frameon=False, loc='best')
        ax.set_xlim(40, 180)
        _apply_pub_style(ax)
        fig.tight_layout()
        _save_fig(fig, os.path.join(output_dir, f"{prefix}_angles"),
                  dpi, save_pdf)
        plt.close(fig)

        if save_csv:
            angle_csv_path = os.path.join(output_dir, f"{prefix}_angles.csv")
            with open(angle_csv_path, "w") as f:
                f.write("triplet,angle(deg)\n")
                for triplet, angles in all_angle_data.items():
                    for a in angles:
                        f.write(f"{triplet},{a:.2f}\n")
            print(f"  Saved: {angle_csv_path}")

    # ── 4. Per-structure density violin ──────────────────────────────
    # New in v1.0.0: matches the density panel in
    # amorphgen.analysis.comparison_plots.plot_density.
    density_dict = analyser.density()
    rho_values = np.array(density_dict.get("values", []))
    if len(rho_values) >= 2:
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
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
        m, s = rho_values.mean(), rho_values.std()
        ax.hlines(m, x_pos - 0.22, x_pos + 0.22, color="black", lw=1.6,
                  zorder=4)
        ax.errorbar(x_pos, m, yerr=s, color="black", lw=1.0, capsize=4,
                    fmt="none", zorder=4)
        ax.text(x_pos, rho_values.max()
                + 0.03 * (rho_values.max() - rho_values.min() + 0.1) + 0.02,
                f"{m:.2f} ± {s:.2f}",
                ha="center", va="bottom", fontsize=9,
                color=_PALETTE[0], fontweight="bold")
        ax.set_xticks([x_pos])
        ax.set_xticklabels(["AmorphGen"])
        ax.set_xlim(0.4, 1.6)
        ymin = rho_values.min() - 0.10 * max(0.05,
                                              rho_values.max() - rho_values.min())
        ymax = rho_values.max() + 0.20 * max(0.05,
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
        plt.close(fig)

        if save_csv:
            rho_csv_path = os.path.join(output_dir, f"{prefix}_density.csv")
            with open(rho_csv_path, "w") as f:
                f.write("structure_index,density_g_per_cm3\n")
                for i, rho in enumerate(rho_values):
                    f.write(f"{i},{rho:.4f}\n")
            print(f"  Saved: {rho_csv_path}")


def plot_sq(sq_result, output_dir=".", prefix="analysis", dpi=300,
            save_pdf=False, weighting="xray", show_title=False):
    """Plot the direct-method total structure factor S(q) + write a CSV.

    Direct (Debye) S(q) with Faber-Ziman normalisation (S(q→∞)=1). The FSDP
    region is resolvable down to q_min ≈ 2π/L, so small boxes leave the
    low-q part noisy — the CSV includes ``n_per_bin`` so shells built from
    only 1-3 reciprocal vectors can be identified.
    """
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    q = np.array(sq_result["q"], dtype=float)
    s = np.array(sq_result["s_q"], dtype=float)
    n = np.array(sq_result["n_per_bin"], dtype=int)
    m = ~np.isnan(s) & (n > 0)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(q[m], s[m], lw=1.4, color=_PALETTE[0])
    ax.axhline(1.0, ls=":", color="grey", alpha=0.6)
    ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel(r"$S(q)$")
    _apply_pub_style(ax)
    if show_title:
        ax.set_title(f"Total S(q) — direct method, {weighting} weighting")
    base = os.path.join(output_dir, f"{prefix}_sq")
    _save_fig(fig, base, dpi, save_pdf)

    with open(f"{base}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["q_invA", "s_q", "n_per_bin"])
        for qi, si, ni in zip(q, s, n):
            w.writerow([f"{qi:.5f}", "" if np.isnan(si) else f"{si:.6f}", ni])
    print(f"  Saved: {base}.csv")
