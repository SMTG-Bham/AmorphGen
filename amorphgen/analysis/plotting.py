"""Plotting functions for structure analysis."""

from __future__ import annotations

import os
import numpy as np

# Okabe-Ito colour-blind-safe palette (RGB hex)
_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
            "#F0E442", "#56B4E9", "#E69F00", "#000000"]


def _apply_pub_style(ax, label_fs=13, tick_fs=11):
    """Apply publication-style cosmetics: hide top/right spines, inward ticks,
    minor ticks, consistent font sizing on tick labels."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_linewidth(1.0)
    ax.tick_params(axis='both', which='major', labelsize=tick_fs,
                   direction='in', length=5, width=1.0, top=False, right=False)
    ax.tick_params(axis='both', which='minor', direction='in', length=3,
                   width=0.8, top=False, right=False)
    ax.minorticks_on()


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

    ax.set_xlabel(r"r (Å)", fontsize=13)
    ax.set_ylabel("g(r)" if normalise else "Count", fontsize=13)
    if show_title:
        ax.set_title(f"RDF — {formula}", fontsize=13)
    ax.legend(fontsize=11, frameon=False, loc='best')
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

    # ── 2. CN distribution bar chart ─────────────────────────────────
    cn_data = analyser.coordination()
    cn_pairs = {k: v for k, v in cn_data.items() if v["mean"] > 0.5}

    if cn_pairs:
        n_panels = len(cn_pairs)
        fig, axes = plt.subplots(1, n_panels, figsize=(3.8 * n_panels, 3.8),
                                 squeeze=False)
        for idx, (pair, data) in enumerate(cn_pairs.items()):
            ax = axes[0, idx]
            cn_vals = sorted(data["distribution"].keys())
            pcts = [data["distribution"][cn] for cn in cn_vals]
            bars = ax.bar(cn_vals, pcts,
                          color=_PALETTE[idx % len(_PALETTE)],
                          edgecolor='black', linewidth=0.8)
            ax.set_xlabel(f"{pair} CN", fontsize=12)
            ax.set_ylabel("Fraction (%)", fontsize=12)
            ax.set_xticks(cn_vals)
            ax.text(0.97, 0.95, f"mean = {data['mean']:.1f}",
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white',
                              ec='0.7', alpha=0.85))
            for bar, pct in zip(bars, pcts):
                if pct > 2:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 1,
                            f"{pct:.0f}%", ha='center', va='bottom',
                            fontsize=9)
            _apply_pub_style(ax)
        if show_title:
            fig.suptitle(f"CN Distribution — {formula}", fontsize=13, y=1.02)
        fig.tight_layout()
        _save_fig(fig, os.path.join(output_dir, f"{prefix}_cn"), dpi, save_pdf)
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

        ax.set_xlabel("Angle (°)", fontsize=13)
        ax.set_ylabel("Probability density" if normalise else "Count",
                      fontsize=13)
        if show_title:
            ax.set_title(f"Bond Angle Distribution — {formula}", fontsize=13)
        ax.legend(fontsize=11, frameon=False, loc='best')
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
