"""Multi-ensemble comparison plots for amorphous structures.

Produces publication-style figures that overlay two or more ensembles
(e.g. Random vs Hybrid vs DFT reference) for the four standard
descriptors:

    (a) partial RDFs
    (b) coordination distributions (mirrored for AB systems)
    (c) bond-angle distributions
    (d) per-structure density vs experimental reference

Each panel is saved as its own PNG / PDF / CSV file rather than a
combined figure. Use :func:`compare_ensembles` for the high-level API
or call the panel functions individually.

Example
-------
::

    from amorphgen.analysis import compare_ensembles, EnsembleSpec

    compare_ensembles(
        ensembles=[
            EnsembleSpec("Random", ["rnd/*.vasp"], color="#D55E00"),
            EnsembleSpec("Hybrid", ["hyb/*.xyz"],  color="#0072B2"),
        ],
        rdf_pairs=[("Si-O", "-"), ("Si-Si", "--"), ("O-O", ":")],
        cn_top_key="Si-O",
        cn_bot_key="O-Si",
        angle_keys=[("O-Si-O", "-"), ("Si-O-Si", "--")],
        exp_density=(2.18, 2.22),
        output_dir="comparison_plots/",
        prefix="sio2",
    )

Outputs (in ``output_dir/``)::

    sio2_rdf.{png,pdf,csv}
    sio2_coordination.{png,pdf,csv}
    sio2_angles.{png,pdf,csv}
    sio2_density.{png,pdf,csv}
"""
from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase.io import read
from ase.units import _Nav

# matplotlib imports happen lazily inside the plot functions (mirrors the
# existing plotting.py pattern) so that simply importing this module does
# not pull matplotlib — keeps doc builds and lightweight scripts fast.

from .analyser import StructureAnalyser


# ─── Okabe-Ito colour-blind-safe palette ──────────────────────────────────
DEFAULT_COLORS = [
    "#0072B2",   # blue        (reference / DFT)
    "#D55E00",   # orange      (random / first AmorphGen ensemble)
    "#009E73",   # green       (hybrid)
    "#CC79A7",   # pink        (full MQ)
    "#F0E442",   # yellow      (spare)
    "#56B4E9",   # light blue  (spare)
]

EXP_COLOR = "#222222"


# ─── Specs ────────────────────────────────────────────────────────────────
@dataclass
class EnsembleSpec:
    """Specification of one ensemble to include in a comparison.

    Parameters
    ----------
    label
        Display name (used in legend and axis labels).
    files
        Either a list of structure-file paths, or a single glob string
        like ``"hybrid_runs/run_*/final_amorphous.xyz"``.
    color
        Matplotlib colour. If None, one is drawn from
        :data:`DEFAULT_COLORS` in registration order.
    cutoff
        Cutoff mode for :class:`StructureAnalyser` (``"auto"``,
        ``"auto-rdf"``, or a numeric value). Defaults to ``"auto-rdf"`` (the
        first RDF minimum) — the correct neighbour cutoff for coordination
        counting. The plain ``"auto"`` mode can land near the bond peak and
        undercount CN (spurious CN 0/1), so it is not the default here.
    """
    label: str
    files: list | str
    color: str | None = None
    cutoff: str = "auto-rdf"
    _analyser: StructureAnalyser | None = field(default=None, init=False, repr=False)
    _file_list: list = field(default_factory=list, init=False, repr=False)

    def resolve_files(self) -> list[str]:
        """Expand a glob string to a sorted list of file paths."""
        if self._file_list:
            return self._file_list
        if isinstance(self.files, str):
            self._file_list = sorted(glob.glob(self.files))
        else:
            self._file_list = list(self.files)
        if not self._file_list:
            raise FileNotFoundError(f"No files matched for ensemble '{self.label}'")
        return self._file_list

    def analyser(self) -> StructureAnalyser:
        """Lazy-built :class:`StructureAnalyser` for this ensemble."""
        if self._analyser is None:
            self._analyser = StructureAnalyser(self.resolve_files(),
                                               cutoff=self.cutoff)
        return self._analyser

    @classmethod
    def from_analyser(cls, label: str,
                      analyser: "StructureAnalyser",
                      color: str | None = None) -> "EnsembleSpec":
        """Wrap an already-built :class:`StructureAnalyser`.

        Useful when you have an analyser object already (e.g., from
        :meth:`StructureAnalyser.plot`) and want to reuse its loaded
        atoms/cutoff rather than re-reading files.
        """
        spec = cls(label=label, files=[], color=color)
        spec._analyser = analyser
        # Cache file paths from the analyser if available
        spec._file_list = getattr(analyser, "_file_list", []) or []
        return spec


# ─── Shared style ─────────────────────────────────────────────────────────
def _style(ax, fs_label=11, fs_tick=10):
    """Apply the AmorphGen publication style to one axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", length=4, width=0.9,
                   labelsize=fs_tick, top=False, right=False)
    ax.tick_params(which="minor", direction="in", length=2.5, width=0.7,
                   top=False, right=False)
    ax.minorticks_on()
    ax.xaxis.label.set_size(fs_label)
    ax.yaxis.label.set_size(fs_label)


def _assign_colours(ensembles: list[EnsembleSpec]) -> None:
    """In-place: fill missing colours from DEFAULT_COLORS."""
    palette_iter = iter(DEFAULT_COLORS)
    for ens in ensembles:
        if ens.color is None:
            try:
                ens.color = next(palette_iter)
            except StopIteration:
                ens.color = "#888888"


def _save(fig, output_dir: str, prefix: str, name: str,
          save_pdf: bool = True, dpi: int = 300) -> None:
    """Save fig as PNG and (optionally) PDF under output_dir/prefix_name.*."""
    import matplotlib.pyplot as plt
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    base = os.path.join(output_dir, f"{prefix}_{name}" if prefix else name)
    fig.savefig(base + ".png", dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _per_structure_density(files: list[str]) -> np.ndarray:
    """Density in g/cm^3 for each structure file."""
    rho = []
    for f in files:
        atoms = read(f)
        rho.append((atoms.get_masses().sum() / _Nav)
                   / (atoms.get_volume() * 1e-24))
    return np.array(rho)


# ─── Panel (a): partial RDFs ──────────────────────────────────────────────
def plot_partial_rdf(ensembles: list[EnsembleSpec],
                     pairs: list[tuple[str, str]],
                     output_dir: str,
                     prefix: str = "",
                     sigma: float = 0.1,
                     rmax: float | None = None,
                     save_pdf: bool = True) -> None:
    """Partial RDFs for one or more pairs, overlaid across ensembles.

    Parameters
    ----------
    ensembles
        List of :class:`EnsembleSpec` (colours auto-assigned if None).
    pairs
        ``[(pair_label, linestyle), ...]`` — e.g. ``[("Si-O", "-")]``.
    output_dir
        Directory to save plots into (created if missing).
    prefix
        Filename prefix; final file is ``<prefix>_rdf.png``.
    sigma
        Gaussian smearing width in Å (default 0.1).
    rmax
        Max r in Å. ``None`` = auto from cell.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    _assign_colours(ensembles)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    csv_rows = [["ensemble", "pair", "r_A", "g_r"]]

    auto_rmax = None
    for ens in ensembles:
        analyser = ens.analyser()
        for pair, ls in pairs:
            try:
                rdf = analyser.rdf(pair=pair, rmax=rmax, sigma=sigma)
            except Exception:
                continue
            ax.plot(rdf["r"], rdf["g_r"], color=ens.color, lw=1.4,
                    ls=ls, alpha=0.9)
            if auto_rmax is None and len(rdf["r"]):
                auto_rmax = float(rdf["r"][-1])
            for r, g in zip(rdf["r"], rdf["g_r"]):
                csv_rows.append([ens.label, pair, float(r), float(g)])

    ax.axhline(1, color="0.5", lw=0.7, ls="--", alpha=0.6, zorder=0)
    ax.set_xlim(0, auto_rmax or 6.0)
    ax.set_ylim(0, None)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("g(r)")

    leg = [Line2D([0], [0], color=e.color, lw=1.6, label=e.label)
           for e in ensembles]
    leg += [Line2D([0], [0], color="0.3", lw=1.4, ls=ls, label=pair)
            for pair, ls in pairs]
    ax.legend(handles=leg, frameon=False, fontsize=8,
              loc="upper right", handlelength=2.0, labelspacing=0.2)
    ax.text(0.03, 0.97, "(a)", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")
    _style(ax)

    _save(fig, output_dir, prefix, "rdf", save_pdf=save_pdf)
    _write_csv(output_dir, prefix, "rdf", csv_rows)


# ─── Panel (b): coordination distributions ────────────────────────────────
def plot_coordination(ensembles: list[EnsembleSpec],
                      top_key: str,
                      bot_key: str | None,
                      output_dir: str,
                      prefix: str = "",
                      save_pdf: bool = True) -> None:
    """Coordination-number distribution as paired bars.

    For systems with two centres of interest (AB), set
    ``bot_key`` (e.g. ``"O-Si"``) to mirror the second distribution
    on the negative half-plane. For mono-element systems, set
    ``bot_key=None``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    _assign_colours(ensembles)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    csv_rows = [["ensemble", "site", "CN", "fraction_percent"]]

    cn_tops = [(e, e.analyser().coordination()[top_key]["distribution"])
               for e in ensembles]
    cn_bots = [(e, e.analyser().coordination()[bot_key]["distribution"]
                if bot_key else {}) for e in ensembles]
    all_cn = sorted({c for _, d in cn_tops for c in d}
                    | {c for _, d in cn_bots for c in d})
    x = np.array(all_cn, dtype=float)
    n = len(ensembles)
    width = 0.8 / n if n > 0 else 0.8

    for i, (ens, dist) in enumerate(cn_tops):
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, [dist.get(c, 0) for c in all_cn], width,
               color=ens.color, edgecolor="black", lw=0.4, label=ens.label)
        for c in all_cn:
            csv_rows.append([ens.label, top_key, c, dist.get(c, 0.0)])

    if bot_key:
        for i, (ens, dist) in enumerate(cn_bots):
            offset = (i - (n - 1) / 2) * width
            ax.bar(x + offset, [-dist.get(c, 0) for c in all_cn], width,
                   color=ens.color, edgecolor="black", lw=0.4, alpha=0.55)
            for c in all_cn:
                csv_rows.append([ens.label, bot_key, c, dist.get(c, 0.0)])
        ax.axhline(0, color="black", lw=0.9, zorder=4)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, p: f"{abs(v):.0f}"))
        ymax = max(
            max((max(d.values(), default=0) for _, d in cn_tops), default=0),
            max((max(d.values(), default=0) for _, d in cn_bots), default=0),
        )
        ax.set_ylim(-1.20 * ymax, 1.20 * ymax)
        ax.text(0.97, 0.93, top_key, transform=ax.transAxes,
                ha="right", va="top", fontsize=9, fontweight="bold", color="0.25")
        ax.text(0.97, 0.07, bot_key, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, fontweight="bold", color="0.25")
    else:
        ymax = max((max(d.values(), default=0) for _, d in cn_tops), default=1)
        ax.set_ylim(0, ymax * 1.20)
        ax.text(0.97, 0.93, top_key, transform=ax.transAxes,
                ha="right", va="top", fontsize=9, fontweight="bold", color="0.25")

    ax.set_xlabel("Coordination number")
    ax.set_ylabel("Fraction (%)")
    ax.set_xticks(all_cn)
    ax.legend(frameon=False, fontsize=8, loc="upper left",
              bbox_to_anchor=(0.10, 0.98), labelspacing=0.25)
    ax.text(0.03, 0.97, "(b)", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")
    _style(ax)

    _save(fig, output_dir, prefix, "coordination", save_pdf=save_pdf)
    _write_csv(output_dir, prefix, "coordination", csv_rows)


# ─── Panel (c): bond-angle distributions ──────────────────────────────────
def plot_bond_angles(ensembles: list[EnsembleSpec],
                     angle_keys: list[tuple[str, str]],
                     output_dir: str,
                     prefix: str = "",
                     bins: np.ndarray | None = None,
                     save_pdf: bool = True) -> None:
    """Histogram of bond angles, overlaid across ensembles.

    Parameters
    ----------
    angle_keys
        ``[(triplet, linestyle), ...]`` — e.g. ``[("O-Si-O", "-")]``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    _assign_colours(ensembles)
    if bins is None:
        bins = np.arange(40, 180, 2)
    centres = 0.5 * (bins[:-1] + bins[1:])
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    csv_rows = [["ensemble", "triplet", "angle_deg", "probability_density"]]

    for ens in ensembles:
        angles_dict = ens.analyser()._compute_all_angles()
        for key, ls in angle_keys:
            angles = angles_dict.get(key, [])
            if len(angles):
                h, _ = np.histogram(angles, bins=bins, density=True)
                ax.plot(centres, h, color=ens.color, lw=1.5, ls=ls, alpha=0.9)
                for c, v in zip(centres, h):
                    csv_rows.append([ens.label, key, float(c), float(v)])

    ax.set_xlim(float(bins[0]), float(bins[-1]))
    ax.set_xlabel("Angle (°)")
    ax.set_ylabel("Probability density")

    leg = [Line2D([0], [0], color=e.color, lw=1.6, label=e.label)
           for e in ensembles]
    leg += [Line2D([0], [0], color="0.3", lw=1.4, ls=ls, label=k)
            for k, ls in angle_keys]
    ax.legend(handles=leg, frameon=False, fontsize=8,
              loc="upper right", handlelength=2.0, labelspacing=0.2)
    ax.text(0.03, 0.97, "(c)", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")
    _style(ax)

    _save(fig, output_dir, prefix, "angles", save_pdf=save_pdf)
    _write_csv(output_dir, prefix, "angles", csv_rows)


# ─── Panel (d): per-structure density vs experimental reference ───────────
def plot_density(ensembles: list[EnsembleSpec],
                 exp_density: tuple[float, float] | None,
                 output_dir: str,
                 prefix: str = "",
                 exp_label: str = "Expt.",
                 save_pdf: bool = True) -> None:
    """Per-structure density as violins, with optional experimental band.

    Parameters
    ----------
    exp_density
        ``(rho_lo, rho_hi)`` in g/cm³, drawn as a cap-bar at x=1. Pass
        ``None`` to hide the experiment column.
    """
    import matplotlib.pyplot as plt
    _assign_colours(ensembles)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    csv_rows = [["ensemble", "structure_index", "density_g_per_cm3"]]

    has_exp = exp_density is not None
    x_exp = 1 if has_exp else None
    if has_exp:
        exp_lo, exp_hi = exp_density
        exp_mid = 0.5 * (exp_lo + exp_hi)
        ax.vlines(x_exp, exp_lo, exp_hi, color=EXP_COLOR, lw=2.2, zorder=3)
        for y in (exp_lo, exp_hi):
            ax.hlines(y, x_exp - 0.18, x_exp + 0.18,
                      color=EXP_COLOR, lw=1.4, zorder=3)
        ax.scatter([x_exp], [exp_mid], marker="D", s=46, color=EXP_COLOR,
                   edgecolor="white", lw=0.8, zorder=4)
        ax.text(x_exp, exp_hi + 0.03 * (exp_hi - exp_lo + 0.1) + 0.02,
                f"{exp_mid:.2f}", ha="center", va="bottom",
                fontsize=9, color=EXP_COLOR, fontweight="bold")
        csv_rows.append([exp_label, "exp_lo", exp_lo])
        csv_rows.append([exp_label, "exp_hi", exp_hi])

    rho_data = [_per_structure_density(e.resolve_files()) for e in ensembles]
    start = (x_exp + 1) if has_exp else 1
    positions = list(range(start, start + len(ensembles)))

    vp = ax.violinplot(rho_data, positions=positions, widths=0.65,
                       showmeans=False, showmedians=False, showextrema=False)
    for body, ens in zip(vp["bodies"], ensembles):
        body.set_facecolor(ens.color)
        body.set_alpha(0.32)
        body.set_edgecolor("black")
        body.set_linewidth(0.9)

    rng = np.random.default_rng(0)
    for x0, vals, ens in zip(positions, rho_data, ensembles):
        jx = x0 + 0.06 * rng.standard_normal(len(vals))
        ax.scatter(jx, vals, color=ens.color, s=22, alpha=0.9,
                   edgecolor="black", lw=0.4, zorder=3)
        m, s = vals.mean(), vals.std()
        ax.hlines(m, x0 - 0.22, x0 + 0.22, color="black", lw=1.6, zorder=4)
        ax.errorbar(x0, m, yerr=s, color="black", lw=1.0, capsize=4,
                    fmt="none", zorder=4)
        ax.text(x0, vals.max() + 0.03 * (vals.max() - vals.min() + 0.1) + 0.02,
                f"{m:.2f}±{s:.2f}", ha="center", va="bottom",
                fontsize=9, color=ens.color, fontweight="bold")
        for i, v in enumerate(vals):
            csv_rows.append([ens.label, i, float(v)])

    xticks = ([x_exp] if has_exp else []) + positions
    xticklabels = ([exp_label] if has_exp else []) + [e.label for e in ensembles]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(0.4, positions[-1] + 0.6)

    all_rho = np.concatenate(rho_data + ([np.array([exp_lo, exp_hi])]
                                          if has_exp else []))
    ymin = all_rho.min() - 0.10 * max(0.05, all_rho.max() - all_rho.min())
    ymax = all_rho.max() + 0.20 * max(0.05, all_rho.max() - all_rho.min())
    span = max(0.1, ymax - ymin)
    ax.set_ylim(ymin - 0.05 * span, ymax + 0.10 * span)
    ax.set_ylabel(r"Density (g cm$^{-3}$)")
    ax.text(0.03, 0.97, "(d)", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")
    _style(ax)

    _save(fig, output_dir, prefix, "density", save_pdf=save_pdf)
    _write_csv(output_dir, prefix, "density", csv_rows)


# ─── CSV helper ───────────────────────────────────────────────────────────
def _write_csv(output_dir: str, prefix: str, name: str, rows: list) -> None:
    """Write a CSV alongside the figure for downstream re-plotting."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = f"{prefix}_{name}.csv" if prefix else f"{name}.csv"
    with open(os.path.join(output_dir, fname), "w", newline="") as f:
        csv.writer(f).writerows(rows)


# ─── High-level convenience wrapper ───────────────────────────────────────
def compare_ensembles(
    ensembles: list[EnsembleSpec],
    rdf_pairs: list[tuple[str, str]] | None = None,
    cn_top_key: str | None = None,
    cn_bot_key: str | None = None,
    angle_keys: list[tuple[str, str]] | None = None,
    exp_density: tuple[float, float] | None = None,
    output_dir: str = "comparison_plots",
    prefix: str = "",
    exp_label: str = "Expt.",
    save_pdf: bool = True,
) -> None:
    """Run all four panel functions for the supplied ensembles.

    Each descriptor produces three files:
    ``{prefix}_{descriptor}.png``,
    ``{prefix}_{descriptor}.pdf``,
    ``{prefix}_{descriptor}.csv``.

    Skip any descriptor by passing ``None`` for its key arguments.
    """
    _assign_colours(ensembles)
    if rdf_pairs:
        plot_partial_rdf(ensembles, rdf_pairs, output_dir, prefix,
                         save_pdf=save_pdf)
    if cn_top_key:
        plot_coordination(ensembles, cn_top_key, cn_bot_key,
                          output_dir, prefix, save_pdf=save_pdf)
    if angle_keys:
        plot_bond_angles(ensembles, angle_keys, output_dir, prefix,
                         save_pdf=save_pdf)
    if exp_density is not None or len(ensembles) >= 1:
        plot_density(ensembles, exp_density, output_dir, prefix,
                     exp_label=exp_label, save_pdf=save_pdf)
