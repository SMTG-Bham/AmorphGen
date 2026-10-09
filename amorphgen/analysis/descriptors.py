"""CLI reporting and export for optional material descriptors."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ._serialization import numpy_json_default

def _format_descriptor(name, result):
    """Return a compact, unit-labelled report for a computed descriptor."""
    if name == "bond_order":
        params = result["parameters"]
        lines = ["\n  Crystal-like order (Steinhardt q6 / Lechner-Dellago qbar6):",
                 f"    Criterion: qbar6 >= {params['qbar6_threshold']:g}; "
                 f"neighbors >= {params['min_neighbors']}",
                 f"    Neighbor cutoff (A): {params['cutoff']}",
                 "    Structure    mean q6   mean qbar6    ordered     largest cluster"]
        for row in result["per_structure"]:
            lines.append(
                f"    {row['index']:9d}    {row['q6_mean']:.5f}      "
                f"{row['qbar6_mean']:.5f}    {100 * row['ordered_fraction']:6.2f}%"
                f"    {row['largest_cluster_size']:6d}/{row['n_atoms']} "
                f"({100 * row['largest_cluster_fraction']:.2f}%)")
        lines.append("    Geometric order criterion; calibrate the cutoff and threshold "
                     "against crystal and liquid references.")
        return "\n".join(lines)
    if name == "voids":
        def clearance(value):
            return f"{value:.4f}" if value is not None else "unavailable"

        quantiles = result["clearance_quantiles"]
        lines = ["\n  Void distribution (sampled point clearance):",
                 f"    Accessible fraction: {result['accessible_fraction']:.6f} "
                 f"+/- {result['accessible_fraction_stderr']:.6f} (Monte Carlo sampling stderr)",
                 f"    Probe radius: {result['probe_radius']:.4f} A; "
                 f"radii: {result['radius_source']}",
                 f"    Mean accessible volume: {result['accessible_volume']:.4f} A^3",
                 "    Accessible-point clearance (A): "
                 f"mean={clearance(result['mean_clearance'])}; "
                 f"sample maximum={clearance(result['max_clearance'])}",
                 "    Accessible-point clearance quantiles (A): " + "; ".join(
                     f"{key}={clearance(quantiles[key])}" for key in ("p10", "p50", "p90"))]
        curve = result["probe_curve"]
        lines.append(f"    Probe accessibility curve: {len(curve['radii'])} radii; "
                     "fractions are weighted by cell volume.")
        lines.append(f"    Sampling: {result['n_samples']} points per structure; "
                     "Monte Carlo errors exclude between-structure variation.")
        lines.append("    Sample maxima are lower bounds on maximum clearance; "
                     "this is not a connected-pore or maximal-cavity analysis.")
        return "\n".join(lines)
    if name == "rings":
        pair = "-".join(result["bond_pair"])
        cutoff = result["cutoff"]
        cutoff_text = (", ".join(f"{key}={value:g}" for key, value in cutoff.items())
                       if isinstance(cutoff, dict) else f"{cutoff:g}")
        lines = [f"\n  Ring statistics ({pair}; network nodes):",
                 f"    nodes-bridge: {pair}",
                 f"    Shortest closure per network edge; cutoff={cutoff_text} A; "
                 f"maximum ring size={result['max_ring']}.",
                 "    Counts are edge assignments, not unique cycles; "
                 "size fractions use resolved edges."]
        for size, count, fraction in zip(result["ring_sizes"], result["counts"],
                                         result["fractions"]):
            lines.append(f"    {size:2d}-ring: {count:6d} ({fraction:6.2f}%)")
        if result["mean_ring_size"] is not None:
            lines.append(f"    Ring size: mean={result['mean_ring_size']:.3f}; "
                         f"std={result['std_ring_size']:.3f}; "
                         f"range={result['min_ring_size']}-{result['max_ring_size']} nodes")
        else:
            lines.append("    No ring closures resolved within the requested maximum size.")
        coverage = result["ring_edge_fraction"]
        coverage_text = f" ({100 * coverage:.2f}%)" if coverage is not None else ""
        lines.append(f"    Resolved network edges: {result['n_ring_edges']}/"
                     f"{result['n_network_edges']}{coverage_text}")
        lines.append(f"    Unresolved edges: {result['n_unresolved_edges']} "
                     f"(no closure found with size <= {result['max_ring']}; "
                     "larger rings may exist).")
        return "\n".join(lines)
    if name == "oxygen_speciation":
        lines = ["\n  Oxygen speciation (network formers: "
                 + ", ".join(result['network_formers']) + "):"]
        for species, count in result["counts"].items():
            lines.append(f"    {species:20s}: {count:6d} "
                         f"({100 * result['fractions'][species]:6.2f}%)")
        return "\n".join(lines)
    if name == "elastic":
        kind = "relaxed ions" if result["relaxed_ions"] else "clamped ions"
        lines = [f"\n  Elastic moduli ({kind}; GPa):"]
        for label, values in result["ensemble"]["moduli"].items():
            parts = []
            for key, short in (("bulk_modulus_gpa", "K"), ("shear_modulus_gpa", "G"),
                               ("young_modulus_gpa", "E"), ("poisson_ratio", "nu")):
                value = values[key]
                parts.append(f"{short}={value['mean']:.4f}" if value["mean"] is not None
                             else f"{short}=unavailable")
            lines.append(f"    {label}: " + ", ".join(parts))
        for item in result["per_structure"]:
            lines.append(f"    Structure {item['index']}: max residual stress "
                         f"{item['max_residual_stress_gpa']:.4f} GPa; "
                         f"stable={item['mechanically_stable']}")
            lines.extend(f"      {w}" for w in item["warnings"])
        return "\n".join(lines)
    if name == "vdos":
        return ("\n  Vibrational density of states (harmonic, Gamma-point cell modes):\n"
                f"    Modes: {result['total_modes']}; imaginary: {result['imaginary_modes']}\n"
                f"    Frequency: THz; Gaussian width: {result['sigma_thz']:.4f} THz\n"
                "    Negative frequencies denote imaginary modes; DOS integrates to one.")
    raise ValueError(f"Unknown descriptor: {name}")


def format_descriptor(name, result):
    """Report pooled fractions separately from ensemble uncertainty."""
    from .analyser import _format_uncertainty
    text = _format_descriptor(name, result)
    lines = [text]
    for field, label in (("fraction_of_sites", "Fraction of sites"),
                         ("fraction_of_structures", "Fraction of structures (any such site)")):
        if field in result:
            values = result[field]
            if isinstance(values, dict):
                lines.append("    " + label + ": " + ", ".join(
                    f"{key}={100 * value:.2f}%" for key, value in values.items()
                    if value is not None))
            elif values is not None:
                lines.append(f"    {label}: {100 * values:.2f}%")
    summaries = result.get("uncertainty", {})
    if summaries:
        lines.append("    Ensemble uncertainty (equal weight per independent structure):")
        for key, summary in summaries.items():
            if not isinstance(summary, dict) or "sem" not in summary:
                continue
            if isinstance(summary["mean"], list):
                lines.append(f"      {key}: per-point SEM, t CI and bootstrap bounds in JSON "
                             f"(n={summary['n_structures']})")
            else:
                lines.append(f"      {key}: " + _format_uncertainty(summary))
    return "\n".join(lines)


def _json_default(value):
    try:
        return numpy_json_default(value)
    except TypeError:
        raise TypeError(f"Cannot encode {type(value).__name__}") from None


def save_descriptor(name, result, output_dir, *, dpi=300, save_pdf=False,
                    show_title=False):
    """Write full JSON, tabular CSV and a PNG (optionally PDF) figure."""
    from .plotting import _figure, _save_fig, _apply_pub_style, _PALETTE

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / f"analysis_{name}"
    with base.with_suffix(".json").open("w") as handle:
        json.dump(result, handle, default=_json_default, indent=2, allow_nan=False)
        handle.write("\n")

    def write_csv(path, header, rows):
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    def source_file(index):
        files = result.get("structure_files", [])
        return str(files[index]) if index < len(files) else ""

    if name == "rings":
        from .plotting import plot_rings
        # Retain the original pooled CSV interface for existing workflows.
        write_csv(base.with_suffix(".csv"), ["ring_size", "count", "fraction_percent"],
                  zip(result["ring_sizes"], result["counts"], result["fractions"]))
        columns = ["mean_ring_size", "std_ring_size", "min_ring_size", "max_ring_size",
                   "n_network_nodes", "n_network_edges", "n_ring_edges",
                   "n_unresolved_edges", "ring_edge_fraction", "total_rings"]
        write_csv(directory / "analysis_rings_structures.csv", ["index", "source_file", *columns],
                  ([row["index"], source_file(row["index"]), *[row[key] for key in columns]]
                   for row in result["per_structure"]))
        write_csv(directory / "analysis_rings_per_structure.csv",
                  ["structure_index", "source_file", "ring_size", "count", "fraction_percent"],
                  ([row["index"], source_file(row["index"]), size,
                    row["counts"].get(size, row["counts"].get(str(size), 0)),
                    100 * row["counts"].get(size, row["counts"].get(str(size), 0)) / row["total_rings"]
                    if row["total_rings"] else None]
                   for row in result["per_structure"] for size in result["ring_sizes"]))
        plot_rings(result, directory, label="-".join(result["bond_pair"]), dpi=dpi,
                   save_pdf=save_pdf, show_title=show_title)
        print(f"  Saved: {base}.csv / .json")
        return

    fig, ax = _figure(figsize=(6.0, 4.0))
    if name == "bond_order":
        columns = ["index", "n_atoms", "q6_mean", "qbar6_mean", "ordered_count",
                   "ordered_fraction", "largest_cluster_size", "largest_cluster_fraction"]
        write_csv(base.with_suffix(".csv"), columns,
                  ([row[key] for key in columns] for row in result["per_structure"]))
        write_csv(directory / "analysis_bond_order_atoms.csv",
                  ["structure_index", "atom_index", "q6", "qbar6", "neighbor_count",
                   "ordered", "cluster_id"],
                  ([row["index"], i, q6, qbar6, count, int(ordered), cluster]
                   for row in result["per_structure"]
                   for i, (q6, qbar6, count, ordered, cluster) in enumerate(zip(
                       row["q6"], row["qbar6"], row["neighbor_counts"],
                       row["ordered"], row["cluster_ids"]))))
        values = [value for row in result["per_structure"] for value in row["qbar6"]]
        ax.hist(values, bins=np.linspace(0, 1, 51), color=_PALETTE[0])
        ax.axvline(result["parameters"]["qbar6_threshold"], color=_PALETTE[1],
                   linestyle="--", label="Order threshold")
        ax.set(xlabel="Lechner–Dellago averaged q₆", ylabel="Atom count")
        ax.legend(frameon=False)
    elif name == "voids":
        write_csv(base.with_suffix(".csv"),
                  ["clearance_A", "probability_density_per_A", "bin_volume_fraction"],
                  zip(result["radius"], result["probability_density"],
                      result["bin_volume_fraction"]))
        ax.plot(result["radius"], result["probability_density"], color=_PALETTE[0])
        ax.set(xlabel="Point clearance (Å)", ylabel="Accessible-space density (Å⁻¹)")
        columns = ["index", "source_file", "cell_volume_A3", "n_samples", "n_accessible",
                   "probe_radius_A", "accessible_fraction", "accessible_fraction_stderr",
                   "accessible_fraction_interval_95_low", "accessible_fraction_interval_95_high",
                   "accessible_volume_A3", "accessible_volume_stderr_A3",
                   "mean_clearance_A", "max_clearance_A",
                   "clearance_p10_A", "clearance_p50_A", "clearance_p90_A"]
        write_csv(directory / "analysis_voids_structures.csv", columns,
                  ([row["index"], source_file(row["index"]), row["cell_volume"],
                    row["n_samples"], row["n_accessible"], result["probe_radius"],
                    row["accessible_fraction"], row["accessible_fraction_stderr"],
                    *row["accessible_fraction_interval_95"], row["accessible_volume"],
                    row["accessible_volume_stderr"], row["mean_clearance"], row["max_clearance"],
                    *[row["clearance_quantiles"][key] for key in ("p10", "p50", "p90")]]
                   for row in result["per_structure"]))
        curve = result["probe_curve"]
        write_csv(directory / "analysis_voids_probe.csv",
                  ["probe_radius_A", "accessible_fraction", "accessible_fraction_stderr",
                   "accessible_volume_A3", "accessible_volume_stderr_A3"],
                  zip(curve["radii"], curve["accessible_fraction"],
                      curve["accessible_fraction_stderr"], curve["accessible_volume"],
                      curve["accessible_volume_stderr"]))
        curve_fig, curve_ax = _figure(figsize=(6.0, 4.0))
        probe_radii = np.asarray(curve["radii"])
        fractions = np.asarray(curve["accessible_fraction"])
        stderr = np.asarray(curve["accessible_fraction_stderr"])
        curve_ax.plot(probe_radii, fractions, color=_PALETTE[0],
                      marker="o" if len(probe_radii) == 1 else None,
                      label="Volume-weighted accessible fraction")
        curve_ax.fill_between(probe_radii, np.maximum(0, fractions - stderr),
                              np.minimum(1, fractions + stderr), color=_PALETTE[0],
                              alpha=0.2, label="±1 Monte Carlo sampling SE")
        if len(probe_radii) == 1:
            curve_ax.errorbar(probe_radii, fractions, yerr=stderr, color=_PALETTE[0],
                              fmt="none", capsize=3)
        curve_ax.set(xlabel="Probe radius (Å)", ylabel="Accessible volume fraction", ylim=(0, 1))
        curve_ax.legend(frameon=False, fontsize=9)
        _apply_pub_style(curve_ax)
        if show_title:
            curve_ax.set_title("Probe accessibility")
        curve_fig.tight_layout()
        _save_fig(curve_fig, str(directory / "analysis_voids_probe"),
                  dpi=dpi, save_pdf=save_pdf)
    elif name == "oxygen_speciation":
        labels = list(result["counts"])
        write_csv(base.with_suffix(".csv"),
                  ["species", "count", "fraction_of_sites", "fraction_of_structures"],
                  ((k, result["counts"][k], result["fractions"][k],
                    result["fraction_of_structures"][k]) for k in labels))
        ax.bar(range(len(labels)), [100 * result["fractions"][k] for k in labels],
               color=_PALETTE[0])
        ax.set_xticks(range(len(labels)), [k.replace("_", "\n") for k in labels])
        ax.set_ylabel("Fraction of oxygen sites (%)")
    elif name == "elastic":
        write_csv(base.with_suffix(".csv"),
                  ["average", "quantity", "mean", "std", "valid_structures"],
                  ((label, key, v["mean"], v["std"], v["count"])
                   for label, values in result["ensemble"]["moduli"].items()
                   for key, v in values.items()))
        tensor = np.asarray(result["ensemble"]["stiffness_tensor_mean_gpa"])
        order = result["voigt_order"]
        write_csv(directory / "analysis_elastic_tensor.csv", ["component", *order],
                  ([label, *row] for label, row in zip(order, tensor)))
        im = ax.imshow(tensor, cmap="coolwarm")
        ax.set_xticks(range(6), order)
        ax.set_yticks(range(6), order)
        ax.set(xlabel="Engineering strain component", ylabel="Stress component")
        fig.colorbar(im, ax=ax, label="Mean stiffness (GPa)")
    elif name == "vdos":
        elements = sorted(result["projected_dos"])
        write_csv(base.with_suffix(".csv"), ["frequency_THz", "dos_per_THz", *elements],
                  zip(result["frequencies_thz"], result["dos"],
                      *(result["projected_dos"][s] for s in elements)))
        ax.plot(result["frequencies_thz"], result["dos"], color=_PALETTE[0], label="Total")
        for i, s in enumerate(elements):
            ax.plot(result["frequencies_thz"], result["projected_dos"][s],
                    color=_PALETTE[(i + 1) % len(_PALETTE)], label=s, alpha=0.8)
        ax.set(xlabel="Frequency (THz; negative = imaginary)", ylabel="DOS (THz⁻¹)")
        ax.legend(frameon=False)
    else:
        raise ValueError(f"Unknown descriptor: {name}")
    _apply_pub_style(ax)
    if show_title:
        ax.set_title(name.replace("_", " ").capitalize())
    fig.tight_layout()
    _save_fig(fig, str(base), dpi=dpi, save_pdf=save_pdf)
    print(f"  Saved: {base}.csv / .json")


def bond_order_options(args, config):
    """Shared analysis/MQ settings, with explicit CLI values ahead of YAML."""
    from ..cli import _typed

    def option(key, default):
        value = getattr(args, key, None)
        return config.get(key, default) if value is None else value

    cutoff = getattr(args, "cutoff", "auto-rdf")
    if not _typed("--cutoff") and cutoff == "auto-rdf":
        cutoff = config.get("cutoff", cutoff)
    cutoff = option("order_cutoff", cutoff)
    return {"cutoff": cutoff,
            "qbar6_threshold": option("qbar6_threshold", 0.3),
            "min_neighbors": option("order_min_neighbors", 4)}


def run_descriptor_analysis(sa, args, config, override, *, plot_dir=None,
                            report_path=None, plot_kwargs=None):
    """Run and return selected descriptors; CLI options override YAML values."""
    def enabled(key):
        return bool(getattr(args, key, False) or config.get(key, False))

    def option(key, default):
        value = getattr(args, key, None)
        return config.get(key, default) if value is None else value

    selected = [name for name in ("bond_order", "voids", "oxygen_speciation", "elastic", "vdos")
                if enabled(name)]
    if not selected:
        return {}
    results = {}
    calculator = None
    if "elastic" in selected or "vdos" in selected:
        from ..utils import get_calculator
        from ..utils.calculators import potential_kwargs
        calculator = get_calculator(
            model=override.get("model", args.model),
            model_path=override.get("model_path", args.model_path),
            device=override.get("device", args.device),
            default_dtype=override.get("default_dtype", args.default_dtype),
            **potential_kwargs(override))
    for name in selected:
        print(f"\n  Computing {name.replace('_', ' ')}...")
        if name == "bond_order":
            options = bond_order_options(args, config)
            # Without a separate order cutoff, reuse the resolved shells.
            if getattr(args, "order_cutoff", None) is None and "order_cutoff" not in config:
                options["cutoff"] = sa.cutoff
            result = sa.bond_order(**options)
        elif name == "voids":
            result = sa.void_distribution(
                n_samples=option("void_samples", 10000),
                probe_radius=option("void_probe_radius", 0.0),
                probe_radii=option("void_probe_radii", None),
                nbins=option("void_bins", 50), seed=option("void_seed", 0),
                radii=config.get("void_radii"))
        elif name == "oxygen_speciation":
            result = sa.oxygen_speciation(network_formers=option("network_formers", None))
        elif name == "elastic":
            opt = override.get("opt", {})
            result = sa.elastic_moduli(
                calculator=calculator, strain=option("elastic_strain", 0.005),
                relax=enabled("elastic_relax"), fmax=opt.get("fmax", args.fmax),
                steps=opt.get("max_steps", args.opt_steps))
        else:
            result = sa.vibrational_dos(
                calculator=calculator, displacement=option("vdos_displacement", 0.01),
                sigma=option("vdos_sigma", 0.1), npoints=option("vdos_npoints", 400))
        if sa._file_list:
            result["structure_files"] = [str(f) for f in sa._file_list]
        results[name] = result
        report = format_descriptor(name, result)
        print(report)
        if report_path:
            with open(report_path, "a") as handle:
                handle.write(report + "\n")
        if plot_dir:
            settings = {k: v for k, v in (plot_kwargs or {}).items()
                        if k in ("dpi", "save_pdf", "show_title")}
            save_descriptor(name, result, plot_dir, **settings)
    return results
