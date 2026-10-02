"""CLI reporting and export for optional material descriptors."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def format_descriptor(name, result):
    """Return a compact, unit-labelled report for a computed descriptor."""
    if name == "voids":
        return ("\n  Void distribution (sampled point clearance):\n"
                f"    Accessible fraction: {result['accessible_fraction']:.6f} "
                f"+/- {result['accessible_fraction_stderr']:.6f} (sampling stderr)\n"
                f"    Probe radius: {result['probe_radius']:.4f} A; "
                f"radii: {result['radius_source']}\n"
                f"    Mean accessible volume: {result['accessible_volume']:.4f} A^3")
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


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__}")


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

    fig, ax = _figure(figsize=(6.0, 4.0))
    if name == "voids":
        write_csv(base.with_suffix(".csv"),
                  ["clearance_A", "probability_density_per_A", "bin_volume_fraction"],
                  zip(result["radius"], result["probability_density"],
                      result["bin_volume_fraction"]))
        ax.plot(result["radius"], result["probability_density"], color=_PALETTE[0])
        ax.set(xlabel="Point clearance (Å)", ylabel="Accessible-space density (Å⁻¹)")
    elif name == "oxygen_speciation":
        labels = list(result["counts"])
        write_csv(base.with_suffix(".csv"), ["species", "count", "fraction"],
                  ((k, result["counts"][k], result["fractions"][k]) for k in labels))
        ax.bar(range(len(labels)), [100 * result["fractions"][k] for k in labels],
               color=_PALETTE[0])
        ax.set_xticks(range(len(labels)), [k.replace("_", "\n") for k in labels])
        ax.set_ylabel("Oxygen fraction (%)")
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


def run_descriptor_analysis(sa, args, config, override, *, plot_dir=None,
                            report_path=None, plot_kwargs=None):
    """Run explicitly selected descriptors; CLI options override YAML values."""
    def enabled(key):
        return bool(getattr(args, key, False) or config.get(key, False))

    def option(key, default):
        value = getattr(args, key, None)
        return config.get(key, default) if value is None else value

    selected = [name for name in ("voids", "oxygen_speciation", "elastic", "vdos")
                if enabled(name)]
    if not selected:
        return
    calculator = None
    if "elastic" in selected or "vdos" in selected:
        from ..utils import get_calculator
        calculator = get_calculator(
            model=override.get("model", args.model),
            model_path=override.get("model_path", args.model_path),
            device=override.get("device", args.device),
            default_dtype=override.get("default_dtype", args.default_dtype),
            **({"classical_params": override["classical_params"]}
               if "classical_params" in override else {}))
    for name in selected:
        print(f"\n  Computing {name.replace('_', ' ')}...")
        if name == "voids":
            result = sa.void_distribution(
                n_samples=option("void_samples", 10000),
                probe_radius=option("void_probe_radius", 0.0),
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
        report = format_descriptor(name, result)
        print(report)
        if report_path:
            with open(report_path, "a") as handle:
                handle.write(report + "\n")
        if plot_dir:
            settings = {k: v for k, v in (plot_kwargs or {}).items()
                        if k in ("dpi", "save_pdf", "show_title")}
            save_descriptor(name, result, plot_dir, **settings)
