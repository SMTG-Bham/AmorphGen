"""Retained-input FEFF convergence study and digitised Ga2O3 comparison.

The experimental file must contain k and k**2*chi in its first two columns.
Its arbitrary amplitude is fitted, never treated as an absolute calibration.

The command-line defaults are the validated protocol (docs/guides/exafs.md,
"Validated settings"): RPATH 8 A, NLEG 4, CRITERIA 1.0 0.5, parent-cell
potential populations, and radii 8, 10 and 12 A around the 10 A cluster. The
original study used 5-8 A radii, RPATH 6 A, FEFF's default path filters and
cluster-inferred populations, which are not path-converged; reproduce it with

    --radii 5 6 7 8 --rpath 6 --criteria default --stoichiometry cluster

The legacy_60 tree always uses that original 3-site recipe, as a fixed
reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
from ase.io import read, write
from scipy.optimize import least_squares

import amorphgen
from amorphgen.analysis import feff


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def prepare(args):
    root = args.output.resolve()
    paths = sorted(args.structures.glob(args.pattern))
    if not paths:
        raise ValueError(f"No structures matching {args.pattern} in {args.structures}")
    radii = sorted(set(args.radii))
    if args.max_sites < 3 or any(r <= 0 for r in radii):
        raise ValueError("Need at least three sites per structure and positive radii")
    source_hashes = {p.name: sha(p) for p in paths}
    # Namespaces without these attributes get the original study's settings.
    nleg = int(getattr(args, "nleg", 4))
    criteria = getattr(args, "criteria", None)
    criteria = None if criteria in (None, "default") else criteria
    stoichiometry = getattr(args, "stoichiometry", "cluster")
    config = {
        "sources": source_hashes, "experiment_sha256": sha(args.experiment),
        "radii_A": radii, "max_sites": args.max_sites, "seed": args.seed,
        "rpath_A": args.rpath, "nleg": nleg, "criteria": criteria,
        "stoichiometry": stoichiometry, "s02": 1.0,
        "comparison_k_range_Ainv": [3.0, 12.0],
        "convergence_relative_rms_target": 0.02,
        "target_note": "2% is an operational numerical tolerance, not an experimental uncertainty",
        "feff_module_sha256": sha(feff.__file__),
        "amorphgen_version": amorphgen.__version__,
    }
    if (root / "study.json").exists():
        previous = json.loads((root / "study.json").read_text())
        if previous["config"] != config:
            raise ValueError("Study inputs/settings changed; use a new output directory")
        print(f"Existing identical study: {len(previous['tasks'])} tasks")
        return
    if root.exists() and any(root.iterdir()):
        raise ValueError("Nonempty output without study.json; choose a new directory")
    (root / "inputs").mkdir(parents=True)
    atoms = [read(p) for p in paths]
    if any(sum(a.symbols == "Ga") < args.max_sites for a in atoms):
        raise ValueError("Not enough Ga sites in every structure")
    for p in paths:
        shutil.copy2(p, root / "inputs" / p.name)
    shutil.copy2(args.experiment, root / "inputs/experiment_k2chi.dat")
    write(root / "inputs/ensemble.extxyz", atoms)
    cards = {"RPATH": str(args.rpath), "NLEG": str(nleg)}
    if criteria is not None:
        cards["CRITERIA"] = criteria
    definitions = [(f"r{r:g}", r, args.max_sites, cards, stoichiometry) for r in radii]
    # The original 3-site recipe, unchanged: RPATH 6 A, default filters, cluster populations.
    definitions.append(("legacy_60", 6.0, 3, {"RPATH": "6.0", "NLEG": "4"}, "cluster"))
    tasks, trees, ranks = [], {}, {}
    for name, radius, count, tree_cards, tree_stoichiometry in definitions:
        manifest = feff.write_feff_jobs(
            atoms, "Ga", str(root / name), n_sites=count, radius=radius,
            seed=args.seed, cards=tree_cards, stoichiometry=tree_stoichiometry,
        )
        trees[name] = {"radius_A": radius, "sites_per_structure": count,
                       "manifest_sha256": sha(root / name / "manifest.json")}
        # Random nested ordering, identical at every radius, independent of
        # the sorted atom indices used to name FEFF job directories.
        ordered = {}
        for si in range(len(atoms)):
            sites = [j["site"] for j in manifest["jobs"] if j["structure"] == si]
            ordered[si] = np.random.default_rng(args.seed + 104729 + si).permutation(sites).tolist()
        ranks[name] = {j["dir"]: ordered[j["structure"]].index(j["site"]) for j in manifest["jobs"]}
        for job in manifest["jobs"]:
            tasks.append({"tree": name, "job": job["dir"], "structure": job["structure"], "site": job["site"]})
    save_json(root / "study.json", {
        "config": config, "trees": trees, "tasks": tasks, "site_ranks": ranks,
        "structures": [{"file": p.name, "atoms": len(a),
                        "density_g_cm3": float(a.get_masses().sum()*1.66053906660/a.get_volume())}
                       for p, a in zip(paths, atoms)],
        "sampling_note": "Nested samples per structure; all radii use matched sites. legacy_60 independently uses the original n_sites=3, seed=0 recipe.",
        "experiment_note": "Reis et al. 2023, DOI 10.1002/pssb.202200535, Fig. 1a, digitised arbitrary-amplitude as-deposited a-Ga2O3 curve; CC BY 4.0 per source notes",
    })
    print(f"Prepared {len(tasks)} tasks across {len(atoms)} structures")


def run(args):
    root = args.output.resolve()
    study = json.loads((root / "study.json").read_text())
    if sha(feff.__file__) != study["config"]["feff_module_sha256"]:
        raise ValueError("FEFF implementation changed after preparing the study")
    tasks = study["tasks"]
    if args.pilot:
        chosen = [next(i for i, t in enumerate(tasks) if t["tree"] == name)
                  for name in study["trees"]]
    elif args.task_index is not None:
        start = args.task_index * args.batch_size
        if start >= len(tasks) or start < 0:
            raise ValueError("Task index outside study")
        chosen = range(start, min(start + args.batch_size, len(tasks)))
    else:
        chosen = range(len(tasks))
    (root / "timings").mkdir(exist_ok=True)
    failures = []
    executable = shutil.which(args.executable)
    if not executable:
        raise FileNotFoundError(args.executable)
    for i in chosen:
        task = tasks[i]
        before = time.monotonic()
        outcome = feff.run_feff_jobs(str(root / task["tree"]), jobs=[task["job"]],
                                     executable=executable, timeout=args.timeout)
        record = {**task, "seconds": time.monotonic()-before, "outcome": outcome,
                  "python": platform.python_version(), "host": platform.node(),
                  "executable": executable, "executable_sha256": sha(executable)}
        save_json(root / "timings" / f"{i:06d}.json", record)
        print(json.dumps(record), flush=True)
        failures.extend(outcome["failed"])
    if failures:
        raise SystemExit(f"{len(failures)} FEFF jobs failed")


def relative_rms(k, actual, expected):
    mask = (k >= 3) & (k <= 12)
    weight = k[mask]**2
    denom = np.linalg.norm(expected[mask]*weight)
    if denom == 0:
        raise ValueError("Zero reference spectrum")
    return float(np.linalg.norm((actual[mask]-expected[mask])*weight)/denom)


def fit_curve(k, chi, observed, upper=12):
    kd, y = observed.T
    mask = (kd >= 2.5) & (kd <= upper)
    kd, y = kd[mask], y[mask]
    if len(kd) < 20:
        raise ValueError("Insufficient experimental coverage")
    # All trial energy shifts must remain within actual computed coverage.
    if np.sqrt(kd.min()**2-15*.2625) < k.min() or np.sqrt(kd.max()**2+15*.2625) > k.max():
        raise ValueError("Model lacks coverage for trial energy shifts")

    def model(p):
        scale, de0, sigma2, offset = p
        shifted = np.sqrt(kd**2 - de0*.2625)
        return scale*kd**2*np.interp(shifted, k, chi)*np.exp(-2*sigma2*kd**2)+offset

    base = kd**2*np.interp(kd, k, chi)
    linear = least_squares(lambda p: p[0]*base+p[1]-y, [1, 0], bounds=([0, -np.inf], [np.inf, np.inf]))
    best = None
    for de0 in (-10, 0, 10):
        for sigma2 in (0, .003):
            trial = least_squares(lambda p: model(p)-y, [max(linear.x[0], .01), de0, sigma2, linear.x[1]],
                                  bounds=([0, -15, -.01, -np.inf], [np.inf, 15, .03, np.inf]),
                                  max_nfev=3000, x_scale="jac")
            if best is None or trial.cost < best.cost:
                best = trial
    if not best.success:
        raise ValueError(f"Spectral fit did not converge: {best.message}")
    scale, de0, sigma2, offset = best.x
    fitted = model(best.x)
    unshifted = linear.x[0]*base + linear.x[1]
    denom = np.sum((y-y.mean())**2)
    result = {"k_range_Ainv": [2.5, upper], "n_points": len(kd),
              "scale": float(scale), "delta_E0_eV": float(de0),
              "extra_sigma2_A2": float(sigma2), "offset": float(offset),
              "normalized_squared_residual": float(np.sum((y-fitted)**2)/denom),
              "scale_offset_only_residual": float(np.sum((y-unshifted)**2)/denom),
              "at_bound": bool(np.any(best.active_mask)),
              "note": "Arbitrary-amplitude digitised data; fitted adjustments are not independent validation, no experimental errors or parameter confidence intervals available"}
    return result, np.c_[kd, y, unshifted, fitted]


def analyse(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = args.output.resolve()
    study = json.loads((root / "study.json").read_text())
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    config = study["config"]
    if sha(root / "inputs/experiment_k2chi.dat") != config["experiment_sha256"]:
        raise ValueError("Retained experimental curve changed")
    matrices, spectra, curves = {}, {}, {}
    nstructures = len(study["structures"])
    counts = sorted(set([1, 3, 5, 10, 20, 40, 80, 120, config["max_sites"]]))
    counts = [n for n in counts if n <= config["max_sites"]]
    for tree in study["trees"]:
        path = root / tree
        if sha(path / "manifest.json") != study["trees"][tree]["manifest_sha256"]:
            raise ValueError(f"Manifest changed for {tree}")
        result = feff.collect_feff_results(path, kmin=3, kmax=13.5, dk=1, kweight=2)
        if result["missing"]:
            raise ValueError(f"Incomplete {tree}: {len(result['missing'])} missing spectra")
        feff.write_feff_csv(result, str(out / f"{tree}_all.csv"))
        spectra[tree] = result
        manifest = feff.read_manifest(path)
        k = result["k"]
        by_structure = []
        for si in range(nstructures):
            jobs = [j for j in manifest["jobs"] if j["structure"] == si]
            jobs.sort(key=lambda j: study["site_ranks"][tree][j["dir"]])
            by_structure.append([np.interp(k, *feff.read_chi(path / j["dir"] / "chi.dat")) for j in jobs])
        matrices[tree] = np.asarray(by_structure)
        if tree != "legacy_60":
            curves[tree] = {n: matrices[tree][:, :n].mean(axis=1).mean(axis=0) for n in counts}
    # Every comparison uses exactly the same grid, no per-curve fitting.
    k = spectra[next(iter(curves))]["k"]
    if any(not np.array_equal(result["k"], k) for result in spectra.values()):
        raise ValueError("Unequal shared grids across trees; choose explicit common coverage")
    radii = config["radii_A"]
    largest = f"r{max(radii):g}"
    max_sites = config["max_sites"]
    reference = curves[largest][max_sites]
    rows = [{"radius_A": radius, "sites_per_structure": n, "n_structures": nstructures,
             "relative_rms_to_largest_radius_max_sites": relative_rms(k, curves[f"r{radius:g}"][n], reference)}
            for radius in radii for n in counts]
    np.savetxt(out / "convergence.csv", np.array([[row[key] for key in rows[0]] for row in rows]),
               delimiter=",", header=",".join(rows[0]), comments="")
    # Sampling variability conditional on the finite computed site pool.
    # The full pool has zero self-difference; that is not proof of convergence.
    rng = np.random.default_rng(4729)
    sampling = []
    pool = matrices[largest]
    for n in counts:
        errors = []
        for _ in range(100):
            mean = np.mean([row[rng.choice(len(row), n, replace=False)].mean(axis=0) for row in pool], axis=0)
            errors.append(relative_rms(k, mean, reference))
        sampling.append({"sites_per_structure": n, "median_relative_rms": float(np.median(errors)),
                         "p95_relative_rms": float(np.percentile(errors, 95))})
    radius_last = relative_rms(k, curves[f"r{radii[-2]:g}"][max_sites], reference) if len(radii)>1 else None
    site_last = relative_rms(k, curves[largest][counts[-2]], reference) if len(counts)>1 else None
    last_p95 = sampling[-2]["p95_relative_rms"] if len(sampling)>1 else None
    tol = config["convergence_relative_rms_target"]
    summary = {"radius_last_increment_relative_rms": radius_last,
               "sites_last_increment_relative_rms": site_last,
               "subsampling_previous_count_p95": last_p95,
               "operational_target": tol,
               "radius_increment_below_target": radius_last is not None and radius_last < tol,
               "site_increment_and_p95_below_target": site_last is not None and site_last < tol and last_p95 < tol,
               "conditional_site_resampling": sampling,
               "interpretation": "Finite-study convergence diagnostics, not proof of physical accuracy or population convergence. RPATH fixed at 6 A by default; path cutoff/NLEG/thermal effects not converged here."}
    save_json(out / "convergence.json", summary)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for radius in radii:
        axes[0].plot(counts, [100*relative_rms(k, curves[f"r{radius:g}"][n], reference) for n in counts],
                     "o-", label=f"{radius:g} Å cluster")
    axes[0].axhline(100*tol, color="k", ls=":", label="2% operational target")
    axes[0].set(xlabel="Absorbers per structure", ylabel="Relative RMS to largest computed model (%)")
    axes[0].legend(fontsize=8)
    axes[1].plot(counts, [100*s["median_relative_rms"] for s in sampling], "o-", label="Median")
    axes[1].plot(counts, [100*s["p95_relative_rms"] for s in sampling], "o-", label="95th percentile")
    axes[1].axhline(100*tol, color="k", ls=":")
    axes[1].set(xlabel="Absorbers per structure", ylabel="Subsampling deviation from finite pool (%)")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle(f"Ga₂O₃: {nstructures} structures; raw k²χ(k), k = 3–12 Å⁻¹; no fitted alignment")
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=200)
    fig.savefig(out / "convergence.svg")
    plt.close(fig)

    observed = np.loadtxt(root / "inputs/experiment_k2chi.dat")[:, :2]
    fits = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for tree, label in (("legacy_60", "Original 3-site recipe"), (largest, f"{max_sites} sites/structure, {max(radii):g} Å")):
        result = spectra[tree]
        fit, values = fit_curve(result["k"], result["chi"], observed, upper=12)
        fits[tree] = fit
        historical, _ = fit_curve(result["k"], result["chi"], observed, upper=13)
        fits[tree]["historical_range_2p5_to_13"] = historical
        np.savetxt(out / f"{tree}_experiment_comparison.csv", values, delimiter=",",
                   header="k,experiment_k2chi,scale_offset_only_k2chi,adjusted_model_k2chi", comments="")
        kd, y, unshifted, adjusted = values.T
        if tree == "legacy_60":
            axes[0].plot(kd, y, "k", lw=2, label="Reis 2023, digitised experiment")
            axes[1].plot(kd, y, "k", lw=2, label="Experiment")
        axes[0].plot(kd, unshifted, lw=1.2, label=label)
        axes[1].plot(kd, adjusted, lw=1.2, label=label)
    axes[0].set_title("Scale and offset only")
    axes[1].set_title("Also fitted: energy shift and extra damping")
    for ax in axes:
        ax.set(xlabel="k (Å⁻¹)", ylabel="k²χ(k) (arbitrary units)", xlim=(2.5, 12))
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle("Fresh FEFF8.5L comparison with a-Ga₂O₃ experiment; arbitrary amplitude")
    fig.tight_layout()
    fig.savefig(out / "experiment_comparison.png", dpi=200)
    fig.savefig(out / "experiment_comparison.svg")
    plt.close(fig)
    save_json(out / "fits.json", fits)
    # Transform experiment and adjusted models on the same observed grid.
    # The extracted experimental display offset is already removed. Remove
    # the fitted model offset before transforming, as in the historical recipe.
    take = (observed[:, 0] >= 2) & (observed[:, 0] <= 12)
    kd, yd = observed[take].T
    ft_data = feff.chi_fourier_transform(kd, yd/kd**2, kmin=2.5, kmax=11.5, dk=1, kweight=2)
    keep = ft_data["r"] <= 5
    norm = float(ft_data["chir_mag"][keep].max())
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(ft_data["r"][keep], ft_data["chir_mag"][keep]/norm, "k", lw=2, label="Experiment")
    export = [ft_data["r"], ft_data["chir_mag"]/norm]
    headers = ["R_A", "experiment_magnitude_normalized"]
    for tree in ("legacy_60", largest):
        fit = fits[tree]
        shifted = np.sqrt(kd**2-fit["delta_E0_eV"]*.2625)
        model = fit["scale"]*np.interp(shifted, spectra[tree]["k"], spectra[tree]["chi"])*np.exp(-2*fit["extra_sigma2_A2"]*kd**2)
        transformed = feff.chi_fourier_transform(kd, model, kmin=2.5, kmax=11.5, dk=1, kweight=2)
        export.append(transformed["chir_mag"]/norm)
        headers.append(tree+"_magnitude_normalized")
        ax.plot(transformed["r"][keep], transformed["chir_mag"][keep]/norm, label=tree+" (adjusted)")
    np.savetxt(out / "experiment_fourier.csv", np.array(export).T, delimiter=",", header=",".join(headers), comments="")
    save_json(out / "experiment_fourier_settings.json", {
        "kmin": ft_data["kmin"], "kmax": ft_data["kmax"], "dk": ft_data["dk"],
        "kweight": 2, "normalization": "All curves divided by experimental peak magnitude over R=0..5 A",
        "offset": "Fitted model constant removed before transform; digitised experimental display offset already removed",
    })
    ax.set(xlabel="R (Å), phase-uncorrected", ylabel="Fourier magnitude / experimental peak",
           title="Ga₂O₃: common-window Fourier comparison", xlim=(0, 5))
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "experiment_fourier.png", dpi=200)
    fig.savefig(out / "experiment_fourier.svg")
    plt.close(fig)
    print(json.dumps({"convergence": summary, "fits": fits}, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("prepare", "run", "analyse"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--structures", type=Path)
    parser.add_argument("--pattern", default="*.vasp")
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--radii", type=float, nargs="+", default=[8, 10, 12])
    parser.add_argument("--max-sites", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rpath", type=float, default=8)
    parser.add_argument("--nleg", type=int, default=4)
    parser.add_argument("--criteria", default="1.0 0.5",
                        help='FEFF CRITERIA card; "default" leaves FEFF\'s own filters')
    parser.add_argument("--stoichiometry", choices=("structure", "cluster"), default="structure")
    parser.add_argument("--executable", default="feff")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--pilot", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.stage == "prepare" and (args.structures is None or args.experiment is None):
        parser.error("prepare requires --structures and --experiment")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    {"prepare": prepare, "run": run, "analyse": analyse}[args.stage](args)


if __name__ == "__main__":
    main()
