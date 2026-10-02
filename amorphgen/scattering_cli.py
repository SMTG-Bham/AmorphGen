"""CLI options and reporting for measured scattering curves and XRD."""

from __future__ import annotations


def add_scattering_arguments(group):
    """Register experimental comparison options on the analysis group."""
    group.add_argument("--experiment-sq", metavar="FILE",
                       help="Measured S(q): q [1/A], S(q), optional sigma. "
                            "Enables --sq and reports agreement with ensemble bands.")
    group.add_argument("--experiment-tr", metavar="FILE",
                       help="Measured T(r): r [A], T(r) [1/A^2], optional sigma. "
                            "Enables --tr; match --tr-qrange and --tr-window.")
    group.add_argument("--experiment-skiprows", type=int, default=0, metavar="N",
                       help="Header lines to skip in experimental files (default 0).")
    group.add_argument("--experiment-columns", type=int, nargs="+", metavar="COL",
                       help="Two or three zero-based columns: x, value, optional sigma. "
                            "Default: exactly two or three columns; CSV/whitespace autodetected.")
    for kind, unit in (("sq", "1/A"), ("tr", "A")):
        group.add_argument(f"--{kind}-fit-range", type=float, nargs=2,
                           metavar=("MIN", "MAX"),
                           help=f"Inclusive experimental {kind} comparison range in {unit}.")
    group.add_argument("--sq-qmax", type=float, default=15.0, metavar="QMAX",
                       help="Maximum calculated S(q) momentum transfer in 1/A (default 15).")
    group.add_argument("--sq-nq", type=int, default=300, metavar="N",
                       help="Number of S(q) points/bins (default 300).")
    group.add_argument("--xrd", action="store_true",
                       help="Coherent X-ray intensity per atom versus 2theta; "
                            "writes analysis_xrd JSON/CSV/PNG with --save-plot.")
    group.add_argument("--xrd-wavelength", type=float, default=1.5406, metavar="A",
                       help="X-ray wavelength in Angstrom (default 1.5406, Cu K-alpha).")
    group.add_argument("--xrd-qmax", type=float, metavar="QMAX",
                       help="XRD q maximum in 1/A; default limited by wavelength.")
    group.add_argument("--xrd-nq", type=int, default=300, metavar="N",
                       help="Number of reciprocal shells for XRD (default 300).")


def run_scattering_comparisons(sa, args, config, typed, *, sq=None, tr=None,
                               plot_dir=None, report_path=None, dpi=300,
                               save_pdf=False):
    """Reuse calculated curves; append metrics and optionally save artifacts."""
    from .analysis.experiment import load_experiment, compare_experiment, format_experiment_report
    from .analysis.scattering_output import save_experiment_comparison, save_xrd_pattern

    def option(name):
        return (getattr(args, name) if typed("--" + name.replace("_", "-"))
                else config.get(name, getattr(args, name)))

    def report(text):
        print(text)
        if report_path:
            with open(report_path, "a") as handle:
                handle.write("\n" + text + "\n")

    for kind, calculated in (("sq", sq), ("tr", tr)):
        path = option(f"experiment_{kind}")
        if not path:
            continue
        if calculated is None:
            raise ValueError(f"cannot compare experimental {kind}: calculation unavailable")
        experiment = load_experiment(path, kind=kind,
                                     skiprows=option("experiment_skiprows"),
                                     columns=option("experiment_columns"))
        result = compare_experiment(calculated, experiment,
                                     x_range=option(f"{kind}_fit_range"))
        result["structure_files"] = list(sa._file_list)
        result["calculation"] = calculated.get("calculation", {})
        report(format_experiment_report(result))
        if plot_dir:
            save_experiment_comparison(result, output_dir=plot_dir,
                                       dpi=dpi, save_pdf=save_pdf)

    if args.xrd or config.get("xrd", False):
        result = sa.xrd_pattern(wavelength=option("xrd_wavelength"),
                                qmax=option("xrd_qmax"), nq=option("xrd_nq"))
        result["structure_files"] = list(sa._file_list)
        report(f"  XRD: coherent intensity per atom; wavelength = "
               f"{result['wavelength']:g} A, {len(result['q'])} q bins. "
               "No instrument or sample corrections.")
        if plot_dir:
            save_xrd_pattern(result, output_dir=plot_dir, dpi=dpi, save_pdf=save_pdf)
