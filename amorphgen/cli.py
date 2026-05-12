"""
amorphgen.cli
--------------
Command-line interface for AmorphGen.

Examples
--------
Full pipeline with MACE (default):
    amorphgen POSCAR

Use CHGNet:
    amorphgen POSCAR --model chgnet --device cpu

Use a custom fine-tuned model:
    amorphgen POSCAR --model-path /data/InO_finetuned.model

List all available models:
    amorphgen --list-models

Random structure generation:
    amorphgen --random-gen --composition "In2O3*16" --relax
    amorphgen --random-gen --composition In=32,O=48 --target-density 5.5

Random generation with custom minsep:
    amorphgen --random-gen --composition "In2O3*8" --target-density 5.5 \
        --minsep In-In=2.8,In-O=1.9,O-O=2.5

Optimise with cubic cell constraint:
    amorphgen POSCAR --stages 1 --cell-filter cubic --model mace-mpa-0-medium

Batch quench from snapshots:
    amorphgen --batch-quench --snapshot-dir snapshots/ --n-runs 20

Convert structure files between formats (xyz/extxyz/vasp/cif):
    amorphgen --convert snapshots/ --format vasp -o snapshots_vasp/
    amorphgen --convert traj_frame.xyz --format cif
    amorphgen --config convert.yaml          # YAML-driven (uses convert: block)
"""

from __future__ import annotations

import argparse
import sys
import os


def _get_parser():
    """Build and return the argument parser (without parsing)."""
    p = argparse.ArgumentParser(
        description="AmorphGen: amorphous structure generation via melt-quench MD and random placement with universal MLIPs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    return _add_arguments(p)


def _add_arguments(p):
    """Add all arguments to the parser and return it."""

    # ── Positional + global ───────────────────────────────────────────────────
    p.add_argument("input_file", nargs="?",
                   help="Input structure (POSCAR/xyz/cif). Pipeline mode only.")
    p.add_argument("--config", default=None, metavar="FILE",
                   help="YAML config file (CLI overrides YAML).")
    p.add_argument("-o", "--work-dir", default=None,
                   help="Output directory (auto-named per mode).")
    p.add_argument("--format", default="xyz", choices=["xyz", "vasp", "cif"],
                   help="Output format for generated/optimised structures.")
    p.add_argument("--resume", action="store_true",
                   help="Skip completed work in --random-gen / --batch-quench "
                        "/ pipeline modes.")

    # ── Mode selectors ────────────────────────────────────────────────────────
    g_mode = p.add_argument_group(
        "modes",
        "Pick one mode (default: full melt-quench pipeline if input_file given).")
    g_mode.add_argument("--random-gen", action="store_true",
                        help="Generate random structures.")
    g_mode.add_argument("--batch-quench", action="store_true",
                        help="Quench multiple snapshots independently.")
    g_mode.add_argument("--batch-opt", action="store_true",
                        help="Optimise all structures in --input-dir.")
    g_mode.add_argument("--analyse", action="store_true",
                        help="Analyse structures (RDF, CN, angles, density).")
    g_mode.add_argument("--rank-from-log", default=None, metavar="LOG",
                        help="Parse a random-gen log and rank by total energy.")
    g_mode.add_argument("--extract-snapshots", default=None, metavar="TRAJ",
                        help="Extract N uniform snapshots from a trajectory "
                             "file (use with --n-runs N --select {uniform,last}).")
    g_mode.add_argument("--mq-ensemble", action="store_true",
                        help="Full MQ-ensemble workflow: stages 1-4 from a "
                             "crystalline input, extract N snapshots from "
                             "stage 4, run stages 5-6-7 on each independently. "
                             "Use with input_file, --n-structures N.")
    g_mode.add_argument("--hybrid-ensemble", action="store_true",
                        help="Hybrid ensemble: starts from a directory of "
                             "disordered structures (e.g. --random-gen "
                             "outputs), runs stages 4-5-6-7 on each. "
                             "Use with --input-dir DIR.")
    g_mode.add_argument("--list-models", action="store_true",
                        help="Print available foundation models and exit.")
    g_mode.add_argument("--convert", default=None, metavar="PATH",
                        help="Convert one structure file or every "
                             "ASE-readable file in a directory to the format "
                             "given by --format. Output goes to --work-dir "
                             "(defaults to <PATH>_<format>/). VASP outputs "
                             "are sorted by species.")

    # ── Calculator ────────────────────────────────────────────────────────────
    g_calc = p.add_argument_group("calculator")
    model_group = g_calc.add_mutually_exclusive_group()
    model_group.add_argument("-m", "--model", default="mace-mpa-0", metavar="NAME",
                             help="Foundation model (mace-mpa-0, chgnet, sevennet, ...).")
    model_group.add_argument("--model-path", default=None, metavar="PATH",
                             help="Path to a local .model file.")
    g_calc.add_argument("-d", "--device", default="auto",
                        choices=["auto", "cuda", "cpu", "mps"],
                        help="Device.")
    g_calc.add_argument("--default-dtype", default="float64",
                        choices=["float32", "float64"],
                        help="MLIP precision (float32 ~2x faster MD).")

    # ── Optimisation (stages 1, 7; also random-gen --relax) ───────────────────
    g_opt = p.add_argument_group("optimisation")
    g_opt.add_argument("-f", "--fmax", type=float, default=0.01,
                       help="Force convergence (eV/A).")
    g_opt.add_argument("--opt-steps", type=int, default=1000,
                       help="Max optimisation steps.")
    g_opt.add_argument("-O", "--optimizer", default="LBFGS",
                       choices=["LBFGS", "FIRE", "BFGSLineSearch", "BFGS", "MDMin"],
                       help="Optimizer.")
    g_opt.add_argument("-C", "--cell-filter", default="FrechetCellFilter",
                       choices=["FrechetCellFilter", "UnitCellFilter",
                                "ExpCellFilter", "StrainFilter", "cubic", "none"],
                       help="Cell filter ('cubic' = isotropic V; 'none' = fixed cell).")

    # ── Melt-quench pipeline ──────────────────────────────────────────────────
    g_pipe = p.add_argument_group(
        "pipeline (melt-quench MD)",
        "Stages: 1=opt, 2=eq-premelt, 3=melt, 4=eq-high, 5=quench, "
        "6=eq-low, 7=final-opt.")
    g_pipe.add_argument("--stages", nargs="+", type=int,
                        default=[1, 2, 3, 4, 5, 6, 7], metavar="N",
                        help="Stages to run.")
    g_pipe.add_argument("--timestep", type=float, default=0.5,
                        help="MD timestep in fs (applies to all MD stages).")
    # Stage 2
    g_pipe.add_argument("--eq-premelt-ensemble", default="NVT",
                        choices=["NVT", "NPT"], help="Stage 2 ensemble.")
    g_pipe.add_argument("--eq-premelt-T", type=int, default=300,
                        help="Stage 2 T (K).")
    g_pipe.add_argument("--eq-premelt-steps", type=int, default=50000,
                        help="Stage 2 MD steps.")
    # Stage 3
    g_pipe.add_argument("--melt-ensemble", default="NPT",
                        choices=["NVT", "NPT"], help="Stage 3 ensemble.")
    g_pipe.add_argument("--melt-T-start", type=int, default=300,
                        help="Stage 3 T_start (K).")
    g_pipe.add_argument("--melt-T-end", type=int, default=3000,
                        help="Stage 3 T_end (K).")
    g_pipe.add_argument("--melt-T-step", type=int, default=100,
                        help="Stage 3 ramp segment (K).")
    g_pipe.add_argument("--melt-steps-per-T", type=int, default=1000,
                        help="Stage 3 steps per segment.")
    # Stage 4
    g_pipe.add_argument("--eq-high-ensemble", default="NVT",
                        choices=["NVT", "NPT"], help="Stage 4 ensemble.")
    g_pipe.add_argument("--eq-high-T", type=int, default=3000,
                        help="Stage 4 T (K).")
    g_pipe.add_argument("--eq-high-steps", type=int, default=10000,
                        help="Stage 4 MD steps.")
    # Stage 5
    g_pipe.add_argument("--quench-ensemble", default="NVT",
                        choices=["NVT", "NPT"], help="Stage 5 ensemble.")
    g_pipe.add_argument("--quench-T-start", type=int, default=3000,
                        help="Stage 5 T_start (K).")
    g_pipe.add_argument("--quench-T-end", type=int, default=300,
                        help="Stage 5 T_end (K).")
    g_pipe.add_argument("--quench-T-step", type=int, default=-100,
                        help="Stage 5 ramp segment (K, negative = cooling).")
    g_pipe.add_argument("--quench-steps-per-T", type=int, default=1000,
                        help="Stage 5 steps per segment.")
    # Stage 6
    g_pipe.add_argument("--eq-low-ensemble", default="NVT",
                        choices=["NVT", "NPT"], help="Stage 6 ensemble.")
    g_pipe.add_argument("--eq-low-T", type=int, default=300,
                        help="Stage 6 T (K).")
    g_pipe.add_argument("--eq-low-steps", type=int, default=10000,
                        help="Stage 6 MD steps.")

    # ── Random generation ─────────────────────────────────────────────────────
    g_rand = p.add_argument_group("random-gen", "Used with --random-gen.")
    g_rand.add_argument("--composition", default=None, metavar="SPEC",
                        help='"In2O3*16" (formula*N units) or "In=32,O=48".')
    g_rand.add_argument("-n", "--n-structures", type=int, default=1,
                        help="Number of structures.")
    g_rand.add_argument("--relax", action="store_true",
                        help="Relax each generated structure.")
    g_rand.add_argument("--target-density", type=float, default=None,
                        help="Target density (g/cm3); auto if omitted.")
    g_rand.add_argument("--density-scale", type=float, default=1.0,
                        metavar="FACTOR",
                        help="Multiplier on the auto-estimated density "
                             "(default 1.0). Use ~1.2 for tight-network "
                             "compositions (a-Si, a-Ge, metallic glasses) "
                             "where sphere-packing underestimates by 15-25%%. "
                             "Ignored when --target-density is set.")
    g_rand.add_argument("--minsep", default=None, metavar="SPEC",
                        help="Per-pair min separations, e.g. In-In=2.8,In-O=1.9.")
    g_rand.add_argument("--target-cn", default=None, metavar="SPEC",
                        help="Target CNs for coordination-aware placement, "
                             "e.g. Si=4,O=2 (auto-detected if omitted).")
    g_rand.add_argument("--no-sc", action="store_true",
                        help="Disable coordination-aware placement.")
    g_rand.add_argument("--dmax", default=None, metavar="SPEC",
                        help="Per-pair bonding cutoffs (auto: minsep * dmax-factor).")
    g_rand.add_argument("--dmax-factor", type=float, default=1.5,
                        help="Auto dmax multiplier.")
    g_rand.add_argument("--cn-tolerance", type=int, default=None,
                        help="Over-coordination tolerance for "
                             "coordination-aware placement (0 or 1).")
    g_rand.add_argument("--repair-iters", type=int, default=0,
                        metavar="N",
                        help="EXPERIMENTAL: post-placement repair pass for "
                             "under-coordinated atoms; N is the max number of "
                             "single-atom relocation proposals (default 0 = "
                             "off).  Useful for tetrahedral covalent networks "
                             "(a-Si, a-Ge) where greedy placement leaves many "
                             "atoms below target CN.")
    g_rand.add_argument("--max-attempts", type=int, default=500000,
                        help="Max placement attempts per atom.")

    # ── Batch quench ──────────────────────────────────────────────────────────
    g_bq = p.add_argument_group("batch-quench", "Used with --batch-quench.")
    g_bq.add_argument("--snapshot-dir", default="snapshots", metavar="PATH",
                      help="Directory of structures, or a single trajectory "
                           "file (.xyz/.extxyz) — file gets auto-extracted "
                           "into N uniform snapshots.")
    g_bq.add_argument("--n-runs", type=int, default=20,
                      help="Number of quench runs.")
    g_bq.add_argument("--select", default="uniform",
                      choices=["uniform", "last"], help="Snapshot selection.")
    g_bq.add_argument("--burn-in-frames", type=int, default=0, metavar="N",
                      help="Discard the first N frames of the trajectory "
                           "before sampling snapshots (useful for skipping "
                           "the non-equilibrated portion of stage-4 MD; "
                           "default: 0).")
    g_bq.add_argument("--batch-stages", nargs="+", type=int,
                      default=[5, 6, 7], metavar="N",
                      help="Stages to run per snapshot.")

    # ── Batch optimisation ────────────────────────────────────────────────────
    g_bo = p.add_argument_group("batch-opt", "Used with --batch-opt.")
    g_bo.add_argument("--input-dir", default=None, metavar="DIR",
                     help="Directory of structures to optimise "
                          "(also used by --analyse).")

    # ── Analysis ──────────────────────────────────────────────────────────────
    g_an = p.add_argument_group("analyse", "Used with --analyse.")
    g_an.add_argument("--cutoff", default="auto",
                      help="Cutoff: number (A), 'auto', or 'auto-rdf'.")
    g_an.add_argument("--per-structure", action="store_true",
                      help="Per-structure comparison table.")
    g_an.add_argument("--save-report", default=None, metavar="FILE",
                      help="Write text report.")
    g_an.add_argument("--save-plot", default=None, metavar="DIR",
                      help="Save RDF/CN/angles plots (PNG + CSV).")
    g_an.add_argument("--save-pdf", action="store_true",
                      help="Also save plots as vector PDF.")
    g_an.add_argument("--dpi", type=int, default=None, metavar="N",
                      help="Plot DPI (default 300).")
    g_an.add_argument("--show-title", action="store_true",
                      help="Add titles to plots (off by default).")
    g_an.add_argument("--total-rdf", action="store_true",
                      help="Include total g(r) in RDF plot.")
    g_an.add_argument("--smearing", type=float, default=0.0, metavar="SIGMA",
                      help="RDF Gaussian smearing (A); use 0.02-0.05 vs exp.")
    g_an.add_argument("--reference", default=None, metavar="YAML",
                      help="Reference YAML; adds validation table.")

    return p


def parse_args():
    """Parse command-line arguments."""
    return _get_parser().parse_args()


def _parse_composition(spec: str) -> dict[str, int]:
    """Parse composition from two supported formats.

    Format 1 (atom counts):  'In=32,O=48'  -> {'In': 32, 'O': 48}
    Format 2 (formula * N):  'In2O3*16'    -> {'In': 32, 'O': 48}
                             'SiO2*16'     -> {'Si': 16, 'O': 32}
                             'Si64'        -> {'Si': 64}

    The '*N' multiplier scales a chemical formula by N formula units.
    If no '*' and no '=' is present, the formula is used as-is.

    Raises ValueError with a helpful message on malformed input.
    """
    from ase.data import chemical_symbols as _chem_syms
    spec = spec.strip()
    if not spec:
        raise ValueError("Empty composition string.")

    # --- Format 2: chemical formula (no '=' signs) ---
    if "=" not in spec:
        from ase import Atoms as _Atoms
        # Split on '*' for multiplier: e.g. 'In2O3*16'
        if "*" in spec:
            parts = spec.split("*", 1)
            formula = parts[0].strip()
            try:
                multiplier = int(parts[1].strip())
            except ValueError:
                raise ValueError(
                    f"Invalid multiplier in '{spec}'. "
                    f"Expected 'Formula*N' (e.g. 'In2O3*16')."
                )
            if multiplier <= 0:
                raise ValueError(
                    f"Multiplier must be positive, got {multiplier}."
                )
        else:
            formula = spec
            multiplier = 1

        try:
            tmp = _Atoms(formula * multiplier)
        except Exception:
            raise ValueError(
                f"Cannot parse '{formula}' as a chemical formula.\n"
                f"  Accepted formats:\n"
                f"    In=32,O=48      (atom counts)\n"
                f"    In2O3*16        (formula * N formula units)\n"
                f"    SiO2*16         (formula * N)\n"
                f"    Si64            (element + count)"
            )
        syms = tmp.get_chemical_symbols()
        comp = {}
        for s in syms:
            comp[s] = comp.get(s, 0) + 1

        total = sum(comp.values())
        formula_str = tmp.get_chemical_formula(mode="hill")
        print(f"  [Composition] {formula_str}: {comp} ({total} atoms)")
        return comp

    # --- Format 1: Element=count pairs ---
    comp = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"Invalid composition entry: '{part}'. "
                f"Expected 'Element=count' (e.g. 'Si=16,O=32').\n"
                f"  Or use formula format: 'SiO2*16'"
            )
        pieces = part.split("=", 1)
        sym = pieces[0].strip()
        count_str = pieces[1].strip()
        if not sym:
            raise ValueError(f"Empty element symbol in: '{part}'")
        if sym not in _chem_syms:
            raise ValueError(
                f"Unknown element symbol '{sym}'. "
                f"Check spelling (case-sensitive: 'O' not 'o', 'In' not 'IN')."
            )
        try:
            count = int(count_str)
        except ValueError:
            raise ValueError(
                f"Invalid count for '{sym}': '{count_str}' (must be an integer)."
            )
        if count <= 0:
            raise ValueError(
                f"Count for '{sym}' must be positive, got {count}."
            )
        comp[sym] = count
    if not comp:
        raise ValueError(f"No valid entries in composition: '{spec}'")
    return comp


def _classical_kwargs(override: dict) -> dict:
    """Extract classical_params from config override if present."""
    kw = {}
    if override.get("classical_params"):
        kw["classical_params"] = override["classical_params"]
    return kw


def _parse_minsep(spec: str) -> dict[str, float]:
    """Parse 'In-In=2.8,In-O=1.9,O-O=2.5' -> {'In-In': 2.8, ...}.

    Raises ValueError on malformed input.
    """
    minsep = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"Invalid minsep entry: '{part}'. "
                f"Expected 'A-B=distance' (e.g. 'Si-O=1.6')."
            )
        pair, val_str = part.split("=", 1)
        pair = pair.strip()
        if "-" not in pair:
            raise ValueError(
                f"Invalid pair format: '{pair}'. Expected 'A-B' (e.g. 'Si-O')."
            )
        try:
            val = float(val_str.strip())
        except ValueError:
            raise ValueError(
                f"Invalid minsep value for '{pair}': '{val_str.strip()}'."
            )
        if val <= 0:
            raise ValueError(
                f"Minsep for '{pair}' must be positive, got {val}."
            )
        minsep[pair] = val
    return minsep


def _parse_target_cn(spec: str) -> dict[str, int]:
    """Parse 'Si=4,O=2' -> {'Si': 4, 'O': 2}.

    Raises ValueError on malformed input.
    """
    target_cn = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"Invalid target-cn entry: '{part}'. "
                f"Expected 'Element=CN' (e.g. 'Si=4,O=2')."
            )
        sym, cn_str = part.split("=", 1)
        sym = sym.strip()
        try:
            cn = int(cn_str.strip())
        except ValueError:
            raise ValueError(
                f"Invalid CN for '{sym}': '{cn_str.strip()}' (must be an integer)."
            )
        if cn <= 0:
            raise ValueError(
                f"CN for '{sym}' must be positive, got {cn}."
            )
        target_cn[sym] = cn
    return target_cn


def _parse_dmax(spec: str) -> dict[str, float]:
    """Parse 'Si-O=2.0,Si-Si=3.2' -> {'Si-O': 2.0, ...}.

    Raises ValueError on malformed input.
    """
    dmax = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"Invalid dmax entry: '{part}'. "
                f"Expected 'A-B=distance' (e.g. 'Si-O=2.0')."
            )
        pair, val_str = part.split("=", 1)
        pair = pair.strip()
        if "-" not in pair:
            raise ValueError(
                f"Invalid pair format: '{pair}'. Expected 'A-B' (e.g. 'Si-O')."
            )
        try:
            val = float(val_str.strip())
        except ValueError:
            raise ValueError(
                f"Invalid dmax value for '{pair}': '{val_str.strip()}'."
            )
        if val <= 0:
            raise ValueError(
                f"Dmax for '{pair}' must be positive, got {val}."
            )
        dmax[pair] = val
    return dmax


def _build_override(args, parser, explicit_only: bool = False) -> dict:
    """
    Build a config override dict from CLI args.

    Parameters
    ----------
    args : argparse.Namespace
    parser : argparse.ArgumentParser
    explicit_only : bool
        If True, only include args the user explicitly set (for YAML mode,
        so defaults don't overwrite YAML values). If False, include all
        args with their defaults (for non-YAML mode).

    Returns
    -------
    dict — config override ready to merge with DEFAULT_CONFIG or YAML.
    """
    if explicit_only:
        explicit = {k for k, v in vars(args).items()
                    if v != parser.get_default(k)}
        def get(key):
            return getattr(args, key) if key in explicit else None
    else:
        def get(key):
            return getattr(args, key)

    mapping = {
        "model": get("model"),
        "model_path": get("model_path"),
        "device": get("device"),
        "default_dtype": get("default_dtype"),
        "opt": {
            "fmax": get("fmax"),
            "max_steps": get("opt_steps"),
            "optimizer": get("optimizer"),
            "cell_filter": get("cell_filter"),
            "output_format": get("format"),
        },
        "eq_premelt": {
            "ensemble": get("eq_premelt_ensemble"),
            "T": get("eq_premelt_T"),
            "steps": get("eq_premelt_steps"),
            "timestep": get("timestep"),
        },
        "melt": {
            "ensemble": get("melt_ensemble"),
            "T_start": get("melt_T_start"),
            "T_end": get("melt_T_end"),
            "T_step": get("melt_T_step"),
            "steps_per_T": get("melt_steps_per_T"),
            "timestep": get("timestep"),
        },
        "eq_high": {
            "ensemble": get("eq_high_ensemble"),
            "T": get("eq_high_T"),
            "steps": get("eq_high_steps"),
            "timestep": get("timestep"),
        },
        "quench": {
            "ensemble": get("quench_ensemble"),
            "T_start": get("quench_T_start"),
            "T_end": get("quench_T_end"),
            "T_step": get("quench_T_step"),
            "steps_per_T": get("quench_steps_per_T"),
            "timestep": get("timestep"),
        },
        "eq_low": {
            "ensemble": get("eq_low_ensemble"),
            "T": get("eq_low_T"),
            "steps": get("eq_low_steps"),
            "timestep": get("timestep"),
        },
        "final_opt": {
            "fmax": get("fmax"),
            "max_steps": get("opt_steps"),
            "optimizer": get("optimizer"),
            "cell_filter": get("cell_filter"),
            "output_format": get("format"),
        },
    }

    if explicit_only:
        # Strip None values so they don't overwrite YAML/defaults
        def _strip_none(d):
            out = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    nested = _strip_none(v)
                    if nested:
                        out[k] = nested
                elif v is not None:
                    out[k] = v
            return out
        return _strip_none(mapping)

    return mapping


# ═════════════════════════════════════════════════════════════════════════════
# Convert mode (--convert)
# ═════════════════════════════════════════════════════════════════════════════

def _run_convert(args, yaml_cfg: dict | None = None) -> None:
    """Thin CLI wrapper around :func:`amorphgen.utils.convert.convert`.

    Resolves the (input_path, output_format, output_dir) tuple from CLI
    flags first, falling back to a ``convert:`` block in the YAML config
    when the CLI flag is at its default (i.e. user did not set it).
    """
    from .utils import convert as _convert

    yaml_block = (yaml_cfg or {}).get("convert", {}) if yaml_cfg else {}

    # Detect "user explicitly set --format" by comparing to the parser default.
    # If the user did NOT pass --format, fall through to the YAML value.
    parser = _get_parser()
    fmt_default = parser.get_default("format")

    input_path = args.convert or yaml_block.get("input")
    if args.format and args.format != fmt_default:
        output_format = args.format
    else:
        output_format = yaml_block.get("format", args.format or "vasp")

    # Ignore the "melt_quench_run" fallback when YAML drives convert mode
    # without an explicit -o flag, so the auto "<input>_<format>/" default
    # in convert() can take over.
    work_dir = args.work_dir
    if work_dir == "melt_quench_run":
        work_dir = None
    output_dir = work_dir or yaml_block.get("output_dir")

    if not input_path:
        print("Error: --convert needs an input PATH (or `convert.input` in "
              "the YAML config).")
        sys.exit(1)

    try:
        _convert(input_path, output_format=output_format,
                 output_dir=output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# Ensemble workflow helpers (--mq-ensemble, --hybrid-ensemble)
# ═════════════════════════════════════════════════════════════════════════════

def _run_mq_ensemble(args, override: dict) -> None:
    """Full melt-quench ensemble: stages 1-4 once + N independent quenches.

    Output layout under args.work_dir:
        shared/      stages 1-4 outputs (incl. stage4_eq_traj.xyz trajectory)
        snapshots/   N uniform snapshots extracted from stage 4 trajectory
        quench_runs/ per-snapshot stages 5-6-7 outputs (run_0000, run_0001, ...)
        final/       collected final amorphous structures (mq_NNNN.<fmt>)
    """
    import glob as _glob
    from .pipeline.run_pipeline import MeltQuenchPipeline
    from .pipeline import batch_quench
    from .utils import get_calculator, extract_snapshots
    from .pipeline.random_gen import _FORMAT_MAP

    work_dir   = args.work_dir or "mq_ensemble"
    shared_dir = os.path.join(work_dir, "shared")
    snap_dir   = os.path.join(work_dir, "snapshots")
    quench_dir = os.path.join(work_dir, "quench_runs")
    final_dir  = os.path.join(work_dir, "final")
    os.makedirs(work_dir, exist_ok=True)

    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  AmorphGen MQ-ensemble: {args.input_file} -> "
          f"{args.n_structures} amorphous structures")
    print(f"  Output: {work_dir}/")
    print(bar)

    # ── Phase 1: shared stages 1-4 (with resume) ─────────────────────────────
    print(f"\n[Phase 1/3] Stages 1-4 (shared) -> {shared_dir}/")
    pipe = MeltQuenchPipeline(args.input_file, work_dir=shared_dir,
                              cfg_override=override)
    pipe.run(stages=[1, 2, 3, 4], resume=args.resume)

    # ── Phase 2: extract N snapshots from stage 4 trajectory ─────────────────
    traj = os.path.join(shared_dir, "stage4_eq_traj.xyz")
    if not os.path.isfile(traj):
        # Backwards-compat: older runs wrote stage4_eq.xyz as the trajectory
        legacy = os.path.join(shared_dir, "stage4_eq.xyz")
        if os.path.isfile(legacy):
            traj = legacy
        else:
            print(f"Error: stage 4 trajectory not found "
                  f"({traj} or {legacy})")
            sys.exit(1)
    print(f"\n[Phase 2/3] Extracting {args.n_structures} snapshots from {traj}")
    extract_snapshots(traj, n_snapshots=args.n_structures,
                      select=args.select, output_dir=snap_dir,
                      burn_in_frames=args.burn_in_frames)

    # ── Phase 3: stages 5-6-7 per snapshot (with resume) ─────────────────────
    print(f"\n[Phase 3/3] Stages 5-6-7 (per snapshot) -> {quench_dir}/")
    snap_files = sorted(
        _glob.glob(os.path.join(snap_dir, "*.xyz"))
        + _glob.glob(os.path.join(snap_dir, "*.extxyz"))
    )
    calc = get_calculator(
        **_classical_kwargs(override),
        model=override.get("model", args.model),
        device=override.get("device", args.device),
        model_path=override.get("model_path", args.model_path),
        default_dtype=override.get("default_dtype", args.default_dtype),
    )
    batch_quench.run(
        snapshot_files=snap_files,
        n_runs=len(snap_files),
        cfg_override=override,
        work_dir=quench_dir,
        stages=[5, 6, 7],
        calc=calc,
        resume=args.resume,
    )

    _collect_ensemble_final(quench_dir, final_dir, args.format,
                            prefix="mq", fmt_map=_FORMAT_MAP)
    print(f"\n{bar}")
    print(f"  MQ ensemble complete -> {final_dir}/")
    print(bar)


def _run_hybrid_ensemble(args, override: dict) -> None:
    """Hybrid ensemble: stages 4-7 per disordered input structure.

    Output layout under args.work_dir:
        quench_runs/ per-input stages 4-7 outputs (run_0000, run_0001, ...)
        final/       collected final amorphous structures (hybrid_NNNN.<fmt>)
    """
    import glob as _glob
    from .pipeline import batch_quench
    from .utils import get_calculator
    from .pipeline.random_gen import _FORMAT_MAP

    work_dir   = args.work_dir or "hybrid_ensemble"
    quench_dir = os.path.join(work_dir, "quench_runs")
    final_dir  = os.path.join(work_dir, "final")
    os.makedirs(work_dir, exist_ok=True)

    # Find input structures (any ASE-readable format)
    snap_files = []
    for pattern in ("*.xyz", "*.extxyz", "*.vasp", "*.cif", "POSCAR*"):
        snap_files = sorted(_glob.glob(os.path.join(args.input_dir, pattern)))
        if snap_files:
            break
    if not snap_files:
        print(f"Error: no structure files in {args.input_dir}/ "
              f"(looked for *.xyz, *.extxyz, *.vasp, *.cif, POSCAR*)")
        sys.exit(1)

    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  AmorphGen hybrid-ensemble: {len(snap_files)} structures from "
          f"{args.input_dir}/")
    print(f"  Output: {work_dir}/")
    print(bar)

    calc = get_calculator(
        **_classical_kwargs(override),
        model=override.get("model", args.model),
        device=override.get("device", args.device),
        model_path=override.get("model_path", args.model_path),
        default_dtype=override.get("default_dtype", args.default_dtype),
    )
    batch_quench.run(
        snapshot_files=snap_files,
        n_runs=len(snap_files),
        cfg_override=override,
        work_dir=quench_dir,
        stages=[4, 5, 6, 7],
        calc=calc,
        resume=args.resume,
    )

    _collect_ensemble_final(quench_dir, final_dir, args.format,
                            prefix="hybrid", fmt_map=_FORMAT_MAP)
    print(f"\n{bar}")
    print(f"  Hybrid ensemble complete -> {final_dir}/")
    print(bar)


def _collect_ensemble_final(quench_dir: str, final_dir: str, output_format: str,
                            prefix: str, fmt_map: dict) -> None:
    """Copy each run_NNNN/final_amorphous.xyz to final_dir/<prefix>_NNNN.<fmt>."""
    import glob as _glob
    from ase.io import read, write

    if output_format not in fmt_map:
        print(f"Warning: unknown format '{output_format}', using 'xyz'")
        output_format = "xyz"
    ase_format, ext = fmt_map[output_format]

    os.makedirs(final_dir, exist_ok=True)
    n_collected = 0
    for run_dir in sorted(_glob.glob(os.path.join(quench_dir, "run_*"))):
        idx = os.path.basename(run_dir).replace("run_", "")
        src = os.path.join(run_dir, "final_amorphous.xyz")
        if not os.path.isfile(src):
            # Backwards-compat: older runs wrote .extxyz extension
            legacy = os.path.join(run_dir, "final_amorphous.extxyz")
            if os.path.isfile(legacy):
                src = legacy
            else:
                continue
        dest = os.path.join(final_dir, f"{prefix}_{idx}{ext}")
        atoms = read(src)
        if ase_format == "vasp":
            atoms = atoms[atoms.numbers.argsort()]
            write(dest, atoms, format=ase_format, sort=True)
        else:
            write(dest, atoms, format=ase_format)
        n_collected += 1
    print(f"  Collected {n_collected} final structures -> {final_dir}/")


def main():
    args = parse_args()

    # ── Smart default for --work-dir based on mode ───────────────────────────
    if args.work_dir is None:
        if args.random_gen:
            from ase import Atoms
            comp = args.composition or ""
            try:
                symbols = []
                for pair in comp.replace(" ", "").split(","):
                    el, n = pair.split("=")
                    symbols.extend([el] * int(n))
                formula = Atoms(symbols).get_chemical_formula(mode="hill")
                args.work_dir = f"random_{formula}"
            except Exception:
                args.work_dir = "random_structures"
        elif args.batch_quench:
            args.work_dir = "batch_quench"
        elif args.batch_opt:
            args.work_dir = "batch_opt"
        elif getattr(args, "analyse", False):
            args.work_dir = "analysis"
        elif args.convert:
            # Leave None so the convert helper picks
            # "<input>_<format>/" as its automatic default.
            pass
        else:
            args.work_dir = "melt_quench_run"

    # ── List models ───────────────────────────────────────────────────────────
    if args.list_models:
        from .utils import list_models
        list_models()
        sys.exit(0)

    # ── Build config override ─────────────────────────────────────────────────
    # Precedence: CLI args > YAML config > DEFAULT_CONFIG
    if args.config is not None:
        from .configs import load_yaml_config
        from .utils import merge_config

        yaml_cfg = load_yaml_config(args.config)
        print(f"[Config] Loaded: {args.config}")

        # Build override from only explicitly-set CLI args
        # We need the parser to detect defaults — re-parse to get it
        cli_override = _build_override(args, _get_parser(), explicit_only=True)

        # Merge: YAML first, then CLI on top
        override = merge_config(yaml_cfg, cli_override)
    else:
        override = _build_override(args, _get_parser())

    # ── Rank structures from a random-gen log file ────────────────────────────
    if args.rank_from_log:
        from .analysis.energy import rank_from_log, format_log_ranking
        result = rank_from_log(args.rank_from_log)
        print(format_log_ranking(result, logfile=args.rank_from_log))
        return

    # ── Convert one or many structure files to a different format ────────────
    yaml_convert_block = override.get("convert") if isinstance(override, dict) else None
    if args.convert or (yaml_convert_block and yaml_convert_block.get("input")):
        _run_convert(args, yaml_cfg=override if isinstance(override, dict) else None)
        return

    # ── Extract snapshots from a trajectory file ──────────────────────────────
    if args.extract_snapshots:
        from .utils import extract_snapshots
        out_dir = args.work_dir or "snapshots"
        # Snapshot count: prefer --n-structures (the unified count flag);
        # fall back to --n-runs for backwards compatibility.  --n-structures
        # has default 1, --n-runs has default 20, so pick the one the user
        # actually changed.
        if args.n_structures != 1:
            n_snap = args.n_structures
        elif args.n_runs != 20:
            n_snap = args.n_runs
        else:
            n_snap = 20  # historical default
        extract_snapshots(
            args.extract_snapshots,
            n_snapshots=n_snap,
            select=args.select,
            output_dir=out_dir,
            burn_in_frames=args.burn_in_frames,
            output_format=args.format,
        )
        return

    # ── MQ-ensemble mode ──────────────────────────────────────────────────────
    # Full melt-quench ensemble workflow in one command:
    #   stages 1-4 (shared, from crystal) -> extract N snapshots -> stages 5-7
    #   independently per snapshot. Resume-aware at every step.
    if args.mq_ensemble:
        if args.input_file is None:
            print("Error: input_file (crystal structure) is required for --mq-ensemble.")
            sys.exit(1)
        _run_mq_ensemble(args, override)
        return

    # ── Hybrid-ensemble mode ──────────────────────────────────────────────────
    # Take all structures in --input-dir and run stages 4-5-6-7 on each.
    # Useful for "AmorphGen random + chgnet quench" workflows.
    if args.hybrid_ensemble:
        if args.input_dir is None:
            print("Error: --input-dir is required for --hybrid-ensemble.")
            sys.exit(1)
        _run_hybrid_ensemble(args, override)
        return

    # ── Structure analysis mode ────────────────────────────────────────────────
    if args.analyse:
        from .analysis import StructureAnalyser

        source = args.input_dir or args.input_file
        if source is None:
            print("Error: --input-dir or input_file is required for --analyse.")
            print("  Example: amorphgen --analyse --input-dir optimised_structures/")
            sys.exit(1)

        # Read analysis block from YAML config (if present)
        an_cfg = override.get("analysis", {})

        # Parse cutoff: CLI > YAML > default "auto"
        cutoff = args.cutoff
        parser = _get_parser()
        if cutoff == parser.get_default("cutoff") and "cutoff" in an_cfg:
            cutoff = an_cfg["cutoff"]
        if cutoff not in ("auto", "auto-rdf"):
            try:
                cutoff = float(cutoff)
            except ValueError:
                print(f"Error: invalid cutoff '{cutoff}'. "
                      f"Use a number, 'auto', or 'auto-rdf'.")
                sys.exit(1)

        sa = StructureAnalyser(source, cutoff=cutoff)

        # Per-structure or grouped analysis
        per_structure = args.per_structure or an_cfg.get("per_structure", False)
        if per_structure:
            text = sa.per_structure_summary()
        else:
            text = sa.summary()

        # Save report: CLI > YAML
        report_path = args.save_report
        if report_path is None and "save_report" in an_cfg:
            report_path = an_cfg["save_report"]
        if report_path:
            sa.save_report(report_path, text=text)

        # Save plots: CLI > YAML
        plot_dir = args.save_plot
        if plot_dir is None and "save_plot" in an_cfg:
            plot_dir = an_cfg["save_plot"]

        # Plot settings from YAML
        plot_kwargs = {}
        if "rdf_pairs" in an_cfg:
            plot_kwargs["rdf_pairs"] = an_cfg["rdf_pairs"]
        if "angle_triplets" in an_cfg:
            plot_kwargs["angle_triplets"] = an_cfg["angle_triplets"]
        if "angle_style" in an_cfg:
            plot_kwargs["angle_style"] = an_cfg["angle_style"]
        if "rmax" in an_cfg:
            plot_kwargs["rmax"] = an_cfg["rmax"]

        # Smearing: CLI > YAML > default (0.0)
        smearing = args.smearing
        if smearing == 0.0 and "smearing" in an_cfg:
            smearing = an_cfg["smearing"]
        if smearing > 0:
            plot_kwargs["smearing"] = smearing

        # Total RDF: CLI flag or YAML
        if args.total_rdf or an_cfg.get("total_rdf", False):
            plot_kwargs["show_total_rdf"] = True
        # Publication-quality knobs (CLI > YAML)
        if args.save_pdf or an_cfg.get("save_pdf", False):
            plot_kwargs["save_pdf"] = True
        if args.dpi is not None:
            plot_kwargs["dpi"] = args.dpi
        elif "dpi" in an_cfg:
            plot_kwargs["dpi"] = an_cfg["dpi"]
        if args.show_title or an_cfg.get("show_title", False):
            plot_kwargs["show_title"] = True
        if plot_dir:
            sa.plot(output_dir=plot_dir, **plot_kwargs)

        # Extra analysis from YAML
        if "ring_bond_pair" in an_cfg:
            pair = tuple(an_cfg["ring_bond_pair"])
            rings = sa.ring_statistics(bond_pair=pair)
            print(f"\n  Ring statistics ({pair[0]}-{pair[1]}):")
            for s, c, f in zip(rings['ring_sizes'], rings['counts'],
                               rings['fractions']):
                print(f"    {s}-ring: {c} ({f:.1f}%)")

        if "voronoi_element" in an_cfg:
            elem = an_cfg["voronoi_element"]
            vor = sa.voronoi(element=elem)
            print(f"\n  Voronoi ({elem}): {vor['total_atoms']} atoms, "
                  f"mean faces={vor['mean_faces']:.1f}")
            for idx, count, pct in vor['top_10'][:5]:
                print(f"    {idx}: {pct:.1f}%")

        # Validation against literature reference YAML
        ref_path = args.reference or an_cfg.get("reference")
        if ref_path:
            from .analysis.validate import (validate_against_reference,
                                            format_validation_report)
            import yaml
            with open(ref_path) as f:
                reference = yaml.safe_load(f)
            v_result = validate_against_reference(sa, reference)
            v_text = format_validation_report(v_result)
            print(v_text)
            if report_path:
                with open(report_path, "a") as rf:
                    rf.write("\n" + v_text + "\n")

        if an_cfg.get("energy_ranking", False):
            er = sa.energy_ranking()
            if "error" not in er:
                print(f"\n  Energy ranking:")
                print(f"    Best: {er['best_energy']:.4f} eV/atom")
                print(f"    Worst: {er['worst_energy']:.4f} eV/atom")
                print(f"    Spread: {er['spread']:.4f} eV/atom")

        return

    # ── Random generation mode ────────────────────────────────────────────────
    if args.random_gen:
        from .pipeline.random_gen import batch_random
        from .utils import get_calculator

        # Read random_gen block from YAML config (if present)
        rg_cfg = override.get("random_gen", {})

        # Composition: CLI > YAML > error
        if args.composition is not None:
            composition = _parse_composition(args.composition)
        elif "composition" in rg_cfg:
            composition = rg_cfg["composition"]
        else:
            print("Error: --composition is required for --random-gen mode.")
            print("  Examples:")
            print('    --composition "In2O3*16"       (formula * N units = 80 atoms)')
            print("    --composition In=32,O=48     (explicit atom counts)")
            print("  Or in YAML:  random_gen: { composition: {In: 32, O: 48} }")
            sys.exit(1)

        # n_structures: CLI > YAML > default (10)
        parser = _get_parser()
        n_structures = args.n_structures
        if n_structures == parser.get_default("n_structures") and "n_structures" in rg_cfg:
            n_structures = rg_cfg["n_structures"]

        # target_density: CLI > YAML > None
        target_density = args.target_density
        if target_density is None and "target_density" in rg_cfg:
            target_density = rg_cfg["target_density"]

        # density_scale: CLI > YAML > 1.0 (parser default)
        density_scale = args.density_scale
        if density_scale == parser.get_default("density_scale") \
                and "density_scale" in rg_cfg:
            density_scale = rg_cfg["density_scale"]

        # output_format: CLI > YAML > default
        output_format = args.format
        if output_format == parser.get_default("format") and "output_format" in rg_cfg:
            output_format = rg_cfg["output_format"]

        # minsep: CLI > YAML > None (auto-generated)
        minsep = None
        if args.minsep is not None:
            minsep = _parse_minsep(args.minsep)
        elif "minsep" in rg_cfg:
            minsep = rg_cfg["minsep"]

        # target_cn: --no-sc > CLI > YAML > auto
        target_cn = None
        if args.no_sc:
            target_cn = {}  # empty dict disables coordination-aware placement
        elif args.target_cn is not None:
            target_cn = _parse_target_cn(args.target_cn)
        elif "target_cn" in rg_cfg:
            target_cn = rg_cfg["target_cn"]

        # dmax: CLI > YAML > None (auto-generated if target_cn set)
        dmax_dict = None
        if args.dmax is not None:
            dmax_dict = _parse_dmax(args.dmax)
        elif "dmax" in rg_cfg:
            dmax_dict = rg_cfg["dmax"]

        # cn_tolerance: CLI > YAML > auto (from composition)
        cn_tolerance = args.cn_tolerance
        if cn_tolerance is None and "cn_tolerance" in rg_cfg:
            cn_tolerance = rg_cfg["cn_tolerance"]
        # If still None, generate_random will auto-detect from composition

        # dmax_factor: CLI > YAML > default (1.5)
        if args.dmax_factor == 1.5 and "dmax_factor" in rg_cfg:
            args.dmax_factor = rg_cfg["dmax_factor"]

        # cell_filter: CLI > YAML random_gen > "cubic" for random-gen
        # Random gen produces cubic cells, so default to cubic (not FrechetCellFilter)
        cell_filter = args.cell_filter
        if cell_filter == parser.get_default("cell_filter"):
            if "cell_filter" in rg_cfg:
                cell_filter = rg_cfg["cell_filter"]
            else:
                cell_filter = "cubic"

        # relax: CLI flag or YAML (default: no relaxation)
        do_relax = args.relax
        if not do_relax and "relax" in rg_cfg:
            do_relax = rg_cfg["relax"]

        calc = None
        if do_relax:
            calc = get_calculator(
                **_classical_kwargs(override),
                model=override.get("model", args.model),
                device=override.get("device", args.device),
                model_path=override.get("model_path", args.model_path),
                default_dtype=override.get("default_dtype", args.default_dtype),

            )

        files = batch_random(
            composition=composition,
            n_structures=n_structures,
            output_dir=args.work_dir,
            output_format=output_format,
            relax=do_relax,
            calc=calc,
            fmax=args.fmax if args.fmax != 0.01 else 0.05,
            max_relax_steps=args.opt_steps,
            optimizer=args.optimizer,
            cell_filter=cell_filter,
            target_density=target_density,
            density_scale=density_scale,
            minsep=minsep,
            max_attempts_per_atom=args.max_attempts,
            target_cn=target_cn,
            dmax=dmax_dict,
            cn_tolerance=cn_tolerance,
            dmax_factor=args.dmax_factor,
            repair_iters=args.repair_iters,
            resume=args.resume,
        )
        return

    # ── Batch optimisation mode ──────────────────────────────────────────────
    if args.batch_opt:
        if args.input_dir is None:
            print("Error: --input-dir is required for --batch-opt mode.")
            print("  Example: amorphgen --batch-opt --input-dir random_Ga2O3/")
            sys.exit(1)

        from .pipeline.opt_cell import batch_optimize
        from .utils import get_calculator

        calc = get_calculator(
            **_classical_kwargs(override),
            model=override.get("model", args.model),
            device=override.get("device", args.device),
            model_path=override.get("model_path", args.model_path),
            default_dtype=override.get("default_dtype", args.default_dtype),

        )

        batch_optimize(
            input_dir=args.input_dir,
            output_dir=args.work_dir,
            cfg_override=override,
            calc=calc,
        )
        return

    # ── Batch quench mode ─────────────────────────────────────────────────────
    if args.batch_quench:
        from .pipeline import batch_quench
        from .utils import get_calculator
        import glob

        snap_source = args.snapshot_dir

        # Polymorphic input: if --snapshot-dir is a single trajectory file
        # (e.g. shared/stage4_eq.xyz from a Stage 4 run), extract --n-runs
        # uniformly-spaced frames into a 'snapshots_extracted/' subdir of
        # the work directory and use that as the snapshot source.
        # Putting the extracted dir inside work_dir avoids race conditions
        # when several array tasks point at files in the same source dir.
        if os.path.isfile(snap_source):
            from .utils import extract_snapshots
            os.makedirs(args.work_dir, exist_ok=True)
            extracted_dir = os.path.join(args.work_dir, "snapshots_extracted")
            print(f"[batch-quench] '{snap_source}' is a file — extracting "
                  f"{args.n_runs} uniform snapshots to {extracted_dir}/")
            extract_snapshots(snap_source, n_snapshots=args.n_runs,
                              select=args.select, output_dir=extracted_dir,
                              burn_in_frames=args.burn_in_frames)
            snap_source = extracted_dir

        # Accept any ASE-readable structure format. extxyz/xyz are the
        # original use case (snapshots from MD trajectory); vasp/cif/POSCAR
        # let users feed in pre-relaxed structures from --random-gen or DFT.
        snap_files: list = []
        for pattern in ("*.xyz", "*.extxyz", "*.vasp", "*.cif", "POSCAR*"):
            snap_files = sorted(glob.glob(
                os.path.join(snap_source, pattern)))
            if snap_files:
                break
        if not snap_files:
            print(f"Error: no snapshot files found in {snap_source}/ "
                  f"(looked for *.xyz, *.extxyz, *.vasp, *.cif, POSCAR*)")
            sys.exit(1)

        calc = get_calculator(
            **_classical_kwargs(override),
            model=override.get("model", args.model),
            device=override.get("device", args.device),
            model_path=override.get("model_path", args.model_path),
            default_dtype=override.get("default_dtype", args.default_dtype),

        )

        batch_quench.run(
            snapshot_files=snap_files,
            n_runs=args.n_runs,
            select=args.select,
            cfg_override=override,
            work_dir=args.work_dir,
            stages=args.batch_stages,
            calc=calc,
            resume=args.resume,
        )
        return

    # ── Standard pipeline mode ────────────────────────────────────────────────
    if args.input_file is None:
        print("Error: input_file is required for melt-quench pipeline mode.")
        print("  Usage: amorphgen POSCAR [--model NAME] [--stages 1 2 3 4 5 6 7]")
        print("  Other modes that don't need an input file:")
        print('    amorphgen --random-gen --composition "SiO2*16"')
        print("    amorphgen --batch-opt --input-dir structures/")
        print("    amorphgen --analyse structures/")
        print("  Run 'amorphgen --help' for full options.")
        sys.exit(1)

    # For optimisation-only (--stages 1 or --stages 7), call opt_cell directly
    # so that output filenames are derived from the input file name.
    if args.stages == [1] or args.stages == [7]:
        from .pipeline.opt_cell import run as opt_run
        from .utils import get_calculator

        os.makedirs(args.work_dir, exist_ok=True)
        orig_dir = os.getcwd()
        os.chdir(args.work_dir)

        try:
            input_path = os.path.join(orig_dir, args.input_file)
            calc = get_calculator(
                **_classical_kwargs(override),
                model=override.get("model", args.model),
                device=override.get("device", args.device),
                model_path=override.get("model_path", args.model_path),
                default_dtype=override.get("default_dtype", args.default_dtype),

            )
            stage_key = "opt" if args.stages == [1] else "final_opt"
            opt_run(input_path, cfg_override=override, calc=calc,
                    stage_key=stage_key)
        finally:
            os.chdir(orig_dir)
        return

    from .pipeline.run_pipeline import MeltQuenchPipeline

    pipe = MeltQuenchPipeline(
        input_file=args.input_file,
        work_dir=args.work_dir,
        cfg_override=override,
    )
    pipe.run(stages=args.stages, resume=args.resume)


if __name__ == "__main__":
    main()
