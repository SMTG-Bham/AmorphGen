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

Use CHGNet:
    amorphgen POSCAR --model chgnet --device cpu

Use a custom fine-tuned model:
    amorphgen POSCAR --model-path /data/InO_finetuned.model

List all available models:
    amorphgen --list-models

Random structure generation:
    amorphgen --random-gen --composition In=32,O=48 --target-density 5.5

Batch quench from snapshots:
    amorphgen --batch-quench --snapshot-dir snapshots/ --n-runs 20
"""

from __future__ import annotations

import argparse
import sys
import os


def parse_args():
    p = argparse.ArgumentParser(
        description="AmorphGen: model-agnostic amorphous structure generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── positional ────────────────────────────────────────────────────────────
    p.add_argument("input_file", nargs="?",
                   help="Input structure file (POSCAR, .xyz, .cif, ...)")

    # ── info flags ────────────────────────────────────────────────────────────
    p.add_argument("--list-models", action="store_true",
                   help="Print all available foundation models and exit")

    # ── pipeline control ──────────────────────────────────────────────────────
    p.add_argument("--stages", nargs="+", type=int,
                   default=[1, 2, 3, 4, 5, 6, 7], metavar="N",
                   help="Stages to run (1-7)")
    p.add_argument("--work-dir", default="melt_quench_run",
                   help="Output directory")

    # ── model ─────────────────────────────────────────────────────────────────
    model_group = p.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model", default="mace-mpa-0", metavar="NAME",
        help="Foundation model: mace-mpa-0, chgnet, m3gnet, etc. "
             "(use --list-models to see all)"
    )
    model_group.add_argument(
        "--model-path", default=None, metavar="PATH",
        help="Path to a local .model file (fine-tuned / custom model)"
    )
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    # ── Stage 1 & 7: optimisation ─────────────────────────────────────────────
    p.add_argument("--fmax", type=float, default=0.01,
                   help="Force convergence (eV/A)")
    p.add_argument("--opt-steps", type=int, default=1000,
                   help="Max optimisation steps")
    p.add_argument("--optimizer", default="LBFGS",
                   choices=["LBFGS", "FIRE", "BFGSLineSearch", "BFGS", "MDMin"],
                   help="Optimizer for structure optimisation")

    # ── Stage 2: pre-melt equilibration ────────────────────────────────────────
    p.add_argument("--eq-premelt-ensemble", default="NVT", choices=["NVT", "NPT"])
    p.add_argument("--eq-premelt-T", type=int, default=300)
    p.add_argument("--eq-premelt-steps", type=int, default=50000)

    # ── Stage 3: melt ─────────────────────────────────────────────────────────
    p.add_argument("--melt-ensemble", default="NPT", choices=["NVT", "NPT"])
    p.add_argument("--melt-T-start", type=int, default=300)
    p.add_argument("--melt-T-end", type=int, default=3000)
    p.add_argument("--melt-T-step", type=int, default=100)
    p.add_argument("--melt-steps-per-T", type=int, default=1000)

    # ── Stage 4: high-T equilibration ─────────────────────────────────────────
    p.add_argument("--eq-high-ensemble", default="NVT", choices=["NVT", "NPT"])
    p.add_argument("--eq-high-T", type=int, default=3000)
    p.add_argument("--eq-high-steps", type=int, default=10000)

    # ── Stage 5: quench ───────────────────────────────────────────────────────
    p.add_argument("--quench-ensemble", default="NVT", choices=["NVT", "NPT"])
    p.add_argument("--quench-T-start", type=int, default=3000)
    p.add_argument("--quench-T-end", type=int, default=300)
    p.add_argument("--quench-T-step", type=int, default=-100)
    p.add_argument("--quench-steps-per-T", type=int, default=1000)

    # ── Stage 6: low-T equilibration ──────────────────────────────────────────
    p.add_argument("--eq-low-ensemble", default="NVT", choices=["NVT", "NPT"])
    p.add_argument("--eq-low-T", type=int, default=300)
    p.add_argument("--eq-low-steps", type=int, default=10000)

    # ── Batch quench mode ─────────────────────────────────────────────────────
    p.add_argument("--batch-quench", action="store_true",
                   help="Run batch quench on snapshot files")
    p.add_argument("--snapshot-dir", default="snapshots", metavar="DIR")
    p.add_argument("--n-runs", type=int, default=20)
    p.add_argument("--select", default="uniform",
                   choices=["uniform", "last"])
    p.add_argument("--batch-stages", nargs="+", type=int,
                   default=[5, 6, 7], metavar="N")
    p.add_argument("--resume", action="store_true",
                   help="Skip batch runs whose final output already exists")

    # ── Random generation mode ────────────────────────────────────────────────
    p.add_argument("--random-gen", action="store_true",
                   help="Generate random structures (AIRSS-style)")
    p.add_argument("--composition", default=None, metavar="SPEC",
                   help="Composition, e.g. In=32,O=48")
    p.add_argument("--target-density", type=float, default=None,
                   help="Target density in g/cm3")
    p.add_argument("--n-structures", type=int, default=10)
    p.add_argument("--no-relax", action="store_true",
                   help="Skip relaxation of random structures")

    return p.parse_args()


def _parse_composition(spec: str) -> dict[str, int]:
    """Parse 'In=32,O=48' -> {'In': 32, 'O': 48}."""
    comp = {}
    for part in spec.split(","):
        sym, count = part.strip().split("=")
        comp[sym.strip()] = int(count.strip())
    return comp


def main():
    args = parse_args()

    # ── List models ───────────────────────────────────────────────────────────
    if args.list_models:
        from .utils import list_models
        list_models()
        sys.exit(0)

    # ── Build config override from CLI args ───────────────────────────────────
    override = {
        "model": args.model,
        "model_path": args.model_path,
        "device": args.device,
        "opt": {
            "fmax": args.fmax,
            "max_steps": args.opt_steps,
            "optimizer": args.optimizer,
        },
        "eq_premelt": {
            "ensemble": args.eq_premelt_ensemble,
            "T": args.eq_premelt_T,
            "steps": args.eq_premelt_steps,
        },
        "melt": {
            "ensemble": args.melt_ensemble,
            "T_start": args.melt_T_start,
            "T_end": args.melt_T_end,
            "T_step": args.melt_T_step,
            "steps_per_T": args.melt_steps_per_T,
        },
        "eq_high": {
            "ensemble": args.eq_high_ensemble,
            "T": args.eq_high_T,
            "steps": args.eq_high_steps,
        },
        "quench": {
            "ensemble": args.quench_ensemble,
            "T_start": args.quench_T_start,
            "T_end": args.quench_T_end,
            "T_step": args.quench_T_step,
            "steps_per_T": args.quench_steps_per_T,
        },
        "eq_low": {
            "ensemble": args.eq_low_ensemble,
            "T": args.eq_low_T,
            "steps": args.eq_low_steps,
        },
    }

    # ── Random generation mode ────────────────────────────────────────────────
    if args.random_gen:
        if args.composition is None:
            print("Error: --composition is required for --random-gen mode.")
            print("  Example: --composition In=32,O=48")
            sys.exit(1)

        from .pipeline.random_gen import batch_random
        from .utils import get_calculator

        composition = _parse_composition(args.composition)
        calc = None
        if not args.no_relax:
            calc = get_calculator(
                model=args.model,
                device=args.device,
                model_path=args.model_path,
            )

        batch_random(
            composition=composition,
            n_structures=args.n_structures,
            output_dir=args.work_dir,
            relax=not args.no_relax,
            calc=calc,
            target_density=args.target_density,
        )
        return

    # ── Batch quench mode ─────────────────────────────────────────────────────
    if args.batch_quench:
        from .pipeline import batch_quench
        from .utils import get_calculator, extract_snapshots
        import glob

        snap_files = sorted(glob.glob(
            os.path.join(args.snapshot_dir, "*.extxyz")
        ))
        if not snap_files:
            snap_files = sorted(glob.glob(
                os.path.join(args.snapshot_dir, "*.xyz")
            ))
        if not snap_files:
            print(f"Error: no snapshot files found in {args.snapshot_dir}/")
            sys.exit(1)

        calc = get_calculator(
            model=args.model,
            device=args.device,
            model_path=args.model_path,
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
        print("Error: input_file is required for pipeline mode.")
        print("  Usage: amorphgen POSCAR [--model NAME] [--stages 1 2 3 4 5 6 7]")
        print("  Run 'amorphgen --help' for full options.")
        sys.exit(1)

    from .pipeline.run_pipeline import MeltQuenchPipeline

    pipe = MeltQuenchPipeline(
        input_file=args.input_file,
        work_dir=args.work_dir,
        cfg_override=override,
    )
    pipe.run(stages=args.stages)


if __name__ == "__main__":
    main()
