#!/usr/bin/env python3
"""
run_melt_quench.py
------------------
Command-line entry point for the 7-stage melt-and-quench amorphous oxide pipeline.

Stage 1 : Optimise crystalline input               (LBFGS)
Stage 2 : Pre-melt equilibration  300 K            (NVT/NPT, 50 ps)
Stage 3 : Melt  –  heat ramp  300 → T_melt        (NPT, 100 K/ps)
Stage 4 : High-T equilibration at T_melt           (NVT, 100 ps)
Stage 5 : Quench  –  cooling ramp  T_melt → 300 K (NVT, 100 K/ps)
Stage 6 : Low-T equilibration  300 K               (NVT, 50 ps)
Stage 7 : Final optimisation  →  amorphous         (LBFGS)

Examples
--------
Full 7-stage pipeline:
    run_melt_quench.py POSCAR --model mace-mpa-0-medium --device cuda

Cell optimisation only:
    run_melt_quench.py POSCAR --stages 1

Pre-melt eq at 300 K with NPT:
    run_melt_quench.py stage1_optimised.xyz --stages 2 --eq-premelt-ensemble NPT

Melt to 2500 K:
    run_melt_quench.py stage2_eq_300K.xyz --stages 3 --melt-T-end 2500

Resume from Stage 5:
    run_melt_quench.py stage4_eq_high.xyz --stages 5 6 7 --work-dir my_run

List all available MACE foundation models:
    run_melt_quench.py --list-models
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amorphMD import MeltQuenchPipeline
from amorphMD.utils import list_models


def parse_args():
    p = argparse.ArgumentParser(
        description="7-stage melt-and-quench amorphous oxide generator (MACE-MP)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── positional ────────────────────────────────────────────────────────────
    p.add_argument("input_file", nargs="?",
                   help="Input structure file (POSCAR, .xyz, .cif, extxyz …)")

    # ── info ──────────────────────────────────────────────────────────────────
    p.add_argument("--list-models", action="store_true",
                   help="Print all available MACE foundation models and exit")

    # ── pipeline control ──────────────────────────────────────────────────────
    p.add_argument("--stages", nargs="+", type=int,
                   default=[1, 2, 3, 4, 5, 6, 7], metavar="N",
                   help="Stages to run (1–7)")
    p.add_argument("--work-dir", default="melt_quench_run",
                   help="Output directory (created if absent)")

    # ── model ─────────────────────────────────────────────────────────────────
    model_group = p.add_mutually_exclusive_group()
    model_group.add_argument("--model", default="mace-mpa-0", metavar="NAME",
                             help="Foundation model short name (--list-models to see all)")
    model_group.add_argument("--model-path", default=None, metavar="PATH",
                             help="Path to a local .model file (fine-tuned model)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--traj-format", default="extxyz",
                   choices=["extxyz", "xyz", "traj", "lammps-dump"],
                   help="Trajectory format for all MD stages")

    # ── optimisation (stages 1 & 7) ───────────────────────────────────────────
    p.add_argument("--fmax",      type=float, default=0.01, help="Force convergence eV/Å")
    p.add_argument("--opt-steps", type=int,   default=1000, help="Max optimisation steps")
    p.add_argument("--optimizer", default="LBFGS",
                   choices=["LBFGS", "FIRE", "BFGSLineSearch", "BFGS", "MDMin"],
                   help="Optimizer for Stage 1 crystalline opt (default: LBFGS)")
    p.add_argument("--final-optimizer", default=None,
                   choices=["LBFGS", "FIRE", "BFGSLineSearch", "BFGS", "MDMin"],
                   help="Optimizer for Stage 7 final opt (default: same as --optimizer)")
    p.add_argument("--cell-filter", default="UnitCellFilter",
                   choices=["UnitCellFilter", "ExpCellFilter", "StrainFilter", "cubic"],
                   help="Cell filter for Stage 1 opt (default: UnitCellFilter). "
                        "'cubic' reshapes to cube and fixes angles at 90 deg.")
    p.add_argument("--final-cell-filter", default=None,
                   choices=["UnitCellFilter", "ExpCellFilter", "StrainFilter", "cubic"],
                   help="Cell filter for Stage 7 final opt (default: same as --cell-filter). "
                        "Use 'cubic' to enforce orthogonal amorphous cell.")

    # ── global + per-stage timesteps ──────────────────────────────────────────
    p.add_argument("--timestep",          type=float, default=None, metavar="FS",
                   help="MD timestep (fs) for ALL stages. Per-stage flags override this.")
    p.add_argument("--melt-timestep",     type=float, default=None, metavar="FS")
    p.add_argument("--eq-premelt-timestep", type=float, default=None, metavar="FS")
    p.add_argument("--eq-high-timestep",  type=float, default=None, metavar="FS")
    p.add_argument("--quench-timestep",   type=float, default=None, metavar="FS")
    p.add_argument("--eq-low-timestep",   type=float, default=None, metavar="FS")

    # ── Stage 2 : Pre-melt equilibration at 300 K ────────────────────────────
    p.add_argument("--eq-premelt-ensemble", default="NVT", choices=["NVT", "NPT"],
                   help="Ensemble for pre-melt eq (default: NVT)")
    p.add_argument("--eq-premelt-T",        type=int, default=300,
                   help="Temperature for pre-melt eq in K (default: 300)")
    p.add_argument("--eq-premelt-steps",    type=int, default=50000,
                   help="Steps for pre-melt eq (default: 50000 = 50 ps)")

    # ── Stage 3 : Melt (heat ramp) ────────────────────────────────────────────
    p.add_argument("--melt-ensemble",    default="NPT", choices=["NVT", "NPT"])
    p.add_argument("--melt-T-start",     type=int, default=300)
    p.add_argument("--melt-T-end",       type=int, default=2500,
                   help="Melt temperature in K (default: 2500)")
    p.add_argument("--melt-T-step",      type=int, default=100)
    p.add_argument("--melt-steps-per-T", type=int, default=1000,
                   help="Steps per temperature increment (1000 = 1 ps → 100 K/ps)")
    p.add_argument("--no-cubic",         action="store_true",
                   help="Skip cubic cell reshape before melting")

    # ── Stage 4 : High-T equilibration ───────────────────────────────────────
    p.add_argument("--eq-high-ensemble", default="NVT", choices=["NVT", "NPT"])
    p.add_argument("--eq-high-T",        type=int, default=2500,
                   help="High-T equilibration temperature in K (default: 2500)")
    p.add_argument("--eq-high-steps",    type=int, default=100000,
                   help="Steps for high-T eq (default: 100000 = 100 ps)")
    p.add_argument("--eq-high-cell-mode", default="free",
                   choices=["free", "fix_volume", "keep_cubic", "target_density"],
                   help="Cell control during high-T eq (default: free). "
                        "free=cell evolves with ensemble, "
                        "fix_volume=freeze cell (NVT only), "
                        "keep_cubic=fix angles at 90deg (NVT only), "
                        "target_density=rescale to target density before MD.")
    p.add_argument("--eq-high-target-density", type=float, default=None,
                   metavar="G_CM3",
                   help="Target density in g/cm3 for cell_mode=target_density "
                        "(e.g. 3.0 for amorphous In2O3)")

    # ── Stage 5 : Quench (cooling ramp) ──────────────────────────────────────
    p.add_argument("--quench-ensemble",    default="NVT", choices=["NVT", "NPT"])
    p.add_argument("--quench-T-start",     type=int, default=2500,
                   help="Start temperature for quench in K (default: 2500)")
    p.add_argument("--quench-T-end",       type=int, default=300)
    p.add_argument("--quench-T-step",      type=int, default=-100)
    p.add_argument("--quench-steps-per-T", type=int, default=1000,
                   help="Steps per temperature decrement (1000 = 1 ps → 100 K/ps)")

    # ── Stage 6 : Low-T equilibration ────────────────────────────────────────
    p.add_argument("--eq-low-ensemble", default="NVT", choices=["NVT", "NPT"],
                   help="Ensemble for low-T eq (default: NVT)")
    p.add_argument("--eq-low-T",        type=int, default=300)
    p.add_argument("--eq-low-steps",    type=int, default=50000,
                   help="Steps for low-T eq (default: 50000 = 50 ps)")

    # ── Snapshot sampling (Stage 4) ───────────────────────────────────────────
    p.add_argument("--sample-interval", type=float, default=None, metavar="PS",
                   help="Save snapshot every N ps during Stage 4 high-T eq")
    p.add_argument("--snapshot-dir", default="snapshots", metavar="DIR")

    # ── Batch quench mode ─────────────────────────────────────────────────────
    p.add_argument("--batch-quench", action="store_true",
                   help="Run stages 5→6→7 on a set of snapshots")
    p.add_argument("--traj-file", default=None, metavar="FILE",
                   help="Trajectory file to extract snapshots from (batch mode)")
    p.add_argument("--n-runs",  type=int, default=20)
    p.add_argument("--select",  default="uniform",
                   choices=["uniform", "random", "first", "last", "all"])
    p.add_argument("--batch-stages", nargs="+", type=int, default=[5, 6, 7], metavar="N")
    p.add_argument("--resume", action="store_true",
                   help="Skip batch runs whose final output already exists "
                        "(use to continue a timed-out batch job)")

    # ── Snapshot extraction mode ──────────────────────────────────────────────
    p.add_argument("--extract-snapshots", action="store_true",
                   help="Extract N snapshots from an existing trajectory file")
    p.add_argument("--n-snapshots", type=int, default=10, metavar="N",
                   help="Number of snapshots to extract (default: 10)")

    return p.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        resolved = "cpu"
    print(f"[device] auto-detected → {resolved}")
    return resolved


def build_override(args, device: str) -> dict:
    # device already resolved and passed in from main()

    def pick_dt(stage_val):
        if stage_val is not None:
            return stage_val
        if args.timestep is not None:
            return args.timestep
        return None

    def ts(val):
        return {"timestep_fs": val} if val is not None else {}

    return {
        "mace_model":  args.model,
        "model_path":  args.model_path,
        "device":      device,
        "traj_format": args.traj_format,

        # Stage 1 & 7
        "opt": {
            "fmax":        args.fmax,
            "max_steps":   args.opt_steps,
            "optimizer":   args.optimizer,
            "cell_filter": args.cell_filter,
        },
        "final_opt": {
            "fmax":        args.fmax,
            "max_steps":   args.opt_steps,
            "optimizer":   args.final_optimizer or args.optimizer,
            "cell_filter": args.final_cell_filter or args.cell_filter,
        },

        # Stage 2 – pre-melt equilibration at 300 K
        "eq_premelt": {
            "ensemble":      args.eq_premelt_ensemble,
            "temperature_K": args.eq_premelt_T,
            "steps":         args.eq_premelt_steps,
            **ts(pick_dt(args.eq_premelt_timestep)),
        },

        # Stage 3 – melt heat ramp
        "melt": {
            "ensemble":    args.melt_ensemble,
            "T_start":     args.melt_T_start,
            "T_end":       args.melt_T_end,
            "T_step":      args.melt_T_step,
            "steps_per_T": args.melt_steps_per_T,
            "make_cubic":  not args.no_cubic,
            **ts(pick_dt(args.melt_timestep)),
        },

        # Stage 4 – high-T equilibration
        "eq_high": {
            "ensemble":              args.eq_high_ensemble,
            "temperature_K":         args.eq_high_T,
            "steps":                 args.eq_high_steps,
            "cell_mode":             args.eq_high_cell_mode,
            "target_density_g_cm3":  args.eq_high_target_density,
            "sample_interval_ps":    args.sample_interval,
            "snapshot_dir":          args.snapshot_dir,
            **ts(pick_dt(args.eq_high_timestep)),
        },

        # Stage 5 – quench cooling ramp
        "quench": {
            "ensemble":    args.quench_ensemble,
            "T_start":     args.quench_T_start,
            "T_end":       args.quench_T_end,
            "T_step":      args.quench_T_step,
            "steps_per_T": args.quench_steps_per_T,
            **ts(pick_dt(args.quench_timestep)),
        },

        # Stage 6 – low-T equilibration
        "eq_low": {
            "ensemble":      args.eq_low_ensemble,
            "temperature_K": args.eq_low_T,
            "steps":         args.eq_low_steps,
            **ts(pick_dt(args.eq_low_timestep)),
        },
    }


def main():
    args = parse_args()

    if args.list_models:
        list_models()
        return

    # ── Extract snapshots mode ────────────────────────────────────────────────
    if args.extract_snapshots:
        if not args.traj_file:
            print("Error: --extract-snapshots requires --traj-file PATH")
            return
        from amorphMD.utils import extract_snapshots
        extract_snapshots(
            traj_file = args.traj_file,
            n         = args.n_snapshots,
            select    = args.select,
            out_dir   = args.snapshot_dir,
        )
        return

    # ── Early exit if no input and not batch mode ────────────────────────────
    if not args.input_file and not args.batch_quench:
        print("Error: input_file is required (or use --list-models / --batch-quench).")
        return

    device   = resolve_device(args.device)
    override = build_override(args, device)

    # ── Batch quench mode ─────────────────────────────────────────────────────
    if args.batch_quench:
        from amorphMD.pipeline import batch_quench
        from amorphMD.utils import get_mace_calculator

        calc = get_mace_calculator(
            model=override["mace_model"],
            device=override["device"],
            model_path=override.get("model_path"),
        )

        if args.traj_file:
            batch_quench.run_from_traj(
                traj_file=args.traj_file,
                sample_interval_ps=args.sample_interval or 10.0,
                timestep_fs=override["melt"].get("timestep_fs", 1.0),
                n_runs=args.n_runs,
                select=args.select,
                cfg_override=override,
                work_dir=args.work_dir,
                stages=args.batch_stages,
                calc=calc,
                snapshot_dir=args.snapshot_dir,
            )
        else:
            import glob
            snap_files = sorted(
                glob.glob(os.path.join(args.snapshot_dir, "snapshot_*.extxyz")) +
                glob.glob(os.path.join(args.snapshot_dir, "snapshot_*.xyz"))
            )
            if not snap_files:
                print(f"Error: no snapshot_*.extxyz / *.xyz files found in '{args.snapshot_dir}'.")
                print("  Run Stage 4 with --sample-interval first, or use --traj-file.")
                return
            print(f"Found {len(snap_files)} snapshots in '{args.snapshot_dir}/'")
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

    # ── Normal pipeline mode ──────────────────────────────────────────────────
    pipe = MeltQuenchPipeline(
        input_file=args.input_file,
        work_dir=args.work_dir,
        cfg_override=override,
        share_calc=True,
    )
    pipe.run(stages=args.stages)


if __name__ == "__main__":
    main()
