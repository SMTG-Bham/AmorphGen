"""Small, resumable FEFF example using two Ga2O3 cells shipped with Tutorial 8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ase.io import read, write

from amorphgen.analysis.feff import (
    collect_feff_results, format_feff_report, plot_feff,
    run_feff_jobs, write_feff_csv, write_feff_jobs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "collect", "all"))
    parser.add_argument("--output", type=Path, default=Path("ga2o3_exafs_example"))
    parser.add_argument("--executable", default="feff")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    root = args.output.resolve()
    jobs = root / "jobs"
    if args.stage in ("prepare", "all"):
        fixtures = Path(__file__).resolve().parents[2] / "Tutorials/T8_validation_workflow/structures/relaxed"
        paths = [fixtures / f"random_{n:04d}_opt.xyz" for n in range(2)]
        structures = [read(p) for p in paths]
        manifest = write_feff_jobs(structures, "Ga", str(jobs), n_sites=1, radius=6, seed=0)
        root.mkdir(parents=True, exist_ok=True)
        write(root / "structures.extxyz", structures)
        (root / "sources.json").write_text(json.dumps({
            "description": "Two relaxed a-Ga2O3 cells shipped with upstream Tutorial 8; execution example only",
            "sources": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
            "jobs": manifest["n_jobs"], "seed": 0,
        }, indent=2) + "\n")
        print(f"Prepared {manifest['n_jobs']} jobs in {jobs}")
    if args.stage in ("run", "all"):
        outcome = run_feff_jobs(str(jobs), executable=args.executable, timeout=args.timeout)
        print(json.dumps(outcome))
        if outcome["failed"]:
            raise SystemExit(1)
    if args.stage in ("collect", "all"):
        result = collect_feff_results(jobs, kmin=3, kmax=12, dk=1, kweight=2)
        if result["missing"]:
            raise SystemExit(f"Incomplete example: {result['missing']}")
        write_feff_csv(result, str(root / "spectrum.csv"))
        report = format_feff_report(result)
        (root / "report.txt").write_text(report)
        plot_feff(result, output_dir=str(root), prefix="ga2o3")
        print(report)


if __name__ == "__main__":
    main()
