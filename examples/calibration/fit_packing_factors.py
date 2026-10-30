"""Fit AmorphGen packing factors from a table of reference densities.

    python fit_packing_factors.py [--csv packing_reference_densities.csv] [--ratio 0.80]
                                  [--all-rows] [--column density_on_file] [--json out.json]

With the defaults (ratio 0.80, the 100 rows with a verified source) this reproduces the
calibration of the SoftwareX paper: leave-one-out error 9.8 %, 11.9 % for the factors in the code.

Rule (the whole method):
  1. each compound is assigned to a class by amorphgen.utils.radii._classify_compound
  2. target density = the reference if it is amorphous, else crystal density x ratio
  3. back-solved factor  f_i = f_code * target_i / rho_est_i   (density is proportional to f)
  4. class factor = median of f_i over the class
  5. leave-one-out: predict each compound from the median of the OTHER class members

The script only reads; it never changes PACKING_FACTORS.
"""
import argparse, csv, json, re, statistics, subprocess, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import amorphgen
from ase.data import atomic_masses, atomic_numbers
from amorphgen.utils import radii

AMU_A3_TO_G_CM3 = 1.66054


def parse(spec):
    """'SiO2*72' or 'Cu=138,Zr=78' -> {'Si': 72, 'O': 144}"""
    if "=" in spec:
        return {a.split("=")[0]: int(a.split("=")[1]) for a in spec.split(",")}
    formula, n = spec.split("*")
    return {e: int(k or 1) * int(n) for e, k in re.findall(r"([A-Z][a-z]?)(\d*)", formula)}


def load(path, ratio, verified_only, column="density_g_cm3"):
    out = []
    for row in csv.DictReader(open(path)):
        if verified_only and row["verified"].strip().lower() != "yes":
            continue
        comp = parse(row["composition"])
        cls = radii._classify_compound(comp)
        f_code = radii.get_packing_factor(cls, comp)
        mass = sum(atomic_masses[atomic_numbers[e]] * n for e, n in comp.items()) * AMU_A3_TO_G_CM3
        rho_est = mass / radii.estimate_cell_length(comp) ** 3
        ref = float(row[column])
        target = ref if row["kind"] == "amorphous" else ref * ratio
        out.append(dict(compound=row["compound"], cls=cls, kind=row["kind"], ref=ref, target=target,
                        rho_est=rho_est, f_code=f_code, f_fit=f_code * target / rho_est,
                        verified=row["verified"].strip().lower() == "yes"))
    return out


def fit(rows):
    classes = {}
    for r in rows:
        classes.setdefault(r["cls"], []).append(r)
    table = []
    for cls, members in classes.items():
        f = [m["f_fit"] for m in members]
        loo = []
        if len(members) >= 2:
            for i, m in enumerate(members):
                others = statistics.median(f[:i] + f[i + 1:])
                loo.append(abs(m["rho_est"] * others / m["f_code"] / m["target"] - 1))
        code_err = [abs(m["rho_est"] / m["target"] - 1) for m in members]
        codes = sorted({round(m["f_code"], 3) for m in members})
        table.append(dict(cls=cls, n=len(members), f_code=codes, f_median=statistics.median(f),
                          f_min=min(f), f_max=max(f), loo_mean=statistics.mean(loo) if loo else None,
                          loo_max=max(loo) if loo else None, code_err_mean=statistics.mean(code_err),
                          n_amorphous=sum(m["kind"] == "amorphous" for m in members),
                          compounds=[m["compound"] for m in members]))
    table.sort(key=lambda t: -t["f_median"])
    return table


def overall(rows, table):
    med = {t["cls"]: t for t in table}
    loo, code = [], []
    for cls in med:
        members = [r for r in rows if r["cls"] == cls]
        f = [m["f_fit"] for m in members]
        for i, m in enumerate(members):
            code.append(abs(m["rho_est"] / m["target"] - 1))
            if len(members) >= 2:
                loo.append(abs(m["rho_est"] * statistics.median(f[:i] + f[i + 1:]) / m["f_code"] / m["target"] - 1))
    frac = lambda xs, t: sum(x <= t for x in xs) / len(xs)
    return dict(n=len(rows), n_loo=len(loo), loo_mean=statistics.mean(loo), loo_within10=frac(loo, 0.10),
                loo_within20=frac(loo, 0.20), code_mean=statistics.mean(code), code_within10=frac(code, 0.10),
                code_within20=frac(code, 0.20))


def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=here / "packing_reference_densities.csv")
    ap.add_argument("--ratio", type=float, default=0.80, help="amorphous / crystal density ratio")
    ap.add_argument("--all-rows", action="store_true",
                    help="also use the rows without a verified source (default: verified rows only)")
    ap.add_argument("--column", default="density_g_cm3",
                    help="density column: density_g_cm3 (sourced where verified) or density_on_file")
    ap.add_argument("--json")
    a = ap.parse_args()

    repo = Path(amorphgen.__file__).parent.parent
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
    print(f"amorphgen at {repo} (commit {commit})")
    verified_only = not a.all_rows
    rows = load(a.csv, a.ratio, verified_only, a.column)
    table = fit(rows)
    print(f"{len(rows)} reference densities ({sum(r['verified'] for r in rows)} with a verified source), "
          f"amorphous/crystal ratio {a.ratio}\n")
    print(f"{'class':26s} {'n':>2s} {'amorph':>6s} {'code':>10s} {'fitted':>6s} {'min':>5s} {'max':>5s} {'LOO err':>7s}")
    for t in table:
        code = "-".join(f"{c:.2f}" for c in (t["f_code"][0], t["f_code"][-1])) if len(t["f_code"]) > 1 else f"{t['f_code'][0]:.2f}"
        loo = f"{t['loo_mean'] * 100:6.1f}%" if t["loo_mean"] is not None else "    n/a"
        print(f"{t['cls']:26s} {t['n']:2d} {t['n_amorphous']:6d} {code:>10s} {t['f_median']:6.2f} "
              f"{t['f_min']:5.2f} {t['f_max']:5.2f} {loo}")
    o = overall(rows, table)
    print(f"\nleave-one-out ({o['n_loo']} compounds in classes of 2+): mean error {o['loo_mean'] * 100:.1f}%, "
          f"within 10%: {o['loo_within10'] * 100:.0f}%, within 20%: {o['loo_within20'] * 100:.0f}%")
    print(f"current code values ({o['n']} compounds): mean error {o['code_mean'] * 100:.1f}%, "
          f"within 10%: {o['code_within10'] * 100:.0f}%, within 20%: {o['code_within20'] * 100:.0f}%")
    print("\nsensitivity of the fitted factor to the amorphous/crystal ratio (classes with crystal references):")
    for ratio in (0.80, 0.85, 0.90, 0.95):
        t2 = {t["cls"]: t["f_median"] for t in fit(load(a.csv, ratio, verified_only, a.column))}
        print(f"  ratio {ratio:.2f}: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(t2.items())
              if any(r["cls"] == k and r["kind"] == "crystal" for r in rows)))
    if a.json:
        json.dump(dict(commit=commit, ratio=a.ratio, overall=o, classes=table, rows=rows), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
