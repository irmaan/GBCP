#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict

def as_float(x, default=float("nan")):
    if x is None:
        return default
    s = str(x).strip()
    if s.lower() in {"inf", "infinity"}:
        return float("inf")
    if s.lower() in {"-inf", "-infinity"}:
        return float("-inf")
    if s == "":
        return default
    try:
        return float(s)
    except Exception:
        return default

def as_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "1", "yes", "y"}

def latex_escape(s):
    s = str(s)
    reps = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in reps.items():
        s = s.replace(k, v)
    return s

def classify(row):
    frac = as_float(row.get("feasible_fraction"), 0.0)
    oracle_feas = as_bool(row.get("oracle_feasible"))
    det_feas = as_bool(row.get("deterministically_feasible"))

    if det_feas or frac > 0:
        if frac >= 0.10:
            return "feasible-easy"
        return "feasible-hard"
    if oracle_feas:
        return "blind-infeasible / oracle-feasible"
    return "structural infeasible"

def short_instance_name(name):
    name = str(name)
    for prefix in ["tiger_transfer_", "rocksample_transfer_", "sysadmin_transfer_"]:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name

def fmt_float(x, nd=3):
    x = as_float(x)
    if math.isinf(x):
        return r"$\infty$"
    if math.isnan(x):
        return "--"
    return f"{x:.{nd}f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="paper_tables/table_public_extra_transfer.csv")
    ap.add_argument("--out-csv", default="paper_tables/table_public_transfer_regime_safe.csv")
    ap.add_argument("--out-tex", default="paper_tables/table_public_transfer_regime_safe.tex")
    ap.add_argument("--summary-tex", default="paper_tables/table_public_transfer_family_coverage.tex")
    args = ap.parse_args()

    rows = []
    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            row["evidence_regime"] = classify(row)
            row["label_warning"] = ""
            inst = row.get("instance", "")
            if "stress_unobservable" in inst and row["evidence_regime"] != "blind-infeasible / oracle-feasible":
                row["label_warning"] = (
                    "Original filename contains stress_unobservable, but evidence does not support "
                    "informational infeasibility because oracle_feasible is false."
                )
            rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []
    if "evidence_regime" not in fieldnames:
        fieldnames.append("evidence_regime")
    if "label_warning" not in fieldnames:
        fieldnames.append("label_warning")

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Full compact LaTeX table for paper/appendix
    with open(args.out_tex, "w") as f:
        f.write(r"""\begin{table}[t]
\centering
\scriptsize
\caption{Evidence-safe public-transfer summary. Regime labels are assigned from the measured feasible fraction and oracle feasibility, not from filenames.}
\label{tab:public_transfer_regime_safe}
\begin{tabular}{llrrrrl}
\toprule
Family & Instance & Seq. & Feas. frac. & Gov. mean & Oracle mean & Evidence label \\
\midrule
""")
        for r in rows:
            fam = latex_escape(r.get("family", ""))
            inst = latex_escape(short_instance_name(r.get("instance", "")))
            nseq = int(as_float(r.get("num_sequences"), 0))
            frac = fmt_float(r.get("feasible_fraction"), 4)
            gm = fmt_float(r.get("governed_mean"), 1)
            om = fmt_float(r.get("oracle_mean"), 1)
            lab = latex_escape(r.get("evidence_regime", ""))
            f.write(f"{fam} & {inst} & {nseq} & {frac} & {gm} & {om} & {lab} \\\\\n")
        f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")

    # Family x regime coverage table
    fam_counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        fam_counts[r.get("family", "")][r.get("evidence_regime", "")] += 1

    regimes = [
        "feasible-easy",
        "feasible-hard",
        "blind-infeasible / oracle-feasible",
        "structural infeasible",
    ]

    with open(args.summary_tex, "w") as f:
        f.write(r"""\begin{table}[t]
\centering
\small
\caption{Public-transfer coverage by evidence-based regime label.}
\label{tab:public_transfer_family_coverage}
\begin{tabular}{lrrrr}
\toprule
Family & Feas.-easy & Feas.-hard & Blind-infeas./oracle-feas. & Structural infeas. \\
\midrule
""")
        for fam in sorted(fam_counts):
            vals = [fam_counts[fam].get(reg, 0) for reg in regimes]
            f.write(f"{latex_escape(fam)} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} \\\\\n")
        f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")

    warnings = [r for r in rows if r.get("label_warning")]
    print(f"[OK] wrote {args.out_csv}")
    print(f"[OK] wrote {args.out_tex}")
    print(f"[OK] wrote {args.summary_tex}")
    if warnings:
        print("[WARN] Filename/evidence mismatches found; paper should use evidence labels:")
        for r in warnings:
            print(f"  - {r.get('instance')}: {r.get('label_warning')}")

if __name__ == "__main__":
    main()
