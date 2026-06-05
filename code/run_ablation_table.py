#!/usr/bin/env python3
"""
Ablation table for the governed blind-commitment contract.

This script summarizes what is lost when each component is removed:
  - remove governance budgets: best mean may be infeasible;
  - remove oracle diagnosis: blind-infeasible cases cannot be separated;
  - remove mixture audit: deterministic-only conclusions may miss convexification;
  - remove ordered-risk selectors: selector diversity disappears by definition;
  - remove public transfer: evidence is restricted to synthetic core.

The script uses existing exact all-sequence CSVs and public transfer outputs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, List

DEFAULT_BUDGETS: Dict[str, Dict[str, float]] = {
    "bfsp_hard": {"prob_catastrophic": 0.075, "prob_degraded": 0.3, "prob_rejected": 0.36},
    "bcst_hard": {"prob_catastrophic": 0.13, "prob_toxicity": 0.34, "prob_moderate_harm": 0.82},
    "bged_hard": {"prob_catastrophic": 0.22, "prob_no_benefit": 0.24, "prob_offtarget": 0.24, "prob_immune": 0.22},
    "bsp_hard": {"prob_catastrophic": 0.075, "prob_degraded": 0.3, "prob_rejected": 0.36, "prob_shortage": 0.36},
    "bfsp_stress_unobservable": {"prob_catastrophic": 0.0, "prob_degraded": 0.0},
    "bcst_stress_unobservable": {"prob_catastrophic": 0.0, "prob_toxicity": 0.0},
    "bged_stress_unobservable": {"prob_catastrophic": 0.0, "prob_offtarget": 0.0, "prob_immune": 0.0},
    "bsp_stress_unobservable": {"prob_catastrophic": 0.0, "prob_degraded": 0.0, "prob_shortage": 0.0},
    "sysadmin_transfer_hard_H6": {"prob_catastrophic": 0.075, "prob_degraded": 0.3, "prob_shortage": 0.3},
    "sysadmin_transfer_stress_unobservable_H3": {"prob_catastrophic": 0.0, "prob_degraded": 0.0, "prob_shortage": 0.0},
    "tiger_transfer_hard_H4": {"prob_catastrophic": 0.20, "prob_degraded": 0.45},
    "tiger_transfer_stress_unobservable_H2": {"prob_catastrophic": 0.0, "prob_degraded": 0.0},
    "rocksample_transfer_hard_H5": {"prob_catastrophic": 0.15, "prob_degraded": 0.35, "prob_shortage": 0.35},
    "rocksample_transfer_stress_unobservable_H3": {"prob_catastrophic": 0.0, "prob_degraded": 0.0},
}

SELECTORS = ["mean", "cvar05", "cvar25", "distorted_ordered", "wowa_balanced", "wowa_severe"]


def instance_name(path: str) -> str:
    return os.path.basename(path).replace("_all_sequences.csv", "")


def fnum(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def feasible(row, budgets):
    return all(fnum(row.get(k, 0.0)) <= v + 1e-12 for k, v in budgets.items())


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def seq_col(rows):
    for c in ["seq", "sequence", "policy", "actions"]:
        if rows and c in rows[0]:
            return c
    return "seq"


def selector_col(rows, preferred):
    if not rows:
        return None
    cols = rows[0].keys()
    if preferred in cols:
        return preferred
    aliases = {"cvar05": ["cvar_0.05", "CVaR_0.05"], "cvar25": ["cvar_0.25", "CVaR_0.25"], "distorted_ordered": ["wowa_balanced", "ordered_risk"]}
    for a in aliases.get(preferred, []):
        if a in cols:
            return a
    return None


def summarize_instance(path: str) -> Dict[str, object]:
    name = instance_name(path)
    rows = read(path)
    budgets = DEFAULT_BUDGETS.get(name)
    if not budgets:
        return {"instance": name, "status": "skipped_no_budgets"}
    sc = seq_col(rows)
    mean_col = selector_col(rows, "mean") or "mean"
    best_mean = min(rows, key=lambda r: fnum(r.get(mean_col)))
    feasible_rows = [r for r in rows if feasible(r, budgets)]
    best_gov = min(feasible_rows, key=lambda r: fnum(r.get(mean_col))) if feasible_rows else None
    selector_choices = set()
    for sel in SELECTORS:
        col = selector_col(rows, sel)
        if col and feasible_rows:
            selector_choices.add(min(feasible_rows, key=lambda r: fnum(r.get(col))).get(sc, ""))
    return {
        "instance": name,
        "num_sequences": len(rows),
        "feasible_fraction": len(feasible_rows) / len(rows) if rows else 0.0,
        "best_unconstrained_mean": fnum(best_mean.get(mean_col)),
        "best_unconstrained_feasible": feasible(best_mean, budgets),
        "governed_mean": fnum(best_gov.get(mean_col)) if best_gov else "NA",
        "governed_exists": bool(best_gov),
        "governance_changes_decision": bool(best_gov and best_mean.get(sc) != best_gov.get(sc)),
        "distinct_selector_choices": len(selector_choices) if feasible_rows else 0,
    }


def write_latex(rows: List[Dict[str, object]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    total = len(rows)
    gov_changes = sum(1 for r in rows if str(r.get("governance_changes_decision")) == "True")
    unconstrained_infeasible = sum(1 for r in rows if str(r.get("best_unconstrained_feasible")) == "False")
    selector_diverse = sum(1 for r in rows if int(float(r.get("distinct_selector_choices", 0))) > 1)
    no_feasible = sum(1 for r in rows if str(r.get("governed_exists")) == "False")
    public = sum(1 for r in rows if any(x in r.get("instance", "") for x in ["sysadmin", "tiger", "rocksample"]))
    with open(path, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Ablation summary: how often each component changes the scientific conclusion.}\n")
        f.write("\\label{tab:ablation_summary}\n")
        f.write("\\begin{tabular}{lcc}\n\\toprule\n")
        f.write("Removed component & Observable consequence & Count \\\\\n\\midrule\n")
        f.write(f"Governance budgets & Unconstrained best-mean row is infeasible & {unconstrained_infeasible}/{total} \\\\\n")
        f.write(f"Governance-first choice & Governed choice differs from unconstrained mean & {gov_changes}/{total} \\\\\n")
        f.write(f"Ordered-risk selectors & More than one selector choice exists & {selector_diverse}/{total} \\\\\n")
        f.write(f"Oracle diagnosis & Blind feasible set is empty and needs diagnosis & {no_feasible}/{total} \\\\\n")
        f.write(f"Public transfers & Transfer rows included beyond synthetic core & {public}/{total} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"[OK] wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/*_all_sequences.csv,results_public/*_all_sequences.csv")
    ap.add_argument("--out", default="paper_tables/table_ablation_rows.csv")
    ap.add_argument("--latex", default="paper_tables/table_ablation_summary.tex")
    args = ap.parse_args()
    paths = []
    for g in args.glob.split(","):
        paths.extend(glob.glob(g.strip()))
    rows = [summarize_instance(p) for p in sorted(set(paths))]
    rows = [r for r in rows if r.get("status") != "skipped_no_budgets"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        fieldnames = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] wrote {args.out}")
    write_latex(rows, args.latex)

if __name__ == "__main__":
    main()
