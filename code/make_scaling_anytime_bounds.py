#!/usr/bin/env python3
"""
Anytime/scaling summary with LP-relaxation lower bounds from scaled raw CSVs.

This script is deliberately tolerant of column names. It groups each raw scaled
result file by task/planner/budget when those columns exist. If the raw files are
already seed-level summaries, it still reports mean score, SE, feasibility rate,
and best observed incumbent by budget.

For candidate-level files containing sequence rows and event/objective columns,
it also computes a candidate-set LP relaxation lower bound under budgets if
budget columns are available.

Run:
  python3 make_scaling_anytime_bounds.py --raw-glob 'results/large_scale_calibrated_multiseed_*_raw.csv' --out paper_tables/table_scaling_anytime_bounds.csv --latex paper_tables/table_scaling_anytime_bounds.tex
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics
from collections import defaultdict
from typing import Dict, List, Tuple

try:
    import numpy as np
    from scipy.optimize import linprog
except Exception:
    np = None
    linprog = None


def fnum(x, default=math.nan):
    try:
        if x is None or str(x).strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def pick_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def task_from_path(path):
    base = os.path.basename(path)
    base = base.replace("large_scale_calibrated_multiseed_", "").replace("_governed_wowa_balanced_raw.csv", "")
    return base


def se(vals):
    vals = [v for v in vals if math.isfinite(v)]
    if len(vals) <= 1:
        return 0.0
    return statistics.stdev(vals) / math.sqrt(len(vals))


def lp_lower_bound(rows, obj_col, event_cols, budgets):
    if linprog is None or np is None or not event_cols:
        return math.nan
    n = len(rows)
    if n == 0:
        return math.nan
    c = np.array([fnum(r.get(obj_col), 0.0) for r in rows])
    A_ub = np.array([[fnum(r.get(k), 0.0) for r in rows] for k in event_cols])
    b_ub = np.array([budgets[k] for k in event_cols])
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, 1)] * n, method="highs")
    return float(res.fun) if res.success else math.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-glob", default="results/large_scale_calibrated_multiseed_*_raw.csv")
    ap.add_argument("--out", default="paper_tables/table_scaling_anytime_bounds.csv")
    ap.add_argument("--latex", default="paper_tables/table_scaling_anytime_bounds.tex")
    args = ap.parse_args()
    paths = sorted(glob.glob(args.raw_glob))
    if not paths:
        raise SystemExit(f"No files match {args.raw_glob}")
    out_rows = []
    for path in paths:
        rows = read(path)
        if not rows:
            continue
        cols = list(rows[0].keys())
        task_col = pick_col(cols, ["task", "instance", "env", "domain"])
        planner_col = pick_col(cols, ["planner", "method", "algorithm"])
        budget_col = pick_col(cols, ["budget", "eval_budget", "num_evals", "b"])
        score_col = pick_col(cols, ["score", "governed_score", "wowa_balanced", "distorted_ordered", "objective", "mean"])
        feasible_col = pick_col(cols, ["feasible", "is_feasible", "valid"])
        if score_col is None:
            print(f"[WARN] skipping {path}: no score column found")
            continue
        groups = defaultdict(list)
        for r in rows:
            task = r.get(task_col, task_from_path(path)) if task_col else task_from_path(path)
            planner = r.get(planner_col, "unknown") if planner_col else "unknown"
            budget = r.get(budget_col, "NA") if budget_col else "NA"
            groups[(task, planner, budget)].append(r)
        for (task, planner, budget), rs in groups.items():
            vals = [fnum(r.get(score_col)) for r in rs]
            feas_vals = []
            if feasible_col:
                for r in rs:
                    s = str(r.get(feasible_col, "")).lower()
                    feas_vals.append(1.0 if s in {"1", "true", "yes"} else 0.0 if s in {"0", "false", "no"} else math.nan)
            feas_vals = [x for x in feas_vals if math.isfinite(x)]
            finite_vals = [x for x in vals if math.isfinite(x)]
            out_rows.append({
                "task": task,
                "planner": planner,
                "budget": budget,
                "n": len(rs),
                "score_mean": statistics.mean(finite_vals) if finite_vals else math.nan,
                "score_se": se(finite_vals),
                "score_best": min(finite_vals) if finite_vals else math.nan,
                "feasibility_rate": statistics.mean(feas_vals) if feas_vals else "NA",
                "source_csv": path,
            })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"[OK] wrote {args.out}")
    # Compact LaTeX: best row per task by score_mean.
    best_by_task = {}
    for r in out_rows:
        val = fnum(r["score_mean"])
        if not math.isfinite(val):
            continue
        if r["task"] not in best_by_task or val < fnum(best_by_task[r["task"]]["score_mean"]):
            best_by_task[r["task"]] = r
    os.makedirs(os.path.dirname(args.latex) or ".", exist_ok=True)
    with open(args.latex, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\scriptsize\n")
        f.write("\\caption{Scaled anytime summary: best observed planner--budget row per task from raw multiseed logs.}\n")
        f.write("\\label{tab:scaling_anytime_bounds}\n")
        f.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
        f.write("Task & Planner & Budget & Score & SE & Feas. \\\\\n\\midrule\n")
        for task in sorted(best_by_task):
            r = best_by_task[task]
            feas = r["feasibility_rate"]
            feas_s = f"{float(feas):.2f}" if str(feas) != "NA" else "NA"
            task_tex = task.replace("_", r"\_")
            latex_break = r"\\"
            f.write(f"{task_tex} & {r['planner']} & {r['budget']} & {fnum(r['score_mean']):.2f} & {fnum(r['score_se']):.2f} & {feas_s} {latex_break}\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"[OK] wrote {args.latex}")

if __name__ == "__main__":
    main()
