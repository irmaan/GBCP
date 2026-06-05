#!/usr/bin/env python3
"""
Randomized generator study to reduce the "hand-engineered benchmark" objection.

This script samples many compact latent-response GBCP instances and reports how
often each regime appears as scenario entropy and budget strictness vary.

It is intentionally generic and independent of the custom domain families. It
uses a random latent compatibility model:
  - H stages, A actions, S scenarios.
  - Each scenario has a preferred action at each stage.
  - Loss and harm probabilities depend on mismatch count plus noise.
  - Oracle can choose scenario-specific sequences; blind planner cannot.

Run:
  python3 random_regime_generator_study.py --out results/random_generator_study.csv --latex paper_tables/table_random_generator_study.tex
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

try:
    import numpy as np
except Exception as e:
    raise SystemExit("This script needs numpy. Try: python3 -m pip install numpy") from e


def entropy(p: np.ndarray) -> float:
    return float(-np.sum(p * np.log(p + 1e-15)) / math.log(len(p)))


def classify_instance(rng: np.random.Generator, H: int, A: int, S: int, alpha: float, budget_scale: float, noise: float) -> Dict[str, object]:
    # Scenario probabilities: lower alpha => low entropy, higher alpha => nearly uniform.
    p = rng.dirichlet(np.full(S, alpha))
    preferred = rng.integers(0, A, size=(S, H))
    scenario_harshness = rng.lognormal(mean=0.0, sigma=0.4, size=S)
    seqs = list(itertools.product(range(A), repeat=H))

    # Compute event rates for every blind sequence.
    rows = []
    for seq in seqs:
        mismatch = np.array([sum(seq[t] != preferred[s, t] for t in range(H)) for s in range(S)], dtype=float)
        # Two harm classes: catastrophic is stricter; degraded is broader.
        cat_ind = (mismatch >= max(1, int(math.ceil(0.65 * H)))).astype(float)
        deg_ind = (mismatch >= max(1, int(math.ceil(0.35 * H)))).astype(float)
        # Mild random flips near boundaries to avoid deterministic artifacts.
        if noise > 0:
            flip_cat = rng.random(S) < noise * 0.03
            flip_deg = rng.random(S) < noise * 0.05
            cat_ind = np.where(flip_cat, 1.0 - cat_ind, cat_ind)
            deg_ind = np.where(flip_deg, 1.0 - deg_ind, deg_ind)
        mean_loss = float(np.sum(p * (10.0 * mismatch * scenario_harshness + 500.0 * cat_ind + 80.0 * deg_ind)))
        rows.append((seq, float(np.sum(p * cat_ind)), float(np.sum(p * deg_ind)), mean_loss))

    # Budget strictness is relative to a simple baseline event rate.
    # Smaller budget_scale => stricter governance.
    cat_budget = min(1.0, budget_scale * 0.20)
    deg_budget = min(1.0, budget_scale * 0.45)
    feasible = [r for r in rows if r[1] <= cat_budget + 1e-12 and r[2] <= deg_budget + 1e-12]

    # Scenario-wise oracle: choose best sequence for each scenario. In this model, the
    # scenario's preferred sequence has zero mismatch and zero harm.
    oracle_cat = 0.0
    oracle_deg = 0.0
    oracle_feasible = (oracle_cat <= cat_budget + 1e-12 and oracle_deg <= deg_budget + 1e-12)

    if feasible:
        regime = "feasible_hard" if len(feasible) / len(rows) < 0.10 else "feasible_easy"
    else:
        regime = "stress_unobservable" if oracle_feasible else "stress_impossible"

    best_mean = min(r[3] for r in feasible) if feasible else min(r[3] for r in rows)
    least_violation = min(max(0.0, r[1] - cat_budget) + max(0.0, r[2] - deg_budget) for r in rows)
    return {
        "H": H,
        "A": A,
        "S": S,
        "alpha": alpha,
        "scenario_entropy": entropy(p),
        "budget_scale": budget_scale,
        "noise": noise,
        "regime": regime,
        "feasible_fraction": len(feasible) / len(rows),
        "oracle_feasible": oracle_feasible,
        "best_mean_or_least_violation_mean": best_mean,
        "least_violation": least_violation,
    }


def write_latex_summary(csv_path: str, latex_path: str) -> None:
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    groups = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    for r in rows:
        key = (r["alpha"], r["budget_scale"])
        groups[key][r["regime"]] += 1
        totals[key] += 1
    os.makedirs(os.path.dirname(latex_path) or ".", exist_ok=True)
    with open(latex_path, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Randomized generator study: regime frequencies over sampled latent-response instances.}\n")
        f.write("\\label{tab:random_generator_study}\n")
        f.write("\\begin{tabular}{ccrrrr}\n\\toprule\n")
        f.write("Dirichlet $\\alpha$ & Budget scale & Easy & Hard & Unobs. & Impossible \\\\\n")
        f.write("\\midrule\n")
        for key in sorted(groups, key=lambda x: (float(x[0]), float(x[1]))):
            g = groups[key]
            n = totals[key]
            vals = [g.get("feasible_easy", 0), g.get("feasible_hard", 0), g.get("stress_unobservable", 0), g.get("stress_impossible", 0)]
            f.write(f"{float(key[0]):.2g} & {float(key[1]):.2g} & " + " & ".join(f"{v}/{n}" for v in vals) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"[OK] wrote {latex_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/random_generator_study.csv")
    ap.add_argument("--latex", default="paper_tables/table_random_generator_study.tex")
    ap.add_argument("--seeds", type=int, default=50, help="instances per grid cell")
    ap.add_argument("--H", type=int, default=5)
    ap.add_argument("--A", type=int, default=4)
    ap.add_argument("--S", type=int, default=24)
    ap.add_argument("--alphas", default="0.2,1.0,5.0")
    ap.add_argument("--budget-scales", default="0.0,0.5,1.0,1.5,2.0")
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    alphas = [float(x) for x in args.alphas.split(",")]
    bscales = [float(x) for x in args.budget_scales.split(",")]
    rows: List[Dict[str, object]] = []
    for alpha in alphas:
        for bs in bscales:
            for _ in range(args.seeds):
                rows.append(classify_instance(rng, args.H, args.A, args.S, alpha, bs, args.noise))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] wrote {args.out} ({len(rows)} rows)")
    write_latex_summary(args.out, args.latex)

if __name__ == "__main__":
    main()
