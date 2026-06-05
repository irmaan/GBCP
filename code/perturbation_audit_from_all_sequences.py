#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Edit paths if needed. These defaults assume you run the script from the project root.
CASES = {
    "bfsp_hard": {
        "csv": "results/bfsp_hard_all_sequences.csv",
        "budgets": {"prob_catastrophic": 0.075, "prob_degraded": 0.30, "prob_rejected": 0.36},
    },
    "bcst_hard": {
        "csv": "results/bcst_hard_all_sequences.csv",
        "budgets": {"prob_catastrophic": 0.13, "prob_toxicity": 0.34, "prob_moderate_harm": 0.82},
    },
    "bged_hard": {
        "csv": "results/bged_hard_all_sequences.csv",
        "budgets": {"prob_catastrophic": 0.22, "prob_no_benefit": 0.24, "prob_offtarget": 0.24, "prob_immune": 0.22},
    },
    "bsp_hard": {
        "csv": "results/bsp_hard_all_sequences.csv",
        "budgets": {"prob_catastrophic": 0.075, "prob_degraded": 0.30, "prob_rejected": 0.36, "prob_shortage": 0.36},
    },
    "bfsp_stress_unobservable": {
        "csv": "results/bfsp_stress_unobservable_all_sequences.csv",
        "budgets": {"prob_catastrophic": 0.0, "prob_degraded": 0.0},
    },
    "bcst_stress_unobservable": {
        "csv": "results/bcst_stress_unobservable_all_sequences.csv",
        "budgets": {"prob_catastrophic": 0.0, "prob_toxicity": 0.0},
    },
    "bged_stress_unobservable": {
        "csv": "results/bged_stress_unobservable_all_sequences.csv",
        "budgets": {"prob_catastrophic": 0.0, "prob_offtarget": 0.0, "prob_immune": 0.0},
    },
    "bsp_stress_unobservable": {
        "csv": "results/bsp_stress_unobservable_all_sequences.csv",
        "budgets": {"prob_catastrophic": 0.0, "prob_degraded": 0.0, "prob_shortage": 0.0},
    },
}

BUDGET_MULTS = [0.98, 1.00, 1.02]
LOSS_SCALES = [0.90, 1.00, 1.10]

def feasible_mask(df, budgets):
    mask = np.ones(len(df), dtype=bool)
    for col, bound in budgets.items():
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in CSV")
        mask &= (df[col].astype(float).to_numpy() <= float(bound) + 1e-12)
    return mask

def best_feasible_sequence(df, budgets, loss_scale=1.0):
    mask = feasible_mask(df, budgets)
    if not mask.any():
        return None
    sub = df.loc[mask].copy()
    sub["_scaled_mean"] = sub["mean"].astype(float) * float(loss_scale)
    return str(sub.sort_values(["_scaled_mean", "sequence"]).iloc[0]["sequence"])

def least_violation_sequence(df, budgets, weight_overrides=None):
    weight_overrides = weight_overrides or {}
    viol = np.zeros(len(df), dtype=float)
    for col, bound in budgets.items():
        weights = float(weight_overrides.get(col, 1.0))
        excess = np.maximum(df[col].astype(float).to_numpy() - float(bound), 0.0)
        viol += weights * excess
    tmp = df.copy()
    tmp["_viol"] = viol
    tmp["_mean"] = tmp["mean"].astype(float)
    return str(tmp.sort_values(["_viol", "_mean", "sequence"]).iloc[0]["sequence"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/perturbation_preservation_summary.csv")
    args = ap.parse_args()

    rows = []
    for name, cfg in CASES.items():
        csv_path = Path(cfg["csv"])
        if not csv_path.exists():
            print(f"[SKIP] {name}: missing {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        budgets = dict(cfg["budgets"])

        baseline_feasible = best_feasible_sequence(df, budgets, loss_scale=1.0)
        if baseline_feasible is not None:
            baseline_type = "deterministic_feasible"
            baseline_seq = baseline_feasible
        else:
            baseline_type = "deterministic_infeasible"
            baseline_seq = least_violation_sequence(df, budgets)

        # Budget perturbation
        budget_preserved = 0
        budget_total = 0
        for mult in BUDGET_MULTS:
            pert = {k: v * mult for k, v in budgets.items()}
            seq = best_feasible_sequence(df, pert, loss_scale=1.0)
            lab = "deterministic_feasible" if seq is not None else "deterministic_infeasible"
            if seq is None:
                seq = least_violation_sequence(df, pert)
            budget_preserved += int((lab == baseline_type) and (seq == baseline_seq))
            budget_total += 1

        # Loss scaling
        loss_preserved = 0
        loss_total = 0
        for scale in LOSS_SCALES:
            seq = best_feasible_sequence(df, budgets, loss_scale=scale)
            if seq is None:
                seq = least_violation_sequence(df, budgets)
            loss_preserved += int(seq == baseline_seq)
            loss_total += 1

        # Event-weight sensitivity on infeasible cases only
        weight_preserved = 0
        weight_total = 0
        if baseline_type == "deterministic_infeasible":
            for col in budgets.keys():
                seq = least_violation_sequence(df, budgets, {col: 10.0})
                weight_preserved += int(seq == baseline_seq)
                weight_total += 1

        rows.append({
            "instance": name,
            "baseline_type": baseline_type,
            "baseline_sequence": baseline_seq,
            "budget_preserved": f"{budget_preserved}/{budget_total}",
            "lossscale_preserved": f"{loss_preserved}/{loss_total}",
            "eventweight_preserved": f"{weight_preserved}/{weight_total}" if weight_total else "---",
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[OK] wrote {out}")

if __name__ == "__main__":
    main()
