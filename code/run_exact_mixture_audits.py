import os
import csv
import subprocess
import pandas as pd

INSTANCES = [
    "bfsp_hard",
    "bcst_hard",
    "bged_hard",
    "bsp_hard",
    "bfsp_stress_unobservable",
    "bcst_stress_unobservable",
    "bged_stress_unobservable",
    "bsp_stress_unobservable",
]

def infer_budgets_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    budgets = []
    for vcol in [c for c in df.columns if c.startswith("viol_")]:
        suffix = vcol[len("viol_"):]
        pcol = f"prob_{suffix}"
        if pcol not in df.columns:
            continue
        v = df[vcol].astype(float)
        p = df[pcol].astype(float)
        mask = v > 1e-12
        if mask.any():
            vals = (p[mask] - v[mask]).round(12).unique()
            if len(vals) != 1:
                raise RuntimeError(f"{csv_path}: non-unique inferred bound for {pcol}: {vals}")
            bound = float(vals[0])
        else:
            bound = float(p.max())
        budgets.append((pcol, bound))
    return budgets

def deterministic_best(df, budgets):
    mask = pd.Series(True, index=df.index)
    for pcol, bound in budgets:
        mask &= (df[pcol].astype(float) <= bound + 1e-12)
    feas = df[mask].copy()
    if len(feas) == 0:
        return 0, None, None
    best = feas.sort_values("mean").iloc[0]
    return len(feas), float(best["mean"]), str(best["sequence"])

def mixed_lp(csv_path, budgets):
    cmd = ["python3", "randomized_blind_mixture_audit.py", "--csv", csv_path]
    for pcol, bound in budgets:
        cmd.extend(["--budget", f"{pcol}={bound}"])
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    txt = res.stdout

    out = {
        "mixed_feasible": "mixed feasible             : yes" in txt,
        "mixed_mean": None,
        "status_line": None,
    }

    for line in txt.splitlines():
        if line.startswith("status"):
            out["status_line"] = line
        if line.startswith("mixed mean"):
            out["mixed_mean"] = float(line.split(":")[1].strip())
    return out, txt

rows = []
os.makedirs("results_mixture", exist_ok=True)

for name in INSTANCES:
    csv_path = f"results/{name}_all_sequences.csv"
    if not os.path.exists(csv_path):
        subprocess.run(
            ["python3", "run_experiments.py", "--instance", name, "--save_all"],
            check=True,
        )

    budgets = infer_budgets_from_csv(csv_path)
    df = pd.read_csv(csv_path)
    det_count, det_mean, det_seq = deterministic_best(df, budgets)
    mixed, raw_txt = mixed_lp(csv_path, budgets)

    with open(f"results_mixture/{name}_audit.txt", "w") as f:
        f.write(raw_txt)

    row = {
        "instance": name,
        "num_sequences": len(df),
        "deterministically_feasible_count": det_count,
        "best_deterministic_mean": det_mean,
        "best_deterministic_sequence": det_seq,
        "mixed_feasible": mixed["mixed_feasible"],
        "mixed_mean": mixed["mixed_mean"],
    }

    if det_mean is not None and mixed["mixed_mean"] is not None:
        row["mixed_minus_best_deterministic"] = mixed["mixed_mean"] - det_mean
    else:
        row["mixed_minus_best_deterministic"] = None

    for pcol, bound in budgets:
        row[f"bound_{pcol}"] = bound

    rows.append(row)

pd.DataFrame(rows).to_csv("results_mixture/exact_mixture_audit_summary.csv", index=False)
print("Wrote results_mixture/exact_mixture_audit_summary.csv")