#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import math
import random
import statistics
import zlib
from collections import Counter, defaultdict

def stable_seed(*parts):
    s = "|".join(map(str, parts)).encode("utf-8")
    return zlib.crc32(s) & 0xffffffff

def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))

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
    }
    for k, v in reps.items():
        s = s.replace(k, v)
    return s

def make_safe_sequences(rng, entropy, H, A, S):
    if entropy == "low":
        K = 2
    elif entropy == "medium":
        K = 5
    elif entropy == "high":
        K = S
    else:
        raise ValueError(entropy)

    bases = [tuple(rng.randrange(A) for _ in range(H)) for _ in range(K)]

    if entropy == "low":
        # One dominant latent response mode; easy feasibility under loose budgets.
        return [bases[0] if rng.random() < 0.80 else bases[1] for _ in range(S)]
    if entropy == "medium":
        return [rng.choice(bases) for _ in range(S)]
    return bases[:S]

def budget_and_tolerances(budget_mode):
    if budget_mode == "tight":
        return {"prob_catastrophic": 0.0, "prob_degraded": 0.0}, 0, 0
    if budget_mode == "medium":
        return {"prob_catastrophic": 0.18, "prob_degraded": 0.45}, 2, 1
    if budget_mode == "loose":
        return {"prob_catastrophic": 0.45, "prob_degraded": 0.75}, 3, 2
    raise ValueError(budget_mode)

def classify(feasible_count, feasible_fraction, oracle_feasible):
    if feasible_count == 0:
        return "informational" if oracle_feasible else "structural"
    if feasible_fraction >= 0.10:
        return "easy"
    return "hard"

def run_one(seed, entropy, budget_mode, forced_mode, H, A, S, seqs):
    rng = random.Random(stable_seed(seed, entropy, budget_mode, forced_mode, H, A, S))
    safe = make_safe_sequences(rng, entropy, H, A, S)
    budgets, cat_tol, deg_tol = budget_and_tolerances(budget_mode)

    forced_set = set()
    if forced_mode == "forced":
        # Forced scenarios create structural infeasibility only when budgets are tight
        # enough; otherwise they test robustness under unavoidable small event mass.
        forced_set = set(rng.sample(range(S), max(1, int(0.20 * S))))

    feasible_count = 0
    best_mean = float("inf")
    best_violation = float("inf")

    # Uniform scenario weights.
    invS = 1.0 / S

    for seq in seqs:
        cat = 0.0
        deg = 0.0
        loss = 0.0
        for s, safe_seq in enumerate(safe):
            d = hamming(seq, safe_seq)
            c = 1 if (s in forced_set or d > cat_tol) else 0
            g = 1 if d > deg_tol else 0
            cat += invS * c
            deg += invS * g
            loss += invS * (10.0 * d + 1000.0 * c + 100.0 * g)
        violation = max(0.0, cat - budgets["prob_catastrophic"]) + max(0.0, deg - budgets["prob_degraded"])
        if violation <= 1e-12:
            feasible_count += 1
            best_mean = min(best_mean, loss)
        best_violation = min(best_violation, violation)

    # Scenario-wise oracle chooses the safe sequence for each scenario. It has
    # zero degradation by construction and only suffers forced catastrophic mass.
    oracle_cat = len(forced_set) * invS
    oracle_deg = 0.0
    oracle_feasible = (
        oracle_cat <= budgets["prob_catastrophic"] + 1e-12
        and oracle_deg <= budgets["prob_degraded"] + 1e-12
    )

    feasible_fraction = feasible_count / len(seqs)
    regime = classify(feasible_count, feasible_fraction, oracle_feasible)

    return {
        "seed": seed,
        "entropy": entropy,
        "budget_mode": budget_mode,
        "forced_mode": forced_mode,
        "H": H,
        "A": A,
        "S": S,
        "num_sequences": len(seqs),
        "feasible_count": feasible_count,
        "feasible_fraction": feasible_fraction,
        "oracle_feasible": oracle_feasible,
        "oracle_catastrophic": oracle_cat,
        "oracle_degraded": oracle_deg,
        "best_feasible_mean": "" if math.isinf(best_mean) else best_mean,
        "best_violation": best_violation,
        "regime": regime,
        "budgets_json": json.dumps(budgets, sort_keys=True),
    }

def write_latex_summary(rows, path):
    groups = defaultdict(list)
    for r in rows:
        key = (r["entropy"], r["budget_mode"], r["forced_mode"])
        groups[key].append(r)

    with open(path, "w") as f:
        f.write(r"""\begin{table}[t]
\centering
\scriptsize
\caption{Randomized generator study with evidence-based regime counts. The sweep varies latent-response entropy, budget strictness, and unavoidable forced-harm mass.}
\label{tab:random_generator_study_v2}
\begin{tabular}{lllrrrrr}
\toprule
Entropy & Budget & Forced & Easy & Hard & Informational & Structural & Mean feas. frac. \\
\midrule
""")
        for key in sorted(groups):
            rs = groups[key]
            cnt = Counter(r["regime"] for r in rs)
            mean_frac = statistics.mean(float(r["feasible_fraction"]) for r in rs)
            f.write(
                f"{latex_escape(key[0])} & {latex_escape(key[1])} & {latex_escape(key[2])} & "
                f"{cnt.get('easy',0)} & {cnt.get('hard',0)} & "
                f"{cnt.get('informational',0)} & {cnt.get('structural',0)} & "
                f"{mean_frac:.4f} \\\\\n"
            )
        f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")

def write_compact_latex(rows, path):
    cnt = Counter(r["regime"] for r in rows)
    total = len(rows)
    with open(path, "w") as f:
        f.write(r"""\begin{table}[t]
\centering
\small
\caption{Aggregate randomized-generator regime coverage.}
\label{tab:random_generator_study_v2_compact}
\begin{tabular}{lrrrrr}
\toprule
Study & Total & Easy & Hard & Informational & Structural \\
\midrule
""")
        f.write(
            f"Random generator v2 & {total} & {cnt.get('easy',0)} & {cnt.get('hard',0)} & "
            f"{cnt.get('informational',0)} & {cnt.get('structural',0)} \\\\\n"
        )
        f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=80)
    ap.add_argument("--H", type=int, default=5)
    ap.add_argument("--A", type=int, default=4)
    ap.add_argument("--S", type=int, default=24)
    ap.add_argument("--out", default="results/random_generator_study_v2.csv")
    ap.add_argument("--latex", default="paper_tables/table_random_generator_study_v2.tex")
    ap.add_argument("--summary-latex", default="paper_tables/table_random_generator_study_v2_compact.tex")
    args = ap.parse_args()

    seqs = list(itertools.product(range(args.A), repeat=args.H))
    rows = []

    for entropy in ["low", "medium", "high"]:
        for budget_mode in ["tight", "medium", "loose"]:
            for forced_mode in ["none", "forced"]:
                for seed in range(args.seeds):
                    rows.append(run_one(seed, entropy, budget_mode, forced_mode, args.H, args.A, args.S, seqs))

    with open(args.out, "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_latex_summary(rows, args.latex)
    write_compact_latex(rows, args.summary_latex)

    cnt = Counter(r["regime"] for r in rows)
    print(f"[OK] wrote {args.out} ({len(rows)} rows)")
    print(f"[OK] wrote {args.latex}")
    print(f"[OK] wrote {args.summary_latex}")
    print("[SUMMARY]", dict(cnt))

if __name__ == "__main__":
    main()
