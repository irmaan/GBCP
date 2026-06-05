#!/usr/bin/env python3
import argparse
import pandas as pd
import re

def planner_short(p):
    return {'beam':'beam','cem':'CEM','evolutionary':'evo.','open_loop_mcts':'MCTS'}.get(p, p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    summ = df[df["score_mean"].notna()].copy()

    best_struct = []
    best_mcts = []
    for inst, sub in summ.groupby("instance"):
        ss = sub[sub["planner"].isin(["beam", "cem", "evolutionary"])]
        ms = sub[sub["planner"] == "open_loop_mcts"]
        if ss.empty or ms.empty:
            continue
        b1 = ss.loc[ss["score_mean"].idxmin()].copy()
        b2 = ms.loc[ms["score_mean"].idxmin()].copy()
        dom = inst.split("_")[0].upper()
        hz = int(re.search(r"_H(\d+)_", inst).group(1))
        best_struct.append((dom, hz, b1["planner"], int(b1["budget"]), float(b1["score_mean"]), float(b1["feasible_rate"]), float(b1["score_sem"])))
        best_mcts.append((dom, hz, b2["planner"], int(b2["budget"]), float(b2["score_mean"]), float(b2["feasible_rate"]), float(b2["score_sem"])))

    best_struct = sorted(best_struct)
    best_mcts = sorted(best_mcts)

    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\scriptsize")
    print(r"\caption{Full 12-instance scaled hard-suite summary at the best budget for each planner family. Lower score is better.}")
    print(r"\label{tab:scaled_full_best}")
    print(r"\begin{tabular}{l l c c c l c c c}")
    print(r"\toprule")
    print(r"Task & Best structured & Score & Feas. & SE & Best MCTS & Score & Feas. & SE \\")
    print(r"\midrule")
    for s, m in zip(best_struct, best_mcts):
        dom, hz, p1, b1, sc1, fr1, se1 = s
        _, _, p2, b2, sc2, fr2, se2 = m
        print(f"\\textsc{{{dom}}}-hard-$H{hz}$ & {planner_short(p1)} ({b1}) & {sc1:.1f} & {fr1:.2f} & {se1:.1f} & {planner_short(p2)} ({b2}) & {sc2:.1f} & {fr2:.2f} & {se2:.1f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

if __name__ == "__main__":
    main()
