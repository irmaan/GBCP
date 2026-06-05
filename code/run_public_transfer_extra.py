#!/usr/bin/env python3
"""
Two additional public-domain transfer templates for Reviewer 3.

The goal is not to claim these are canonical benchmark scores. The goal is to
show that the governed blind-commitment contract can be instantiated on
recognizable planning templates beyond the custom synthetic core and SysAdmin:

  1. Tiger-style hidden-door commitment with observations suppressed.
  2. RockSample-style hidden-rock commitment with observations suppressed.

Each template exports exact all-sequence tables and summary rows for easy,
hard, stress-unobservable, and stress-impossible variants when available.

Run:
  python3 run_public_transfer_extra.py --variant all

Then audit with:
  python3 exact_governed_sp_baseline.py --glob 'results_public/*transfer*_all_sequences.csv' --out paper_tables/table_public_extra_exact_sp.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

@dataclass(frozen=True)
class Scenario:
    name: str
    p: float
    kind: str

@dataclass(frozen=True)
class Instance:
    name: str
    actions: List[str]
    horizon: int
    scenarios: List[Scenario]
    budgets: Dict[str, float]
    family: str
    notes: str


def seq_str(seq: Sequence[str]) -> str:
    return "|".join(seq)


def cvar_tail(losses: List[float], probs: List[float], alpha: float) -> float:
    pairs = sorted(zip(losses, probs), key=lambda x: x[0], reverse=True)
    remain = alpha
    acc = 0.0
    for loss, p in pairs:
        take = min(remain, p)
        acc += take * loss
        remain -= take
        if remain <= 1e-12:
            break
    if remain > 1e-9:
        acc += remain * pairs[-1][0]
    return acc / alpha


def distorted_ordered(losses: List[float], probs: List[float], gamma: float = 0.55) -> float:
    # Transparent rank-sensitive probability distortion over increasing losses.
    pairs = sorted(zip(losses, probs), key=lambda x: x[0])
    prev = 0.0
    total = 0.0
    def g(u: float) -> float:
        return u ** gamma
    for loss, p in pairs:
        cur = min(1.0, prev + p)
        w = g(cur) - g(prev)
        total += w * loss
        prev = cur
    return total


def tiger_loss_and_events(seq: Sequence[str], sc: Scenario, instance: Instance) -> Tuple[float, Dict[str, int]]:
    # Tiger template: a hidden dangerous door. Listening is uninformative during commitment
    # and costs time. Opening the tiger door is catastrophic; opening treasure gives reward.
    tiger_left = sc.kind == "tiger_left"
    opened_wrong = False
    opened_safe = False
    wasted = 0
    for a in seq:
        if a == "listen":
            wasted += 1
        elif a == "open_left":
            if tiger_left:
                opened_wrong = True
            else:
                opened_safe = True
        elif a == "open_right":
            if not tiger_left:
                opened_wrong = True
            else:
                opened_safe = True
        elif a == "wait":
            wasted += 1
    if instance.name.endswith("stress_impossible_H2"):
        # Every commitment faces an exogenous governance failure in one mode; oracle cannot fix it.
        forced = sc.kind in {"tiger_left", "tiger_right"}
    else:
        forced = False
    catastrophic = int(opened_wrong or forced)
    degraded = int((not opened_safe) or wasted >= max(1, instance.horizon - 1))
    loss = 20.0 * wasted + (1000.0 if catastrophic else 0.0) + (80.0 if degraded else 0.0) - (120.0 if opened_safe and not catastrophic else 0.0)
    loss = max(0.0, loss)
    return loss, {"prob_catastrophic": catastrophic, "prob_degraded": degraded}


def rock_loss_and_events(seq: Sequence[str], sc: Scenario, instance: Instance) -> Tuple[float, Dict[str, int]]:
    # RockSample-style: hidden good rock; observations are suppressed, so the commitment
    # must choose collect/check/move actions without knowing the scenario.
    good = set(sc.kind.split("+")) if sc.kind else set()
    collected_good = 0
    collected_bad = 0
    checks = 0
    moves = 0
    for a in seq:
        if a.startswith("sample_"):
            r = a.split("_", 1)[1]
            if r in good:
                collected_good += 1
            else:
                collected_bad += 1
        elif a.startswith("check_"):
            checks += 1  # check is uninformative before commitment closes
        elif a == "move":
            moves += 1
    if instance.name.endswith("stress_impossible_H3"):
        forced_cat = 1
    else:
        forced_cat = 0
    catastrophic = int(forced_cat or collected_bad >= 2)
    degraded = int(collected_good == 0)
    shortage = int(moves < 1 and collected_good < 2)
    loss = 15.0 * checks + 10.0 * moves + 120.0 * collected_bad + 600.0 * catastrophic + 160.0 * degraded + 90.0 * shortage - 80.0 * collected_good
    return max(0.0, loss), {"prob_catastrophic": catastrophic, "prob_degraded": degraded, "prob_shortage": shortage}


def make_instances() -> List[Instance]:
    tiger_scen = [Scenario("TL", 0.5, "tiger_left"), Scenario("TR", 0.5, "tiger_right")]
    tiger_actions = ["listen", "open_left", "open_right", "wait"]
    rock_scen = [
        Scenario("R1_good", 0.25, "r1"),
        Scenario("R2_good", 0.25, "r2"),
        Scenario("Both_good", 0.25, "r1+r2"),
        Scenario("None_good", 0.25, ""),
    ]
    rock_actions = ["check_r1", "check_r2", "sample_r1", "sample_r2", "move"]
    return [
        Instance("tiger_transfer_easy_H4", tiger_actions, 4, tiger_scen, {"prob_catastrophic": 0.55, "prob_degraded": 0.75}, "Tiger", "No-observation Tiger-style easy transfer."),
        Instance("tiger_transfer_hard_H4", tiger_actions, 4, tiger_scen, {"prob_catastrophic": 0.20, "prob_degraded": 0.45}, "Tiger", "No-observation Tiger-style hard transfer."),
        Instance("tiger_transfer_stress_unobservable_H2", tiger_actions, 2, tiger_scen, {"prob_catastrophic": 0.0, "prob_degraded": 0.0}, "Tiger", "Oracle can open the safe door, but blind commitment cannot know which door is safe."),
        Instance("tiger_transfer_stress_impossible_H2", tiger_actions, 2, tiger_scen, {"prob_catastrophic": 0.0, "prob_degraded": 0.0}, "Tiger", "Forced hidden failure makes even the oracle infeasible."),
        Instance("rocksample_transfer_easy_H5", rock_actions, 5, rock_scen, {"prob_catastrophic": 0.50, "prob_degraded": 0.75, "prob_shortage": 0.75}, "RockSample", "No-observation RockSample-style easy transfer."),
        Instance("rocksample_transfer_hard_H5", rock_actions, 5, rock_scen, {"prob_catastrophic": 0.15, "prob_degraded": 0.35, "prob_shortage": 0.35}, "RockSample", "No-observation RockSample-style hard transfer."),
        Instance("rocksample_transfer_stress_unobservable_H3", rock_actions, 3, rock_scen, {"prob_catastrophic": 0.0, "prob_degraded": 0.0}, "RockSample", "Scenario-contingent sampling can avoid bad rocks; blind commitment cannot."),
        Instance("rocksample_transfer_stress_impossible_H3", rock_actions, 3, rock_scen, {"prob_catastrophic": 0.0, "prob_degraded": 0.0}, "RockSample", "Forced catastrophic failure makes the oracle infeasible."),
    ]


def evaluate_sequence(seq: Sequence[str], inst: Instance) -> Dict[str, object]:
    losses = []
    probs = []
    event_rates = {k: 0.0 for k in inst.budgets}
    for sc in inst.scenarios:
        if inst.family == "Tiger":
            loss, events = tiger_loss_and_events(seq, sc, inst)
        else:
            loss, events = rock_loss_and_events(seq, sc, inst)
        losses.append(loss)
        probs.append(sc.p)
        for k in event_rates:
            event_rates[k] += sc.p * float(events.get(k, 0))
    mean = sum(p * l for p, l in zip(probs, losses))
    feasible = all(event_rates[k] <= b + 1e-12 for k, b in inst.budgets.items())
    row = {
        "seq": seq_str(seq),
        "mean": mean,
        "cvar05": cvar_tail(losses, probs, 0.05),
        "cvar25": cvar_tail(losses, probs, 0.25),
        "distorted_ordered": distorted_ordered(losses, probs),
        "feasible": feasible,
        "losses_json": json.dumps(losses),
        "probs_json": json.dumps(probs),
    }
    row.update(event_rates)
    return row


def oracle_stats(inst: Instance) -> Tuple[bool, float, Dict[str, float]]:
    # Scenario-contingent oracle: choose a different full sequence per scenario.
    oracle_event_rates = {k: 0.0 for k in inst.budgets}
    oracle_mean = 0.0
    all_seqs = list(itertools.product(inst.actions, repeat=inst.horizon))
    for sc in inst.scenarios:
        best_loss = math.inf
        best_events = None
        for seq in all_seqs:
            if inst.family == "Tiger":
                loss, events = tiger_loss_and_events(seq, sc, inst)
            else:
                loss, events = rock_loss_and_events(seq, sc, inst)
            if loss < best_loss:
                best_loss = loss
                best_events = events
        oracle_mean += sc.p * best_loss
        assert best_events is not None
        for k in oracle_event_rates:
            oracle_event_rates[k] += sc.p * float(best_events.get(k, 0))
    oracle_feasible = all(oracle_event_rates[k] <= b + 1e-12 for k, b in inst.budgets.items())
    return oracle_feasible, oracle_mean, oracle_event_rates


def run_instance(inst: Instance, outdir: str) -> Dict[str, object]:
    os.makedirs(outdir, exist_ok=True)
    rows = [evaluate_sequence(seq, inst) for seq in itertools.product(inst.actions, repeat=inst.horizon)]
    all_path = os.path.join(outdir, f"{inst.name}_all_sequences.csv")
    fieldnames = ["seq", "mean", "cvar05", "cvar25", "distorted_ordered"] + list(inst.budgets.keys()) + ["feasible", "losses_json", "probs_json"]
    with open(all_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    feasible_rows = [r for r in rows if r["feasible"]]
    governed = min(feasible_rows, key=lambda r: r["mean"]) if feasible_rows else min(rows, key=lambda r: sum(max(0.0, r[k] - inst.budgets[k]) for k in inst.budgets))
    oracle_feasible, oracle_mean, oracle_event_rates = oracle_stats(inst)
    feasible_fraction = len(feasible_rows) / len(rows)
    blind_tax_ratio = float(governed["mean"]) / oracle_mean if oracle_mean > 0 else math.inf
    summary = {
        "instance": inst.name,
        "family": inst.family,
        "num_sequences": len(rows),
        "horizon": inst.horizon,
        "num_actions": len(inst.actions),
        "num_scenarios": len(inst.scenarios),
        "feasible_fraction": feasible_fraction,
        "deterministically_feasible": bool(feasible_rows),
        "governed_mean": governed["mean"],
        "governed_seq": governed["seq"],
        "oracle_mean": oracle_mean,
        "oracle_feasible": oracle_feasible,
        "blind_tax_ratio": blind_tax_ratio,
        "budgets_json": json.dumps(inst.budgets, sort_keys=True),
        "oracle_event_rates_json": json.dumps(oracle_event_rates, sort_keys=True),
        "notes": inst.notes,
    }
    with open(os.path.join(outdir, f"{inst.name}_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    with open(os.path.join(outdir, f"{inst.name}_meta.json"), "w") as f:
        json.dump({
            "instance": inst.name,
            "family": inst.family,
            "actions": inst.actions,
            "horizon": inst.horizon,
            "scenarios": [sc.__dict__ for sc in inst.scenarios],
            "budgets": inst.budgets,
            "notes": inst.notes,
        }, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all", choices=["all", "tiger", "rocksample"])
    ap.add_argument("--outdir", default="results_public")
    ap.add_argument("--summary-out", default="paper_tables/table_public_extra_transfer.csv")
    args = ap.parse_args()
    instances = make_instances()
    if args.variant == "tiger":
        instances = [x for x in instances if x.family == "Tiger"]
    elif args.variant == "rocksample":
        instances = [x for x in instances if x.family == "RockSample"]
    summaries = [run_instance(inst, args.outdir) for inst in instances]
    os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
    with open(args.summary_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    print(f"[OK] wrote {args.summary_out}")

if __name__ == "__main__":
    main()
