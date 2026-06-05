from __future__ import annotations
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


NODES = ["A", "B", "C", "D"]
ACTIONS = ["noop"] + [f"reboot_{n}" for n in NODES]

SELECTORS = [
    "mean",
    "null_pomdp_mean",
    "replan_no_feedback",
    "worst",
    "chance",
    "robust20",
    "cvar01",
    "cvar05",
    "cvar10",
    "cvar25",
    "wowa_mild",
    "wowa_balanced",
    "wowa_severe",
    "governed_mean",
    "governed_cvar05",
    "governed_cvar25",
    "governed_wowa_balanced",
    "governed_wowa_severe",
]


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def weighted_mean(x: np.ndarray, p: np.ndarray) -> float:
    return float(np.dot(x, p))


def weighted_var_cvar(losses: np.ndarray, probs: np.ndarray, alpha: float) -> Tuple[float, float]:
    # Tail-mass convention: CVaR_0.05 means worst 5% mass.
    order = np.argsort(losses)[::-1]  # worst first
    l = np.asarray(losses, dtype=float)[order]
    p = np.asarray(probs, dtype=float)[order]

    mass = 0.0
    tail_sum = 0.0
    var = float(l[0])

    for li, pi in zip(l, p):
        if mass < alpha - 1e-15:
            take = min(pi, alpha - mass)
            tail_sum += li * take
            mass += take
            var = float(li)

    return var, float(tail_sum / max(alpha, 1e-12))


def omega_desc(m: int, power: float) -> np.ndarray:
    # Worst-first rank weights. Larger power emphasizes the worst ranks more.
    vals = np.array([(m - i) ** power for i in range(m)], dtype=float)
    vals /= vals.sum()
    return vals


def distortion_value(y: float, omega: np.ndarray) -> float:
    m = len(omega)
    if y <= 0.0:
        return 0.0
    if y >= 1.0:
        return 1.0

    cum = np.concatenate([[0.0], np.cumsum(omega)])
    pos = y * m
    k = int(math.floor(pos))
    if k >= m:
        return 1.0
    frac = pos - k
    return float(cum[k] + frac * (cum[k + 1] - cum[k]))


def wowa(losses: np.ndarray, probs: np.ndarray, omega: np.ndarray) -> float:
    # Descending-loss WOWA-like aggregation using a piecewise-linear rank distortion.
    order = np.argsort(losses)[::-1]  # worst first
    l = np.asarray(losses, dtype=float)[order]
    p = np.asarray(probs, dtype=float)[order]

    cum = 0.0
    out = 0.0
    for li, pi in zip(l, p):
        prev = cum
        cum += pi
        w = distortion_value(cum, omega) - distortion_value(prev, omega)
        out += w * li
    return float(out)


def robust_tail_mean(losses: np.ndarray, probs: np.ndarray, beta: float = 0.20) -> float:
    return weighted_var_cvar(losses, probs, beta)[1]


def summarize(losses: np.ndarray, probs: np.ndarray, events: Dict[str, List[int]]) -> dict:
    losses = np.asarray(losses, dtype=float)
    probs = np.asarray(probs, dtype=float)
    m = len(losses)

    out = {
        "mean": weighted_mean(losses, probs),
        "worst": float(losses.max()),
    }

    for alpha, tag in [(0.01, "01"), (0.05, "05"), (0.10, "10"), (0.25, "25")]:
        v, c = weighted_var_cvar(losses, probs, alpha)
        out[f"var{tag}"] = v
        out[f"cvar{tag}"] = c

    out["wowa_mild"] = wowa(losses, probs, omega_desc(m, 0.5))
    out["wowa_balanced"] = wowa(losses, probs, omega_desc(m, 1.5))
    out["wowa_severe"] = wowa(losses, probs, omega_desc(m, 4.0))
    out["robust20"] = robust_tail_mean(losses, probs, 0.20)

    for k, vals in events.items():
        out[f"prob_{k}"] = float(np.dot(np.asarray(vals, dtype=float), probs))

    return out


def constraint_violations(metrics: dict, bounds: Dict[str, float], lambdas: Dict[str, float]) -> dict:
    out = {}
    score = 0.0
    feasible = True

    for pcol, bound in bounds.items():
        suffix = pcol[len("prob_"):]
        v = max(0.0, metrics.get(pcol, 0.0) - bound)
        out[f"viol_{suffix}"] = v
        score += lambdas.get(pcol, 1e6) * v
        if v > 1e-12:
            feasible = False

    out["violation_score"] = float(score)
    out["feasible"] = feasible
    return out


def governed(sel: str) -> bool:
    return sel.startswith("governed_")


def raw_score(metrics: dict, sel: str) -> float:
    if sel in ("mean", "null_pomdp_mean", "replan_no_feedback", "chance", "governed_mean"):
        return float(metrics["mean"])
    if sel == "worst":
        return float(metrics["worst"])
    if sel == "robust20":
        return float(metrics["robust20"])
    if sel == "cvar01":
        return float(metrics["cvar01"])
    if sel in ("cvar05", "governed_cvar05"):
        return float(metrics["cvar05"])
    if sel == "cvar10":
        return float(metrics["cvar10"])
    if sel in ("cvar25", "governed_cvar25"):
        return float(metrics["cvar25"])
    if sel == "wowa_mild":
        return float(metrics["wowa_mild"])
    if sel in ("wowa_balanced", "governed_wowa_balanced"):
        return float(metrics["wowa_balanced"])
    if sel in ("wowa_severe", "governed_wowa_severe"):
        return float(metrics["wowa_severe"])
    raise KeyError(f"Unknown selector: {sel}")


def simulate_sysadmin_transfer(seq: Tuple[str, ...], scen: dict) -> Tuple[float, dict]:
    """
    Public-domain transfer benchmark derived from the classic SysAdmin maintenance / reboot setting.
    We preserve the ring-machine semantics and reboot action semantics, but evaluate under a
    null-observation governed blind-commitment contract.
    """
    kind = scen["kind"]

    # Easy / hard variants: latent vulnerable machine + latent load regime.
    if kind in ("easy", "hard"):
        state = [1, 1, 1, 1]  # 1=up, 0=down
        protection = [0, 0, 0, 0]
        total_loss = 0.0
        down_per_stage: List[int] = []

        v = scen["v"]
        shock = scen["shock"]
        high = scen["load"] == "high"

        for t, a in enumerate(seq):
            protection = [max(0, p - 1) for p in protection]

            if a != "noop":
                i = NODES.index(a.split("_", 1)[1])
                state[i] = 1
                protection[i] = 2
                total_loss += 6.0  # reboot cost

            # Hidden vulnerability becomes active after shock time.
            if t >= shock and protection[v] == 0:
                state[v] = 0

            # Under high load, failure propagates to neighbors if vulnerable node stays down.
            if high and t >= shock + 1 and state[v] == 0:
                for nb in ((v - 1) % 4, (v + 1) % 4):
                    if protection[nb] == 0:
                        state[nb] = 0

            # Hard variant: opposite node also becomes endangered later.
            if kind == "hard" and t >= shock + 2 and state[v] == 0:
                opp = (v + 2) % 4
                if protection[opp] == 0:
                    state[opp] = 0

            down = 4 - sum(state)
            total_loss += 35.0 * down
            if high and down >= 2:
                total_loss += 15.0
            down_per_stage.append(down)

        arr = np.array(down_per_stage, dtype=float)
        events = {
            "catastrophic": int(arr.max() >= 3),
            "degraded": int(arr.mean() >= 1.0),
            "shortage": int(arr[-1] >= 2),
        }
        return total_loss, events

    # Stress-unobservable: latent vulnerable node requires a conflicting first action.
    if kind == "stress_unobservable":
        state = [1, 1, 1, 1]
        total_loss = 0.0
        down_per_stage: List[int] = []
        v = scen["v"]

        for t, a in enumerate(seq):
            if a != "noop":
                i = NODES.index(a.split("_", 1)[1])
                state[i] = 1
                total_loss += 6.0

            # If the blind commitment does not reboot the hidden vulnerable node immediately,
            # the node fails before any information could arrive.
            if t == 0 and a != f"reboot_{NODES[v]}":
                state[v] = 0

            # Persistent hidden failure cascades to the opposite node.
            if t >= 1 and state[v] == 0:
                state[(v + 2) % 4] = 0

            down = 4 - sum(state)
            total_loss += 40.0 * down
            down_per_stage.append(down)

        arr = np.array(down_per_stage, dtype=float)
        events = {
            "catastrophic": int(arr.max() >= 2),
            "degraded": int(arr.mean() > 0.0),
            "shortage": int(arr[-1] >= 1),
        }
        return total_loss, events

    # Stress-impossible: simultaneous hidden pair failure that no open-loop commitment can prevent.
    if kind == "stress_impossible":
        state = [1, 1, 1, 1]
        total_loss = 0.0
        down_per_stage: List[int] = []
        pair = scen["vs"]

        for t, a in enumerate(seq):
            if a != "noop":
                total_loss += 6.0

            if t == 0:
                for v in pair:
                    state[v] = 0
            if t >= 1:
                for v in pair:
                    state[v] = 0

            down = 4 - sum(state)
            total_loss += 40.0 * down
            down_per_stage.append(down)

        arr = np.array(down_per_stage, dtype=float)
        events = {
            "catastrophic": int(arr.max() >= 2),
            "degraded": int(arr.mean() > 0.0),
            "shortage": int(arr[-1] >= 1),
        }
        return total_loss, events

    raise ValueError(f"Unknown scenario kind: {kind}")


def make_variant(variant: str, horizon: int | None = None):
    """
    Public-domain transfer family derived from the classic SysAdmin ring benchmark.
    """
    if variant == "easy":
        H = 6 if horizon is None else horizon
        scenarios = [
            {"kind": "easy", "v": v, "shock": 2, "load": load}
            for v in range(4)
            for load in ("low", "high")
        ]
        probs = [1.0 / 8.0] * 8
        bounds = {
            "prob_catastrophic": 0.25,
            "prob_degraded": 0.50,
            "prob_shortage": 0.50,
        }
        lambdas = {
            "prob_catastrophic": 2e6,
            "prob_degraded": 5e5,
            "prob_shortage": 5e5,
        }

    elif variant == "hard":
        H = 6 if horizon is None else horizon
        scenarios = [
            {"kind": "hard", "v": v, "shock": 1, "load": load}
            for v in range(4)
            for load in ("low", "high")
        ]
        probs = [1.0 / 8.0] * 8
        bounds = {
            "prob_catastrophic": 0.075,
            "prob_degraded": 0.30,
            "prob_shortage": 0.30,
        }
        lambdas = {
            "prob_catastrophic": 2e6,
            "prob_degraded": 7e5,
            "prob_shortage": 7e5,
        }

    elif variant == "stress_unobservable":
        H = 3 if horizon is None else horizon
        scenarios = [
            {"kind": "stress_unobservable", "v": 0},
            {"kind": "stress_unobservable", "v": 2},
        ]
        probs = [0.5, 0.5]
        bounds = {
            "prob_catastrophic": 0.0,
            "prob_degraded": 0.0,
            "prob_shortage": 0.0,
        }
        lambdas = {
            "prob_catastrophic": 2e6,
            "prob_degraded": 1e6,
            "prob_shortage": 1e6,
        }

    elif variant == "stress_impossible":
        H = 3 if horizon is None else horizon
        scenarios = [
            {"kind": "stress_impossible", "vs": [0, 2]},
            {"kind": "stress_impossible", "vs": [1, 3]},
        ]
        probs = [0.5, 0.5]
        bounds = {
            "prob_catastrophic": 0.0,
            "prob_degraded": 0.0,
            "prob_shortage": 0.0,
        }
        lambdas = {
            "prob_catastrophic": 2e6,
            "prob_degraded": 1e6,
            "prob_shortage": 1e6,
        }

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return scenarios, probs, bounds, lambdas, H


def run_variant(variant: str, horizon: int | None = None) -> Tuple[List[dict], List[dict], dict]:
    scenarios, probs, bounds, lambdas, H = make_variant(variant, horizon)
    sequences = list(itertools.product(ACTIONS, repeat=H))
    instance_name = f"sysadmin_transfer_{variant}"

    all_rows: List[dict] = []
    cached: List[tuple] = []
    feasible_count = 0

    best = {
        s: {"score": float("inf"), "seq": None, "metrics": None, "viol": None}
        for s in SELECTORS
    }

    for seq in sequences:
        losses = []
        event_cols = {"catastrophic": [], "degraded": [], "shortage": []}

        for sc in scenarios:
            loss, events = simulate_sysadmin_transfer(seq, sc)
            losses.append(loss)
            for k in event_cols:
                event_cols[k].append(events[k])

        losses_arr = np.array(losses, dtype=float)
        metrics = summarize(losses_arr, np.array(probs, dtype=float), event_cols)
        viol = constraint_violations(metrics, bounds, lambdas)

        if viol["feasible"]:
            feasible_count += 1

        row = {"sequence": "|".join(seq)}
        row.update(metrics)
        row.update(viol)
        all_rows.append(row)

        cached.append((seq, losses_arr, event_cols))

        for sel in SELECTORS:
            score = raw_score(metrics, sel) + (viol["violation_score"] if governed(sel) else 0.0)
            if score < best[sel]["score"] - 1e-12:
                best[sel] = {
                    "score": score,
                    "seq": seq,
                    "metrics": metrics,
                    "viol": viol,
                }

    rows: List[dict] = []
    for sel in SELECTORS:
        row = {
            "instance": instance_name,
            "selector": sel,
            "sequence": "|".join(best[sel]["seq"]),
            "selector_score": best[sel]["score"],
            "num_sequences": len(sequences),
            "feasible_count": feasible_count,
            "feasible_fraction": feasible_count / max(1, len(sequences)),
            "runtime_sec": 0.0,
        }
        row.update(best[sel]["metrics"])
        row.update(best[sel]["viol"])
        rows.append(row)

    # Oracle benchmark: best sequence per scenario.
    oracle_losses = []
    oracle_events = {"catastrophic": [], "degraded": [], "shortage": []}

    for j in range(len(scenarios)):
        best_loss = float("inf")
        best_events = None
        for seq, losses_arr, event_cols in cached:
            if losses_arr[j] < best_loss:
                best_loss = float(losses_arr[j])
                best_events = {k: event_cols[k][j] for k in oracle_events}
        oracle_losses.append(best_loss)
        for k, v in best_events.items():
            oracle_events[k].append(v)

    om = summarize(np.array(oracle_losses, dtype=float), np.array(probs, dtype=float), oracle_events)
    ov = constraint_violations(om, bounds, lambdas)

    oracle_row = {
        "instance": instance_name,
        "selector": "oracle_hidden_state",
        "sequence": "<scenario-dependent>",
        "selector_score": om["mean"],
        "num_sequences": len(sequences),
        "feasible_count": feasible_count,
        "feasible_fraction": feasible_count / max(1, len(sequences)),
        "runtime_sec": 0.0,
    }
    oracle_row.update(om)
    oracle_row.update(ov)
    rows.append(oracle_row)

    meta = {
        "instance": instance_name,
        "variant": variant,
        "horizon": H,
        "actions": ACTIONS,
        "num_sequences": len(sequences),
        "bounds": bounds,
        "lambdas": lambdas,
        "scenarios": scenarios,
        "probs": probs,
    }

    return rows, all_rows, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant",
        choices=["easy", "hard", "stress_unobservable", "stress_impossible", "all"],
        default="all",
    )
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--outdir", default="results_public")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    variants = (
        ["easy", "hard", "stress_unobservable", "stress_impossible"]
        if args.variant == "all"
        else [args.variant]
    )

    for variant in variants:
        rows, all_rows, meta = run_variant(variant, args.horizon)
        H = meta["horizon"]
        stem = f"sysadmin_transfer_{variant}_H{H}"

        write_csv(outdir / f"{stem}_summary.csv", rows)
        write_csv(outdir / f"{stem}_all_sequences.csv", all_rows)

        with open(outdir / f"{stem}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n== {stem} ==")
        gm = next(r for r in rows if r["selector"] == "governed_mean")
        oc = next(r for r in rows if r["selector"] == "oracle_hidden_state")
        print(json.dumps({
            "instance": stem,
            "num_sequences": gm["num_sequences"],
            "feasible_fraction": gm["feasible_fraction"],
            "governed_mean": gm["mean"],
            "oracle_mean": oc["mean"],
            "oracle_feasible": oc["feasible"],
            "blind_tax_ratio": (gm["mean"] / oc["mean"]) if oc["mean"] > 0 else None,
        }, indent=2))


if __name__ == "__main__":
    main()