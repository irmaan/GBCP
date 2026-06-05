from __future__ import annotations
import time
import numpy as np
from .risk import weighted_mean, weighted_cvar, wowa, robust20, summarize, weighted_event_prob


def governance_score(metrics, event_names=None):
    """Composite safety metric for calibrated large-scale regimes."""
    if not event_names:
        event_names = sorted(k.replace("prob_", "") for k in metrics if k.startswith("prob_"))
    score = 0.0
    for ev in event_names:
        p = float(metrics.get(f"prob_{ev}", 0.0))
        if ev == "catastrophic":
            score += 10.0 * p
        elif ev in ("toxicity", "offtarget", "immune", "degraded"):
            score += 3.0 * p
        elif ev in ("rejected", "moderate_harm", "no_benefit", "shortage"):
            score += 1.5 * p
        else:
            score += p
    score += 1e-5 * float(metrics.get("mean", 0.0))
    score += 1e-6 * float(metrics.get("worst", 0.0))
    return float(score)

SELECTORS = [
    "mean", "null_pomdp_mean", "replan_no_feedback", "worst", "chance", "robust20",
    "cvar01", "cvar05", "cvar10", "cvar25",
    "wowa_mild", "wowa_balanced", "wowa_severe",
    "governed_mean", "governed_cvar05", "governed_cvar25",
    "governed_wowa_balanced", "governed_wowa_severe",
]

def constraint_violations(metrics, constraints):
    out = {}
    score = 0.0
    for ev, lim in constraints.event_max.items():
        val = metrics.get(f"prob_{ev}", 0.0)
        viol = max(0.0, val - lim)
        out[f"viol_{ev}"] = viol
        score += 1e6 * viol
    if constraints.mean_max is not None:
        viol = max(0.0, metrics["mean"] - constraints.mean_max)
        out["viol_mean"] = viol
        score += viol
    if constraints.worst_max is not None:
        viol = max(0.0, metrics["worst"] - constraints.worst_max)
        out["viol_worst"] = viol
        score += viol
    for metric, lim in getattr(constraints, "metric_max", {}).items():
        val = float(metrics.get(metric, 0.0))
        viol = max(0.0, val - float(lim))
        out[f"viol_{metric}"] = viol
        score += 1e6 * viol
    out["violation_score"] = score
    out["feasible"] = score <= 1e-12
    return out

def objective_score(losses, probs, events, selector):
    if selector in ("mean", "null_pomdp_mean", "replan_no_feedback", "governed_mean"):
        return weighted_mean(losses, probs)
    if selector == "worst":
        return float(np.max(losses))
    if selector == "chance":
        cat = events.get("catastrophic", np.zeros_like(losses, dtype=bool))
        return 1e8 * weighted_event_prob(cat, probs) + weighted_mean(losses, probs)
    if selector == "robust20":
        return robust20(losses, probs)
    if selector.startswith("cvar") or selector.startswith("governed_cvar"):
        if "01" in selector:
            a = 0.01
        elif "05" in selector:
            a = 0.05
        elif "10" in selector:
            a = 0.10
        elif "25" in selector:
            a = 0.25
        else:
            a = 0.05
        return weighted_cvar(losses, probs, a)
    if selector.startswith("wowa") or selector.startswith("governed_wowa"):
        if "mild" in selector:
            profile = "mild"
        elif "severe" in selector:
            profile = "severe"
        else:
            profile = "balanced"
        return wowa(losses, probs, profile)
    raise ValueError(selector)

def is_governed(selector):
    return selector.startswith("governed_")

def evaluate_sequence_with_metrics(env, seq):
    ev = env.evaluate_sequence(seq)
    metrics = summarize(ev.losses, env.probs, ev.events)
    if getattr(env.constraints, "metric_max", None):
        metrics["calibration_score"] = governance_score(
            metrics, getattr(env.constraints, "calibration_event_names", tuple())
        )
    viol = constraint_violations(metrics, env.constraints)
    return ev, metrics, viol

def selector_total_score(env, seq, selector):
    ev, metrics, viol = evaluate_sequence_with_metrics(env, seq)
    raw = objective_score(ev.losses, env.probs, ev.events, selector)
    score = viol["violation_score"] + raw if is_governed(selector) else raw
    return score, ev, metrics, viol

def run_selectors(env, selectors=None, max_sequences=None, seed=0, save_all=False):
    selectors = selectors or SELECTORS
    t0 = time.time()
    sequences = list(env.sequences(max_sequences=max_sequences, seed=seed))
    cached = []
    feasible_count = 0
    all_rows = []
    best = {s: {"score": float("inf"), "seq": None, "eval": None} for s in selectors}

    for seq in sequences:
        ev, metrics, viol = evaluate_sequence_with_metrics(env, seq)
        cached.append((seq, ev, metrics, viol))
        if viol["feasible"]:
            feasible_count += 1
        if save_all:
            row = {"sequence": "|".join(seq)}
            row.update(metrics)
            row.update(viol)
            all_rows.append(row)

        for sel in selectors:
            raw = objective_score(ev.losses, env.probs, ev.events, sel)
            score = viol["violation_score"] + raw if is_governed(sel) else raw
            if score < best[sel]["score"]:
                best[sel] = {"score": score, "seq": seq, "eval": ev}

    rows = []
    for sel in selectors:
        ev = best[sel]["eval"]
        metrics = summarize(ev.losses, env.probs, ev.events)
        viol = constraint_violations(metrics, env.constraints)
        row = {
            "instance": env.name,
            "selector": sel,
            "sequence": "|".join(best[sel]["seq"]),
            "selector_score": best[sel]["score"],
            "num_sequences": len(sequences),
            "feasible_count": feasible_count,
            "feasible_fraction": feasible_count / max(1, len(sequences)),
            "runtime_sec": time.time() - t0,
        }
        row.update(metrics)
        row.update(viol)
        rows.append(row)

    oracle_losses = []
    oracle_events = {}
    for j, sc in enumerate(env.scenarios):
        best_loss = float("inf")
        best_events = None
        for seq, ev, metrics, viol in cached:
            if ev.losses[j] < best_loss:
                best_loss = ev.losses[j]
                best_events = {k: v[j] for k, v in ev.events.items()}
        oracle_losses.append(best_loss)
        for k, v in best_events.items():
            oracle_events.setdefault(k, []).append(v)

    oracle_losses = np.array(oracle_losses, dtype=float)
    oracle_events = {k: np.array(v, dtype=bool) for k, v in oracle_events.items()}
    metrics = summarize(oracle_losses, env.probs, oracle_events)
    viol = constraint_violations(metrics, env.constraints)
    row = {
        "instance": env.name,
        "selector": "oracle_hidden_state",
        "sequence": "<scenario-dependent>",
        "selector_score": weighted_mean(oracle_losses, env.probs),
        "num_sequences": len(sequences),
        "feasible_count": feasible_count,
        "feasible_fraction": feasible_count / max(1, len(sequences)),
        "runtime_sec": time.time() - t0,
    }
    row.update(metrics)
    row.update(viol)
    rows.append(row)

    return rows, all_rows
