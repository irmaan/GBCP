from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from .algorithms import evaluate_sequence_with_metrics, constraint_violations, objective_score, selector_total_score
from .risk import weighted_mean, weighted_cvar, robust20, summarize, weighted_event_prob

@dataclass
class BaselineResult:
    baseline: str
    sequence: tuple
    score: float
    evaluations: int
    runtime_sec: float
    metrics: dict
    violations: dict

def _row_result(env, name, seq, score, evals, start):
    ev, metrics, viol = evaluate_sequence_with_metrics(env, seq)
    return BaselineResult(name, tuple(seq), float(score), evals, time.time() - start, metrics, viol)

def _all_cached(env, max_sequences=None, seed=0):
    cached = []
    for seq in env.sequences(max_sequences=max_sequences, seed=seed):
        ev, metrics, viol = evaluate_sequence_with_metrics(env, seq)
        cached.append((tuple(seq), ev, metrics, viol))
    return cached

def conformant_first_feasible(env, max_sequences=None, seed=0):
    start = time.time()
    cached = _all_cached(env, max_sequences, seed)
    best = None
    for seq, ev, metrics, viol in cached:
        if viol["feasible"]:
            best = (seq, metrics["mean"])
            break
    if best is None:
        seq, ev, metrics, viol = min(cached, key=lambda x: (x[3]["violation_score"], x[2]["mean"]))
        best = (seq, viol["violation_score"] + metrics["mean"])
    return _row_result(env, "conformant_first_feasible", best[0], best[1], len(cached), start)

def conformant_min_cost_feasible(env, max_sequences=None, seed=0):
    start = time.time()
    cached = _all_cached(env, max_sequences, seed)
    feasible = [x for x in cached if x[3]["feasible"]]
    pool = feasible if feasible else cached
    seq, ev, metrics, viol = min(pool, key=lambda x: (0 if x[3]["feasible"] else x[3]["violation_score"], x[2]["mean"]))
    score = metrics["mean"] if viol["feasible"] else viol["violation_score"] + metrics["mean"]
    return _row_result(env, "conformant_min_cost_feasible", seq, score, len(cached), start)

def lexicographic_safety(env, max_sequences=None, seed=0):
    start = time.time()
    cached = _all_cached(env, max_sequences, seed)
    def key(x):
        seq, ev, metrics, viol = x
        cat = metrics.get("prob_catastrophic", 0.0)
        worst = metrics.get("worst", 0.0)
        mean = metrics.get("mean", 0.0)
        return (viol["violation_score"], cat, worst, mean)
    seq, ev, metrics, viol = min(cached, key=key)
    return _row_result(env, "lexicographic_safety", seq, key((seq, ev, metrics, viol))[0] + metrics["mean"], len(cached), start)

def chance_constrained_mean(env, max_sequences=None, seed=0, cat_limit=None):
    start = time.time()
    cached = _all_cached(env, max_sequences, seed)
    if cat_limit is None:
        cat_limit = env.constraints.event_max.get("catastrophic", 0.10)
    feasible = [x for x in cached if x[2].get("prob_catastrophic", 0.0) <= cat_limit]
    pool = feasible if feasible else cached
    seq, ev, metrics, viol = min(pool, key=lambda x: (max(0.0, x[2].get("prob_catastrophic", 0.0) - cat_limit), x[2]["mean"]))
    score = metrics["mean"] + 1e6 * max(0.0, metrics.get("prob_catastrophic", 0.0) - cat_limit)
    return _row_result(env, "chance_constrained_mean", seq, score, len(cached), start)

def cvar_constrained_mean(env, max_sequences=None, seed=0, alpha=0.05, cvar_limit=None):
    start = time.time()
    cached = _all_cached(env, max_sequences, seed)
    key_name = f"cvar{int(alpha*100):02d}"
    values = [x[2][key_name] for x in cached]
    if cvar_limit is None:
        # Set data-driven limit at the lower quartile of achievable CVaR.
        cvar_limit = float(np.quantile(values, 0.25))
    feasible = [x for x in cached if x[2][key_name] <= cvar_limit]
    pool = feasible if feasible else cached
    seq, ev, metrics, viol = min(pool, key=lambda x: (max(0.0, x[2][key_name] - cvar_limit), x[2]["mean"]))
    score = metrics["mean"] + 1e3 * max(0.0, metrics[key_name] - cvar_limit)
    return _row_result(env, f"cvar{int(alpha*100):02d}_constrained_mean", seq, score, len(cached), start)

def robust_cvar(env, max_sequences=None, seed=0):
    start = time.time()
    cached = _all_cached(env, max_sequences, seed)
    seq, ev, metrics, viol = min(cached, key=lambda x: x[2]["robust20"])
    return _row_result(env, "distributionally_robust_cvar20", seq, metrics["robust20"], len(cached), start)

def null_observation_pomdp(env, max_sequences=None, seed=0):
    # A POMDP with singleton observation cannot condition on observations, so this is open-loop mean planning.
    start = time.time()
    cached = _all_cached(env, max_sequences, seed)
    seq, ev, metrics, viol = min(cached, key=lambda x: x[2]["mean"])
    return _row_result(env, "null_observation_pomdp", seq, metrics["mean"], len(cached), start)

def minimax_regret(env, max_sequences=None, seed=0):
    start = time.time()
    cached = _all_cached(env, max_sequences, seed)
    n_scen = len(env.scenarios)
    best_by_scenario = np.full(n_scen, np.inf)
    for seq, ev, metrics, viol in cached:
        best_by_scenario = np.minimum(best_by_scenario, ev.losses)
    def regret_score(x):
        seq, ev, metrics, viol = x
        regret = ev.losses - best_by_scenario
        return float(np.max(regret))
    seq, ev, metrics, viol = min(cached, key=regret_score)
    return _row_result(env, "minimax_regret", seq, regret_score((seq, ev, metrics, viol)), len(cached), start)

def run_baselines(env, max_sequences=None, seed=0):
    funcs = [
        conformant_first_feasible,
        conformant_min_cost_feasible,
        lexicographic_safety,
        chance_constrained_mean,
        cvar_constrained_mean,
        robust_cvar,
        minimax_regret,
        null_observation_pomdp,
    ]
    return [fn(env, max_sequences=max_sequences, seed=seed) for fn in funcs]

def result_to_row(res):
    row = {
        "baseline": res.baseline,
        "sequence": "|".join(res.sequence),
        "score": res.score,
        "evaluations": res.evaluations,
        "runtime_sec": res.runtime_sec,
    }
    row.update(res.metrics)
    row.update(res.violations)
    return row
