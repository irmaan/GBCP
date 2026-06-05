from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from .algorithms import selector_total_score, evaluate_sequence_with_metrics

@dataclass
class PlannerResult:
    planner: str
    selector: str
    sequence: tuple
    score: float
    evaluations: int
    runtime_sec: float
    metrics: dict
    violations: dict

def _result(env, planner, selector, seq, score, evaluations, start_time):
    ev, metrics, viol = evaluate_sequence_with_metrics(env, seq)
    return PlannerResult(
        planner=planner,
        selector=selector,
        sequence=tuple(seq),
        score=float(score),
        evaluations=evaluations,
        runtime_sec=time.time() - start_time,
        metrics=metrics,
        violations=viol,
    )

def _evaluate(env, seq, selector):
    score, ev, metrics, viol = selector_total_score(env, tuple(seq), selector)
    return score, ev, metrics, viol

def exact_search(env, selector, budget=None, seed=0):
    start = time.time()
    best_seq = None
    best_score = float("inf")
    n = 0
    for seq in env.sequences(max_sequences=budget, seed=seed):
        score, _, _, _ = _evaluate(env, seq, selector)
        n += 1
        if score < best_score:
            best_score = score
            best_seq = seq
    return _result(env, "exact" if budget is None else "exact_budgeted", selector, best_seq, best_score, n, start)

def random_search(env, selector, budget=500, seed=0):
    start = time.time()
    rng = np.random.default_rng(seed)
    actions = list(env.actions)
    best_seq = None
    best_score = float("inf")
    for i in range(max(1, budget)):
        seq = tuple(rng.choice(actions, size=env.horizon))
        score, _, _, _ = _evaluate(env, seq, selector)
        if score < best_score:
            best_score = score
            best_seq = seq
    return _result(env, "random", selector, best_seq, best_score, max(1, budget), start)

def beam_search(env, selector, budget=500, beam_width=16, seed=0):
    start = time.time()
    # Prefix score: complete prefix with the first action to make it evaluable.
    actions = list(env.actions)
    beam = [()]
    evals = 0
    for t in range(env.horizon):
        candidates = []
        for prefix in beam:
            for a in actions:
                pref = prefix + (a,)
                fill = pref + tuple([actions[0]] * (env.horizon - len(pref)))
                score, _, _, _ = _evaluate(env, fill, selector)
                candidates.append((score, pref))
                evals += 1
                if evals >= budget:
                    break
            if evals >= budget:
                break
        candidates.sort(key=lambda x: x[0])
        beam = [p for _, p in candidates[:beam_width]]
        if evals >= budget:
            break
    best_seq = None
    best_score = float("inf")
    for pref in beam:
        seq = pref + tuple([actions[0]] * (env.horizon - len(pref)))
        score, _, _, _ = _evaluate(env, seq, selector)
        evals += 1
        if score < best_score:
            best_score = score
            best_seq = seq
    return _result(env, "beam", selector, best_seq, best_score, evals, start)

def cem_search(env, selector, budget=500, seed=0, elite_frac=0.20, smoothing=0.70):
    start = time.time()
    rng = np.random.default_rng(seed)
    actions = list(env.actions)
    n_actions = len(actions)
    action_to_idx = {a: i for i, a in enumerate(actions)}
    probs = np.ones((env.horizon, n_actions), dtype=float) / n_actions
    batch = max(20, min(100, budget // 5 if budget >= 100 else budget))
    evals = 0
    best_seq = None
    best_score = float("inf")

    while evals < budget:
        cur_batch = min(batch, budget - evals)
        samples = []
        for _ in range(cur_batch):
            seq = []
            for t in range(env.horizon):
                seq.append(actions[rng.choice(n_actions, p=probs[t])])
            seq = tuple(seq)
            score, _, _, _ = _evaluate(env, seq, selector)
            samples.append((score, seq))
            evals += 1
            if score < best_score:
                best_score = score
                best_seq = seq
        samples.sort(key=lambda x: x[0])
        elite_n = max(1, int(len(samples) * elite_frac))
        elite = [seq for _, seq in samples[:elite_n]]
        new_probs = np.zeros_like(probs) + 1e-6
        for seq in elite:
            for t, a in enumerate(seq):
                new_probs[t, action_to_idx[a]] += 1.0
        new_probs = new_probs / new_probs.sum(axis=1, keepdims=True)
        probs = smoothing * probs + (1.0 - smoothing) * new_probs
        probs = probs / probs.sum(axis=1, keepdims=True)

    return _result(env, "cem", selector, best_seq, best_score, evals, start)

def evolutionary_search(env, selector, budget=500, seed=0, pop_size=40, mutation_rate=0.20):
    start = time.time()
    rng = np.random.default_rng(seed)
    actions = list(env.actions)
    pop_size = max(6, min(pop_size, max(6, budget)))
    population = [tuple(rng.choice(actions, size=env.horizon)) for _ in range(pop_size)]
    evals = 0
    best_seq = None
    best_score = float("inf")

    def mutate(seq):
        out = list(seq)
        for i in range(len(out)):
            if rng.random() < mutation_rate:
                out[i] = rng.choice(actions)
        return tuple(out)

    def crossover(a, b):
        if env.horizon <= 1:
            return a
        cut = rng.integers(1, env.horizon)
        return tuple(a[:cut] + b[cut:])

    while evals < budget:
        scored = []
        for seq in population:
            if evals >= budget:
                break
            score, _, _, _ = _evaluate(env, seq, selector)
            scored.append((score, seq))
            evals += 1
            if score < best_score:
                best_score = score
                best_seq = seq
        scored.sort(key=lambda x: x[0])
        elites = [seq for _, seq in scored[:max(2, len(scored)//4)]]
        new_pop = list(elites)
        while len(new_pop) < pop_size:
            p1 = elites[rng.integers(0, len(elites))]
            p2 = elites[rng.integers(0, len(elites))]
            child = mutate(crossover(p1, p2))
            new_pop.append(child)
        population = new_pop
    return _result(env, "evolutionary", selector, best_seq, best_score, evals, start)

def open_loop_mcts(env, selector, budget=500, seed=0, c=1.4):
    start = time.time()
    rng = np.random.default_rng(seed)
    actions = list(env.actions)
    stats = {}

    def key(prefix):
        return tuple(prefix)

    def rollout(prefix):
        seq = list(prefix)
        while len(seq) < env.horizon:
            seq.append(rng.choice(actions))
        return tuple(seq)

    best_seq = None
    best_score = float("inf")
    evals = 0

    while evals < budget:
        prefix = []
        # Select / expand
        for depth in range(env.horizon):
            children = [key(prefix + [a]) for a in actions]
            unvisited = [ch for ch in children if ch not in stats]
            if unvisited:
                chosen = unvisited[rng.integers(0, len(unvisited))]
                prefix = list(chosen)
                break
            parent_visits = max(1, stats[key(prefix)]["n"]) if key(prefix) in stats else sum(stats[ch]["n"] for ch in children)
            def ucb(ch):
                st = stats[ch]
                # Scores are losses; convert to reward by negating mean score.
                return -st["mean"] + c * np.sqrt(np.log(parent_visits + 1) / st["n"])
            chosen = max(children, key=ucb)
            prefix = list(chosen)

        seq = rollout(prefix)
        score, _, _, _ = _evaluate(env, seq, selector)
        evals += 1
        if score < best_score:
            best_score = score
            best_seq = seq

        # Backpropagate along prefixes of the sampled sequence.
        for d in range(env.horizon + 1):
            k = key(seq[:d])
            if k not in stats:
                stats[k] = {"n": 0, "mean": 0.0}
            st = stats[k]
            st["n"] += 1
            st["mean"] += (score - st["mean"]) / st["n"]

    return _result(env, "open_loop_mcts", selector, best_seq, best_score, evals, start)

def run_planner(name, env, selector, budget=500, seed=0):
    if name == "exact":
        return exact_search(env, selector, budget=None, seed=seed)
    if name == "exact_budgeted":
        return exact_search(env, selector, budget=budget, seed=seed)
    if name == "random":
        return random_search(env, selector, budget=budget, seed=seed)
    if name == "beam":
        return beam_search(env, selector, budget=budget, seed=seed)
    if name == "cem":
        return cem_search(env, selector, budget=budget, seed=seed)
    if name == "evolutionary":
        return evolutionary_search(env, selector, budget=budget, seed=seed)
    if name == "open_loop_mcts":
        return open_loop_mcts(env, selector, budget=budget, seed=seed)
    raise ValueError(name)

def planner_names():
    return ["exact", "exact_budgeted", "random", "beam", "cem", "evolutionary", "open_loop_mcts"]

def result_to_row(result):
    row = {
        "planner": result.planner,
        "selector": result.selector,
        "sequence": "|".join(result.sequence),
        "score": result.score,
        "evaluations": result.evaluations,
        "runtime_sec": result.runtime_sec,
    }
    row.update(result.metrics)
    row.update(result.violations)
    return row
