from __future__ import annotations
from itertools import product
import math
from .core import Scenario, SealedNOMDP, ConstraintSpec
from .instances import DOMAIN, VARIANT, make_instance

def _renormalize(scenarios):
    total = sum(s.prob for s in scenarios)
    return [Scenario(s.name, s.prob / total, s.params) for s in scenarios]

def _make_grid_scenarios(target_scenarios=128):
    target_scenarios = max(8, int(target_scenarios))
    candidates = [
        (3, 2, 2, 2), (4, 3, 2, 2), (4, 3, 3, 2), (5, 3, 3, 2),
        (5, 4, 3, 2), (5, 4, 3, 3), (6, 4, 3, 3), (6, 4, 4, 3),
        (6, 5, 4, 3), (7, 5, 4, 3)
    ]
    H, R, G, F = min(candidates, key=lambda x: abs(math.prod(x) - target_scenarios))
    scenarios = []
    for h, r, g, f in product(range(H), range(R), range(G), range(F)):
        p = (0.62 ** h) * (0.72 ** r) * (0.78 ** g) * (0.70 ** f)
        params = {
            "H": h / max(1, H - 1) * 2.0,
            "R": r / max(1, R - 1),
            "G": g / max(1, G - 1),
            "F": f / max(1, F - 1),
        }
        scenarios.append(Scenario(f"H{h}_R{r}_G{g}_F{f}", p, params))
    scenarios.sort(key=lambda s: s.prob, reverse=True)
    return _renormalize(scenarios[:target_scenarios])

def _extend_actions(cfg, action_count):
    action_count = max(4, min(8, int(action_count)))
    actions = list(cfg["actions"])
    treat, damage, ops = dict(cfg["treat"]), dict(cfg["damage"]), dict(cfg["ops"])
    extras = [
        ("mild_plus", 1.18, 0.34, 2.4),
        ("standard_plus", 1.78, 0.72, 3.2),
        ("targeted_strong", 2.38, 0.86, 4.8),
        ("salvage", 2.75, 1.40, 6.0),
    ]
    for name, tr, dmg, op in extras:
        if len(actions) >= action_count:
            break
        actions.append(name); treat[name] = tr; damage[name] = dmg; ops[name] = op
    return actions, treat, damage, ops

def make_scaled_instance(domain="bcst", variant="hard", horizon=8, action_count=6, scenario_count=128):
    if variant in ("stress_unobservable", "stress_impossible"):
        return make_instance(f"{domain}_{variant}")
    if domain not in DOMAIN or variant not in VARIANT:
        raise ValueError(f"Unknown domain/variant: {domain}/{variant}")
    cfg, var = DOMAIN[domain], VARIANT[variant]
    scenarios = _make_grid_scenarios(scenario_count)
    actions, treat, damage, ops = _extend_actions(cfg, action_count)

    def init(sc):
        p = sc.params
        hazard = cfg["base"] + var["hazard_shift"] + 0.92*p["H"] + 0.78*p["R"] + 0.58*p["G"]
        return {"hazard": hazard, "damage": 0.0}

    def trans(st, a, sc, t):
        p = sc.params
        st = dict(st)
        fatigue = 1.0 + 0.015 * t
        treatment = treat[a] * (1.0 - 0.16*p["R"] - 0.12*p["G"])
        dmg = damage[a] * var["damage_scale"] * fatigue * (1.0 + 0.50*p["F"] + 0.08*p["G"])
        st["hazard"] = max(0.0, st["hazard"] - treatment)
        st["damage"] = max(0.0, st["damage"] + dmg)
        return st, ops[a]

    def term(st, sc):
        p = sc.params
        catastrophic = st["hazard"] > 0.72
        harm = st["damage"] > (cfg["robust_threshold"] if p["F"] < 0.5 else cfg["fragile_threshold"])
        medium = st["hazard"] > 1.22 or harm
        events = {ev: False for ev in cfg["events"]}
        events["catastrophic"] = catastrophic
        events[cfg["harm_event"]] = harm
        events[cfg["medium_event"]] = medium
        if cfg["extra_medium"]:
            if domain == "bged":
                events["immune"] = harm and p["G"] > 0.5
                events["offtarget"] = harm and p["F"] > 0.5
                events["catastrophic"] = events["immune"] or events["offtarget"]
            else:
                events[cfg["extra_medium"]] = medium
        loss = (
            cfg["cat_penalty"] * events["catastrophic"]
            + cfg["harm_penalty"] * events[cfg["harm_event"]]
            + cfg["medium_penalty"] * events[cfg["medium_event"]]
            + cfg["hazard_coeff"] * st["hazard"]
            + cfg["damage_coeff"] * st["damage"]
        )
        return loss, events

    cons = {ev: lim for ev, lim in var["constraints"].items() if ev in cfg["events"]}
    if domain == "bcst" and variant == "hard":
        cons.update({"catastrophic": 0.13, "toxicity": 0.34, "moderate_harm": 0.82})
    if domain == "bged" and variant == "hard":
        cons.update({"catastrophic": 0.22, "no_benefit": 0.24, "offtarget": 0.30, "immune": 0.30})

    return SealedNOMDP(
        f"{domain}_{variant}_H{horizon}_A{len(actions)}_S{len(scenarios)}",
        actions, scenarios, int(horizon), trans, term, init, ConstraintSpec(event_max=cons),
        f"Scaled {cfg['label']} ({variant}), horizon={horizon}, actions={len(actions)}, scenarios={len(scenarios)}."
    )
