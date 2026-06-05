from __future__ import annotations
from itertools import product
from .core import Scenario, SealedNOMDP, ConstraintSpec

def _scenario_grid(names, levels, prob_fn):
    scenarios = []
    total = 0.0
    for vals in product(*levels):
        params = dict(zip(names, vals))
        p = float(prob_fn(params))
        scenarios.append(Scenario("_".join(f"{k}{v}" for k, v in params.items()), p, params))
        total += p
    return [Scenario(s.name, s.prob / total, s.params) for s in scenarios]

def make_diagnostic_cvar_blindspot(horizon=1):
    scenarios = [Scenario(f"mode{i:02d}", 1.0/20.0, {"i": i}) for i in range(20)]
    actions = ["mean_risky", "tail_guard", "rank_balanced", "overkill"]

    def init(sc):
        return {"i": sc.params["i"]}

    def trans(st, a, sc, t):
        st = dict(st)
        st["action"] = a
        return st, 0.0

    def term(st, sc):
        i = sc.params["i"]
        a = st["action"]
        if a == "mean_risky":
            loss = 10 if i < 19 else 300
        elif a == "tail_guard":
            loss = 78 if i == 19 else (72 if i >= 10 else 18)
        elif a == "rank_balanced":
            loss = 82 if i == 19 else (28 if i >= 10 else 20)
        elif a == "overkill":
            loss = 95
        else:
            raise ValueError(a)
        return loss, {"catastrophic": loss >= 250, "medium_harm": loss >= 60, "degraded": loss >= 60}

    return SealedNOMDP(
        "diagnostic_cvar_blindspot", actions, scenarios, horizon, trans, term, init,
        ConstraintSpec(event_max={"catastrophic": 0.0, "medium_harm": 0.60}),
        "Diagnostic CVaR single-tail blindspot."
    )

DOMAIN = {
    "bsp": {
        "label": "Blind Sterilization Protocol",
        "actions": ["gentle", "standard", "strong", "protect"],
        "treat": {"gentle": 1.02, "standard": 1.55, "strong": 2.20, "protect": 0.30},
        "damage": {"gentle": 0.22, "standard": 0.56, "strong": 1.05, "protect": -0.18},
        "ops": {"gentle": 2.0, "standard": 2.6, "strong": 4.0, "protect": 1.5},
        "cat_penalty": 9000.0, "harm_penalty": 460.0, "medium_penalty": 230.0,
        "hazard_coeff": 28.0, "damage_coeff": 35.0,
        "events": ("catastrophic", "degraded", "rejected", "shortage"),
        "harm_event": "degraded", "medium_event": "rejected", "extra_medium": "shortage",
        "base": 2.95, "fragile_threshold": 2.35, "robust_threshold": 3.15,
    },
    "bfsp": {
        "label": "Blind Food Safety Processing",
        "actions": ["mild_heat", "scheduled_heat", "extra_heat", "protect"],
        "treat": {"mild_heat": 1.00, "scheduled_heat": 1.55, "extra_heat": 2.15, "protect": 0.28},
        "damage": {"mild_heat": 0.22, "scheduled_heat": 0.58, "extra_heat": 1.00, "protect": -0.12},
        "ops": {"mild_heat": 1.5, "scheduled_heat": 2.3, "extra_heat": 3.6, "protect": 1.2},
        "cat_penalty": 10000.0, "harm_penalty": 390.0, "medium_penalty": 220.0,
        "hazard_coeff": 30.0, "damage_coeff": 24.0,
        "events": ("catastrophic", "degraded", "rejected"),
        "harm_event": "degraded", "medium_event": "rejected", "extra_medium": None,
        "base": 2.85, "fragile_threshold": 2.30, "robust_threshold": 3.10,
    },
    "bcst": {
        "label": "Blind Cell-State Treatment",
        "actions": ["spare", "standard", "boost", "protect"],
        "treat": {"spare": 1.00, "standard": 1.58, "boost": 2.22, "protect": 0.25},
        "damage": {"spare": 0.22, "standard": 0.64, "boost": 1.15, "protect": -0.10},
        "ops": {"spare": 2.0, "standard": 2.6, "boost": 4.2, "protect": 1.4},
        "cat_penalty": 8000.0, "harm_penalty": 1600.0, "medium_penalty": 350.0,
        "hazard_coeff": 55.0, "damage_coeff": 70.0,
        "events": ("catastrophic", "toxicity", "moderate_harm"),
        "harm_event": "toxicity", "medium_event": "moderate_harm", "extra_medium": None,
        "base": 2.90, "fragile_threshold": 2.18, "robust_threshold": 3.00,
    },
    "bged": {
        "label": "Blind Genome-Editor Delivery",
        "actions": ["low_vector", "standard_vector", "high_vector", "targeted_vector"],
        "treat": {"low_vector": 0.95, "standard_vector": 1.45, "high_vector": 2.05, "targeted_vector": 1.62},
        "damage": {"low_vector": 0.16, "standard_vector": 0.45, "high_vector": 0.98, "targeted_vector": 0.32},
        "ops": {"low_vector": 3.0, "standard_vector": 5.0, "high_vector": 8.0, "targeted_vector": 10.0},
        "cat_penalty": 6500.0, "harm_penalty": 5200.0, "medium_penalty": 4500.0,
        "hazard_coeff": 60.0, "damage_coeff": 115.0,
        "events": ("catastrophic", "no_benefit", "offtarget", "immune"),
        "harm_event": "offtarget", "medium_event": "no_benefit", "extra_medium": "immune",
        "base": 2.55, "fragile_threshold": 1.45, "robust_threshold": 2.10,
    },
}

VARIANT = {
    "easy": {
        "kind": "normal",
        "hazard_shift": -0.30,
        "damage_scale": 0.78,
        "constraints": {"catastrophic": 0.25, "degraded": 0.70, "rejected": 0.80, "shortage": 0.80,
                        "toxicity": 0.58, "moderate_harm": 0.95, "no_benefit": 0.45, "offtarget": 0.55, "immune": 0.55},
    },
    "hard": {
        # v0.5 hard is deliberately stricter than v0.4 to make governance meaningful.
        "kind": "normal",
        "hazard_shift": 0.28,
        "damage_scale": 1.05,
        "constraints": {"catastrophic": 0.075, "degraded": 0.30, "rejected": 0.36, "shortage": 0.36,
                        "toxicity": 0.24, "moderate_harm": 0.66, "no_benefit": 0.18, "offtarget": 0.26, "immune": 0.26},
    },
    "stress_unobservable": {"kind": "conflict"},
    "stress_impossible": {"kind": "impossible"},
}

def make_normal(domain, variant, horizon=4):
    cfg = DOMAIN[domain]
    var = VARIANT[variant]
    names, levels = ["H", "R", "G", "F"], [[0,1,2], [0,1], [0,1], [0,1]]
    def pf(p):
        return [0.58,0.30,0.12][p["H"]] * [0.82,0.18][p["R"]] * [0.78,0.22][p["G"]] * [0.76,0.24][p["F"]]
    scenarios = _scenario_grid(names, levels, pf)

    def init(sc):
        p = sc.params
        hazard = cfg["base"] + var["hazard_shift"] + 0.92*p["H"] + 0.78*p["R"] + 0.58*p["G"]
        return {"hazard": hazard, "damage": 0.0}

    def trans(st, a, sc, t):
        p = sc.params
        st = dict(st)
        treatment = cfg["treat"][a] * (1.0 - 0.16*p["R"] - 0.12*p["G"])
        dmg = cfg["damage"][a] * var["damage_scale"] * (1.0 + 0.50*p["F"] + 0.08*p["G"])
        st["hazard"] = max(0.0, st["hazard"] - treatment)
        st["damage"] = max(0.0, st["damage"] + dmg)
        return st, cfg["ops"][a]

    def term(st, sc):
        p = sc.params
        catastrophic = st["hazard"] > 0.72
        harm = st["damage"] > (cfg["robust_threshold"] if p["F"] == 0 else cfg["fragile_threshold"])
        medium = st["hazard"] > 1.22 or harm
        events = {ev: False for ev in cfg["events"]}
        events["catastrophic"] = catastrophic
        events[cfg["harm_event"]] = harm
        events[cfg["medium_event"]] = medium
        if cfg["extra_medium"]:
            if domain == "bged":
                events["immune"] = harm and p["G"] == 1
                events["offtarget"] = harm and p["F"] == 1
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
    # Domain-specific v0.5 calibration: hard should be sparse but not empty.
    # The cell-treatment hard setting is biologically harsher, so allow a small
    # recurrence/toxicity window to keep a few feasible blind policies.
    if domain == "bcst" and variant == "hard":
        cons.update({"catastrophic": 0.13, "toxicity": 0.34, "moderate_harm": 0.82})
    # Genome delivery hard is intentionally quite tight but should retain a few feasible policies.
    if domain == "bged" and variant == "hard":
        cons.update({"catastrophic": 0.22, "no_benefit": 0.24, "offtarget": 0.30, "immune": 0.30})
    return SealedNOMDP(
        f"{domain}_{variant}", cfg["actions"], scenarios, horizon, trans, term, init,
        ConstraintSpec(event_max=cons),
        f"{cfg['label']} ({variant})."
    )

def make_conflict(domain, impossible=False):
    cfg = DOMAIN[domain]
    actions = {
        "bsp": ["cycle_A", "cycle_B", "overkill", "hold"],
        "bfsp": ["process_A", "process_B", "overprocess", "hold"],
        "bcst": ["beam_A", "beam_B", "whole_region", "pause"],
        "bged": ["vector_A", "vector_B", "broad_vector", "pause"],
    }[domain]

    scenarios = [
        Scenario("mode_A", 0.50 if not impossible else 0.45, {"mode": "A", "untreatable": False}),
        Scenario("mode_B", 0.50 if not impossible else 0.45, {"mode": "B", "untreatable": False}),
    ]
    if impossible:
        scenarios.append(Scenario("untreatable", 0.10, {"mode": "C", "untreatable": True}))

    def init(sc):
        return {"match": 0, "damage": 0.0, "broad": 0, "wrong": 0}

    def trans(st, a, sc, t):
        st = dict(st)
        mode = sc.params["mode"]
        first, second, broad, pause = actions
        if sc.params.get("untreatable", False):
            if a == broad:
                st["damage"] += 1.8
            elif a != pause:
                st["damage"] += 0.6
            return st, 2.0

        if (mode == "A" and a == first) or (mode == "B" and a == second):
            st["match"] += 1
            st["damage"] += 0.25
        elif a == broad:
            st["match"] += 1
            st["broad"] += 1
            st["damage"] += 1.4
        elif a == pause:
            st["damage"] = max(0.0, st["damage"] - 0.05)
        else:
            st["wrong"] += 1
            st["damage"] += 0.75
        return st, 2.0

    def term(st, sc):
        if sc.params.get("untreatable", False):
            catastrophic = True
        else:
            catastrophic = st["match"] < 2
        harm = st["damage"] > 1.1
        medium = catastrophic or harm
        events = {ev: False for ev in cfg["events"]}
        events["catastrophic"] = catastrophic
        events[cfg["harm_event"]] = harm
        events[cfg["medium_event"]] = medium
        if cfg["extra_medium"]:
            if domain == "bged":
                events["immune"] = harm
                events["offtarget"] = harm
                events["catastrophic"] = catastrophic or harm
            else:
                events[cfg["extra_medium"]] = medium
        loss = (
            cfg["cat_penalty"] * events["catastrophic"]
            + cfg["harm_penalty"] * events[cfg["harm_event"]]
            + cfg["medium_penalty"] * events[cfg["medium_event"]]
            + 50.0 * st["damage"]
            + 80.0 * max(0, 2 - st["match"])
        )
        return loss, events

    if impossible:
        cons = {"catastrophic": 0.0}
    else:
        cons = {"catastrophic": 0.0, cfg["harm_event"]: 0.0}
        if cfg["extra_medium"] and cfg["extra_medium"] in cfg["events"]:
            cons[cfg["extra_medium"]] = 0.0

    return SealedNOMDP(
        f"{domain}_{'stress_impossible' if impossible else 'stress_unobservable'}",
        actions, scenarios, 2, trans, term, init, ConstraintSpec(event_max=cons),
        f"{cfg['label']} ({'stress_impossible' if impossible else 'stress_unobservable'})."
    )

def list_instances():
    names = ["diagnostic_cvar_blindspot"]
    for domain in ["bsp", "bfsp", "bcst", "bged"]:
        for variant in ["easy", "hard", "stress_unobservable", "stress_impossible"]:
            names.append(f"{domain}_{variant}")
    return names

def make_instance(name, horizon=None):
    if name == "diagnostic_cvar_blindspot":
        return make_diagnostic_cvar_blindspot(horizon or 1)
    parts = name.split("_")
    domain = parts[0]
    variant = "_".join(parts[1:])
    if domain not in DOMAIN or variant not in VARIANT:
        raise ValueError(f"Unknown instance {name}. Options: {list_instances()}")
    kind = VARIANT[variant]["kind"]
    if kind == "normal":
        return make_normal(domain, variant, horizon or 4)
    if kind == "conflict":
        return make_conflict(domain, impossible=False)
    if kind == "impossible":
        return make_conflict(domain, impossible=True)
    raise ValueError(name)
