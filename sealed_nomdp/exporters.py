from __future__ import annotations
from pathlib import Path
import json
from collections import deque
from .algorithms import evaluate_sequence_with_metrics

def _state_name(prefix, scenario_idx):
    if prefix == ("terminal",):
        return "terminal"
    p = "start" if not prefix else "_".join(prefix)
    return f"s{scenario_idx}_{p}"

def export_history_pomdp(env, out_prefix, discount=1.0):
    """Export a finite-horizon, history-expanded POMDP with singleton observation.

    The state includes the hidden scenario and the action prefix. This is not meant as a compact
    engineering POMDP; it is a transparent external-solver test showing that observations are null.
    Rewards are negative losses: step rewards are represented as 0 and terminal reward is negative
    terminal total loss for the complete sequence. This is enough for small instances and for showing
    that observation-conditioned policies cannot branch on hidden scenario.
    """
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    pomdp_path = out_prefix.with_suffix(".pomdp")
    meta_path = out_prefix.with_name(out_prefix.name + "_metadata.json")

    actions = list(env.actions)
    obs = ["null"]
    states = []
    prefixes = [()]
    for _ in range(env.horizon):
        new = []
        for p in prefixes:
            for a in actions:
                new.append(p + (a,))
        prefixes.extend(new)
    prefixes_by_len = {}
    for p in prefixes:
        if len(p) <= env.horizon:
            prefixes_by_len.setdefault(len(p), []).append(p)

    for j, _ in enumerate(env.scenarios):
        for length in range(env.horizon + 1):
            for p in prefixes_by_len.get(length, []):
                if len(p) == length:
                    states.append(_state_name(p, j))
    states.append("terminal")

    start_states = [_state_name((), j) for j in range(len(env.scenarios))]
    start_probs = env.probs

    with pomdp_path.open("w") as f:
        f.write("# History-expanded singleton-observation POMDP export\n")
        f.write(f"discount: {discount}\n")
        f.write("values: reward\n")
        f.write("states: " + " ".join(states) + "\n")
        f.write("actions: " + " ".join(actions) + "\n")
        f.write("observations: null\n")
        f.write("start: ")
        start_vec = []
        start_set = set(start_states)
        for s in states:
            if s in start_set:
                j = start_states.index(s)
                start_vec.append(str(float(start_probs[j])))
            else:
                start_vec.append("0")
        f.write(" ".join(start_vec) + "\n\n")

        # Transitions
        for a in actions:
            f.write(f"# transitions for {a}\n")
            for j in range(len(env.scenarios)):
                for length in range(env.horizon + 1):
                    for p in prefixes_by_len.get(length, []):
                        s = _state_name(p, j)
                        if length >= env.horizon:
                            sp = "terminal"
                        else:
                            sp = _state_name(p + (a,), j)
                        f.write(f"T: {a} : {s} : {sp} 1\n")
            f.write(f"T: {a} : terminal : terminal 1\n")
        f.write("\n")

        # Observations: always null.
        for a in actions:
            for sp in states:
                f.write(f"O: {a} : {sp} : null 1\n")
        f.write("\n")

        # Rewards. Give terminal negative total loss when action completes final prefix.
        # For transitions that lead to terminal, reward equals -loss of resulting sequence.
        for j in range(len(env.scenarios)):
            for p in prefixes_by_len.get(env.horizon - 1, []):
                for a in actions:
                    seq = p + (a,)
                    ev = env.evaluate_sequence(seq)
                    loss = float(ev.losses[j])
                    s = _state_name(p, j)
                    f.write(f"R: {a} : {s} : terminal : null {-loss}\n")

    meta = {
        "instance": env.name,
        "horizon": env.horizon,
        "num_actions": len(actions),
        "num_scenarios": len(env.scenarios),
        "num_states": len(states),
        "observations": obs,
        "null_observation_claim": "All observations are the singleton symbol 'null'; an observation-conditioned solver has no online signal to distinguish hidden scenarios.",
        "external_solver_note": "This export is history-expanded for transparency and small-instance solver tests, not for compact large-scale POMDP solving.",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return pomdp_path, meta_path
