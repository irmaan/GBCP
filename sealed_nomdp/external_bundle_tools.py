from __future__ import annotations
from pathlib import Path
import itertools, json
from typing import Sequence, Tuple, List

def _read_csv_matrix(path: Path, cast=float):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([cast(x) for x in line.split(",")])
    return rows

def load_bundle(bundle_dir: str | Path) -> dict:
    bundle_dir = Path(bundle_dir)
    next_state_1 = _read_csv_matrix(bundle_dir / "next_state.csv", int)
    reward = _read_csv_matrix(bundle_dir / "reward.csv", float)
    with open(bundle_dir / "start.csv") as f:
        start = [float(x) for x in f.read().strip().split(",") if x.strip()]
    with open(bundle_dir / "terminal.csv") as f:
        terminal = [int(line.strip().split(",")[0]) == 1 for line in f if line.strip()]
    actions = [line.strip() for line in (bundle_dir / "actions.txt").read_text().splitlines() if line.strip()]
    horizon = int((bundle_dir / "horizon.txt").read_text().strip())
    discount = float((bundle_dir / "discount.txt").read_text().strip())
    state_meta_path = bundle_dir / "state_meta.jsonl"
    state_meta = []
    if state_meta_path.exists():
        state_meta = [json.loads(line) for line in state_meta_path.read_text().splitlines() if line.strip()]

    next_state = [[int(x) - 1 for x in row] for row in next_state_1]
    action_to_idx = {a:i for i, a in enumerate(actions)}

    start_state_by_scenario = {}
    if state_meta:
        for i, sm in enumerate(state_meta):
            if sm.get("depth", None) == 0 and "scenario_index" in sm:
                start_state_by_scenario[int(sm["scenario_index"])] = i
    if not start_state_by_scenario:
        for i, p in enumerate(start):
            if p > 1e-15:
                start_state_by_scenario[i] = i

    scenario_probs = []
    for scen in sorted(start_state_by_scenario):
        scenario_probs.append(float(start[start_state_by_scenario[scen]]))

    return {
        "bundle_dir": str(bundle_dir),
        "next_state": next_state,
        "reward": reward,
        "start": start,
        "terminal": terminal,
        "actions": actions,
        "action_to_idx": action_to_idx,
        "horizon": horizon,
        "discount": discount,
        "state_meta": state_meta,
        "start_state_by_scenario": start_state_by_scenario,
        "scenario_probs": scenario_probs,
    }

def enumerate_sequences(actions: Sequence[str], horizon: int):
    for seq in itertools.product(actions, repeat=horizon):
        yield tuple(seq)

def scenario_returns(bundle: dict, seq: Sequence[str], gamma: float | None = None) -> Tuple[List[float], List[float]]:
    gamma = bundle["discount"] if gamma is None else float(gamma)
    ns = bundle["next_state"]
    rw = bundle["reward"]
    a2i = bundle["action_to_idx"]
    returns = []
    for scen, s0 in sorted(bundle["start_state_by_scenario"].items()):
        state = int(s0)
        disc = 1.0
        ret = 0.0
        for a in seq:
            ai = a2i[a]
            ret += disc * float(rw[state][ai])
            state = int(ns[state][ai])
            disc *= gamma
        returns.append(ret)
    probs = list(bundle["scenario_probs"])
    return returns, probs

def weighted_mean(vals: Sequence[float], probs: Sequence[float]) -> float:
    z = sum(probs)
    return sum(v*p for v, p in zip(vals, probs)) / max(z, 1e-15)

def weighted_cvar_losses(losses: Sequence[float], probs: Sequence[float], alpha: float) -> float:
    pairs = sorted(zip(losses, probs), key=lambda x: x[0], reverse=True)
    remain = alpha
    acc = 0.0
    total = 0.0
    for loss, p in pairs:
        if remain <= 1e-15:
            break
        take = min(remain, p)
        total += loss * take
        acc += take
        remain -= take
    return total / max(acc, 1e-15)

def score_sequence(bundle: dict, seq: Sequence[str], criterion: str = "mean", gamma: float | None = None) -> dict:
    rets, probs = scenario_returns(bundle, seq, gamma)
    losses = [-r for r in rets]
    out = {
        "sequence": "|".join(seq),
        "expected_return": weighted_mean(rets, probs),
        "mean_loss": weighted_mean(losses, probs),
        "worst_loss": max(losses),
        "discount_gamma": bundle["discount"] if gamma is None else float(gamma),
        "scenario_returns_json": json.dumps(rets),
        "scenario_probs_json": json.dumps(probs),
    }
    out["cvar05_loss"] = weighted_cvar_losses(losses, probs, 0.05)
    out["cvar25_loss"] = weighted_cvar_losses(losses, probs, 0.25)
    if criterion == "mean":
        out["criterion_score"] = -out["mean_loss"]
    elif criterion == "worst":
        out["criterion_score"] = -out["worst_loss"]
    elif criterion == "cvar05":
        out["criterion_score"] = -out["cvar05_loss"]
    elif criterion == "cvar25":
        out["criterion_score"] = -out["cvar25_loss"]
    else:
        raise ValueError(f"unknown criterion {criterion}")
    return out

def exact_search(bundle: dict, criterion: str = "mean", gamma: float | None = None) -> dict:
    best = None
    for seq in enumerate_sequences(bundle["actions"], bundle["horizon"]):
        row = score_sequence(bundle, seq, criterion=criterion, gamma=gamma)
        if best is None or row["criterion_score"] > best["criterion_score"]:
            best = row
    return best
