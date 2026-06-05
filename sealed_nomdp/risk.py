from __future__ import annotations
import numpy as np

def normalize(p):
    p = np.asarray(p, dtype=float)
    s = p.sum()
    if s <= 0:
        return np.ones_like(p, dtype=float) / len(p)
    return p / s

def weighted_mean(losses, probs):
    return float(np.dot(normalize(probs), np.asarray(losses, dtype=float)))

def weighted_event_prob(mask, probs):
    return float(np.dot(normalize(probs), np.asarray(mask, dtype=bool).astype(float)))

def weighted_var(losses, probs, alpha):
    losses = np.asarray(losses, dtype=float)
    probs = normalize(probs)
    order = np.argsort(losses)
    x = losses[order]
    p = probs[order]
    cdf = np.cumsum(p)
    idx = np.searchsorted(cdf, 1.0 - alpha, side="left")
    return float(x[min(idx, len(x) - 1)])

def weighted_cvar(losses, probs, alpha):
    losses = np.asarray(losses, dtype=float)
    probs = normalize(probs)
    order = np.argsort(-losses)
    x = losses[order]
    p = probs[order]
    remaining = float(alpha)
    acc = 0.0
    for xi, pi in zip(x, p):
        take = min(remaining, pi)
        acc += take * xi
        remaining -= take
        if remaining <= 1e-15:
            break
    return float(acc / alpha)

def rank_weights(n, profile="balanced"):
    beta = {"mild": 1.25, "balanced": 3.0, "severe": 7.0}[profile]
    r = np.arange(1, n + 1, dtype=float)
    return normalize(np.exp(-beta * (r - 1) / max(n - 1, 1)))

def wowa(losses, probs, profile="balanced"):
    losses = np.asarray(losses, dtype=float)
    probs = normalize(probs)
    order = np.argsort(-losses)
    x = losses[order]
    p = probs[order]
    w = rank_weights(len(x), profile)
    omega = normalize(np.sqrt(np.maximum(p, 0.0) * np.maximum(w, 0.0)))
    return float(np.dot(omega, x))

def robust20(losses, probs):
    return weighted_cvar(losses, probs, 0.20)

def summarize(losses, probs, events):
    losses = np.asarray(losses, dtype=float)
    out = {"mean": weighted_mean(losses, probs), "worst": float(np.max(losses))}
    for name, mask in sorted(events.items()):
        out[f"prob_{name}"] = weighted_event_prob(mask, probs)
    for a in [0.01, 0.05, 0.10, 0.25]:
        out[f"var{int(a*100):02d}"] = weighted_var(losses, probs, a)
        out[f"cvar{int(a*100):02d}"] = weighted_cvar(losses, probs, a)
    for profile in ["mild", "balanced", "severe"]:
        out[f"wowa_{profile}"] = wowa(losses, probs, profile)
    out["robust20"] = robust20(losses, probs)
    return out
