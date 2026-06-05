from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
import numpy as np

@dataclass(frozen=True)
class Scenario:
    name: str
    prob: float
    params: dict

@dataclass
class EvalResult:
    losses: np.ndarray
    events: dict

@dataclass
class ConstraintSpec:
    event_max: dict = field(default_factory=dict)
    mean_max: float | None = None
    worst_max: float | None = None

class SealedNOMDP:
    def __init__(self, name, actions, scenarios, horizon, transition_fn, terminal_fn, initial_state_fn,
                 constraints=None, description=""):
        self.name = name
        self.actions = tuple(actions)
        self.scenarios = tuple(scenarios)
        self.horizon = int(horizon)
        self.transition_fn = transition_fn
        self.terminal_fn = terminal_fn
        self.initial_state_fn = initial_state_fn
        self.constraints = constraints or ConstraintSpec()
        self.description = description
        self.probs = np.array([s.prob for s in self.scenarios], dtype=float)
        self.probs = self.probs / self.probs.sum()

    def evaluate_sequence(self, seq):
        seq = tuple(seq)
        if len(seq) != self.horizon:
            raise ValueError(f"Expected sequence length {self.horizon}, got {len(seq)}")
        losses = []
        events = {}
        for sc in self.scenarios:
            state = self.initial_state_fn(sc)
            total = 0.0
            for t, action in enumerate(seq):
                state, step_loss = self.transition_fn(state, action, sc, t)
                total += step_loss
            terminal_loss, ev = self.terminal_fn(state, sc)
            total += terminal_loss
            losses.append(float(total))
            for k, v in ev.items():
                events.setdefault(k, []).append(bool(v))
        return EvalResult(np.array(losses, dtype=float), {k: np.array(v, dtype=bool) for k, v in events.items()})

    def sequences(self, max_sequences=None, seed=0):
        total = len(self.actions) ** self.horizon
        if max_sequences is None or total <= max_sequences:
            yield from product(self.actions, repeat=self.horizon)
            return

        emitted = set()
        def emit(seq):
            seq = tuple(seq)
            if seq not in emitted:
                emitted.add(seq)
                return seq
            return None

        templates = []
        for a in self.actions:
            templates.append((a,) * self.horizon)
        for a in self.actions:
            for b in self.actions:
                templates.append(tuple(([a, b] * ((self.horizon + 1) // 2))[:self.horizon]))
        for a in self.actions:
            for b in self.actions:
                for c in self.actions:
                    templates.append(tuple(([a, b, c] * ((self.horizon + 2) // 3))[:self.horizon]))

        for seq in templates:
            out = emit(seq)
            if out:
                yield out
                if len(emitted) >= max_sequences:
                    return

        rng = np.random.default_rng(seed)
        while len(emitted) < max_sequences:
            out = emit(tuple(rng.choice(self.actions, size=self.horizon)))
            if out:
                yield out
