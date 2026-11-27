"""Bayesian knowledge tracing, as plain functions.

BKT keeps one number per concept: the probability the learner knows it. An observation updates it
by Bayes' rule using two error rates (guess: right without knowing; slip: wrong despite knowing),
then allows for learning from the attempt. It needs little data and is explainable, which is why it
is the model here; the update functions are pure so they are easy to test and to swap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BKTParams:
    learn: float = 0.12          # chance of learning the concept from one attempt (with feedback)
    forget_days: float = 45.0    # time constant of forgetting, before evidence slows it down


def _clamp(x: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return min(max(x, lo), hi)


def error_rates(difficulty: float) -> tuple:
    """(guess, slip) for an item of this difficulty: easy items are easier to guess, hard ones easier to slip on."""
    d = _clamp(difficulty, 0.0, 1.0)
    return 0.05 + 0.30 * (1 - d), 0.03 + 0.15 * d


def observe(p: float, correct: bool, params: BKTParams = BKTParams(), difficulty: float = 0.5, weight: float = 1.0) -> float:
    """The new probability of knowing after one observation.

    ``weight`` in [0, 1] says how much to trust the observation: 1 for a graded answer, less for indirect
    signals (weight 0 leaves the belief untouched). A wrong answer never raises the estimate."""
    p = _clamp(p)
    guess, slip = error_rates(difficulty)
    if correct:
        posterior = p * (1 - slip) / (p * (1 - slip) + (1 - p) * guess)
    else:
        posterior = p * slip / (p * slip + (1 - p) * (1 - guess))
    after = posterior + (1 - posterior) * params.learn
    if not correct:
        after = min(after, p)
    w = _clamp(weight, 0.0, 1.0)
    return _clamp((1 - w) * p + w * after)


def exposure(p: float, amount: float = 0.03) -> float:
    """Having read about it counts for a little, never for much."""
    return _clamp(p + amount * (1 - p))


def decay(p: float, prior: float, elapsed_days: float, evidence: int, params: BKTParams = BKTParams()) -> float:
    """Forgetting: drift back towards the prior. Well-evidenced beliefs fade more slowly."""
    if elapsed_days <= 0:
        return p
    tau = params.forget_days * min(1 + evidence / 5.0, 4.0)
    return prior + (p - prior) * math.exp(-elapsed_days / tau)
