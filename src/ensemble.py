"""Vote rules and accuracy-versus-size curves.

The question this module answers: given per-member predictions, how much does
the ensemble actually gain as you add members, and does that curve bend where
the effective sample size says it should?
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .correlation import effective_sample_size, mean_pairwise


def majority_vote(predictions: np.ndarray, confidences: np.ndarray | None = None) -> np.ndarray:
    """Plurality vote over member predictions.

    predictions: (n_models, n_items) array of label strings or ints.
    Ties are broken by summed confidence when available, otherwise by the
    prediction of the first member in the subset (a fixed, documented rule,
    which matters because random tie-breaking inflates the variance you are
    trying to measure).
    """
    P = np.asarray(predictions)
    if P.ndim != 2:
        raise ValueError("predictions must be 2-D (n_models, n_items)")
    n_models, n_items = P.shape
    C = np.asarray(confidences) if confidences is not None else None
    out = np.empty(n_items, dtype=P.dtype)

    for j in range(n_items):
        column = P[:, j]
        counts = Counter(column.tolist())
        top = max(counts.values())
        candidates = [label for label, count in counts.items() if count == top]
        if len(candidates) == 1 or C is None:
            out[j] = candidates[0] if len(candidates) == 1 else column[0]
            continue
        scored = {
            label: float(np.sum(C[:, j][column == label]))
            for label in candidates
        }
        out[j] = max(scored, key=scored.get)
    return out


def confidence_weighted_vote(predictions: np.ndarray, confidences: np.ndarray) -> np.ndarray:
    """Soft vote: each member contributes its confidence to its predicted label."""
    P = np.asarray(predictions)
    C = np.asarray(confidences, dtype=float)
    if P.shape != C.shape:
        raise ValueError(f"shape mismatch: {P.shape} vs {C.shape}")
    n_items = P.shape[1]
    out = np.empty(n_items, dtype=P.dtype)
    for j in range(n_items):
        column = P[:, j]
        weights: dict[object, float] = {}
        for label, weight in zip(column.tolist(), C[:, j].tolist()):
            weights[label] = weights.get(label, 0.0) + float(weight)
        out[j] = max(weights, key=weights.get)
    return out


VOTE_RULES = {
    "majority": lambda P, C: majority_vote(P, C),
    "confidence_weighted": lambda P, C: confidence_weighted_vote(P, C if C is not None else np.ones_like(P, dtype=float)),
}


@dataclass
class ScalingPoint:
    n_members: int
    mean_accuracy: float
    std_accuracy: float
    predicted_ess: float
    n_subsets: int


def accuracy_vs_size(
    predictions: np.ndarray,
    gold: Sequence,
    confidences: np.ndarray | None = None,
    sizes: Sequence[int] = (1, 2, 4, 8, 16),
    subsets_per_size: int = 200,
    rule: str = "majority",
    rng: np.random.Generator | None = None,
) -> list[ScalingPoint]:
    """Monte-Carlo estimate of ensemble accuracy as a function of member count.

    For each size n, draw `subsets_per_size` random subsets of members without
    replacement, vote within each subset, and record accuracy. Subsets larger
    than the member pool are skipped rather than sampled with replacement, so
    the curve never silently flattens because it ran out of models.
    """
    P = np.asarray(predictions)
    y = np.asarray(gold)
    C = np.asarray(confidences, dtype=float) if confidences is not None else None
    n_models = P.shape[0]
    rng = rng if rng is not None else np.random.default_rng(0)
    vote = VOTE_RULES[rule]

    error_matrix = (P != y[None, :]).astype(np.int8)
    rho_bar_full = mean_pairwise(error_matrix, "phi")

    points: list[ScalingPoint] = []
    for n in sizes:
        if n > n_models:
            continue
        n_draws = 1 if n == n_models else subsets_per_size
        accuracies = np.empty(n_draws, dtype=float)
        for d in range(n_draws):
            members = rng.choice(n_models, size=n, replace=False) if n < n_models else np.arange(n_models)
            sub_c = C[members] if C is not None else None
            voted = vote(P[members], sub_c)
            accuracies[d] = float(np.mean(voted == y))
        points.append(
            ScalingPoint(
                n_members=int(n),
                mean_accuracy=float(accuracies.mean()),
                std_accuracy=float(accuracies.std(ddof=0)),
                predicted_ess=effective_sample_size(int(n), rho_bar_full),
                n_subsets=int(n_draws),
            )
        )
    return points


def saturation_point(points: Sequence[ScalingPoint], tolerance: float = 0.002) -> int:
    """Smallest member count beyond which accuracy never improves by `tolerance`.

    This is the empirical counterpart of ESS: if the measured saturation point
    tracks the predicted ESS across ensemble families, the paper's central
    claim holds.
    """
    ordered = sorted(points, key=lambda p: p.n_members)
    best = max(p.mean_accuracy for p in ordered) if ordered else 0.0
    for p in ordered:
        if p.mean_accuracy >= best - tolerance:
            return p.n_members
    return ordered[-1].n_members if ordered else 0


def oracle_and_floor(predictions: np.ndarray, gold: Sequence) -> dict[str, float]:
    """Headroom bounds for the ensemble.

    oracle_accuracy: fraction of items at least one member answers correctly.
    all_correct: fraction every member gets right.
    all_wrong: fraction every member gets wrong, which is the share of error
    that no vote rule can ever recover and the clearest symptom of inherited
    teacher weakness.
    """
    P = np.asarray(predictions)
    y = np.asarray(gold)
    correct = P == y[None, :]
    return {
        "oracle_accuracy": float(np.mean(correct.any(axis=0))),
        "all_correct": float(np.mean(correct.all(axis=0))),
        "all_wrong": float(np.mean(~correct.any(axis=0))),
        "mean_member_accuracy": float(np.mean(correct)),
    }
