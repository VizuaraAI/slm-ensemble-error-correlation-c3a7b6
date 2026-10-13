"""Error-correlation and effective-sample-size estimators.

This module is the scientific core of the project and it depends only on
numpy, so it runs anywhere and is easy to unit test. Everything here operates
on an *error matrix* E of shape (n_models, n_items) with entries in {0, 1},
where 1 means the model got that item wrong.

Definitions used, with the names they have in the literature:

  phi            Pearson correlation between two binary error vectors, also
                 called the mean square contingency coefficient.
  Yule's Q       (n11 n00 - n10 n01) / (n11 n00 + n10 n01). Bounded in
                 [-1, 1], used as a pairwise diversity measure by Kuncheva
                 and Whitaker (2003).
  double fault   fraction of items both models get wrong. The quantity that
                 actually hurts a majority vote.
  disagreement   fraction of items exactly one model gets wrong.
  kappa          Cohen's kappa on the error labels, chance corrected.

  ESS            effective sample size, from the design effect of a
                 correlated mean:

                     Var(mean error) = sigma^2 / N * (1 + (N - 1) * rho_bar)
                     ESS             = N / (1 + (N - 1) * rho_bar)

                 ESS is the number of *independent* models whose averaged
                 error would be as noisy as your N correlated ones. It is the
                 headline number of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------------------
# Pairwise measures
# ---------------------------------------------------------------------------
def contingency(a: np.ndarray, b: np.ndarray) -> tuple[int, int, int, int]:
    """Return (n11, n10, n01, n00) for two binary error vectors."""
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    n11 = int(np.sum(a & b))
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n00 = int(np.sum(~a & ~b))
    return n11, n10, n01, n00


def phi(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two binary vectors. Returns 0.0 if either is constant."""
    n11, n10, n01, n00 = contingency(a, b)
    num = n11 * n00 - n10 * n01
    den = np.sqrt(float(n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    if den < EPS:
        return 0.0
    return float(num / den)


def yule_q(a: np.ndarray, b: np.ndarray) -> float:
    n11, n10, n01, n00 = contingency(a, b)
    num = n11 * n00 - n10 * n01
    den = n11 * n00 + n10 * n01
    if abs(den) < EPS:
        return 0.0
    return float(num / den)


def double_fault(a: np.ndarray, b: np.ndarray) -> float:
    n11, _, _, _ = contingency(a, b)
    return float(n11 / max(len(a), 1))


def disagreement(a: np.ndarray, b: np.ndarray) -> float:
    _, n10, n01, _ = contingency(a, b)
    return float((n10 + n01) / max(len(a), 1))


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    n11, n10, n01, n00 = contingency(a, b)
    total = n11 + n10 + n01 + n00
    if total == 0:
        return 0.0
    observed = (n11 + n00) / total
    p_a = (n11 + n10) / total
    p_b = (n11 + n01) / total
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if abs(1 - expected) < EPS:
        return 0.0
    return float((observed - expected) / (1 - expected))


MEASURES = {
    "phi": phi,
    "yule_q": yule_q,
    "double_fault": double_fault,
    "disagreement": disagreement,
    "kappa": cohen_kappa,
}


def pairwise_matrix(error_matrix: np.ndarray, measure: str = "phi") -> np.ndarray:
    """Symmetric (n_models, n_models) matrix of a pairwise measure, diagonal 1.0."""
    if measure not in MEASURES:
        raise KeyError(f"unknown measure {measure!r}; choose from {sorted(MEASURES)}")
    fn = MEASURES[measure]
    E = np.asarray(error_matrix)
    n = E.shape[0]
    out = np.eye(n, dtype=float)
    for i, j in combinations(range(n), 2):
        value = fn(E[i], E[j])
        out[i, j] = out[j, i] = value
    return out


def mean_pairwise(error_matrix: np.ndarray, measure: str = "phi") -> float:
    """Mean of the off-diagonal entries. This is rho_bar when measure is 'phi'."""
    E = np.asarray(error_matrix)
    n = E.shape[0]
    if n < 2:
        return 0.0
    M = pairwise_matrix(E, measure=measure)
    off = M[np.triu_indices(n, k=1)]
    return float(np.mean(off))


# ---------------------------------------------------------------------------
# Effective sample size
# ---------------------------------------------------------------------------
def effective_sample_size(n_models: int, rho_bar: float) -> float:
    """ESS = N / (1 + (N - 1) rho_bar), with rho_bar clipped to a feasible range.

    For N correlated members the mean pairwise correlation cannot go below
    -1 / (N - 1), otherwise the implied covariance matrix is not positive
    semi-definite. Clipping keeps ESS finite and positive.
    """
    if n_models < 1:
        raise ValueError("n_models must be >= 1")
    if n_models == 1:
        return 1.0
    lower = -1.0 / (n_models - 1) + 1e-6
    rho = float(np.clip(rho_bar, lower, 1.0))
    return float(n_models / (1.0 + (n_models - 1) * rho))


def design_effect(n_models: int, rho_bar: float) -> float:
    """Inflation of the variance of the averaged error relative to independence."""
    return float(n_models / effective_sample_size(n_models, rho_bar))


@dataclass
class EnsembleDiversityReport:
    n_models: int
    n_items: int
    mean_error_rate: float
    rho_bar: float
    ess: float
    design_effect: float
    mean_yule_q: float
    mean_double_fault: float
    mean_disagreement: float
    rho_bar_ci_low: float
    rho_bar_ci_high: float
    ess_ci_low: float
    ess_ci_high: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def diversity_report(
    error_matrix: np.ndarray,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> EnsembleDiversityReport:
    """Full summary for one ensemble, with item-level bootstrap intervals.

    The bootstrap resamples *items*, not models: the models are the fixed
    population under study, the probe items are the sample.
    """
    E = np.asarray(error_matrix).astype(np.int8)
    if E.ndim != 2:
        raise ValueError("error_matrix must be 2-D (n_models, n_items)")
    n_models, n_items = E.shape
    rng = rng if rng is not None else np.random.default_rng(0)

    rho_bar = mean_pairwise(E, "phi")
    point_ess = effective_sample_size(n_models, rho_bar)

    rho_samples = np.empty(n_bootstrap, dtype=float)
    ess_samples = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_items, size=n_items)
        resampled = E[:, idx]
        rho_b = mean_pairwise(resampled, "phi")
        rho_samples[b] = rho_b
        ess_samples[b] = effective_sample_size(n_models, rho_b)

    alpha = (1.0 - ci_level) / 2.0
    if n_bootstrap > 0:
        rho_lo, rho_hi = np.quantile(rho_samples, [alpha, 1 - alpha])
        ess_lo, ess_hi = np.quantile(ess_samples, [alpha, 1 - alpha])
    else:
        rho_lo = rho_hi = rho_bar
        ess_lo = ess_hi = point_ess

    return EnsembleDiversityReport(
        n_models=n_models,
        n_items=n_items,
        mean_error_rate=float(E.mean()),
        rho_bar=rho_bar,
        ess=point_ess,
        design_effect=design_effect(n_models, rho_bar),
        mean_yule_q=mean_pairwise(E, "yule_q"),
        mean_double_fault=mean_pairwise(E, "double_fault"),
        mean_disagreement=mean_pairwise(E, "disagreement"),
        rho_bar_ci_low=float(rho_lo),
        rho_bar_ci_high=float(rho_hi),
        ess_ci_low=float(ess_lo),
        ess_ci_high=float(ess_hi),
    )


# ---------------------------------------------------------------------------
# Teacher-student inheritance
# ---------------------------------------------------------------------------
@dataclass
class InheritanceReport:
    rho_teacher_student: float
    p_student_wrong: float
    p_student_wrong_given_teacher_wrong: float
    p_student_wrong_given_teacher_right: float
    error_lift: float
    inherited_error_share: float
    yule_q: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def inheritance_report(teacher_err: np.ndarray, student_err: np.ndarray) -> InheritanceReport:
    """Quantify how much of a student's error pattern tracks its teacher's.

    error_lift is the ratio of the two conditional error rates. A value of 1.0
    means the student's mistakes are statistically unrelated to the teacher's;
    a large value means the student fails precisely where the teacher failed.
    inherited_error_share is the fraction of the student's errors that land on
    items the teacher also got wrong.
    """
    t = np.asarray(teacher_err).astype(bool)
    s = np.asarray(student_err).astype(bool)
    if t.shape != s.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {s.shape}")

    n_t_wrong = int(t.sum())
    n_t_right = int((~t).sum())
    p_s = float(s.mean()) if s.size else 0.0
    p_s_given_t_wrong = float(s[t].mean()) if n_t_wrong else 0.0
    p_s_given_t_right = float(s[~t].mean()) if n_t_right else 0.0
    lift = p_s_given_t_wrong / p_s_given_t_right if p_s_given_t_right > EPS else float("inf")
    inherited = float(np.sum(s & t) / max(int(s.sum()), 1))

    return InheritanceReport(
        rho_teacher_student=phi(t, s),
        p_student_wrong=p_s,
        p_student_wrong_given_teacher_wrong=p_s_given_t_wrong,
        p_student_wrong_given_teacher_right=p_s_given_t_right,
        error_lift=lift,
        inherited_error_share=inherited,
        yule_q=yule_q(t, s),
    )


# ---------------------------------------------------------------------------
# Variance decomposition
# ---------------------------------------------------------------------------
def error_variance_decomposition(error_matrix: np.ndarray) -> dict[str, float]:
    """Split the variance of the ensemble mean error into variance and covariance.

    Var(mean) = (1/N^2) * [ sum_i Var_i + sum_{i != j} Cov_ij ]

    The covariance share is the part of the ensemble's noise that extra members
    cannot average away. Reporting it alongside ESS makes the mechanism visible
    rather than just the headline number.
    """
    E = np.asarray(error_matrix).astype(float)
    n_models = E.shape[0]
    cov = np.cov(E, bias=True)
    cov = np.atleast_2d(cov)
    var_term = float(np.trace(cov))
    cov_term = float(cov.sum() - np.trace(cov))
    total = (var_term + cov_term) / (n_models**2)
    return {
        "variance_term": var_term / (n_models**2),
        "covariance_term": cov_term / (n_models**2),
        "variance_of_mean": total,
        "covariance_share": float(cov_term / (var_term + cov_term)) if abs(var_term + cov_term) > EPS else 0.0,
    }


def group_rho(error_matrix: np.ndarray, groups: Sequence[str]) -> dict[str, float]:
    """Mean within-group and between-group error correlation.

    Pass the teacher label (or any recipe axis) as `groups` to answer the
    question the paper is built around: which axis of the recipe grid actually
    buys independence?
    """
    E = np.asarray(error_matrix)
    if len(groups) != E.shape[0]:
        raise ValueError("groups must have one entry per model")
    M = pairwise_matrix(E, "phi")
    within: list[float] = []
    between: list[float] = []
    for i, j in combinations(range(E.shape[0]), 2):
        (within if groups[i] == groups[j] else between).append(M[i, j])
    return {
        "rho_within": float(np.mean(within)) if within else float("nan"),
        "rho_between": float(np.mean(between)) if between else float("nan"),
        "n_within_pairs": len(within),
        "n_between_pairs": len(between),
    }


def error_matrix_from_records(
    records: Iterable[dict],
    model_key: str = "model",
    item_key: str = "item_id",
    error_key: str = "is_error",
) -> tuple[np.ndarray, list[str], list[str]]:
    """Pivot flat trace records into (error_matrix, model_order, item_order).

    Items missing for any model are dropped, so every column is comparable
    across all members. The number dropped is worth logging in your notes.
    """
    rows = list(records)
    models = sorted({str(r[model_key]) for r in rows})
    per_model: dict[str, dict[str, int]] = {m: {} for m in models}
    for r in rows:
        per_model[str(r[model_key])][str(r[item_key])] = int(bool(r[error_key]))

    common = set.intersection(*[set(d) for d in per_model.values()]) if models else set()
    items = sorted(common)
    E = np.zeros((len(models), len(items)), dtype=np.int8)
    for i, m in enumerate(models):
        for j, item in enumerate(items):
            E[i, j] = per_model[m][item]
    return E, models, items
