"""End-to-end smoke test with synthetic traces. No GPU, no model downloads.

Run this first, before you touch a real model:

    python scripts/smoke_test.py

It fabricates traces for a teacher, several correlated students and an
uncorrelated control arm, runs the real analysis code over them, and checks
that the pipeline recovers the structure it was given. If this passes, every
piece of the analysis works and the only remaining variable is your models.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.correlation import (  # noqa: E402
    diversity_report,
    effective_sample_size,
    error_matrix_from_records,
    inheritance_report,
    phi,
)
from src.ensemble import accuracy_vs_size, majority_vote, oracle_and_floor  # noqa: E402

N_ITEMS = 400
CHOICES = ["A", "B", "C", "D"]


def synth(rng, n_models, teacher_err, inherit, base_rate=0.35):
    """Students whose errors overlap the teacher's with probability `inherit`."""
    E = np.zeros((n_models, N_ITEMS), dtype=np.int8)
    for i in range(n_models):
        own = rng.random(N_ITEMS) < base_rate
        follow = rng.random(N_ITEMS) < inherit
        E[i] = np.where(follow, teacher_err, own).astype(np.int8)
    return E


def to_records(E, names, gold, rng):
    rows = []
    for i, name in enumerate(names):
        for j in range(E.shape[1]):
            correct = gold[j]
            if E[i, j]:
                wrong = [c for c in CHOICES if c != correct]
                prediction = wrong[int(rng.integers(0, len(wrong)))]
            else:
                prediction = correct
            rows.append({
                "model": name, "model_role": "student", "recipe": "-",
                "item_id": f"item{j:04d}", "task": "synthetic", "kind": "mcq",
                "prediction": prediction, "gold": correct,
                "is_error": int(E[i, j]), "confidence": 0.5,
                "refused": 0, "omitted": 0, "n_chars": 0,
            })
    return rows


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}{(' :: ' + detail) if detail else ''}")
    return bool(condition)


def main() -> int:
    rng = np.random.default_rng(0)
    gold = np.array([CHOICES[int(k)] for k in rng.integers(0, 4, N_ITEMS)], dtype=object)
    teacher_err = (rng.random(N_ITEMS) < 0.30).astype(np.int8)

    ok = True
    print("1. estimators on known inputs")
    a = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    ok &= check("phi(x, x) == 1", abs(phi(a, a) - 1.0) < 1e-9)
    ok &= check("phi(x, 1-x) == -1", abs(phi(a, 1 - a) + 1.0) < 1e-9)
    ok &= check("ESS == N when rho == 0", abs(effective_sample_size(8, 0.0) - 8.0) < 1e-9)
    ok &= check("ESS == 1 when rho == 1", abs(effective_sample_size(8, 1.0) - 1.0) < 1e-9)
    ok &= check("ESS shrinks as rho grows",
                effective_sample_size(8, 0.6) < effective_sample_size(8, 0.2))

    print("2. correlated students versus an independent control arm")
    E_same = synth(rng, 8, teacher_err, inherit=0.75)
    E_ctrl = synth(rng, 8, teacher_err, inherit=0.00)
    r_same = diversity_report(E_same, n_bootstrap=200, rng=rng)
    r_ctrl = diversity_report(E_ctrl, n_bootstrap=200, rng=rng)
    print(f"      same-teacher : rho={r_same.rho_bar:.3f}  ESS={r_same.ess:.2f}")
    print(f"      control      : rho={r_ctrl.rho_bar:.3f}  ESS={r_ctrl.ess:.2f}")
    ok &= check("same-teacher ensemble has higher rho", r_same.rho_bar > r_ctrl.rho_bar)
    ok &= check("same-teacher ESS is far below nominal 8", r_same.ess < 4.0,
                f"ESS={r_same.ess:.2f}")
    ok &= check("control ESS is near nominal 8", r_ctrl.ess > 6.0, f"ESS={r_ctrl.ess:.2f}")
    ok &= check("bootstrap CI brackets the point estimate",
                r_same.ess_ci_low <= r_same.ess <= r_same.ess_ci_high)

    print("3. inheritance report recovers the planted signal")
    high = inheritance_report(teacher_err, E_same[0])
    low = inheritance_report(teacher_err, E_ctrl[0])
    print(f"      lift high={high.error_lift:.2f}  low={low.error_lift:.2f}")
    ok &= check("error lift higher for the inheriting student",
                high.error_lift > low.error_lift)
    ok &= check("teacher-student rho higher for the inheriting student",
                high.rho_teacher_student > low.rho_teacher_student)

    print("4. voting and scaling curves")
    names = [f"m{i}" for i in range(8)]
    records = to_records(E_ctrl, names, gold, rng)
    E_round, order, items = error_matrix_from_records(records)
    ok &= check("pivot round-trips the error matrix", E_round.shape == (8, N_ITEMS))
    by_model = {m: {} for m in order}
    for r in records:
        by_model[r["model"]][r["item_id"]] = r["prediction"]
    P = np.array([[by_model[m][i] for i in items] for m in order], dtype=object)
    g = np.array([gold[int(i[4:])] for i in items], dtype=object)
    voted = majority_vote(P)
    bounds = oracle_and_floor(P, g)
    vote_acc = float(np.mean(voted == g))
    print(f"      member={bounds['mean_member_accuracy']:.3f} "
          f"vote={vote_acc:.3f} oracle={bounds['oracle_accuracy']:.3f}")
    ok &= check("voting beats the average member", vote_acc > bounds["mean_member_accuracy"])
    ok &= check("oracle is at least the vote", bounds["oracle_accuracy"] >= vote_acc)
    points = accuracy_vs_size(P, g, sizes=[1, 2, 4, 8], subsets_per_size=40, rng=rng)
    ok &= check("accuracy grows with members for independent models",
                points[-1].mean_accuracy > points[0].mean_accuracy)

    print("5. config loads and the CLI entry points import")
    from src.config import load_config

    cfg = load_config()
    ok &= check("default config parses", "teachers" in cfg.data)
    ok &= check("recipe names are filesystem safe",
                "/" not in cfg.recipe_name(cfg["distill"]["active_recipes"][0]))
    for module in ("src.train", "src.evaluate", "src.analyze", "src.plots", "src.probes", "src.distill"):
        result = subprocess.run([sys.executable, "-c", f"import {module}"],
                                cwd=REPO_ROOT, capture_output=True, text=True)
        ok &= check(f"import {module}", result.returncode == 0, result.stderr.strip()[:160])

    print()
    print("SMOKE TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
