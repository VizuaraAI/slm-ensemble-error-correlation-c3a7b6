"""Entry point: traces in, paper tables out.

Usage:
    python -m src.analyze --config configs/default.yaml
    python -m src.analyze --traces runs/default/traces --out runs/default/analysis

Produces, under the analysis directory:
    table1_main.csv      rows map to Table 1 of the paper
    table2_inherit.csv   rows map to Table 2 (teacher-student inheritance)
    table3_axes.csv      rows map to Table 3 (which recipe axis buys independence)
    table4_channels.csv  rows map to Table 4 (per behavioural channel)
    scaling.csv          accuracy-versus-size curves, the data behind Figure 2
    summary.json         everything above in one machine-readable blob

Every CSV column is named after the column it fills in the paper, so moving a
result into the draft is a copy, not a judgement call.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import load_config
from .correlation import (
    diversity_report,
    error_matrix_from_records,
    error_variance_decomposition,
    group_rho,
    inheritance_report,
)
from .ensemble import accuracy_vs_size, oracle_and_floor, saturation_point
from .traces import read_trace_dir

# Ensemble strategies reported in Table 1. Each predicate selects members
# from the trace pool by role and recipe name.
STRATEGIES = [
    ("Same teacher, seed only",
     lambda m: m["role"] == "student" and m["teacher"] == "T1" and m["objective"] == "hard_label"),
    ("Same teacher, KD varied",
     lambda m: m["role"] == "student" and m["teacher"] == "T1"),
    ("Two teachers, mixed recipes",
     lambda m: m["role"] == "student"),
    ("Independent SLMs (control)",
     lambda m: m["role"] == "control"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse traces into paper tables.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--traces", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--bootstrap", type=int, default=None)
    return parser.parse_args()


def parse_recipe_name(name: str) -> dict[str, str]:
    """Invert Config.recipe_name: 'teacher-T1__objective-seq_kd__...' to a dict."""
    out: dict[str, str] = {}
    for part in name.split("__"):
        if "-" not in part:
            continue
        key, _, value = part.partition("-")
        out[key] = value
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    analysis_cfg = cfg["analysis"]

    trace_dir = Path(args.traces) if args.traces else cfg.output_dir() / "traces"
    out_dir = Path(args.out) if args.out else cfg.output_dir() / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_boot = args.bootstrap if args.bootstrap is not None else int(analysis_cfg["bootstrap_resamples"])

    rows = read_trace_dir(trace_dir)
    if not rows:
        raise SystemExit(f"no traces found in {trace_dir}; run `python -m src.evaluate` first")
    print(f"loaded {len(rows)} trace rows from {trace_dir}")

    # ---- model metadata ----------------------------------------------------
    meta: dict[str, dict] = {}
    for r in rows:
        name = r["model"]
        if name in meta:
            continue
        recipe = parse_recipe_name(r.get("recipe", "-"))
        meta[name] = {
            "role": r.get("model_role", "student"),
            "teacher": recipe.get("teacher", "-"),
            "objective": recipe.get("objective", "-"),
            "data_shard": recipe.get("data_shard", "-"),
            "student_base": recipe.get("student_base", "-"),
        }
    print(f"models: {len(meta)}  "
          f"({sum(m['role'] == 'student' for m in meta.values())} students, "
          f"{sum(m['role'] == 'teacher' for m in meta.values())} teachers, "
          f"{sum(m['role'] == 'control' for m in meta.values())} controls)")

    mcq_rows = [r for r in rows if r["kind"] == "mcq"]
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    # ---- Table 1: main result ---------------------------------------------
    table1: list[dict] = []
    scaling_rows: list[dict] = []
    for label, predicate in STRATEGIES:
        members = sorted(n for n, m in meta.items() if predicate(m))
        if len(members) < 2:
            print(f"  [skip] '{label}': only {len(members)} member(s) available")
            continue
        subset = [r for r in mcq_rows if r["model"] in members]
        E, model_order, item_order = error_matrix_from_records(subset)
        report = diversity_report(E, n_bootstrap=n_boot, ci_level=float(analysis_cfg["ci_level"]), rng=rng)

        by_model = {m: {} for m in model_order}
        gold_by_item: dict[str, str] = {}
        for r in subset:
            by_model[r["model"]][r["item_id"]] = (r["prediction"], float(r["confidence"]))
            gold_by_item[r["item_id"]] = r["gold"]
        P = np.array([[by_model[m][i][0] for i in item_order] for m in model_order], dtype=object)
        C = np.array([[by_model[m][i][1] for i in item_order] for m in model_order], dtype=float)
        gold = np.array([gold_by_item[i] for i in item_order], dtype=object)

        bounds = oracle_and_floor(P, gold)
        points = accuracy_vs_size(
            P, gold, confidences=C,
            sizes=[s for s in analysis_cfg["ensemble_sizes"] if s <= len(model_order)],
            subsets_per_size=int(analysis_cfg["subsets_per_size"]),
            rule="majority", rng=rng,
        )
        vote_acc = max((p.mean_accuracy for p in points), default=0.0)

        table1.append({
            "ensemble_construction": label,
            "n_members": len(model_order),
            "member_acc_pct": round(100 * bounds["mean_member_accuracy"], 1),
            "vote_acc_pct": round(100 * vote_acc, 1),
            "oracle_acc_pct": round(100 * bounds["oracle_accuracy"], 1),
            "rho_bar": round(report.rho_bar, 3),
            "ess": round(report.ess, 2),
            "design_effect": round(report.design_effect, 2),
            "all_wrong_pct": round(100 * bounds["all_wrong"], 1),
            "ess_ci_low": round(report.ess_ci_low, 2),
            "ess_ci_high": round(report.ess_ci_high, 2),
            "empirical_saturation_n": saturation_point(points),
        })
        for p in points:
            scaling_rows.append({
                "ensemble_construction": label,
                "n_members": p.n_members,
                "mean_accuracy_pct": round(100 * p.mean_accuracy, 2),
                "std_accuracy_pct": round(100 * p.std_accuracy, 2),
                "predicted_ess": round(p.predicted_ess, 2),
                "n_subsets": p.n_subsets,
            })

    write_csv(out_dir / "table1_main.csv", table1)
    write_csv(out_dir / "scaling.csv", scaling_rows)

    # ---- Table 2: teacher-student inheritance ------------------------------
    teachers = [n for n, m in meta.items() if m["role"] == "teacher"]
    # Students record their teacher by config key ("T1"); traces record the
    # teacher by label ("qwen7b"). Map one to the other via the config.
    key_to_label = {key: spec["label"] for key, spec in cfg["teachers"].items()}
    table2: list[dict] = []
    if teachers:
        teacher_errs: dict[str, dict[str, int]] = {
            t: {r["item_id"]: int(r["is_error"]) for r in mcq_rows if r["model"] == t} for t in teachers
        }
        by_objective: dict[str, list[str]] = defaultdict(list)
        for name, m in meta.items():
            if m["role"] == "student":
                by_objective[m["objective"]].append(name)
            elif m["role"] == "control":
                by_objective["no_teacher_control"].append(name)

        for objective, members in sorted(by_objective.items()):
            accumulated = []
            for student in members:
                student_err = {r["item_id"]: int(r["is_error"]) for r in mcq_rows if r["model"] == student}
                own_label = key_to_label.get(meta[student]["teacher"])
                teacher_name = own_label if own_label in teacher_errs else teachers[0]
                shared = sorted(set(student_err) & set(teacher_errs[teacher_name]))
                if len(shared) < 20:
                    continue
                t_vec = np.array([teacher_errs[teacher_name][i] for i in shared])
                s_vec = np.array([student_err[i] for i in shared])
                accumulated.append(inheritance_report(t_vec, s_vec))
            if not accumulated:
                continue
            finite_lift = [a.error_lift for a in accumulated if np.isfinite(a.error_lift)]
            table2.append({
                "distillation_objective": objective,
                "n_students": len(accumulated),
                "rho_ts": round(float(np.mean([a.rho_teacher_student for a in accumulated])), 3),
                "p_student_wrong_given_teacher_wrong":
                    round(float(np.mean([a.p_student_wrong_given_teacher_wrong for a in accumulated])), 3),
                "p_student_wrong_given_teacher_right":
                    round(float(np.mean([a.p_student_wrong_given_teacher_right for a in accumulated])), 3),
                "error_lift": round(float(np.mean(finite_lift)), 2) if finite_lift else "",
                "inherited_share_pct":
                    round(100 * float(np.mean([a.inherited_error_share for a in accumulated])), 1),
            })
    write_csv(out_dir / "table2_inherit.csv", table2)

    # ---- Table 3: which recipe axis buys independence ----------------------
    student_names = sorted(n for n, m in meta.items() if meta[n]["role"] == "student")
    table3: list[dict] = []
    if len(student_names) >= 2:
        subset = [r for r in mcq_rows if r["model"] in student_names]
        E, model_order, _ = error_matrix_from_records(subset)
        for axis in ("data_shard", "objective", "student_base", "teacher"):
            groups = [meta[m][axis] for m in model_order]
            if len(set(groups)) < 2:
                continue
            stats = group_rho(E, groups)
            delta = stats["rho_within"] - stats["rho_between"]
            n = len(model_order)
            from .correlation import effective_sample_size

            table3.append({
                "axis_varied": axis,
                "rho_within": round(stats["rho_within"], 3),
                "rho_between": round(stats["rho_between"], 3),
                "delta_rho": round(delta, 3),
                "delta_ess_at_8": round(
                    effective_sample_size(8, stats["rho_between"]) - effective_sample_size(8, stats["rho_within"]), 2
                ),
                "n_within_pairs": stats["n_within_pairs"],
                "n_between_pairs": stats["n_between_pairs"],
                "gpu_hours": "",  # fill in from your own training logs
            })
    write_csv(out_dir / "table3_axes.csv", table3)

    # ---- Table 4: per behavioural channel ----------------------------------
    table4: list[dict] = []
    if len(student_names) >= 2:
        for channel in ("mcq", "refusal", "omission"):
            subset = [r for r in rows if r["kind"] == channel and r["model"] in student_names]
            if not subset:
                continue
            E, model_order, item_order = error_matrix_from_records(subset)
            if E.shape[1] < 5:
                print(f"  [skip] channel '{channel}': only {E.shape[1]} shared items")
                continue
            report = diversity_report(E, n_bootstrap=min(n_boot, 500),
                                      ci_level=float(analysis_cfg["ci_level"]), rng=rng)
            decomposition = error_variance_decomposition(E)
            table4.append({
                "channel": channel,
                "n_items": E.shape[1],
                "rho_bar": round(report.rho_bar, 3),
                "ess": round(report.ess, 2),
                "ess_ci_low": round(report.ess_ci_low, 2),
                "ess_ci_high": round(report.ess_ci_high, 2),
                "covariance_share_pct": round(100 * decomposition["covariance_share"], 1),
                "mean_error_rate": round(report.mean_error_rate, 3),
            })
    write_csv(out_dir / "table4_channels.csv", table4)

    summary = {
        "n_trace_rows": len(rows),
        "n_models": len(meta),
        "table1_main": table1,
        "table2_inherit": table2,
        "table3_axes": table3,
        "table4_channels": table4,
        "scaling": scaling_rows,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[done] analysis in {out_dir}")


if __name__ == "__main__":
    main()
