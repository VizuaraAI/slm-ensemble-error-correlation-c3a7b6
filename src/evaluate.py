"""Entry point: collect behavioural traces for every ensemble member.

Usage:
    python -m src.evaluate --config configs/default.yaml
    python -m src.evaluate --only teachers
    python -m src.evaluate --only students --limit 50     # quick smoke test

One JSONL trace file per model lands in <output_dir>/traces/. Traces are the
only thing the analysis reads, so once a model is traced you never need to
load it again.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, set_global_seed
from .probes import load_probe_suite
from .traces import ScoredModel, collect_traces, write_traces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect behavioural traces.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--only", choices=["teachers", "students", "controls", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="cap probe items (smoke test)")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_global_seed(int(cfg.get("seed", 0)))

    device = cfg.resolved_device()
    dtype = cfg.get("dtype", "bfloat16")
    out_root = cfg.output_dir()
    trace_dir = out_root / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    probes = load_probe_suite(cfg)
    if args.limit:
        probes = probes[: args.limit]
    print(f"probe suite: {len(probes)} items across "
          f"{len({p.task for p in probes})} tasks  (device={device})")

    targets: list[tuple[str, str, str, str, str | None]] = []  # name, role, recipe, model_id, adapter

    if args.only in ("teachers", "all"):
        for key, spec in cfg["teachers"].items():
            targets.append((spec["label"], "teacher", "-", spec["model_id"], None))

    if args.only in ("controls", "all"):
        for spec in cfg.get("control_models", []):
            targets.append((spec["label"], "control", "-", spec["model_id"], None))

    if args.only in ("students", "all"):
        students_dir = out_root / "students"
        if not students_dir.exists():
            print("[warn] no students found; run `python -m src.train` first")
        else:
            for path in sorted(students_dir.iterdir()):
                if not (path / "recipe.json").exists():
                    continue
                import json

                recipe = json.loads((path / "recipe.json").read_text())
                targets.append((path.name, "student", path.name, recipe["student_model_id"], str(path)))

    if not targets:
        print("[warn] nothing to evaluate")
        return

    for name, role, recipe, model_id, adapter in targets:
        destination = trace_dir / f"{name}.jsonl"
        if destination.exists() and not args.overwrite:
            print(f"[skip] {name} already traced ({destination})")
            continue
        print(f"[trace] {name}  role={role}  model={model_id}")
        model = ScoredModel(model_id, device=device, dtype=dtype, adapter_path=adapter)
        rows = collect_traces(model, probes, model_name=name, model_role=role, recipe=recipe)
        write_traces(rows, destination)
        errors = sum(r.is_error for r in rows)
        print(f"         {len(rows)} rows, error rate {errors / max(len(rows), 1):.3f}")
        del model

    print(f"[done] traces in {trace_dir}")


if __name__ == "__main__":
    main()
