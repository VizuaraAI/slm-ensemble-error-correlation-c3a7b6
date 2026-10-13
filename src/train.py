"""Entry point: build the student factory.

Usage:
    python -m src.train --config configs/default.yaml
    python -m src.train --recipe "T1,seq_kd,0,qwen05b"     # one cell only
    python -m src.train --dry-run                           # print the plan

The teacher corpus is built once per teacher and cached, because it is the
most expensive artefact in the pipeline and every student that shares that
teacher reuses it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config, set_global_seed
from .distill import build_teacher_corpus, distill_student, load_corpus, save_corpus, shard_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distil the grid of student SLMs.")
    parser.add_argument("--config", default=None, help="path to YAML config")
    parser.add_argument("--recipe", default=None,
                        help="single recipe as teacher,objective,shard,student_base")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def recipe_from_string(text: str) -> dict:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("--recipe must be 'teacher,objective,shard,student_base'")
    return {
        "teacher": parts[0],
        "objective": parts[1],
        "data_shard": int(parts[2]),
        "student_base": parts[3],
    }


def collect_prompts(cfg) -> list[str]:
    """Prompts the teacher answers to create the distillation corpus.

    We reuse the probe suite's task families but NOT the probe items
    themselves: training on the evaluation items would manufacture exactly
    the correlation this project is trying to measure.
    """
    from .probes import format_mcq

    from datasets import load_dataset

    limit = int(cfg["distill"]["prompts_per_task"])
    prompts: list[str] = []
    for task_cfg in cfg["probes"]["tasks"]:
        if task_cfg["hf_path"] == "builtin" or task_cfg.get("kind") != "mcq":
            continue
        dataset = load_dataset(task_cfg["hf_path"], task_cfg.get("hf_config"), split="train")
        for i, row in enumerate(dataset):
            if i >= limit:
                break
            row = dict(row)
            choices = row.get("choices")
            texts = list(choices["text"]) if isinstance(choices, dict) else list(choices or [])
            if len(texts) < 2:
                continue
            prompts.append(format_mcq(str(row.get("question", "")), [str(t) for t in texts]))
    return prompts


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    set_global_seed(seed)

    recipes = [recipe_from_string(args.recipe)] if args.recipe else list(cfg["distill"]["active_recipes"])
    out_root = cfg.output_dir()
    device = cfg.resolved_device()

    print(f"device={device}  seed={seed}  recipes={len(recipes)}  output={out_root}")
    for recipe in recipes:
        print(f"  plan: {cfg.recipe_name(recipe)}")
    if args.dry_run:
        return

    # ---- teacher corpora, one per distinct teacher -------------------------
    corpus_dir = out_root / "corpora"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    needed_teachers = sorted({r["teacher"] for r in recipes})
    corpora: dict[str, list] = {}
    prompts: list[str] | None = None

    for teacher_key in needed_teachers:
        path = corpus_dir / f"{teacher_key}.jsonl"
        if path.exists():
            print(f"[corpus] reusing {path}")
            corpora[teacher_key] = load_corpus(path)
            continue
        if prompts is None:
            prompts = collect_prompts(cfg)
            print(f"[corpus] {len(prompts)} prompts collected")
        from .traces import ScoredModel

        print(f"[corpus] sampling from teacher {teacher_key}")
        teacher = ScoredModel(cfg.teacher_model_id(teacher_key), device=device, dtype=cfg.get("dtype", "bfloat16"))
        corpus = build_teacher_corpus(
            teacher,
            prompts,
            temperature=float(cfg["distill"]["teacher_temperature"]),
            max_new_tokens=int(cfg["distill"]["max_new_tokens"]),
        )
        save_corpus(corpus, path)
        corpora[teacher_key] = corpus
        del teacher

    # ---- one student per recipe -------------------------------------------
    manifest = []
    for recipe in recipes:
        name = cfg.recipe_name(recipe)
        target = out_root / "students" / name
        if (target / "recipe.json").exists():
            print(f"[skip] {name} already trained")
            manifest.append({"recipe": recipe, "path": str(target)})
            continue

        shard = shard_examples(corpora[recipe["teacher"]], int(recipe["data_shard"]))
        print(f"[train] {name}  ({len(shard)} examples)")
        distill_student(
            student_model_id=cfg.student_base_model_id(recipe["student_base"]),
            corpus=shard,
            objective=recipe["objective"],
            training_cfg=cfg["training"],
            device=device,
            dtype=cfg.get("dtype", "bfloat16"),
            teacher_model_id=cfg.teacher_model_id(recipe["teacher"]),
            output_dir=target,
            # Seed varies with the shard so that "seed only" is a real axis.
            seed=seed + int(recipe["data_shard"]),
        )
        manifest.append({"recipe": recipe, "path": str(target)})

    with open(out_root / "students.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[done] {len(manifest)} students -> {out_root / 'students.json'}")


if __name__ == "__main__":
    main()
