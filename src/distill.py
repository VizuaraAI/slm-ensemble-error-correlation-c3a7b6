"""Distil one student under one recipe.

Three objectives are supported, chosen because they differ in how much of the
teacher's distribution the student is allowed to see. That difference is the
whole experiment: if error correlation is driven by the distillation signal,
these three should separate cleanly.

  hard_label   plain supervised fine-tuning on the teacher's sampled text.
               The student sees one sequence and nothing about the teacher's
               uncertainty.
  seq_kd       sequence-level KD: the same text, plus a forward-KL term on the
               teacher's token distribution at each position (Kim and Rush
               style, applied token-wise).
  reverse_kl   mode-seeking reverse KL on the student's own samples, in the
               spirit of MiniLLM. Implemented here on teacher-sampled text for
               tractability on a single GPU, which is a documented
               simplification, not an oversight.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

OBJECTIVES = ("hard_label", "seq_kd", "reverse_kl")


@dataclass
class DistillExample:
    prompt: str
    teacher_text: str
    task: str


def shard_examples(examples: Sequence[DistillExample], shard: int, n_shards: int = 3) -> list[DistillExample]:
    """Deterministic disjoint shards. Same shard index always gives the same data."""
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard must be in [0, {n_shards})")
    return [ex for i, ex in enumerate(examples) if i % n_shards == shard]


def build_teacher_corpus(
    teacher,
    prompts: Sequence[str],
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    tasks: Sequence[str] | None = None,
) -> list[DistillExample]:
    """Sample one teacher response per prompt. This is the distillation corpus.

    Cache the result: it is the most expensive artefact in the pipeline and it
    is reused by every student that shares this teacher.
    """
    tasks = list(tasks) if tasks is not None else ["generic"] * len(prompts)
    corpus: list[DistillExample] = []
    for prompt, task in zip(prompts, tasks):
        text = teacher.generate(prompt, max_new_tokens=max_new_tokens)
        corpus.append(DistillExample(prompt=prompt, teacher_text=text, task=task))
    return corpus


def save_corpus(corpus: Sequence[DistillExample], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for ex in corpus:
            handle.write(json.dumps({"prompt": ex.prompt, "teacher_text": ex.teacher_text, "task": ex.task}) + "\n")
    return path


def load_corpus(path: str | Path) -> list[DistillExample]:
    rows: list[DistillExample] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(DistillExample(row["prompt"], row["teacher_text"], row.get("task", "generic")))
    return rows


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def kd_loss(student_logits, teacher_logits, labels, temperature: float = 2.0, direction: str = "forward"):
    """Token-wise KD loss, masked to the response tokens.

    direction="forward"  KL(teacher || student), mean seeking, the classic
                         Hinton formulation.
    direction="reverse"  KL(student || teacher), mode seeking, which makes the
                         student commit to one of the teacher's modes. The
                         hypothesis is that this commitment is what drives
                         students towards identical failures.
    """
    import torch
    import torch.nn.functional as F

    mask = (labels != -100).unsqueeze(-1)
    s = student_logits / temperature
    t = teacher_logits / temperature
    log_p_s = F.log_softmax(s, dim=-1)
    log_p_t = F.log_softmax(t, dim=-1)

    if direction == "forward":
        per_token = (log_p_t.exp() * (log_p_t - log_p_s)).sum(dim=-1)
    elif direction == "reverse":
        per_token = (log_p_s.exp() * (log_p_s - log_p_t)).sum(dim=-1)
    else:
        raise ValueError(f"direction must be 'forward' or 'reverse', got {direction!r}")

    valid = mask.squeeze(-1).float()
    denom = valid.sum().clamp(min=1.0)
    return (per_token * valid).sum() / denom * (temperature**2)


def build_lora_model(model, lora_cfg: dict):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
        task_type="CAUSAL_LM",
        bias="none",
    )
    return get_peft_model(model, config)


def tokenize_example(tokenizer, example: DistillExample, max_seq_len: int = 1024):
    """Prompt tokens are masked out of the loss; only the response is learned."""
    prompt_ids = tokenizer(example.prompt, add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(example.teacher_text, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_seq_len]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_seq_len]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def collate(batch, pad_token_id: int):
    import torch

    width = max(len(row["input_ids"]) for row in batch)
    input_ids, labels, attention = [], [], []
    for row in batch:
        pad = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_token_id] * pad)
        labels.append(row["labels"] + [-100] * pad)
        attention.append(row["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }


def distill_student(
    student_model_id: str,
    corpus: Sequence[DistillExample],
    objective: str,
    training_cfg: dict,
    device: str = "cuda",
    dtype: str = "bfloat16",
    teacher_model_id: str | None = None,
    output_dir: str | Path = "runs/student",
    seed: int = 0,
) -> Path:
    """Train one student and save its adapter (or full weights) to output_dir.

    The teacher is only loaded for the two objectives that need its logits,
    which keeps hard_label runs cheap enough to use as the baseline arm.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")

    torch.manual_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_dtype = getattr(torch, dtype, torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(student_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    student = AutoModelForCausalLM.from_pretrained(student_model_id, torch_dtype=torch_dtype).to(device)
    if training_cfg.get("lora", {}).get("enabled", True):
        student = build_lora_model(student, training_cfg["lora"])
    student.train()

    teacher = None
    if objective in ("seq_kd", "reverse_kl"):
        if teacher_model_id is None:
            raise ValueError(f"objective {objective!r} needs teacher_model_id for its logits")
        teacher = AutoModelForCausalLM.from_pretrained(teacher_model_id, torch_dtype=torch_dtype).to(device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)

    max_seq_len = int(training_cfg.get("max_seq_len", 1024))
    tokenized = [tokenize_example(tokenizer, ex, max_seq_len) for ex in corpus]
    batch_size = int(training_cfg.get("per_device_batch_size", 4))
    loader = DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    accum = int(training_cfg.get("grad_accum_steps", 8))
    epochs = int(training_cfg.get("epochs", 1))
    total_steps = max(1, (len(loader) // accum) * epochs)
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=float(training_cfg.get("lr", 1e-4)),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(training_cfg.get("warmup_ratio", 0.03))),
        num_training_steps=total_steps,
    )

    alpha = float(training_cfg.get("kd_alpha", 0.5))
    temperature = float(training_cfg.get("kd_temperature", 2.0))
    direction = "reverse" if objective == "reverse_kl" else "forward"

    step = 0
    for _ in range(epochs):
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = student(**batch)
            loss = output.loss
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(
                        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                    ).logits
                distil = kd_loss(output.logits, teacher_logits, batch["labels"], temperature, direction)
                loss = (1.0 - alpha) * loss + alpha * distil
            (loss / accum).backward()
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    print(f"step {step}/{total_steps}  loss {float(loss):.4f}")

    student.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(output_dir / "recipe.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "student_model_id": student_model_id,
                "teacher_model_id": teacher_model_id,
                "objective": objective,
                "seed": seed,
                "n_examples": len(corpus),
            },
            handle,
            indent=2,
        )
    return output_dir
