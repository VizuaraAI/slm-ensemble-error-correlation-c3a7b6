"""Config loading. Thin on purpose: one YAML file, light validation, no magic."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


@dataclass
class Config:
    """Dict-backed config with attribute and key access."""

    data: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getattr__(self, key: str) -> Any:
        # Only reached for names not found as real attributes.
        try:
            return self.data[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def resolved_device(self) -> str:
        requested = self.data.get("device", "auto")
        if requested != "auto":
            return requested
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def output_dir(self) -> Path:
        out = Path(self.data.get("output_dir", "runs/default"))
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.mkdir(parents=True, exist_ok=True)
        return out

    def recipe_name(self, recipe: dict[str, Any]) -> str:
        """Stable, filesystem-safe identifier for one grid cell."""
        return "__".join(
            f"{key}-{recipe[key]}" for key in ("teacher", "objective", "data_shard", "student_base")
        )

    def teacher_model_id(self, teacher_key: str) -> str:
        return self.data["teachers"][teacher_key]["model_id"]

    def student_base_model_id(self, label: str) -> str:
        for base in self.data["student_bases"]:
            if base["label"] == label:
                return base["model_id"]
        raise KeyError(f"unknown student_base label: {label!r}")


REQUIRED_TOP_LEVEL = ("teachers", "student_bases", "probes", "distill", "training", "analysis")


def load_config(path: str | os.PathLike[str] | None = None, **overrides: Any) -> Config:
    """Load YAML config, apply shallow overrides, check required sections."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    data = copy.deepcopy(data)
    for key, value in overrides.items():
        if value is not None:
            data[key] = value

    missing = [key for key in REQUIRED_TOP_LEVEL if key not in data]
    if missing:
        raise ValueError(f"config {cfg_path} is missing sections: {', '.join(missing)}")

    return Config(data=data, path=cfg_path)


def set_global_seed(seed: int) -> None:
    """Seed every generator we might touch. Call this first in any entry point."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
