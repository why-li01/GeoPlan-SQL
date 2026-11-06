"""Small helpers for the SFT training shell."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DEFAULT_DEBUG_TRAIN_SAMPLES = 8


def setup_training_imports() -> None:
    root = str(Path(__file__).resolve().parent)
    if root not in sys.path:
        sys.path.insert(0, root)


def resolve_max_train_samples() -> int | None:
    raw = os.environ.get("MAX_TRAIN_SAMPLES", "").strip()
    if raw:
        if raw.lower() in {"0", "none", "false", "no"}:
            return None
        return max(1, int(raw))
    debug = os.environ.get("TRAIN_DEBUG", "").strip().lower()
    if debug in {"1", "true", "yes"}:
        return DEFAULT_DEBUG_TRAIN_SAMPLES
    return None


def limit_train_list(samples: list, *, seed: int) -> list:
    limit = resolve_max_train_samples()
    if limit is None or len(samples) <= limit:
        return samples
    import random

    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    return [samples[i] for i in idx[:limit]]


def resolve_local_model_dir(model_name_or_path: str, cache_dir: str | None = None) -> Path | None:
    candidates = [Path(model_name_or_path)]
    if cache_dir:
        candidates.append(Path(cache_dir) / Path(model_name_or_path).name)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def ensure_model_shard_symlinks(model_dir: Path) -> list[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return []
    weight_map = json.loads(index_path.read_text()).get("weight_map", {})
    created: list[Path] = []
    for shard_name in set(weight_map.values()):
        expected = model_dir / shard_name
        if expected.exists():
            continue
        matches = sorted(model_dir.glob(f"{expected.stem}_*{expected.suffix}"))
        if not matches:
            continue
        expected.symlink_to(matches[0].name)
        created.append(expected)
    return created


def prepare_local_model_dir(model_name_or_path: str, cache_dir: str | None = None) -> None:
    model_dir = resolve_local_model_dir(model_name_or_path, cache_dir)
    if model_dir is not None:
        ensure_model_shard_symlinks(model_dir)
