"""Shared LoRA / QLoRA helpers.

Requires optional deps: torch, transformers, trl, peft, loguru.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

_MODULES_TO_SAVE_DISABLED = frozenset({"none", "0", "false", "skip", "empty"})

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from trl import ModelConfig


def modules_to_save_disabled() -> bool:
    value = os.environ.get("LORA_MODULES_TO_SAVE", "").strip().lower()
    return value in _MODULES_TO_SAVE_DISABLED


def merge_lora_on_save_enabled() -> bool:
    value = os.environ.get("MERGE_LORA_ON_SAVE", "1").lower()
    return value in {"1", "true", "yes"}


def merge_lora_device() -> str:
    value = os.environ.get("MERGE_LORA_DEVICE", "auto").lower()
    return value if value in {"cpu", "cuda", "auto"} else "auto"


_CPU_INCOMPATIBLE_ATTN = frozenset({"flash_attention_2", "flash_attn", "fa2"})


def resolve_merge_attn_implementation(attn_implementation: str | None, merge_device: str) -> str:
    if merge_device == "cpu" and attn_implementation in _CPU_INCOMPATIBLE_ATTN:
        return "sdpa"
    return attn_implementation or "sdpa"


def load_causal_lm(model_config, *, cache_dir: str, local_files_only: bool = True):
    import torch
    from loguru import logger
    from transformers import AutoModelForCausalLM
    from trl.trainer.utils import get_kbit_device_map, get_quantization_config

    load_kwargs: dict = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "attn_implementation": model_config.attn_implementation,
        "low_cpu_mem_usage": True,
    }
    quantization_config = get_quantization_config(model_config)
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
        load_kwargs["device_map"] = get_kbit_device_map()
        logger.info(
            "Loading model with {}-bit quantization",
            4 if model_config.load_in_4bit else 8,
        )
    else:
        load_kwargs["torch_dtype"] = model_config.torch_dtype
    return AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **load_kwargs)


def _dtype_config_name(torch_dtype) -> str:
    if isinstance(torch_dtype, str):
        return torch_dtype.replace("torch.", "")
    if hasattr(torch_dtype, "name"):
        return torch_dtype.name
    return str(torch_dtype).replace("torch.", "")


def _strip_quantization_config(storage_path: Path, torch_dtype) -> None:
    config_path = storage_path / "config.json"
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("quantization_config", None)
    config["torch_dtype"] = _dtype_config_name(torch_dtype)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _pick_merge_device(preference: str, *, min_free_gib: float = 20.0) -> str:
    import torch
    from loguru import logger

    if preference == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    if preference == "cuda":
        return "cuda"
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes >= int(min_free_gib * 1024**3):
        return "cuda"
    logger.warning(
        "Only {:.2f} GiB GPU free; merging on CPU (set MERGE_LORA_DEVICE=cuda to force)",
        free_bytes / 1024**3,
    )
    return "cpu"


def _release_trainer_gpu_memory(trainer) -> None:
    import torch

    model = getattr(trainer, "model", None)
    if model is not None:
        trainer.model = None
        del model
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None:
        if hasattr(accelerator, "free_memory"):
            accelerator.free_memory()
        trainer.accelerator = None
    for attr in ("optimizer", "lr_scheduler", "model_wrapped", "deepspeed"):
        if getattr(trainer, attr, None) is not None:
            setattr(trainer, attr, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_qlora_training(model_config) -> bool:
    from trl.trainer.utils import get_quantization_config

    return model_config.use_peft and get_quantization_config(model_config) is not None


def resolve_peft_config(model_config, *, modules_to_save: list[str] | None = None):
    from loguru import logger
    from trl.trainer.utils import get_peft_config

    if not model_config.use_peft:
        logger.info("PEFT disabled; using full-parameter fine-tuning.")
        return None
    if model_config.lora_target_modules is None:
        model_config.lora_target_modules = DEFAULT_TARGET_MODULES.copy()
    if (
        modules_to_save
        and model_config.lora_modules_to_save is None
        and not modules_to_save_disabled()
    ):
        model_config.lora_modules_to_save = modules_to_save
    peft_config = get_peft_config(model_config)
    logger.info(
        "Using LoRA: r={}, alpha={}, dropout={}, modules_to_save={}",
        model_config.lora_r,
        model_config.lora_alpha,
        model_config.lora_dropout,
        model_config.lora_modules_to_save,
    )
    return peft_config


def _invoke_qlora_merge_subprocess(
    *,
    adapter_path: Path,
    storage_path: Path,
    model_config,
    tokenizer,
    cache_dir: str | None,
    local_files_only: bool,
    merge_device: str,
) -> None:
    from loguru import logger

    tokenizer_path = None
    if tokenizer is not None:
        tokenizer_path = str(adapter_path / "tokenizer")
        tokenizer.save_pretrained(tokenizer_path)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "merge_qlora_adapter.py"),
        "--adapter_path",
        str(adapter_path),
        "--base_path",
        model_config.model_name_or_path,
        "--output_path",
        str(storage_path),
        "--torch_dtype",
        _dtype_config_name(model_config.torch_dtype),
    ]
    if cache_dir is not None:
        cmd.extend(["--cache_dir", cache_dir])
    if tokenizer_path is not None:
        cmd.extend(["--tokenizer_path", tokenizer_path])
    if model_config.attn_implementation:
        cmd.extend(["--attn_implementation", model_config.attn_implementation])
    if local_files_only:
        cmd.append("--local_files_only")
    else:
        cmd.append("--no-local_files_only")

    env = os.environ.copy()
    if merge_device == "cpu":
        env["MERGE_LORA_DEVICE"] = "cpu"
        env["CUDA_VISIBLE_DEVICES"] = ""
    logger.info("Spawning QLoRA merge subprocess (merge_device={})", merge_device)
    subprocess.run(cmd, check=True, env=env)


def _save_qlora_merged_bf16(
    trainer,
    storage_path: Path,
    model_config,
    tokenizer,
    *,
    cache_dir: str | None,
    local_files_only: bool,
) -> None:
    from loguru import logger

    adapter_tmp = Path(tempfile.mkdtemp(prefix="qlora_adapter_"))
    try:
        logger.info("Saving QLoRA adapter to temporary directory: {}", adapter_tmp)
        trainer.model.save_pretrained(adapter_tmp)
        _release_trainer_gpu_memory(trainer)
        del trainer
        merge_device = _pick_merge_device(merge_lora_device())
        _invoke_qlora_merge_subprocess(
            adapter_path=adapter_tmp,
            storage_path=storage_path,
            model_config=model_config,
            tokenizer=tokenizer,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            merge_device=merge_device,
        )
    finally:
        shutil.rmtree(adapter_tmp, ignore_errors=True)


def save_trained_model(
    trainer,
    storage_path: str | Path,
    model_config,
    tokenizer=None,
    *,
    cache_dir: str | None = None,
    local_files_only: bool = True,
) -> None:
    from loguru import logger

    storage_path = Path(storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)

    if not model_config.use_peft or not merge_lora_on_save_enabled():
        trainer.save_model(storage_path)
        if tokenizer is not None:
            tokenizer.save_pretrained(storage_path)
        return

    if is_qlora_training(model_config):
        _save_qlora_merged_bf16(
            trainer,
            storage_path,
            model_config,
            tokenizer,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        return

    model = trainer.model
    if hasattr(model, "merge_and_unload"):
        logger.info("Merging LoRA weights into base model before save: {}", storage_path)
        model = model.merge_and_unload()
        trainer.model = model
    trainer.save_model(storage_path)
    if tokenizer is not None:
        tokenizer.save_pretrained(storage_path)
