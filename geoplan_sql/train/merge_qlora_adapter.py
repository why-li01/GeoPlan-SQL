#!/usr/bin/env python3
"""Merge a QLoRA adapter into a bf16 checkpoint (subprocess entrypoint)."""

from __future__ import annotations

import argparse
from pathlib import Path

from lora_utils import (
    _dtype_config_name,
    _pick_merge_device,
    _strip_quantization_config,
    merge_lora_device,
    resolve_merge_attn_implementation,
)


def merge_qlora_adapter(
    *,
    adapter_path: Path,
    base_path: str,
    output_path: Path,
    cache_dir: str | None,
    local_files_only: bool,
    attn_implementation: str | None,
    torch_dtype: str,
    tokenizer_path: str | None,
) -> None:
    import torch
    from loguru import logger
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merge_device = _pick_merge_device(merge_lora_device())
    logger.info("Using {} for QLoRA merge", merge_device)

    dtype = getattr(torch, torch_dtype.replace("torch.", ""), torch.bfloat16)
    attn_impl = resolve_merge_attn_implementation(attn_implementation, merge_device)
    load_kwargs: dict = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": local_files_only,
        "device_map": merge_device if merge_device == "cpu" else "auto",
        "attn_implementation": attn_impl,
    }
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    logger.info("Loading bf16 base model for merge: {}", base_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_path, **load_kwargs)

    tokenizer = None
    if tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

    logger.info("Merging QLoRA adapter into bf16 base and saving to {}", output_path)
    merged_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        ensure_weight_tying=True,
    )
    merged_model = merged_model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True, max_shard_size="4GB")
    if tokenizer is not None:
        tokenizer.save_pretrained(output_path)
    _strip_quantization_config(output_path, dtype)
    logger.info("Saved bf16 merged checkpoint to {}", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a QLoRA adapter into a bf16 base model.")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--base_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    merge_qlora_adapter(
        adapter_path=Path(args.adapter_path),
        base_path=args.base_path,
        output_path=Path(args.output_path),
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
        torch_dtype=_dtype_config_name(args.torch_dtype),
        tokenizer_path=args.tokenizer_path,
    )


if __name__ == "__main__":
    main()
