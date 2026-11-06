#!/usr/bin/env python3
"""Minimal QLoRA supervised fine-tuning shell.

No training data is included. The input is a JSON list of objects with
``prompt`` and ``label`` fields. Every hyper-parameter is supplied on the
command line; this shell encodes no particular recipe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoplan_sql.train.training_utils import limit_train_list, prepare_local_model_dir


def load_samples(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list: {path}")
    samples = []
    for item in raw:
        prompt, label = item.get("prompt"), item.get("label")
        if not prompt or not label:
            continue
        samples.append({"prompt": prompt, "label": label})
    if not samples:
        raise ValueError(f"No valid prompt/label pairs in {path}")
    return samples


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QLoRA SFT shell")
    parser.add_argument("--train_file", required=True, help="JSON list of {prompt, label}")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument(
        "--response_template",
        required=True,
        help="Completion marker used for loss masking",
    )
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--num_train_epochs", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_seq_length", type=int, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True)
    parser.add_argument("--lora_r", type=int, required=True)
    parser.add_argument("--lora_alpha", type=int, required=True)
    parser.add_argument("--lora_dropout", type=float, required=True)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--no_load_in_4bit", action="store_false", dest="load_in_4bit")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate data and print the configuration; do not train",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    samples = load_samples(Path(args.train_file))
    samples = limit_train_list(samples, seed=args.seed)

    print(
        json.dumps(
            {
                "n_samples": len(samples),
                "model": args.model_name_or_path,
                "learning_rate": args.learning_rate,
                "epochs": args.num_train_epochs,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "max_seq_length": args.max_seq_length,
                "response_template": args.response_template,
            },
            indent=2,
        )
    )
    if args.dry_run:
        print("Dry run complete; no training started.")
        return

    # Heavy imports only when actually training.
    from datasets import Dataset
    from loguru import logger
    from transformers import AutoTokenizer
    from trl import DataCollatorForCompletionOnlyLM, ModelConfig, SFTConfig, SFTTrainer

    from geoplan_sql.train.lora_utils import (
        DEFAULT_TARGET_MODULES,
        load_causal_lm,
        resolve_peft_config,
        save_trained_model,
    )

    prepare_local_model_dir(args.model_name_or_path, args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_text(sample: dict) -> str:
        messages = [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": args.response_template + sample["label"]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    kept = []
    for sample in samples:
        text = format_text(sample)
        if len(tokenizer.encode(text)) <= args.max_seq_length:
            kept.append({"text": text})
    if not kept:
        raise ValueError("All samples exceeded max_seq_length")
    logger.info("Training on {}/{} samples after length filter", len(kept), len(samples))
    dataset = Dataset.from_list(kept)

    model_config = ModelConfig(
        model_name_or_path=args.model_name_or_path,
        torch_dtype="bfloat16",
        attn_implementation=args.attn_implementation,
        use_peft=True,
        load_in_4bit=args.load_in_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=DEFAULT_TARGET_MODULES.copy(),
    )
    model = load_causal_lm(
        model_config,
        cache_dir=args.cache_dir or "",
        local_files_only=False,
    )
    peft_config = resolve_peft_config(model_config)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        bf16=args.bf16,
        report_to=[],
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
    )
    collator = DataCollatorForCompletionOnlyLM(
        response_template=args.response_template,
        tokenizer=tokenizer,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        data_collator=collator,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    save_trained_model(
        trainer,
        args.output_dir,
        model_config,
        tokenizer,
        cache_dir=args.cache_dir,
        local_files_only=False,
    )
    logger.info("Saved checkpoint to {}", args.output_dir)


if __name__ == "__main__":
    main()
