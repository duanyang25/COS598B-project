#!/usr/bin/env python3

# ============================================================
# Full-parameter SFT for Qwen/Qwen3.5-0.8B-Base
#
# Train data:
#   dataset/sft_by_label/sat/train.jsonl
#   dataset/sft_by_label/unsat/train.jsonl
#
# Output:
#   results/sft_qwen35_08b_base_sat_unsat_full/
#
# Supports automatic resumption from latest checkpoint.
# Compatible with transformers versions where Trainer(tokenizer=...)
# is removed.
# ============================================================

import os
import csv
import json
import time
import random
import inspect
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint


# ============================================================
# JSONL loading
# ============================================================

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {e}") from e

    return rows


# ============================================================
# Message formatting
# ============================================================

def split_messages(row: Dict[str, Any]):
    messages = row.get("messages")

    if not isinstance(messages, list):
        raise ValueError(f"Row has no valid messages field. Keys: {list(row.keys())}")

    prompt_messages = []
    assistant_text = None

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "assistant":
            assistant_text = content
        else:
            prompt_messages.append({
                "role": role,
                "content": content,
            })

    if assistant_text is None:
        raise ValueError("No assistant message found in row.")

    if len(prompt_messages) == 0:
        raise ValueError("No prompt messages found before assistant message.")

    return prompt_messages, assistant_text


def format_prompt_fallback(messages: List[Dict[str, str]]) -> str:
    parts = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            parts.append(f"### System:\n{content}")
        elif role == "user":
            parts.append(f"### User:\n{content}")
        else:
            parts.append(f"### {role.capitalize()}:\n{content}")

    parts.append("### Assistant:\n")
    return "\n\n".join(parts)


def build_prompt(tokenizer, prompt_messages: List[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return format_prompt_fallback(prompt_messages)


# ============================================================
# Dataset
# ============================================================

class ChatSFTDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        prompt_messages, assistant_text = split_messages(row)
        prompt_text = build_prompt(self.tokenizer, prompt_messages)

        assistant_text_with_eos = assistant_text + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        assistant_ids = self.tokenizer(
            assistant_text_with_eos,
            add_special_tokens=False,
        )["input_ids"]

        # Ensure at least one assistant token remains trainable.
        available_for_assistant = self.max_length - len(prompt_ids)

        if available_for_assistant <= 0:
            prompt_ids = prompt_ids[-(self.max_length - 1):]
            assistant_ids = assistant_ids[:1]
        else:
            assistant_ids = assistant_ids[:available_for_assistant]

        input_ids = prompt_ids + assistant_ids
        labels = [-100] * len(prompt_ids) + assistant_ids.copy()
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: Any
    label_pad_token_id: int = -100

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        pad_id = self.tokenizer.pad_token_id

        input_ids = []
        attention_mask = []
        labels = []

        for x in features:
            cur_len = len(x["input_ids"])
            pad_len = max_len - cur_len

            input_ids.append(x["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(x["attention_mask"] + [0] * pad_len)
            labels.append(x["labels"] + [self.label_pad_token_id] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ============================================================
# Progress callback
# ============================================================

class ProgressPrinterCallback(TrainerCallback):
    """
    Prints useful progress information during training:
    global step, epoch, loss, LR, elapsed time, ETA, and checkpoint saves.
    """

    def __init__(self, total_steps: Optional[int] = None):
        self.total_steps = total_steps
        self.start_time = None
        self.last_log_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_log_time = self.start_time

        print("\n" + "=" * 80)
        print("Training started")
        print(f"Total optimization steps: {state.max_steps}")
        print(f"Epochs: {args.num_train_epochs}")
        print(f"Per-device train batch size: {args.per_device_train_batch_size}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Effective batch size per GPU: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        print("=" * 80 + "\n", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}

        now = time.time()
        elapsed = now - self.start_time if self.start_time else 0.0

        max_steps = state.max_steps or self.total_steps or 0
        step = state.global_step

        if step > 0 and max_steps > 0:
            progress = step / max_steps
            eta = elapsed * (1 - progress) / progress
        else:
            progress = 0.0
            eta = None

        loss = logs.get("loss")
        grad_norm = logs.get("grad_norm")
        lr = logs.get("learning_rate")
        epoch = logs.get("epoch", state.epoch)

        msg = [
            f"[progress] step={step}/{max_steps}",
            f"epoch={epoch:.4f}" if isinstance(epoch, float) else f"epoch={epoch}",
            f"progress={progress * 100:.2f}%",
        ]

        if loss is not None:
            msg.append(f"loss={loss:.6f}")

        if grad_norm is not None:
            msg.append(f"grad_norm={grad_norm:.4f}")

        if lr is not None:
            msg.append(f"lr={lr:.3e}")

        msg.append(f"elapsed={elapsed / 60:.2f} min")

        if eta is not None:
            msg.append(f"eta={eta / 60:.2f} min")

        print(" | ".join(msg), flush=True)
        self.last_log_time = now

    def on_save(self, args, state, control, **kwargs):
        print(f"[checkpoint] saved checkpoint at global_step={state.global_step}", flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        print("\n" + "=" * 80)
        print("Training finished")
        print(f"Final global step: {state.global_step}")
        print(f"Total elapsed time: {elapsed / 60:.2f} min")
        print("=" * 80 + "\n", flush=True)


# ============================================================
# Version-compatible helpers
# ============================================================

def choose_dtype():
    if not torch.cuda.is_available():
        return torch.float32

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def make_training_args(**kwargs):
    valid_args = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    filtered = {}
    dropped = {}

    for key, value in kwargs.items():
        if key in valid_args:
            filtered[key] = value
        else:
            dropped[key] = value

    if dropped:
        print("\nDropped unsupported TrainingArguments keys:")
        for key in dropped:
            print(f"  - {key}")

    return TrainingArguments(**filtered)


def make_trainer(model, args, train_dataset, data_collator, tokenizer, callbacks):
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
        "callbacks": callbacks,
    }

    valid_trainer_args = set(inspect.signature(Trainer.__init__).parameters.keys())

    if "processing_class" in valid_trainer_args:
        trainer_kwargs["processing_class"] = tokenizer
        print("Trainer will use processing_class=tokenizer")
    elif "tokenizer" in valid_trainer_args:
        trainer_kwargs["tokenizer"] = tokenizer
        print("Trainer will use tokenizer=tokenizer")
    else:
        print("Trainer supports neither tokenizer nor processing_class. Omitting tokenizer argument.")

    filtered_kwargs = {
        key: value for key, value in trainer_kwargs.items()
        if key in valid_trainer_args
    }

    dropped = set(trainer_kwargs.keys()) - set(filtered_kwargs.keys())
    if dropped:
        print("Dropped unsupported Trainer keys:")
        for key in dropped:
            print(f"  - {key}")

    return Trainer(**filtered_kwargs)


# ============================================================
# Metrics saving
# ============================================================

def append_training_run_record(output_dir: Path, record: Dict[str, Any]):
    path = output_dir / "training_runs.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "model_id",
        "sat_train_rows",
        "unsat_train_rows",
        "total_train_rows",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "max_length",
        "last_checkpoint",
        "final_train_loss",
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
    ]

    exists = path.exists()

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(record)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--sat_train_path", type=Path, default=Path("dataset/sft_by_label/sat/train.jsonl"))
    parser.add_argument("--unsat_train_path", type=Path, default=Path("dataset/sft_by_label/unsat/train.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/sft_qwen35_08b_base_sat_unsat_full"))

    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--limit_per_label", type=int, default=None)

    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    parser.add_argument("--max_length", type=int, default=4096)

    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=3)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--start_from_scratch", action="store_true")
    parser.add_argument("--final_model_subdir", type=str, default="final_model")

    args = parser.parse_args()

    # ------------------------------
    # Setup
    # ------------------------------

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir = args.output_dir / args.final_model_subdir

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 80)
    print("Full-parameter SFT script")
    print("=" * 80)
    print(f"transformers version: {transformers.__version__}")
    print(f"torch version: {torch.__version__}")
    print(f"model_id: {args.model_id}")
    print(f"sat_train_path: {args.sat_train_path}")
    print(f"unsat_train_path: {args.unsat_train_path}")
    print(f"output_dir: {args.output_dir}")
    print(f"final_model_dir: {final_model_dir}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")

    print("=" * 80, flush=True)

    # ------------------------------
    # Load train rows
    # ------------------------------

    sat_rows = load_jsonl(args.sat_train_path)
    unsat_rows = load_jsonl(args.unsat_train_path)

    if args.limit_per_label is not None:
        sat_rows = sat_rows[:args.limit_per_label]
        unsat_rows = unsat_rows[:args.limit_per_label]

    for row in sat_rows:
        row["_train_label"] = "SAT"

    for row in unsat_rows:
        row["_train_label"] = "UNSAT"

    train_rows = sat_rows + unsat_rows
    random.shuffle(train_rows)

    print("\nDataset loaded:")
    print(f"SAT train rows:   {len(sat_rows)}")
    print(f"UNSAT train rows: {len(unsat_rows)}")
    print(f"Total train rows: {len(train_rows)}", flush=True)

    if len(train_rows) == 0:
        raise ValueError("No training rows loaded. Check your train JSONL paths.")

    # ------------------------------
    # Tokenizer
    # ------------------------------

    print(f"\nLoading tokenizer: {args.model_id}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print("Tokenizer loaded:")
    print(f"pad_token: {tokenizer.pad_token}")
    print(f"pad_token_id: {tokenizer.pad_token_id}")
    print(f"eos_token: {tokenizer.eos_token}")
    print(f"eos_token_id: {tokenizer.eos_token_id}", flush=True)

    # ------------------------------
    # Dataset + collator
    # ------------------------------

    train_dataset = ChatSFTDataset(
        rows=train_rows,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    collator = DataCollatorForCausalLMWithPadding(tokenizer=tokenizer)

    print("\nDataset sanity check:")
    zero_trainable_count = 0
    lengths = []
    trainable_lengths = []

    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        total_tokens = len(sample["input_ids"])
        trainable_tokens = sum(x != -100 for x in sample["labels"])
        masked_tokens = sum(x == -100 for x in sample["labels"])

        lengths.append(total_tokens)
        trainable_lengths.append(trainable_tokens)

        if trainable_tokens == 0:
            zero_trainable_count += 1

        if i < 3:
            print(
                f"sample={i}, "
                f"total_tokens={total_tokens}, "
                f"masked_prompt_tokens={masked_tokens}, "
                f"trainable_assistant_tokens={trainable_tokens}"
            )

    print(f"Max input length: {max(lengths)}")
    print(f"Avg input length: {sum(lengths) / len(lengths):.2f}")
    print(f"Max trainable assistant tokens: {max(trainable_lengths)}")
    print(f"Avg trainable assistant tokens: {sum(trainable_lengths) / len(trainable_lengths):.2f}")
    print(f"Rows with 0 trainable assistant tokens: {zero_trainable_count}", flush=True)

    if zero_trainable_count > 0:
        raise ValueError(
            "Some rows have 0 trainable assistant tokens. "
            "Increase --max_length or shorten prompts."
        )

    # ------------------------------
    # Model
    # ------------------------------

    dtype = choose_dtype()

    print(f"\nLoading model for full-parameter SFT: {args.model_id}")
    print(f"Using dtype: {dtype}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    for param in model.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print("Model loaded:")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%", flush=True)

    # ------------------------------
    # TrainingArguments
    # ------------------------------

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = make_training_args(
        output_dir=str(args.output_dir),

        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,

        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        bf16=use_bf16,
        fp16=use_fp16,

        optim="adamw_torch",

        report_to="none",
        remove_unused_columns=False,

        seed=args.seed,
        data_seed=args.seed,

        save_safetensors=True,
        disable_tqdm=False,
    )

    # ------------------------------
    # Trainer
    # ------------------------------

    trainer = make_trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[ProgressPrinterCallback()],
    )

    # ------------------------------
    # Resume checkpoint
    # ------------------------------

    last_checkpoint = None

    if args.output_dir.exists() and not args.start_from_scratch:
        last_checkpoint = get_last_checkpoint(str(args.output_dir))

    if last_checkpoint is not None:
        print(f"\nFound checkpoint. Resuming from: {last_checkpoint}", flush=True)
    else:
        print("\nNo checkpoint found. Starting full SFT from the base model.", flush=True)

    # ------------------------------
    # Train
    # ------------------------------

    train_start_time = time.time()
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    train_elapsed = time.time() - train_start_time

    print("\nTraining result:")
    print(train_result, flush=True)

    # ------------------------------
    # Save final model + metrics
    # ------------------------------

    final_model_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    metrics = dict(train_result.metrics)
    metrics["manual_train_elapsed_seconds"] = train_elapsed

    metrics_path = args.output_dir / "train_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    run_record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": args.model_id,
        "sat_train_rows": len(sat_rows),
        "unsat_train_rows": len(unsat_rows),
        "total_train_rows": len(train_rows),
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "last_checkpoint": str(last_checkpoint) if last_checkpoint else "",
        "final_train_loss": metrics.get("train_loss"),
        "train_runtime": metrics.get("train_runtime"),
        "train_samples_per_second": metrics.get("train_samples_per_second"),
        "train_steps_per_second": metrics.get("train_steps_per_second"),
    }

    append_training_run_record(args.output_dir, run_record)

    print("\n" + "=" * 80)
    print("Saved outputs")
    print("=" * 80)
    print(f"Saved final full fine-tuned model to: {final_model_dir}")
    print(f"Saved tokenizer to: {final_model_dir}")
    print(f"Saved training metrics to: {metrics_path}")
    print(f"Saved run history to: {args.output_dir / 'training_runs.csv'}")
    print(f"Training checkpoints are in: {args.output_dir}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()