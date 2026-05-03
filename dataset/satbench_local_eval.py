#!/usr/bin/env python3
"""Minimal local SATBench evaluator.

This mirrors the released SATBench SAT/UNSAT evaluation flow, but swaps the
provider-specific API wrapper for a local Hugging Face causal LM.

Example:
    python satbench_local_eval.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output qwen25_7b_eval.jsonl \
        --limit 50
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SAT_PROMPT_TEMPLATE = """You are a logical reasoning assistant. You are given a logic puzzle.

{scenario}

Conditions:
{conditions}

{question}

Guidelines:
- All constraints come only from the Conditions section.
- The scenario provides background and intuition, but does not impose any additional rules or constraints.
- All variables represent independent decisions; there is no mutual exclusivity or implicit linkage unless stated explicitly in the conditions.
- Variables not mentioned in the conditions are considered unknown and irrelevant to satisfiability.

Your task:
- If the puzzle is satisfiable, propose one valid assignment that satisfies all the conditions.
- If the puzzle is unsatisfiable, explain why some of the conditions cannot all be true at once.
- Think step by step.

At the end of your answer, output exactly one of the following labels on a new line:
[SAT] — if a valid assignment exists
[UNSAT] — if the constraints cannot be satisfied

Do not add any text after the final label.
"""


def format_prompt(entry: dict[str, Any]) -> str:
    return SAT_PROMPT_TEMPLATE.format(
        scenario=entry["scenario"],
        conditions="\n".join(entry["conditions"]),
        question=entry["question"],
    )


def parse_label(text: str) -> str | None:
    upper = text.upper()
    bracketed = re.findall(r"\[(SAT|UNSAT)\]", upper)
    if bracketed:
        # Prefer the final bracketed label, since the prompt asks for it on the last line.
        return bracketed[-1]

    tokens = re.findall(r"\b[A-Z]{3,6}\b", upper)
    if "UNSAT" in tokens:
        return "UNSAT"
    if "SAT" in tokens:
        return "SAT"
    return None


def build_messages(tokenizer: AutoTokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def load_done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            done.add(obj["readable"])
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--output", required=True, help="JSONL output path")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    output_path = Path(args.output)
    done_keys = load_done_keys(output_path)

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.device_map,
    )
    model.eval()

    ds = load_dataset("LLM4Code/SATBench", split=args.split)
    entries = [dict(x) for x in ds]
    if args.start:
        entries = entries[args.start :]
    if args.limit is not None:
        entries = entries[: args.limit]

    with output_path.open("a", encoding="utf-8") as fout:
        for entry in entries:
            if entry["readable"] in done_keys:
                continue

            prompt = format_prompt(entry)
            rendered = build_messages(tokenizer, prompt)
            inputs = tokenizer(rendered, return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            new_tokens = generated[0][inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            pred_label = parse_label(text)
            gold_label = "SAT" if entry["satisfiable"] else "UNSAT"

            record = {
                **entry,
                "model_trace": text,
                "pred_label": pred_label,
                "correct_prediction": pred_label == gold_label,
                "eval_model": args.model,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()


if __name__ == "__main__":
    main()
