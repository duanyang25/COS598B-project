#!/usr/bin/env python3

import argparse
import csv
import json
import re
import time
from pathlib import Path
from threading import Thread
from collections import Counter

import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)


# ============================================================
# Helpers: JSONL
# ============================================================

def load_jsonl(path: Path):
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


def write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "dataset_id",
        "row_id",
        "eval_split",
        "split_index",
        "gold_label",
        "predicted_label",
        "correct",
        "input_tokens",
        "output_tokens",
        "seconds",
        "tokens_per_second",
        "error",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ============================================================
# Helpers: labels and prompts
# ============================================================

def get_ground_truth_label(row):
    label = row.get("ground_truth_label")

    if label is None and isinstance(row.get("ground_truth"), dict):
        label = row["ground_truth"].get("label")

    if label is None and isinstance(row.get("ground_truth"), dict):
        satisfiable = row["ground_truth"].get("satisfiable")
        if satisfiable is True:
            label = "SAT"
        elif satisfiable is False:
            label = "UNSAT"

    if label is None:
        label = row.get("_expected_label_from_file")

    if label is None:
        raise ValueError(f"Cannot find ground-truth label. Row keys: {list(row.keys())}")

    label = str(label).strip().upper()
    if label not in {"SAT", "UNSAT"}:
        raise ValueError(f"Unexpected ground-truth label: {label}")

    return label


def get_prompt_messages(row):
    """
    Use only system/user messages for evaluation.
    Remove the assistant target from the SFT row.
    """
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Row does not contain a valid messages list. Keys: {list(row.keys())}")

    prompt_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "assistant":
            continue
        prompt_messages.append({"role": role, "content": content})

    if not prompt_messages:
        raise ValueError("No non-assistant messages found.")

    return prompt_messages


def messages_to_prompt(tokenizer, messages):
    """
    Try the tokenizer chat template first.
    For a base model, this may fail or may not be useful, so fall back to plain text.
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        parts = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            parts.append(f"{role}:\n{content}")
        parts.append("ASSISTANT:\n")
        return "\n\n".join(parts)


def extract_predicted_label(text):
    """
    Extract final SAT/UNSAT prediction.
    Prefer bracketed labels like [SAT] or [UNSAT].
    """
    if not text:
        return None

    bracket_matches = re.findall(r"\[\s*(UNSAT|SAT)\s*\]", text, flags=re.IGNORECASE)
    if bracket_matches:
        return bracket_matches[-1].upper()

    word_matches = re.findall(r"\b(UNSAT|SAT)\b", text, flags=re.IGNORECASE)
    if word_matches:
        return word_matches[-1].upper()

    return None


# ============================================================
# Dataset preparation
# ============================================================

def prepare_eval_rows(sat_path: Path, unsat_path: Path, limit_per_split=None):
    sat_rows = load_jsonl(sat_path)
    unsat_rows = load_jsonl(unsat_path)

    if limit_per_split is not None:
        sat_rows = sat_rows[:limit_per_split]
        unsat_rows = unsat_rows[:limit_per_split]

    eval_rows = []

    for i, row in enumerate(sat_rows):
        row = dict(row)
        row["_eval_split"] = "sat_test"
        row["_expected_label_from_file"] = "SAT"
        row["_split_index"] = i
        row["_dataset_id"] = f"sat_test_{i:05d}"
        eval_rows.append(row)

    for i, row in enumerate(unsat_rows):
        row = dict(row)
        row["_eval_split"] = "unsat_test"
        row["_expected_label_from_file"] = "UNSAT"
        row["_split_index"] = i
        row["_dataset_id"] = f"unsat_test_{i:05d}"
        eval_rows.append(row)

    print(f"Loaded SAT test rows:   {len(sat_rows)}")
    print(f"Loaded UNSAT test rows: {len(unsat_rows)}")
    print(f"Loaded total rows:      {len(eval_rows)}")

    return eval_rows


def load_existing_results(path: Path):
    if not path.exists():
        return {}

    existing = {}
    for row in load_jsonl(path):
        dataset_id = row.get("dataset_id")
        if dataset_id:
            existing[dataset_id] = row

    return existing


def sort_results(rows):
    split_order = {"sat_test": 0, "unsat_test": 1}
    return sorted(
        rows,
        key=lambda r: (
            split_order.get(r.get("eval_split"), 99),
            int(r.get("split_index", r.get("row_id", 999999))),
        ),
    )


# ============================================================
# Model loading
# ============================================================

def choose_dtype():
    if not torch.cuda.is_available():
        return torch.float32

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def load_model_and_tokenizer(model_id: str, local_files_only: bool):
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = choose_dtype()

    print(f"Loading model: {model_id}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"Using dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    model.eval()

    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")

    print(f"Model input embedding device: {model.get_input_embeddings().weight.device}")

    return model, tokenizer


# ============================================================
# Generation
# ============================================================

@torch.inference_mode()
def generate_one(
    row,
    row_id,
    model,
    tokenizer,
    max_new_tokens,
    do_sample,
    temperature,
    top_p,
    stream_output,
):
    gold = get_ground_truth_label(row)
    messages = get_prompt_messages(row)
    prompt_text = messages_to_prompt(tokenizer, messages)

    inputs = tokenizer(prompt_text, return_tensors="pt")

    input_device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    start_time = time.time()

    if stream_output:
        print("\n" + "=" * 100)
        print(
            f"Example {row_id} | "
            f"dataset_id={row['_dataset_id']} | "
            f"split={row['_eval_split']} | "
            f"gold={gold}"
        )
        print("-" * 100, flush=True)

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60.0,
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        if do_sample:
            generation_kwargs.update(
                temperature=temperature,
                top_p=top_p,
            )

        thread_error = []

        def _run_generate():
            try:
                model.generate(**generation_kwargs)
            except Exception as e:
                thread_error.append(e)
                try:
                    streamer.on_finalized_text("", stream_end=True)
                except Exception:
                    pass

        thread = Thread(target=_run_generate)
        thread.start()

        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            print(new_text, end="", flush=True)

        thread.join()

        if thread_error:
            raise thread_error[0]

        print("\n" + "-" * 100, flush=True)

    else:
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        if do_sample:
            generation_kwargs.update(
                temperature=temperature,
                top_p=top_p,
            )

        output_ids = model.generate(**generation_kwargs)

        generated_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    elapsed = time.time() - start_time

    pred = extract_predicted_label(generated_text)
    correct = pred == gold

    input_tokens = int(inputs["input_ids"].shape[-1])
    output_tokens = len(tokenizer.encode(generated_text, add_special_tokens=False))

    return {
        "dataset_id": row["_dataset_id"],
        "row_id": row_id,
        "eval_split": row["_eval_split"],
        "split_index": row["_split_index"],
        "gold_label": gold,
        "predicted_label": pred,
        "correct": correct,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "seconds": elapsed,
        "tokens_per_second": output_tokens / elapsed if elapsed > 0 else None,
        "prompt_text": prompt_text,
        "model_response": generated_text,
        "error": None,
    }


# ============================================================
# Summary
# ============================================================

def accuracy_for(rows):
    if not rows:
        return None
    return sum(bool(r.get("correct")) for r in rows) / len(rows)


def make_summary(results, model_id):
    sat_results = [r for r in results if r.get("eval_split") == "sat_test"]
    unsat_results = [r for r in results if r.get("eval_split") == "unsat_test"]

    tps_values = [
        r.get("tokens_per_second")
        for r in results
        if isinstance(r.get("tokens_per_second"), (int, float))
    ]

    output_tokens = [
        r.get("output_tokens")
        for r in results
        if isinstance(r.get("output_tokens"), (int, float))
    ]

    return {
        "model_id": model_id,
        "num_total": len(results),
        "num_sat_test": len(sat_results),
        "num_unsat_test": len(unsat_results),
        "overall_accuracy": accuracy_for(results),
        "sat_test_accuracy": accuracy_for(sat_results),
        "unsat_test_accuracy": accuracy_for(unsat_results),
        "predicted_label_counts": dict(Counter(str(r.get("predicted_label") or "NONE") for r in results)),
        "gold_label_counts": dict(Counter(str(r.get("gold_label") or "NONE") for r in results)),
        "num_errors": sum(1 for r in results if r.get("error")),
        "avg_output_tokens": sum(output_tokens) / len(output_tokens) if output_tokens else None,
        "avg_tokens_per_second": sum(tps_values) / len(tps_values) if tps_values else None,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--sat_test_path", type=Path, default=Path("dataset/sft_by_label/sat/test.jsonl"))
    parser.add_argument("--unsat_test_path", type=Path, default=Path("dataset/sft_by_label/unsat/test.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/baseline_qwen35_08b_base_results"))

    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--limit_per_split", type=int, default=None)
    parser.add_argument("--local_files_only", action="store_true")

    parser.add_argument("--stream", dest="stream_output", action="store_true")
    parser.add_argument("--no_stream", dest="stream_output", action="store_false")
    parser.set_defaults(stream_output=True)

    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_jsonl = args.output_dir / "qwen35_08b_base_test_predictions.jsonl"
    output_csv = args.output_dir / "qwen35_08b_base_test_predictions.csv"
    summary_path = args.output_dir / "qwen35_08b_base_test_summary.json"

    print("Configuration:")
    print(json.dumps({
        "model_id": args.model_id,
        "sat_test_path": str(args.sat_test_path),
        "unsat_test_path": str(args.unsat_test_path),
        "output_dir": str(args.output_dir),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "limit_per_split": args.limit_per_split,
        "local_files_only": args.local_files_only,
        "stream_output": args.stream_output,
        "overwrite": args.overwrite,
    }, indent=2))

    eval_rows = prepare_eval_rows(
        args.sat_test_path,
        args.unsat_test_path,
        limit_per_split=args.limit_per_split,
    )

    existing_results = {} if args.overwrite else load_existing_results(output_jsonl)
    print(f"Loaded existing completed results: {len(existing_results)}")

    model, tokenizer = load_model_and_tokenizer(
        args.model_id,
        local_files_only=args.local_files_only,
    )

    results_by_id = dict(existing_results)

    for row_id, row in enumerate(tqdm(eval_rows, desc="Evaluating")):
        dataset_id = row["_dataset_id"]

        if dataset_id in results_by_id and not args.overwrite:
            print(f"Skipping existing result: {dataset_id}")
            continue

        try:
            result = generate_one(
                row=row,
                row_id=row_id,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                stream_output=args.stream_output,
            )

        except Exception as e:
            print(f"\nERROR on row_id={row_id}, dataset_id={dataset_id}: {repr(e)}", flush=True)
            result = {
                "dataset_id": dataset_id,
                "row_id": row_id,
                "eval_split": row["_eval_split"],
                "split_index": row["_split_index"],
                "gold_label": get_ground_truth_label(row),
                "predicted_label": None,
                "correct": False,
                "input_tokens": None,
                "output_tokens": None,
                "seconds": None,
                "tokens_per_second": None,
                "prompt_text": None,
                "model_response": "",
                "error": repr(e),
            }

        results_by_id[dataset_id] = result

        sorted_results = sort_results(list(results_by_id.values()))
        write_jsonl(sorted_results, output_jsonl)
        write_csv(sorted_results, output_csv)

        summary = make_summary(sorted_results, args.model_id)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\nCurrent summary:")
        print(json.dumps(summary, indent=2), flush=True)

    final_results = sort_results(list(results_by_id.values()))
    final_summary = make_summary(final_results, args.model_id)

    write_jsonl(final_results, output_jsonl)
    write_csv(final_results, output_csv)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    print("\nFinal summary:")
    print(json.dumps(final_summary, indent=2))

    print(f"\nSaved raw predictions to: {output_jsonl}")
    print(f"Saved CSV to: {output_csv}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()