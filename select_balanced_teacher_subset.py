#!/usr/bin/env python3
"""
Select a balanced, non-truncated subset of teacher-response records.

Typical use after Slurm generation:

  python select_balanced_teacher_subset.py \
    --records_dir teacher_responses_qwen35_2b/records \
    --n_sat 100 \
    --n_unsat 100 \
    --out_jsonl teacher_subset_100sat_100unsat.jsonl \
    --out_sft_jsonl teacher_subset_100sat_100unsat_sft.jsonl

By default, this script filters to records that are:
  * valid JSON files
  * not error records
  * have a non-empty teacher_response
  * are NOT marked likely_truncated_by_max_new_tokens

It balances by ground-truth label by default. You can add
--require_teacher_label_match if you only want examples where the teacher's
extracted final label matches the ground-truth label.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select balanced SAT/UNSAT non-truncated teacher records.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--records_dir", type=str, help="Directory containing row_*.json files, e.g. teacher_responses_x/records")
    src.add_argument("--input_jsonl", type=str, help="Combined teacher response JSONL file")

    p.add_argument("--n_sat", type=int, default=100, help="Number of SAT examples to select")
    p.add_argument("--n_unsat", type=int, default=100, help="Number of UNSAT examples to select")
    p.add_argument("--seed", type=int, default=598, help="Random seed for selection")
    p.add_argument(
        "--selection",
        choices=["first", "random"],
        default="first",
        help="Select first valid examples by row index, or random sample with --seed",
    )
    p.add_argument(
        "--label_source",
        choices=["ground_truth", "teacher"],
        default="ground_truth",
        help="Which label to use for balancing. Usually use ground_truth.",
    )
    p.add_argument(
        "--require_teacher_label_match",
        action="store_true",
        help="Keep only rows where teacher_extracted_label matches ground_truth_label.",
    )
    p.add_argument(
        "--allow_truncated",
        action="store_true",
        help="Do not filter out likely_truncated_by_max_new_tokens records. Default filters them out.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Raise an error if fewer than requested SAT/UNSAT examples are available.",
    )
    p.add_argument("--out_jsonl", type=str, default="teacher_subset_100sat_100unsat.jsonl")
    p.add_argument("--out_sft_jsonl", type=str, default="teacher_subset_100sat_100unsat_sft.jsonl")
    p.add_argument("--out_indices", type=str, default="teacher_subset_100sat_100unsat_indices.txt")
    return p.parse_args()


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def iter_records(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    if args.records_dir:
        records_dir = Path(args.records_dir)
        for path in sorted(records_dir.glob("row_*.json")):
            rec = read_json(path)
            if rec is not None:
                rec.setdefault("_source_file", str(path))
                yield rec
    else:
        with open(args.input_jsonl, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    rec.setdefault("_source_line", line_no)
                    yield rec


def normalize_label(x: Any) -> Optional[str]:
    if isinstance(x, bool):
        return "SAT" if x else "UNSAT"
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in {"SAT", "TRUE", "SATISFIABLE"}:
        return "SAT"
    if s in {"UNSAT", "FALSE", "UNSATISFIABLE"}:
        return "UNSAT"
    return None


def get_ground_truth_label(rec: Dict[str, Any]) -> Optional[str]:
    for key in ("ground_truth_label", "label", "satisfiable"):
        lab = normalize_label(rec.get(key))
        if lab:
            return lab
    gt = rec.get("ground_truth")
    if isinstance(gt, dict):
        for key in ("label", "satisfiable"):
            lab = normalize_label(gt.get(key))
            if lab:
                return lab
    return None


def extract_final_label(text: str) -> Optional[str]:
    if not text:
        return None
    matches = re.findall(r"\[\s*(SAT|UNSAT)\s*\]", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    bare = re.findall(r"\b(UNSAT|SAT)\b", text, flags=re.IGNORECASE)
    if bare:
        return bare[-1].upper()
    return None


def get_teacher_label(rec: Dict[str, Any]) -> Optional[str]:
    lab = normalize_label(rec.get("teacher_extracted_label"))
    if lab:
        return lab
    response = rec.get("teacher_response")
    if isinstance(response, str):
        return extract_final_label(response)
    return None


def row_sort_key(rec: Dict[str, Any]) -> int:
    for key in ("row_index_in_prompt_file", "index"):
        try:
            return int(rec.get(key))
        except Exception:
            pass
    return 10**12


def is_good_record(rec: Dict[str, Any], args: argparse.Namespace) -> Tuple[bool, str]:
    if rec.get("error"):
        return False, "error_record"
    response = rec.get("teacher_response")
    if not isinstance(response, str) or not response.strip():
        return False, "empty_response"
    gen = rec.get("generation_info")
    if isinstance(gen, dict) and gen.get("likely_truncated_by_max_new_tokens") and not args.allow_truncated:
        return False, "truncated"

    gt_label = get_ground_truth_label(rec)
    teacher_label = get_teacher_label(rec)
    if args.require_teacher_label_match and gt_label != teacher_label:
        return False, "teacher_label_mismatch"

    label = gt_label if args.label_source == "ground_truth" else teacher_label
    if label not in {"SAT", "UNSAT"}:
        return False, f"missing_{args.label_source}_label"
    return True, "ok"


def strip_internal_fields(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def make_sft_record(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    messages = rec.get("messages")
    response = rec.get("teacher_response")
    if not isinstance(messages, list) or not isinstance(response, str) or not response.strip():
        return None
    return {
        "prompt_id": rec.get("prompt_id"),
        "index": rec.get("index"),
        "row_index_in_prompt_file": rec.get("row_index_in_prompt_file"),
        "model_id": rec.get("model_id"),
        "ground_truth_label": get_ground_truth_label(rec),
        "teacher_extracted_label": get_teacher_label(rec),
        "teacher_label_matches_ground_truth": get_teacher_label(rec) == get_ground_truth_label(rec),
        "messages": messages + [{"role": "assistant", "content": response}],
        "ground_truth": rec.get("ground_truth"),
        "generation_info": rec.get("generation_info"),
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    stats = {
        "total_read": 0,
        "kept_valid": 0,
        "reasons": {},
        "available_sat": 0,
        "available_unsat": 0,
    }

    valid_sat: List[Dict[str, Any]] = []
    valid_unsat: List[Dict[str, Any]] = []

    for rec in iter_records(args):
        stats["total_read"] += 1
        ok, reason = is_good_record(rec, args)
        stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
        if not ok:
            continue
        stats["kept_valid"] += 1
        label = get_ground_truth_label(rec) if args.label_source == "ground_truth" else get_teacher_label(rec)
        if label == "SAT":
            valid_sat.append(rec)
        elif label == "UNSAT":
            valid_unsat.append(rec)

    valid_sat.sort(key=row_sort_key)
    valid_unsat.sort(key=row_sort_key)
    stats["available_sat"] = len(valid_sat)
    stats["available_unsat"] = len(valid_unsat)

    if args.strict and (len(valid_sat) < args.n_sat or len(valid_unsat) < args.n_unsat):
        raise SystemExit(
            f"Not enough valid examples: SAT available={len(valid_sat)} requested={args.n_sat}; "
            f"UNSAT available={len(valid_unsat)} requested={args.n_unsat}; stats={stats}"
        )

    if args.selection == "random":
        selected_sat = rng.sample(valid_sat, k=min(args.n_sat, len(valid_sat)))
        selected_unsat = rng.sample(valid_unsat, k=min(args.n_unsat, len(valid_unsat)))
        selected = selected_sat + selected_unsat
        selected.sort(key=row_sort_key)
    else:
        selected_sat = valid_sat[: args.n_sat]
        selected_unsat = valid_unsat[: args.n_unsat]
        selected = sorted(selected_sat + selected_unsat, key=row_sort_key)

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for rec in selected:
            f.write(json.dumps(strip_internal_fields(rec), ensure_ascii=False) + "\n")

    sft_written = 0
    with open(args.out_sft_jsonl, "w", encoding="utf-8") as f:
        for rec in selected:
            sft = make_sft_record(rec)
            if sft is not None:
                f.write(json.dumps(sft, ensure_ascii=False) + "\n")
                sft_written += 1

    with open(args.out_indices, "w", encoding="utf-8") as f:
        for rec in selected:
            f.write(f"{row_sort_key(rec)}\t{get_ground_truth_label(rec)}\t{get_teacher_label(rec)}\n")

    summary = {
        **stats,
        "selected_sat": len(selected_sat),
        "selected_unsat": len(selected_unsat),
        "selected_total": len(selected),
        "sft_rows_written": sft_written,
        "out_jsonl": args.out_jsonl,
        "out_sft_jsonl": args.out_sft_jsonl,
        "out_indices": args.out_indices,
        "label_source": args.label_source,
        "require_teacher_label_match": args.require_teacher_label_match,
        "allow_truncated": args.allow_truncated,
        "selection": args.selection,
        "seed": args.seed,
    }
    print("SELECTION_SUMMARY " + json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
