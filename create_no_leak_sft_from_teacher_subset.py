#!/usr/bin/env python3
"""
Create a no-answer-leak SFT JSONL from teacher response records.

Why this is needed:
  The teacher-generation prompt intentionally included ground-truth labels and
  Z3 certificates. That is useful for asking a teacher model to produce a high
  quality explanation, but it should not be used directly as the student SFT
  input, because the user message already contains the answer.

This script keeps the assistant response, but rewrites the system/user messages
so the student sees only the problem information, not the solution/certificate.

By default, the student input keeps the formal CNF fields because your project
plans to use the underlying CNF as part of the input. It removes:
  - <satisfiable>...</satisfiable>
  - <label>...</label>
  - the entire "## Z3 solver output / certificate" section
  - teacher-only system instructions mentioning Z3/certificates as given input

Usage:
  python create_no_leak_sft_from_teacher_subset.py \
    --input teacher_subset_100sat_100unsat_matched_sft.jsonl \
    --output teacher_subset_100sat_100unsat_matched_sft_noleak.jsonl \
    --report leak_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


CLEAN_SYSTEM_PROMPT = """You are an expert in propositional logic and Boolean satisfiability (SAT).
You are solving SATBench-style natural-language logic puzzles.

Important reasoning rules:
- Use only the constraints stated in the conditions and formal CNF information.
- The scenario is background only and adds no hidden constraints.
- Treat all variables as independent Boolean decisions unless the conditions explicitly state otherwise.
- Do not add commonsense assumptions such as mutual exclusivity, exactly-one constraints, or real-world causal links unless they are stated in the conditions.
- Variables not mentioned in the conditions are irrelevant to satisfiability and may be assigned arbitrarily.

Your task:
1. Decide whether the puzzle is SAT or UNSAT.
2. If SAT, give one satisfying assignment or enough assignment information to verify the clauses.
3. If UNSAT, identify a contradiction or an UNSAT core/relevant conflicting clauses.
4. Explain the reasoning briefly and clearly.
5. End with exactly one final label on a new line: [SAT] or [UNSAT].
Do not write anything after the final label."""

TASK_SUFFIX = """\n\n## Task\nSolve the puzzle. Decide whether all conditions can be satisfied simultaneously.\nIf the instance is SAT, provide a satisfying assignment or enough assignment information to check the clauses.\nIf the instance is UNSAT, explain the contradiction or identify the conflicting clauses.\nEnd with exactly one final label on a new line: [SAT] or [UNSAT]."""

LEAK_PATTERNS = [
    r"<satisfiable>\s*(?:true|false)\s*</satisfiable>",
    r"<label>\s*(?:SAT|UNSAT)\s*</label>",
    r"##\s*Z3\s+solver\s+output\s*/\s*certificate",
    r"<sat_assignment>",
    r"<unsat_core_raw_clause_indices>",
    r"<unsat_core_mapped_condition_numbers>",
    r"<unsat_core_clauses>",
]


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e


def get_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Record has no messages list")
    return messages


def get_role_content(messages: List[Dict[str, str]], role: str) -> Optional[str]:
    for msg in messages:
        if msg.get("role") == role:
            return str(msg.get("content", ""))
    return None


def remove_tag_block(text: str, tag: str) -> str:
    pattern = rf"\n?<{tag}>.*?</{tag}>\n?"
    return re.sub(pattern, "\n", text, flags=re.IGNORECASE | re.DOTALL)


def clean_user_prompt(user_prompt: str, keep_formal_cnf: bool = True) -> str:
    text = user_prompt

    # Drop everything from the Z3/certificate section onward. This removes the
    # actual answer certificate while keeping the puzzle and optional CNF info.
    text = re.split(
        r"\n##\s*Z3\s+solver\s+output\s*/\s*certificate\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    # Remove explicit ground-truth answer tags from the formal SAT section.
    text = remove_tag_block(text, "satisfiable")
    text = remove_tag_block(text, "label")

    if not keep_formal_cnf:
        # Keep only the natural-language puzzle, removing formal SAT info.
        text = re.split(
            r"\n##\s*Formal\s+SAT\s+information\b",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

    # Normalize excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text + TASK_SUFFIX


def contains_leak(text: str) -> List[str]:
    hits = []
    for pat in LEAK_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE | re.DOTALL):
            hits.append(pat)
    return hits


def convert_record(record: Dict[str, Any], keep_formal_cnf: bool) -> Dict[str, Any]:
    messages = get_messages(record)
    user_prompt = get_role_content(messages, "user")
    assistant_response = get_role_content(messages, "assistant")

    # Full records may store response separately instead of assistant message.
    if assistant_response is None:
        assistant_response = str(record.get("teacher_response", ""))
    if user_prompt is None:
        user_prompt = str(record.get("user_prompt", ""))

    if not user_prompt.strip():
        raise ValueError("Could not find user prompt")
    if not assistant_response.strip():
        raise ValueError("Could not find assistant/teacher response")

    clean_user = clean_user_prompt(user_prompt, keep_formal_cnf=keep_formal_cnf)

    out = {
        "messages": [
            {"role": "system", "content": CLEAN_SYSTEM_PROMPT},
            {"role": "user", "content": clean_user},
            {"role": "assistant", "content": assistant_response.strip()},
        ]
    }

    # Preserve useful metadata outside messages. The trainer can ignore these.
    for key in [
        "prompt_id",
        "index",
        "row_index_in_prompt_file",
        "model_id",
        "ground_truth_label",
        "teacher_extracted_label",
        "teacher_label_matches_ground_truth",
        "generation_info",
        "ground_truth",
    ]:
        if key in record:
            out[key] = record[key]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input teacher subset JSONL, usually *_sft.jsonl")
    parser.add_argument("--output", required=True, help="Output no-leak SFT JSONL")
    parser.add_argument("--report", default=None, help="Optional JSON report path")
    parser.add_argument(
        "--natural_language_only",
        action="store_true",
        help="Remove formal CNF information too. Default keeps CNF but removes labels/Z3 certificate.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = n_input_leaky = n_output_leaky = 0
    output_leaks = []

    with output_path.open("w", encoding="utf-8") as out_f:
        for line_no, rec in load_jsonl(input_path):
            n_in += 1
            original_messages = get_messages(rec)
            original_text = "\n".join(str(m.get("content", "")) for m in original_messages)
            if contains_leak(original_text):
                n_input_leaky += 1

            converted = convert_record(rec, keep_formal_cnf=not args.natural_language_only)
            converted_text = "\n".join(m["content"] for m in converted["messages"][:2])
            hits = contains_leak(converted_text)
            if hits:
                n_output_leaky += 1
                output_leaks.append({"line_no": line_no, "hits": hits, "prompt_id": rec.get("prompt_id")})

            out_f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            n_out += 1

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "n_input_rows": n_in,
        "n_output_rows": n_out,
        "n_input_rows_with_answer_leak_patterns": n_input_leaky,
        "n_output_rows_with_answer_leak_patterns": n_output_leaky,
        "output_leak_examples": output_leaks[:20],
        "kept_formal_cnf": not args.natural_language_only,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if n_output_leaky:
        raise SystemExit("ERROR: output still appears to contain answer leaks")


if __name__ == "__main__":
    main()
