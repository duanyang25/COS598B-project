#!/usr/bin/env python3
"""
generate_satbench_prompts.py
============================

Create teacher-generation prompts for the COS 598B SATBench certificate project.

This script does NOT load an LLM and does NOT run inference. It only reads the
local enriched SATBench JSONL file, assembles one teacher prompt per row, and
writes all prompts to a new JSONL file. You can later feed the resulting prompts
to a separate Transformers generation script.

Expected input
--------------
A JSONL file such as:

    satbench_with_certificates_full.jsonl

Each line should be one JSON object with the SATBench fields plus your Z3
certificate fields, for example:

    dims, num_vars, num_clauses, clauses, readable, satisfiable,
    scenario, variable_mapping, conditions, question,
    certificate_type, sat_assignment, unsat_core_clause_indices,
    sat_reason, unsat_reason

Default output
--------------
A JSONL file where each line has:

    index:                  row index in the source file
    prompt_id:              stable id, e.g. satbench_00042
    label:                  SAT or UNSAT, from the row's satisfiable field
    certificate_type:       sat_assignment or unsat_core
    messages:               chat-format messages for teacher generation
    system_prompt:          system prompt string
    user_prompt:            user prompt string
    ground_truth:           compact metadata useful for filtering/evaluation

The teacher prompt intentionally contains the ground-truth SAT/UNSAT label and
Z3 certificate because this prompt file is for TEACHER TRACE GENERATION, not for
student evaluation. The later student SFT examples should use the teacher's
answer as the assistant output but should not leak the answer in the user prompt.

Example
-------

    python generate_satbench_prompts_checked.py \
      --dataset satbench_with_certificates_full.jsonl \
      --output satbench_teacher_prompts.jsonl \
      --expected_rows 2100

To inspect the first generated prompt without writing all rows:

    python generate_satbench_prompts_checked.py \
      --dataset satbench_with_certificates_full.jsonl \
      --output /tmp/prompts.jsonl \
      --limit 1 \
      --print_first
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert in propositional logic and Boolean satisfiability (SAT).
You are generating high-quality teacher outputs for supervised fine-tuning of a smaller language model.

You will be given a SATBench-style natural-language logic puzzle, its underlying CNF formula, and a Z3-verified certificate.

Important reasoning rules:
- Use only the constraints stated in the conditions. The scenario is background only and adds no hidden constraints.
- Treat all variables as independent Boolean decisions unless the conditions explicitly state otherwise.
- Do not add commonsense assumptions such as mutual exclusivity, exactly-one constraints, or real-world causal links unless they are stated in the conditions.
- Variables not mentioned in the conditions are irrelevant to satisfiability and may be assigned arbitrarily.

Your task:
1. Solve the puzzle step by step using the natural-language conditions and the formal CNF information.
2. Use the Z3 certificate to anchor the derivation, but explain why the certificate works instead of merely copying it.
3. For SAT, provide one assignment that satisfies every clause.
4. For UNSAT, identify the contradiction and cite the relevant condition numbers or clauses.

Required answer format:

<think>
Write the detailed reasoning trace here.
</think>

Decision: SAT or UNSAT

Certificate:
- If SAT, write: Assignment: <nested 0/1 list matching dims>, where 1 means True and 0 means False.
- If UNSAT, write: UNSAT core condition numbers: <list of 1-indexed condition numbers>, and optionally include the corresponding CNF clauses.

Explanation:
A short justification that connects the certificate to the conditions.

End with exactly one final label on its own line:
[SAT] or [UNSAT]
"""


USER_PROMPT_TEMPLATE = """## Puzzle

<scenario>
{scenario}
</scenario>

<variable_mapping>
{variable_mapping}
</variable_mapping>

<conditions>
{conditions_block}
</conditions>

<question>
{question}
</question>

## Formal SAT information

<dims>
{dims}
</dims>

<num_vars>
{num_vars}
</num_vars>

<num_clauses>
{num_clauses}
</num_clauses>

<clauses_dimacs>
{clauses_json}
</clauses_dimacs>

<readable_cnf>
{readable}
</readable_cnf>

<satisfiable>
{satisfiable_bool}
</satisfiable>

<label>
{label}
</label>

## Z3 solver output / certificate

{certificate_block}

## Instruction

Generate the teacher answer for this SATBench puzzle. The answer should be faithful to the conditions and the CNF formula. It should avoid the common SATBench failure modes: satisfiability bias, context inconsistency, condition omission, and spurious prior assumptions.
"""


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into memory.

    SATBench has only 2,100 rows, so keeping the rows in memory is simple and
    makes validation easier. Malformed lines are treated as fatal because we do
    not want to silently skip training examples.
    """
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Could not parse JSON on line {line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} is not a JSON object")
            rows.append(obj)
    return rows


def json_dumps_compact(obj: Any) -> str:
    """Compact JSON rendering used inside prompts."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def json_dumps_pretty(obj: Any) -> str:
    """Pretty JSON rendering used for human-readable certificate blocks."""
    return json.dumps(obj, ensure_ascii=False, indent=2)


def product(xs: Sequence[int]) -> int:
    total = 1
    for x in xs:
        total *= int(x)
    return total


# ---------------------------------------------------------------------------
# DIMACS variable id <-> SATBench structural variable string
# ---------------------------------------------------------------------------

def dimacs_to_indices(var_id: int, dims: Sequence[int]) -> Tuple[int, ...]:
    """Convert a 1-indexed flat DIMACS variable id to row-major indices.

    Example:
        dims=[3, 5, 2]
        var_id=25  ->  x(2, 2, 0)

    SATBench's `clauses` field uses flat DIMACS-style variable ids, while
    `readable`, `conditions`, and `variable_mapping` use structural variables
    like x(0,), x(1, 2), or x(2, 2, 0). This conversion makes the Z3 output more
    interpretable in the teacher prompt.
    """
    if var_id < 1:
        raise ValueError(f"DIMACS variable ids must be >= 1, got {var_id}")
    if not dims:
        return (var_id - 1,)

    flat = var_id - 1
    indices: List[int] = []
    for axis in range(len(dims)):
        stride = product(dims[axis + 1 :]) if axis + 1 < len(dims) else 1
        idx = flat // stride
        flat %= stride
        indices.append(idx)
    return tuple(indices)


def dimacs_to_struct(var_id: int, dims: Sequence[int]) -> str:
    """Render a DIMACS variable id as SATBench structural notation."""
    indices = dimacs_to_indices(var_id, dims)
    if len(indices) == 1:
        # SATBench writes 1-D variables in Python tuple style: x(0,)
        return f"x({indices[0]},)"
    return "x(" + ", ".join(str(i) for i in indices) + ")"


def literal_to_struct(lit: int, dims: Sequence[int]) -> str:
    """Render a signed DIMACS literal as structural notation."""
    prefix = "¬" if lit < 0 else ""
    return prefix + dimacs_to_struct(abs(int(lit)), dims)


def clause_to_struct(clause: Sequence[int], dims: Sequence[int]) -> str:
    """Render a DIMACS clause as a readable disjunction."""
    return "(" + " ∨ ".join(literal_to_struct(lit, dims) for lit in clause) + ")"


# ---------------------------------------------------------------------------
# Readable CNF parsing for mapping raw clauses to condition numbers
# ---------------------------------------------------------------------------

def split_top_level(text: str, separator: str) -> List[str]:
    """Split text on a separator that appears at parenthesis depth 0.

    We cannot use a simple regex because SATBench clauses contain variable
    strings such as x(0,) and x(2, 1), which themselves contain parentheses.
    """
    parts: List[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == separator and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def strip_outer_parens(text: str) -> str:
    """Remove one pair of outer parentheses if they wrap the whole string."""
    s = text.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return s

    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            # If depth returns to zero before the final char, the outer pair
            # does not wrap the whole expression.
            if depth == 0 and i != len(s) - 1:
                return s
    return s[1:-1].strip()


def normalize_literal(lit: str) -> str:
    """Canonicalize one literal string for matching."""
    # Remove whitespace but keep the negation symbol and x(...) structure.
    return re.sub(r"\s+", "", lit.strip())


def normalize_clause_text(clause_text: str) -> Tuple[str, ...]:
    """Canonicalize a clause into a sorted tuple of literal strings.

    Sorting makes the mapping robust to literal order differences, e.g.
    (x(0,) ∨ ¬x(2,)) and (¬x(2,) ∨ x(0,)) normalize to the same key.
    """
    inner = strip_outer_parens(clause_text)
    literals = split_top_level(inner, "∨")
    return tuple(sorted(normalize_literal(lit) for lit in literals))


def readable_clause_keys(readable: str) -> List[Tuple[str, ...]]:
    """Return normalized clause keys from SATBench's readable CNF string."""
    if not readable:
        return []
    clause_texts = split_top_level(readable, "∧")
    return [normalize_clause_text(c) for c in clause_texts]


def map_raw_clause_index_to_condition_numbers(row: Dict[str, Any]) -> Dict[int, List[int]]:
    """Map each raw `clauses` index to 1-indexed condition numbers.

    In many rows, the raw `clauses` list may not be in the same order as the
    natural-language `conditions` / `readable` CNF order. This function compares
    literal content to recover which condition(s) correspond to each raw clause.
    """
    dims = row.get("dims") or []
    raw_clauses = row.get("clauses") or []
    readable = row.get("readable") or ""

    condition_keys = readable_clause_keys(readable)
    key_to_condition_numbers: Dict[Tuple[str, ...], List[int]] = {}
    for condition_idx, key in enumerate(condition_keys, start=1):
        key_to_condition_numbers.setdefault(key, []).append(condition_idx)

    mapping: Dict[int, List[int]] = {}
    for raw_idx, raw_clause in enumerate(raw_clauses):
        raw_clause_text = clause_to_struct(raw_clause, dims)
        key = normalize_clause_text(raw_clause_text)
        mapping[raw_idx] = list(key_to_condition_numbers.get(key, []))
    return mapping


# ---------------------------------------------------------------------------
# Assignment reshaping and certificate rendering
# ---------------------------------------------------------------------------

def parse_bool(value: Any) -> bool:
    """Parse common bool representations from JSON/Z3 output."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes"}:
            return True
        if v in {"false", "0", "no"}:
            return False
    return bool(value)


def reshape_assignment(assignment: Dict[Any, Any], dims: Sequence[int]) -> Any:
    """Reshape a flat 1-indexed Z3 assignment dict into nested 0/1 lists."""
    if not dims:
        return []

    total = product(dims)
    flat: List[int] = []
    for var_id in range(1, total + 1):
        value = assignment.get(str(var_id), assignment.get(var_id, False))
        flat.append(1 if parse_bool(value) else 0)

    def reshape(values: List[int], shape: Sequence[int]) -> Any:
        if len(shape) == 1:
            return values[: int(shape[0])]
        first = int(shape[0])
        rest = shape[1:]
        chunk_size = product(rest)
        return [
            reshape(values[i * chunk_size : (i + 1) * chunk_size], rest)
            for i in range(first)
        ]

    return reshape(flat, list(dims))


def render_sat_certificate(row: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Render the Z3 satisfying assignment for a SAT row."""
    dims = row.get("dims") or []
    assignment = row.get("sat_assignment") or {}
    structured = reshape_assignment(assignment, dims)

    total_vars = int(row.get("num_vars") or product(dims))
    per_variable_lines: List[str] = []
    for var_id in range(1, total_vars + 1):
        value = assignment.get(str(var_id), assignment.get(var_id, False))
        truth = parse_bool(value)
        try:
            name = dimacs_to_struct(var_id, dims)
        except Exception:
            name = f"x{var_id}"
        per_variable_lines.append(f"- {name} = {truth}")

    sat_reason = row.get("sat_reason")
    sat_reason_block = f"\n<sat_reason>\n{sat_reason}\n</sat_reason>\n" if sat_reason else ""

    block = (
        "<certificate_type>\nSAT satisfying assignment\n</certificate_type>\n\n"
        "<assignment_structured_0_1>\n"
        f"{json_dumps_pretty(structured)}\n"
        "</assignment_structured_0_1>\n\n"
        "<assignment_per_variable>\n"
        f"{chr(10).join(per_variable_lines)}\n"
        "</assignment_per_variable>"
        f"{sat_reason_block}"
    )

    metadata = {
        "assignment_structured_0_1": structured,
        "assignment_flat": assignment,
        "sat_reason": sat_reason,
    }
    return block, metadata


def render_unsat_certificate(row: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Render the Z3 UNSAT core for an UNSAT row."""
    dims = row.get("dims") or []
    clauses = row.get("clauses") or []
    core_indices = row.get("unsat_core_clause_indices") or []
    index_to_condition_numbers = map_raw_clause_index_to_condition_numbers(row)

    core_lines: List[str] = []
    mapped_condition_numbers: List[int] = []
    mapping_details: List[Dict[str, Any]] = []

    for raw_idx in core_indices:
        try:
            raw_idx_int = int(raw_idx)
        except Exception:
            continue

        if 0 <= raw_idx_int < len(clauses):
            clause_dimacs = clauses[raw_idx_int]
            clause_struct = clause_to_struct(clause_dimacs, dims)
        else:
            clause_dimacs = None
            clause_struct = "<out of range>"

        condition_numbers = index_to_condition_numbers.get(raw_idx_int, [])
        for c in condition_numbers:
            if c not in mapped_condition_numbers:
                mapped_condition_numbers.append(c)

        core_lines.append(
            f"- raw clause index {raw_idx_int}: {clause_dimacs} -> {clause_struct}; "
            f"matching condition number(s): {condition_numbers if condition_numbers else 'UNKNOWN'}"
        )
        mapping_details.append(
            {
                "raw_clause_index": raw_idx_int,
                "clause_dimacs": clause_dimacs,
                "clause_structural": clause_struct,
                "condition_numbers": condition_numbers,
            }
        )

    mapped_condition_numbers.sort()
    unsat_reason = row.get("unsat_reason")

    block = (
        "<certificate_type>\nUNSAT core\n</certificate_type>\n\n"
        "<unsat_core_raw_clause_indices>\n"
        f"{json_dumps_compact(core_indices)}\n"
        "</unsat_core_raw_clause_indices>\n\n"
        "<unsat_core_mapped_condition_numbers>\n"
        f"{json_dumps_compact(mapped_condition_numbers)}\n"
        "</unsat_core_mapped_condition_numbers>\n\n"
        "<unsat_core_clauses>\n"
        f"{chr(10).join(core_lines) if core_lines else '(empty)'}\n"
        "</unsat_core_clauses>\n\n"
        "<unsat_reason>\n"
        f"{unsat_reason or '(not provided)'}\n"
        "</unsat_reason>"
    )

    metadata = {
        "unsat_core_clause_indices": core_indices,
        "mapped_condition_numbers": mapped_condition_numbers,
        "mapping_details": mapping_details,
        "unsat_reason": unsat_reason,
    }
    return block, metadata


def render_certificate(row: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Render either SAT or UNSAT certificate, based on row fields."""
    cert_type = row.get("certificate_type")
    satisfiable = parse_bool(row.get("satisfiable"))

    if cert_type == "sat_assignment" or satisfiable:
        return render_sat_certificate(row)
    if cert_type == "unsat_core" or not satisfiable:
        return render_unsat_certificate(row)

    block = (
        "<certificate_type>\nUNKNOWN\n</certificate_type>\n\n"
        "No recognized certificate fields were found in this row."
    )
    return block, {}


# ---------------------------------------------------------------------------
# Prompt construction and row validation
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = [
    "dims",
    "num_vars",
    "num_clauses",
    "clauses",
    "readable",
    "satisfiable",
    "scenario",
    "variable_mapping",
    "conditions",
    "question",
]


def validate_row(row: Dict[str, Any], index: int) -> List[str]:
    """Return a list of validation warnings for one row."""
    warnings: List[str] = []

    for field in REQUIRED_FIELDS:
        if field not in row:
            warnings.append(f"row {index}: missing required field {field!r}")

    dims = row.get("dims") or []
    num_vars = row.get("num_vars")
    if isinstance(dims, list) and num_vars is not None:
        try:
            expected_num_vars = product([int(d) for d in dims])
            if expected_num_vars != int(num_vars):
                warnings.append(
                    f"row {index}: product(dims)={expected_num_vars} but num_vars={num_vars}"
                )
        except Exception as e:
            warnings.append(f"row {index}: could not validate dims/num_vars: {e}")

    clauses = row.get("clauses") or []
    num_clauses = row.get("num_clauses")
    if num_clauses is not None and len(clauses) != int(num_clauses):
        warnings.append(
            f"row {index}: len(clauses)={len(clauses)} but num_clauses={num_clauses}"
        )

    satisfiable = parse_bool(row.get("satisfiable"))
    if satisfiable and not row.get("sat_assignment"):
        warnings.append(f"row {index}: satisfiable=True but sat_assignment is empty/missing")
    if not satisfiable and row.get("unsat_core_clause_indices") is None:
        warnings.append(f"row {index}: satisfiable=False but unsat_core_clause_indices is missing")

    return warnings


def format_conditions(conditions: Any) -> str:
    """Format conditions as a newline-separated block.

    The SATBench examples usually already include numbering, e.g. "1. ...".
    If a condition is unnumbered, we add numbering for clarity.
    """
    if not conditions:
        return "(none)"
    if isinstance(conditions, str):
        return conditions

    lines: List[str] = []
    for i, cond in enumerate(conditions, start=1):
        cond_s = str(cond).strip()
        if re.match(r"^\d+\.", cond_s):
            lines.append(cond_s)
        else:
            lines.append(f"{i}. {cond_s}")
    return "\n".join(lines)


def build_prompt_record(row: Dict[str, Any], index: int, include_original_row: bool) -> Dict[str, Any]:
    """Build one output JSON object containing the teacher prompt."""
    satisfiable_bool = parse_bool(row.get("satisfiable"))
    label = "SAT" if satisfiable_bool else "UNSAT"
    certificate_block, certificate_metadata = render_certificate(row)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        scenario=row.get("scenario", ""),
        variable_mapping=row.get("variable_mapping", ""),
        conditions_block=format_conditions(row.get("conditions")),
        question=row.get("question", ""),
        dims=json_dumps_compact(row.get("dims", [])),
        num_vars=row.get("num_vars", ""),
        num_clauses=row.get("num_clauses", ""),
        clauses_json=json_dumps_pretty(row.get("clauses", [])),
        readable=row.get("readable", ""),
        satisfiable_bool="true" if satisfiable_bool else "false",
        label=label,
        certificate_block=certificate_block,
    )

    record: Dict[str, Any] = {
        "index": index,
        "prompt_id": f"satbench_{index:05d}",
        "label": label,
        "satisfiable": satisfiable_bool,
        "certificate_type": row.get("certificate_type"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "ground_truth": {
            "dims": row.get("dims"),
            "num_vars": row.get("num_vars"),
            "num_clauses": row.get("num_clauses"),
            "satisfiable": satisfiable_bool,
            "label": label,
            "certificate_type": row.get("certificate_type"),
            **certificate_metadata,
        },
    }

    if include_original_row:
        record["satbench_row"] = row

    return record


def write_jsonl(records: Iterable[Dict[str, Any]], output_path: Path) -> int:
    """Write records to JSONL and return the number written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output_path.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SATBench teacher prompts from the enriched JSONL file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("satbench_with_certificates_full.jsonl"),
        help="Local enriched SATBench JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("satbench_teacher_prompts.jsonl"),
        help="Output JSONL file containing assembled teacher prompts.",
    )
    parser.add_argument(
        "--expected_rows",
        type=int,
        default=2100,
        help="Expected row count. Set to 0 to disable this check.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Inclusive start row index. Usually leave at 0 for all prompts.",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="Exclusive end row index. Default: end of dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows to write after start/end filtering.",
    )
    parser.add_argument(
        "--include_original_row",
        action="store_true",
        help="Include the full original SATBench row in each output record.",
    )
    parser.add_argument(
        "--print_first",
        action="store_true",
        help="Print the first generated user prompt to stdout for inspection.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat validation warnings as fatal errors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {args.dataset}\n"
            "Place satbench_with_certificates_full.jsonl in the current directory "
            "or pass --dataset /path/to/file.jsonl."
        )

    rows = load_jsonl(args.dataset)
    if args.expected_rows and len(rows) != args.expected_rows:
        msg = f"Expected {args.expected_rows} rows, but loaded {len(rows)} rows from {args.dataset}"
        if args.strict:
            raise ValueError(msg)
        print(f"[warn] {msg}", file=sys.stderr)

    # Validate all loaded rows before writing so we catch field mismatches early.
    warnings: List[str] = []
    for i, row in enumerate(rows):
        warnings.extend(validate_row(row, i))

    if warnings:
        for w in warnings[:50]:
            print(f"[warn] {w}", file=sys.stderr)
        if len(warnings) > 50:
            print(f"[warn] ... {len(warnings) - 50} more warnings", file=sys.stderr)
        if args.strict:
            raise ValueError(f"Validation failed with {len(warnings)} warning(s); refusing to write output.")

    start = max(0, args.start_idx)
    end = len(rows) if args.end_idx is None else min(len(rows), args.end_idx)
    if end < start:
        raise ValueError(f"Invalid range: start_idx={start}, end_idx={end}")

    selected = list(enumerate(rows[start:end], start=start))
    if args.limit is not None:
        selected = selected[: max(0, args.limit)]

    records = [
        build_prompt_record(row=row, index=i, include_original_row=args.include_original_row)
        for i, row in selected
    ]

    if args.print_first and records:
        print("\n" + "=" * 80)
        print(f"First generated prompt: {records[0]['prompt_id']}")
        print("=" * 80)
        print(records[0]["user_prompt"])
        print("=" * 80 + "\n")

    n_written = write_jsonl(records, args.output)
    print(f"[done] loaded_rows={len(rows)} written_prompts={n_written} output={args.output}")
    if warnings:
        print(f"[done] validation_warnings={len(warnings)}; inspect stderr above", file=sys.stderr)


if __name__ == "__main__":
    main()
