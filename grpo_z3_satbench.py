#!/usr/bin/env python3
"""
Small GRPO fine-tuning experiment for SATBench with Z3-based rewards.

Recommended first run:
python grpo_z3_satbench.py \
  --model_dir results/sft_qwen35_08b_base_sat_unsat_full/final_model \
  --record_dir teacher_responses_qwen35_2b/records \
  --output_dir results/grpo_z3_qwen35_08b_test \
  --n_per_label 8 \
  --max_steps 5 \
  --num_generations 2 \
  --per_device_train_batch_size 2 \
  --max_completion_length 512

After the smoke test works, increase:
  --n_per_label 50
  --max_steps 50
  --max_completion_length 1024
"""

import argparse
import ast
import inspect
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from z3 import Bool, Not, Or, Solver, unsat

from trl import GRPOConfig, GRPOTrainer


# ============================================================
# 1. Two-shot no-leak system prompt
# ============================================================

TWO_SHOT_SYSTEM_PROMPT = """You are a logical reasoning assistant solving SATBench-style natural-language logic puzzles.

Important rules:
- Use only the constraints stated in the <conditions> section.
- The <scenario> section is background only and adds no hidden constraints.
- All variables are independent Boolean decisions unless the conditions explicitly say otherwise.
- Do not add commonsense assumptions such as mutual exclusivity, exactly-one constraints, or real-world causal links.
- Variables not mentioned in the conditions are irrelevant to satisfiability.

Required output format:

<think>
Write concise reasoning here.
</think>

Decision: SAT or UNSAT

Certificate:
- If SAT, write exactly:
  Assignment: <nested 0/1 list matching dims>

  The assignment must contain exactly num_vars values in row-major order according to dims.
  Use 1 for True and 0 for False.
  Do not use ellipses, omitted entries, variable names, or prose inside the assignment list.

- If UNSAT, write exactly:
  UNSAT core condition numbers: <list of 1-indexed condition numbers>

  The condition numbers must refer to the numbered conditions in the <conditions> section.
  The list does not need to be minimal, but the selected conditions must be jointly unsatisfiable.
  Do not use 0-indexed clause indices.

Explanation:
Write a short explanation connecting the certificate to the conditions.

End with exactly one final label on its own line:
[SAT]
or
[UNSAT]

Do not write anything after the final label.

Example 1: SAT output format

Input facts:
<dims>
[2]
</dims>

<num_vars>
2
</num_vars>

<conditions>
1. Alice joins the club.
2. Bob does not join the club.
</conditions>

Correct output:
<think>
Condition 1 requires x(0) to be true. Condition 2 requires x(1) to be false. These requirements do not conflict, so the puzzle is satisfiable.
</think>

Decision: SAT

Certificate:
Assignment: [1, 0]

Explanation:
The assignment sets Alice to True and Bob to False, satisfying both conditions.

[SAT]

Example 2: UNSAT output format

Input facts:
<dims>
[2]
</dims>

<num_vars>
2
</num_vars>

<conditions>
1. Alice joins the club.
2. Alice does not join the club.
</conditions>

Correct output:
<think>
Condition 1 requires x(0) to be true, while condition 2 requires x(0) to be false. The same variable cannot be both true and false, so these conditions are jointly unsatisfiable.
</think>

Decision: UNSAT

Certificate:
UNSAT core condition numbers: [1, 2]

Explanation:
Conditions 1 and 2 directly contradict each other because they require opposite truth values for Alice's decision.

[UNSAT]

Now solve the actual puzzle below. Do not copy the examples. Use the same output format.
"""

SHORT_SYSTEM_PROMPT = """You are a logical reasoning assistant solving SATBench-style natural-language logic puzzles.

Rules:
- Use only the constraints in the <conditions> section.
- The <scenario> section is background only and adds no hidden constraints.
- All variables are independent Boolean decisions unless the conditions explicitly say otherwise.
- Do not add commonsense assumptions such as mutual exclusivity, exactly-one constraints, or real-world causal links.
- Variables not mentioned in the conditions are irrelevant to satisfiability.

Required output format:

<think>
Write concise reasoning here.
</think>

Decision: SAT or UNSAT

Certificate:
If SAT, write exactly:
Assignment: <nested 0/1 list matching dims>

If UNSAT, write exactly:
UNSAT core condition numbers: <list of 1-indexed condition numbers>

Explanation:
Write a short explanation connecting the certificate to the conditions.

Final line:
[SAT] or [UNSAT]

Do not write anything after the final label.
"""

# ============================================================
# 2. Basic parsing helpers
# ============================================================

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def extract_tag(text: str, tag: str) -> Optional[str]:
    m = re.search(
        rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else None


def extract_first_available_tag(text: str, tags: List[str]) -> Optional[str]:
    for tag in tags:
        value = extract_tag(text, tag)
        if value is not None:
            return value
    return None


def parse_literal_object(text: Optional[str]):
    if text is None:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def derive_label_from_record(rec: Dict[str, Any]) -> Optional[str]:
    gt = rec.get("ground_truth") or {}

    label = (
        gt.get("label")
        or rec.get("ground_truth_label")
        or rec.get("label")
        or rec.get("satisfiability_label")
    )

    if isinstance(label, str):
        label = label.upper().strip()
        if label in {"SAT", "UNSAT"}:
            return label

    satisfiable = gt.get("satisfiable", rec.get("satisfiable", None))
    if isinstance(satisfiable, bool):
        return "SAT" if satisfiable else "UNSAT"

    return None


def product(xs: List[int]) -> int:
    out = 1
    for x in xs:
        out *= int(x)
    return out


def struct_indices_to_dimacs(indices: List[int], dims: List[int]) -> Optional[int]:
    """
    Convert x(i,j,k) structural indices to 1-indexed DIMACS variable id.
    This follows row-major order, matching validate_z3_certificates.py.
    """
    if len(indices) != len(dims):
        return None

    flat = 0
    for axis, idx in enumerate(indices):
        if idx < 0 or idx >= dims[axis]:
            return None
        stride = product(dims[axis + 1:]) if axis + 1 < len(dims) else 1
        flat += idx * stride

    return flat + 1


def parse_readable_cnf(readable: Optional[str], dims: List[int]) -> List[List[int]]:
    """
    Parse readable CNF in the order shown to the model, for example:
      (¬x(1, 2) ∨ x(1, 1)) ∧ (x(2, 4) ∨ ¬x(0, 0))

    This is important because UNSAT core condition numbers are 1-indexed
    with respect to the natural-language conditions / readable CNF order.
    """
    if not readable:
        return []

    clause_texts = re.split(r"\s*∧\s*", readable.strip())
    clauses = []

    for clause_text in clause_texts:
        clause_text = clause_text.strip()

        if clause_text.startswith("(") and clause_text.endswith(")"):
            clause_text = clause_text[1:-1]

        lits = []

        for m in re.finditer(r"(¬|-|~)?\s*x\s*\(([^)]*)\)", clause_text):
            neg = bool(m.group(1))
            idx_text = m.group(2)
            parts = [p.strip() for p in idx_text.split(",") if p.strip() != ""]

            try:
                indices = [int(p) for p in parts]
            except Exception:
                continue

            var_id = struct_indices_to_dimacs(indices, dims)
            if var_id is None:
                continue

            lits.append(-var_id if neg else var_id)

        if lits:
            clauses.append(lits)

    return clauses


# ============================================================
# 3. Prompt construction
# ============================================================

def build_no_leak_prompt_from_record(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Build a no-leak RL prompt from teacher response records.

    We keep puzzle/formula fields:
      scenario, variable_mapping, conditions, question,
      dims, num_vars, num_clauses, clauses_dimacs, readable_cnf

    We remove leaked fields:
      satisfiable, label, SAT assignment, UNSAT core, Z3 answer, teacher answer
    """
    user_prompt = rec.get("user_prompt", "")
    gt = rec.get("ground_truth") or {}

    if not user_prompt:
        return None

    scenario = extract_tag(user_prompt, "scenario") or gt.get("scenario")
    variable_mapping = extract_tag(user_prompt, "variable_mapping") or gt.get("variable_mapping")
    conditions = extract_tag(user_prompt, "conditions") or gt.get("conditions")
    question = extract_tag(user_prompt, "question") or gt.get("question")

    dims_text = extract_tag(user_prompt, "dims")
    num_vars_text = extract_tag(user_prompt, "num_vars")
    num_clauses_text = extract_tag(user_prompt, "num_clauses")
    clauses_text = extract_first_available_tag(user_prompt, ["clauses_dimacs", "clauses"])
    readable_cnf = extract_first_available_tag(user_prompt, ["readable_cnf", "readable"])

    dims = gt.get("dims") or parse_literal_object(dims_text)
    clauses = gt.get("clauses") or parse_literal_object(clauses_text)

    num_vars = gt.get("num_vars")
    if num_vars is None and num_vars_text is not None:
        try:
            num_vars = int(num_vars_text)
        except Exception:
            num_vars = None
    if num_vars is None and dims is not None:
        num_vars = product([int(x) for x in dims])

    num_clauses = gt.get("num_clauses")
    if num_clauses is None and num_clauses_text is not None:
        try:
            num_clauses = int(num_clauses_text)
        except Exception:
            num_clauses = None
    if num_clauses is None and clauses is not None:
        num_clauses = len(clauses)

    required = [
        scenario,
        variable_mapping,
        conditions,
        question,
        dims,
        num_vars,
        num_clauses,
        clauses,
        readable_cnf,
    ]

    if any(x is None for x in required):
        return None

    dims = [int(x) for x in dims]
    clauses = [[int(lit) for lit in clause] for clause in clauses]

    # Normalize conditions as numbered text.
    if isinstance(conditions, list):
        conditions_text = "\n".join(str(c) for c in conditions)
    else:
        conditions_text = str(conditions)

    user = f"""## Puzzle

<scenario>
{scenario}
</scenario>

<variable_mapping>
{variable_mapping}
</variable_mapping>

<conditions>
{conditions_text}
</conditions>

<question>
{question}
</question>

## Formal SAT information

<dims>
{json.dumps(dims)}
</dims>

<num_vars>
{int(num_vars)}
</num_vars>

<num_clauses>
{int(num_clauses)}
</num_clauses>

<clauses_dimacs>
{json.dumps(clauses)}
</clauses_dimacs>

<readable_cnf>
{readable_cnf}
</readable_cnf>

## Instruction

Solve the puzzle. Produce a SAT/UNSAT decision and a verifiable certificate in the required format.
"""

    prompt = f"{SHORT_SYSTEM_PROMPT}\n\n{user}\n\nAssistant:\n"

    return {
        "prompt": prompt,
        "dims": dims,
        "num_vars": int(num_vars),
        "num_clauses": int(num_clauses),
        "clauses": clauses,
        "readable_cnf": readable_cnf,
    }


def condition_to_raw_clause_map(gt: Dict[str, Any]) -> Dict[str, int]:
    """
    Optional mapping from 1-indexed condition numbers to raw 0-indexed clause indices.
    Used as a fallback if readable_cnf parsing is unavailable.
    """
    out = {}

    for item in gt.get("mapping_details", []) or []:
        raw_idx = item.get("raw_clause_index")
        if raw_idx is None:
            continue

        for cnum in item.get("condition_numbers", []) or []:
            out[str(int(cnum))] = int(raw_idx)

    return out


def get_used_prompt_ids_from_sft_splits(paths: List[Path]) -> set:
    used = set()

    for p in paths:
        for row in load_jsonl(p):
            for key in ["prompt_id", "dataset_id", "row_id", "row_index"]:
                if row.get(key) is not None:
                    used.add(str(row[key]))

    return used


def build_train_dataset(args) -> Dataset:
    used_prompt_ids = set()

    if args.exclude_sft_splits:
        used_prompt_ids = get_used_prompt_ids_from_sft_splits([
            Path("dataset/sft_by_label/sat/train.jsonl"),
            Path("dataset/sft_by_label/sat/test.jsonl"),
            Path("dataset/sft_by_label/unsat/train.jsonl"),
            Path("dataset/sft_by_label/unsat/test.jsonl"),
        ])

    print(f"Used IDs found from SFT splits: {len(used_prompt_ids)}")

    record_paths = sorted(args.record_dir.glob("row_*.json"))
    print(f"Found record files: {len(record_paths)}")

    records = []

    for p in tqdm(record_paths, desc="Loading teacher record files"):
        try:
            rec = load_json(p)
        except Exception as e:
            print(f"Skipping bad JSON: {p} ({e})")
            continue

        if args.require_teacher_correct:
            if rec.get("teacher_label_matches_ground_truth") is not True:
                continue
            if rec.get("generation_info", {}).get("likely_truncated_by_max_new_tokens") is True:
                continue

        prompt_id = str(rec.get("prompt_id", ""))
        row_index = rec.get("row_index_in_prompt_file", rec.get("row_index", rec.get("index", -1)))

        if args.exclude_sft_splits:
            possible_ids = {str(prompt_id), str(row_index)}
            if any(x in used_prompt_ids for x in possible_ids if x not in {"", "-1", "None"}):
                continue

        label = derive_label_from_record(rec)
        if label not in {"SAT", "UNSAT"}:
            continue

        built = build_no_leak_prompt_from_record(rec)
        if built is None:
            continue

        gt = rec.get("ground_truth") or {}

        records.append({
            "prompt": built["prompt"],
            "prompt_id": prompt_id,
            "row_index": int(row_index) if str(row_index).lstrip("-").isdigit() else -1,
            "label": label,

            # Hidden fields for reward functions.
            "dims_json": json.dumps(built["dims"]),
            "num_vars": built["num_vars"],
            "num_clauses": built["num_clauses"],
            "clauses_json": json.dumps(built["clauses"]),
            "readable_cnf": built["readable_cnf"],
            "condition_to_raw_json": json.dumps(condition_to_raw_clause_map(gt)),
        })

    print(f"Usable records after filtering: {len(records)}")

    sat_rows = [r for r in records if r["label"] == "SAT"]
    unsat_rows = [r for r in records if r["label"] == "UNSAT"]

    random.shuffle(sat_rows)
    random.shuffle(unsat_rows)

    selected = sat_rows[:args.n_per_label] + unsat_rows[:args.n_per_label]
    random.shuffle(selected)

    print(f"Selected SAT rows: {sum(r['label'] == 'SAT' for r in selected)}")
    print(f"Selected UNSAT rows: {sum(r['label'] == 'UNSAT' for r in selected)}")
    print(f"Selected total rows: {len(selected)}")

    if len(selected) == 0:
        raise ValueError("No RL training examples selected. Check --record_dir and filtering options.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_path = args.output_dir / "selected_grpo_train_rows.jsonl"
    with selected_path.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved selected training rows to: {selected_path}")

    return Dataset.from_list(selected)


# ============================================================
# 4. Completion parsing helpers
# ============================================================

def completion_to_text(completion) -> str:
    """
    TRL may pass completions as strings or as chat-style message dictionaries.
    """
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        if len(completion) > 0 and isinstance(completion[0], dict):
            return completion[0].get("content", "")

    return str(completion)


def extract_final_label(text: str) -> Optional[str]:
    # Prefer the final bracketed label.
    labels = re.findall(r"\[(SAT|UNSAT)\]", text, flags=re.IGNORECASE)
    if labels:
        return labels[-1].upper()

    # Fallback to a decision line.
    m = re.search(r"\b(?:Decision|Label)\s*:\s*(SAT|UNSAT)\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


def extract_balanced_bracket(text: str, start_idx: int) -> Optional[str]:
    i = text.find("[", start_idx)
    if i < 0:
        return None

    depth = 0
    for j in range(i, len(text)):
        if text[j] == "[":
            depth += 1
        elif text[j] == "]":
            depth -= 1
            if depth == 0:
                return text[i:j + 1]

    return None


def flatten_assignment(obj) -> List[bool]:
    flat = []

    def rec(x):
        if isinstance(x, list):
            for y in x:
                rec(y)
        elif isinstance(x, bool):
            flat.append(bool(x))
        elif isinstance(x, int):
            flat.append(bool(x))
        elif isinstance(x, str):
            t = x.strip().lower()
            if t in {"1", "true", "t", "yes"}:
                flat.append(True)
            elif t in {"0", "false", "f", "no"}:
                flat.append(False)

    rec(obj)
    return flat


def extract_assignment(text: str, num_vars: int) -> Dict[int, bool]:
    """
    Extracts:
      Assignment: [[0, 1], [1, 0]]
    and returns:
      {1: False, 2: True, 3: True, 4: False}
    """
    for m in re.finditer(r"Assignment\s*:", text, flags=re.IGNORECASE):
        block = extract_balanced_bracket(text, m.end())
        if not block:
            continue

        try:
            obj = ast.literal_eval(block)
        except Exception:
            try:
                obj = json.loads(block)
            except Exception:
                continue

        flat = flatten_assignment(obj)

        if flat:
            return {i + 1: bool(v) for i, v in enumerate(flat[:num_vars])}

    return {}


def extract_unsat_core_condition_numbers(text: str) -> List[int]:
    """
    Preferred:
      UNSAT core condition numbers: [1, 3, 4]
    """
    patterns = [
        r"UNSAT\s+core\s+condition\s+numbers\s*:\s*\[([^\]]+)\]",
        r"UNSAT\s+core\s+conditions?\s*:\s*\[([^\]]+)\]",
        r"core\s+condition\s+numbers\s*:\s*\[([^\]]+)\]",
        r"conflicting\s+conditions?\s*:\s*\[([^\]]+)\]",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
            if nums:
                return nums

    # Fallback: prose like "conditions 1, 3, and 4".
    m = re.search(r"conditions?\s+((?:\d+\s*(?:,|and|&)?\s*){1,20})", text, flags=re.IGNORECASE)
    if m:
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        if nums:
            return nums

    return []


# ============================================================
# 5. Z3 validation helpers
# ============================================================

def z3_vars(num_vars: int):
    return {i: Bool(f"x{i}") for i in range(1, num_vars + 1)}


def z3_clause_expr(clause: List[int], vars_by_id):
    exprs = []

    for lit in clause:
        lit = int(lit)
        v = vars_by_id[abs(lit)]
        exprs.append(v if lit > 0 else Not(v))

    return Or(*exprs)


def check_assignment_satisfies_all_clauses(
    clauses: List[List[int]],
    assignment: Dict[int, bool],
) -> bool:
    if not assignment:
        return False

    for clause in clauses:
        clause_ok = False

        for lit in clause:
            var_id = abs(int(lit))
            if var_id not in assignment:
                continue

            value = assignment[var_id]
            literal_value = value if lit > 0 else (not value)

            if literal_value:
                clause_ok = True
                break

        if not clause_ok:
            return False

    return True


def check_unsat_core_with_z3(
    clauses: List[List[int]],
    num_vars: int,
    indices: List[int],
) -> bool:
    if not indices:
        return False

    if any(i < 0 or i >= len(clauses) for i in indices):
        return False

    vars_by_id = z3_vars(num_vars)
    solver = Solver()

    for idx in indices:
        solver.add(z3_clause_expr(clauses[idx], vars_by_id))

    return solver.check() == unsat


def condition_numbers_to_core_indices(
    condition_numbers: List[int],
    dims: List[int],
    raw_clauses: List[List[int]],
    readable_cnf: str,
    condition_to_raw: Dict[str, int],
) -> Tuple[List[List[int]], List[int], str]:
    """
    Convert 1-indexed condition numbers into the clause subset to check.

    Preferred basis:
      readable_cnf order, because condition numbers correspond to the natural-language
      conditions / readable CNF order. This matches validate_z3_certificates.py.

    Fallback:
      condition_to_raw mapping, then raw clauses with 1-index assumption.
    """
    readable_clauses = parse_readable_cnf(readable_cnf, dims)

    if readable_clauses:
        return readable_clauses, [n - 1 for n in condition_numbers], "readable_condition_1_based"

    if condition_to_raw:
        indices = []
        for n in condition_numbers:
            if str(n) in condition_to_raw:
                indices.append(int(condition_to_raw[str(n)]))
        return raw_clauses, indices, "condition_to_raw_mapping"

    return raw_clauses, [n - 1 for n in condition_numbers], "raw_1_based_fallback"


# ============================================================
# 6. GRPO reward functions
# ============================================================

def format_reward_func(completions, **kwargs):
    """
    Reward parseable output format. Max reward = 1.0.
    """
    rewards = []

    for completion in completions:
        text = completion_to_text(completion).strip()
        r = 0.0

        if re.search(r"\[(SAT|UNSAT)\]\s*$", text, flags=re.IGNORECASE):
            r += 0.3

        if re.search(r"\bDecision\s*:\s*(SAT|UNSAT)\b", text, flags=re.IGNORECASE):
            r += 0.2

        if re.search(r"\bCertificate\s*:", text, flags=re.IGNORECASE):
            r += 0.2

        if re.search(r"\bAssignment\s*:", text, flags=re.IGNORECASE):
            r += 0.3
        elif re.search(r"\bUNSAT\s+core\s+condition\s+numbers\s*:", text, flags=re.IGNORECASE):
            r += 0.3
        elif re.search(r"\bUNSAT\s+core\b", text, flags=re.IGNORECASE):
            r += 0.15

        rewards.append(r)

    return rewards


def label_reward_func(completions, label, **kwargs):
    """
    Reward correct SAT/UNSAT final label.
    """
    rewards = []

    for completion, gold in zip(completions, label):
        text = completion_to_text(completion)
        pred = extract_final_label(text)

        if pred == gold:
            rewards.append(1.0)
        elif pred is None:
            rewards.append(-0.25)
        else:
            rewards.append(-0.5)

    return rewards


def z3_certificate_reward_func(
    completions,
    label,
    clauses_json,
    dims_json,
    num_vars,
    readable_cnf,
    condition_to_raw_json,
    **kwargs,
):
    """
    Reward solver-verifiable certificates.

    SAT:
      +2.0 if Assignment: [...] satisfies every CNF clause.

    UNSAT:
      +2.0 if UNSAT core condition numbers select a subset that Z3 proves UNSAT.
    """
    rewards = []

    for completion, gold, clauses_s, dims_s, nvars, readable, cond_map_s in zip(
        completions,
        label,
        clauses_json,
        dims_json,
        num_vars,
        readable_cnf,
        condition_to_raw_json,
    ):
        text = completion_to_text(completion)
        pred = extract_final_label(text)

        raw_clauses = json.loads(clauses_s)
        dims = json.loads(dims_s)
        nvars = int(nvars)
        condition_to_raw = json.loads(cond_map_s) if cond_map_s else {}

        # No certificate reward if final prediction is wrong.
        if pred != gold:
            rewards.append(0.0)
            continue

        if gold == "SAT":
            assignment = extract_assignment(text, nvars)
            valid = check_assignment_satisfies_all_clauses(raw_clauses, assignment)
            rewards.append(2.0 if valid else 0.0)

        elif gold == "UNSAT":
            condition_numbers = extract_unsat_core_condition_numbers(text)

            core_clauses, indices, _basis = condition_numbers_to_core_indices(
                condition_numbers=condition_numbers,
                dims=dims,
                raw_clauses=raw_clauses,
                readable_cnf=readable,
                condition_to_raw=condition_to_raw,
            )

            valid = check_unsat_core_with_z3(core_clauses, nvars, sorted(set(indices)))
            rewards.append(2.0 if valid else 0.0)

        else:
            rewards.append(0.0)

    return rewards


# ============================================================
# 7. Debug reward sanity checks
# ============================================================

def run_reward_sanity_checks(train_dataset: Dataset):
    print("\nRunning reward sanity checks...")

    print("\nFirst training example:")
    sample0 = train_dataset[0]
    print("label:", sample0["label"])
    print("num_vars:", sample0["num_vars"])
    print("num_clauses:", sample0["num_clauses"])

    # If there is an UNSAT sample, the full list of all conditions should be a valid UNSAT core.
    unsat_sample = None
    for row in train_dataset:
        if row["label"] == "UNSAT":
            unsat_sample = row
            break

    if unsat_sample is not None:
        full_core = list(range(1, int(unsat_sample["num_clauses"]) + 1))
        fake_unsat_completion = f"""
<think>
I will use all conditions as the UNSAT core.
</think>

Decision: UNSAT

Certificate:
UNSAT core condition numbers: {full_core}

Explanation:
The selected conditions are jointly unsatisfiable.

[UNSAT]
"""
        print("\nUNSAT sanity sample:")
        print("prompt_id:", unsat_sample.get("prompt_id"))
        print("full_core:", full_core)
        print("format reward:", format_reward_func([fake_unsat_completion]))
        print("label reward:", label_reward_func([fake_unsat_completion], [unsat_sample["label"]]))
        print(
            "z3 reward:",
            z3_certificate_reward_func(
                [fake_unsat_completion],
                [unsat_sample["label"]],
                [unsat_sample["clauses_json"]],
                [unsat_sample["dims_json"]],
                [unsat_sample["num_vars"]],
                [unsat_sample["readable_cnf"]],
                [unsat_sample["condition_to_raw_json"]],
            ),
        )

    # If there is a SAT sample, we do not know a satisfying assignment here,
    # so we only check that the functions run without crashing.
    sat_sample = None
    for row in train_dataset:
        if row["label"] == "SAT":
            sat_sample = row
            break

    if sat_sample is not None:
        fake_sat_completion = """
<think>
This is only a parser sanity check.
</think>

Decision: SAT

Certificate:
Assignment: [0, 0]

Explanation:
This fake assignment may not be valid.

[SAT]
"""
        print("\nSAT parser sanity sample:")
        print("prompt_id:", sat_sample.get("prompt_id"))
        print("format reward:", format_reward_func([fake_sat_completion]))
        print("label reward:", label_reward_func([fake_sat_completion], [sat_sample["label"]]))
        print(
            "z3 reward:",
            z3_certificate_reward_func(
                [fake_sat_completion],
                [sat_sample["label"]],
                [sat_sample["clauses_json"]],
                [sat_sample["dims_json"]],
                [sat_sample["num_vars"]],
                [sat_sample["readable_cnf"]],
                [sat_sample["condition_to_raw_json"]],
            ),
        )


# ============================================================
# 8. TRL compatibility helpers
# ============================================================

def make_grpo_config(**kwargs) -> GRPOConfig:
    """
    Make the script more robust across TRL versions by dropping unsupported
    GRPOConfig arguments.
    """
    valid = set(inspect.signature(GRPOConfig.__init__).parameters.keys())

    filtered = {}
    dropped = {}

    for k, v in kwargs.items():
        if k in valid:
            filtered[k] = v
        else:
            dropped[k] = v

    if dropped:
        print("\nDropped unsupported GRPOConfig keys for this TRL version:")
        for k in sorted(dropped):
            print("  -", k)

    return GRPOConfig(**filtered)


def build_trainer(model, tokenizer, training_args, train_dataset):
    """
    TRL changed tokenizer/processing_class naming across versions.
    """
    sig = inspect.signature(GRPOTrainer.__init__)
    kwargs = {
        "model": model,
        "args": training_args,
        "reward_funcs": [
            format_reward_func,
            label_reward_func,
            z3_certificate_reward_func,
        ],
        "train_dataset": train_dataset,
    }

    if "processing_class" in sig.parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig.parameters:
        kwargs["tokenizer"] = tokenizer
    else:
        print("Warning: this GRPOTrainer signature has neither processing_class nor tokenizer.")

    return GRPOTrainer(**kwargs)


# ============================================================
# 9. Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("results/sft_qwen35_08b_base_sat_unsat_full/final_model"),
        help="Path to the SFT fine-tuned model. RL should start from the SFT model.",
    )
    parser.add_argument(
        "--record_dir",
        type=Path,
        default=Path("teacher_responses_qwen35_2b/records"),
        help="Folder containing teacher_responses_qwen35_2b/records/row_*.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/grpo_z3_qwen35_08b_test"),
    )

    parser.add_argument("--n_per_label", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=2)

    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_completion_length", type=int, default=512)

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=2)

    parser.add_argument(
        "--require_teacher_correct",
        action="store_true",
        help="Only use records where teacher_label_matches_ground_truth is true and output is not truncated.",
    )
    parser.add_argument(
        "--no_exclude_sft_splits",
        action="store_true",
        help="Do not attempt to exclude examples already used in SFT train/test splits.",
    )
    parser.add_argument(
        "--no_reward_sanity_check",
        action="store_true",
        help="Skip reward sanity checks before training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint inside output_dir if available.",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Build the dataset and run sanity checks, but do not train.",
    )

    args = parser.parse_args()

    args.exclude_sft_splits = not args.no_exclude_sft_splits

    if args.save_steps is None:
        args.save_steps = max(1, args.max_steps)

    return args


def main():
    args = parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 100)
    print("GRPO + Z3 SATBench fine-tuning")
    print("=" * 100)
    print("model_dir:", args.model_dir)
    print("record_dir:", args.record_dir)
    print("output_dir:", args.output_dir)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("bf16 supported:", torch.cuda.is_bf16_supported())

    if not args.model_dir.exists():
        raise FileNotFoundError(f"Cannot find model_dir: {args.model_dir}")

    if not args.record_dir.exists():
        raise FileNotFoundError(f"Cannot find record_dir: {args.record_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save run config.
    with (args.output_dir / "grpo_run_config.json").open("w", encoding="utf-8") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    # Build dataset.
    train_dataset = build_train_dataset(args)
    print(train_dataset)
    print("Dataset columns:", train_dataset.column_names)

    print("\nPrompt preview:")
    print(train_dataset[0]["prompt"][:2500])

    if not args.no_reward_sanity_check:
        run_reward_sanity_checks(train_dataset)

    if args.skip_train:
        print("\n--skip_train was set. Exiting before training.")
        return

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir),
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    print("\nTokenizer loaded.")
    print("pad_token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("eos_token:", tokenizer.eos_token, tokenizer.eos_token_id)

    # Load model.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32
    print("Loading model with dtype:", dtype)

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    if hasattr(model, "config"):
        model.config.use_cache = False

    # GRPO requires the effective batch size to be divisible by num_generations.
    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if effective_batch_size % args.num_generations != 0:
        raise ValueError(
            f"Effective batch size must be divisible by num_generations. "
            f"Got per_device_train_batch_size={args.per_device_train_batch_size}, "
            f"gradient_accumulation_steps={args.gradient_accumulation_steps}, "
            f"effective_batch_size={effective_batch_size}, "
            f"num_generations={args.num_generations}."
        )

    config_kwargs = dict(
        output_dir=str(args.output_dir),

        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,

        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,

        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,

        learning_rate=args.learning_rate,
        warmup_ratio=0.0,
        weight_decay=0.0,
        max_grad_norm=1.0,
        beta=args.beta,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        bf16=use_bf16,
        fp16=use_fp16,

        remove_unused_columns=False,

        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",

        use_vllm=False,

        seed=args.seed,
    )

    # If num_train_epochs is None, avoid passing it on some versions.
    if args.num_train_epochs is None:
        config_kwargs.pop("num_train_epochs", None)

    training_args = make_grpo_config(**config_kwargs)

    print("\nGRPOConfig:")
    print(training_args)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
    )

    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = get_last_checkpoint(str(args.output_dir))
        print("Resume checkpoint:", resume_checkpoint)

    print("\nStarting GRPO training...")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    print("\nTraining result:")
    print(train_result)

    final_dir = args.output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print("\nSaved final GRPO model to:", final_dir)

    metrics = {}
    if hasattr(train_result, "metrics") and train_result.metrics is not None:
        metrics = train_result.metrics

    with (args.output_dir / "train_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved train metrics to:", args.output_dir / "train_metrics.json")
    print("Done.")


if __name__ == "__main__":
    main()