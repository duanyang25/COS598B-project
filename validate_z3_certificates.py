#!/usr/bin/env python3

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from collections import Counter

from tqdm import tqdm
from z3 import Bool, Or, Not, Solver, sat, unsat


# ============================================================
# Basic IO
# ============================================================

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON in {path} line {line_no}: {e}") from e
    return rows


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "dataset_id",
        "row_id",
        "eval_split",
        "gold_label",
        "predicted_label",
        "correct",
        "num_vars",
        "num_clauses",
        "z3_formula_result",
        "certificate_kind",
        "certificate_extracted",
        "certificate_valid",
        "sat_assignment_num_vars",
        "sat_all_clauses_satisfied",
        "sat_assignment_extendable",
        "sat_num_satisfied_clauses",
        "sat_num_undetermined_clauses",
        "sat_num_false_clauses",
        "unsat_core_valid",
        "unsat_core_basis",
        "unsat_core_indices",
        "unsat_core_size",
        "parse_warning",
        "validation_error",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row = dict(row)
            if isinstance(row.get("unsat_core_indices"), list):
                row["unsat_core_indices"] = json.dumps(row["unsat_core_indices"])
            writer.writerow(row)


# ============================================================
# Prompt parsing
# ============================================================

def extract_tag(text, tag):
    m = re.search(
        rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else None


def parse_literal_object(text):
    if text is None:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except Exception as e:
        raise ValueError(f"Could not parse literal object: {text[:200]}") from e


def product(xs):
    out = 1
    for x in xs:
        out *= int(x)
    return out


def struct_indices_to_dimacs(indices, dims):
    """
    Convert structural x(i,j,k) indices to 1-indexed DIMACS variable id.
    SATBench uses row-major layout.
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


def parse_readable_cnf(readable, dims):
    """
    Parse readable CNF such as:
      (¬x(1, 2) ∨ x(1, 1)) ∧ (x(2, 4) ∨ ¬x(0, 0))

    This is useful because the model may cite natural-language condition
    numbers, which follow the readable CNF / conditions order.
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


def parse_prompt_fields(prompt_text):
    dims = parse_literal_object(extract_tag(prompt_text, "dims"))
    clauses = parse_literal_object(extract_tag(prompt_text, "clauses_dimacs"))
    readable_cnf = extract_tag(prompt_text, "readable_cnf")
    num_vars_text = extract_tag(prompt_text, "num_vars")

    if dims is None:
        raise ValueError("Could not find <dims> in prompt_text.")

    if clauses is None:
        raise ValueError("Could not find <clauses_dimacs> in prompt_text.")

    dims = [int(x) for x in dims]
    clauses = [[int(lit) for lit in clause] for clause in clauses]
    num_vars = int(num_vars_text) if num_vars_text is not None else product(dims)

    readable_clauses = parse_readable_cnf(readable_cnf, dims)

    return {
        "dims": dims,
        "num_vars": num_vars,
        "clauses": clauses,
        "readable_clauses": readable_clauses,
        "readable_cnf": readable_cnf,
    }


# ============================================================
# Z3 checks
# ============================================================

def z3_variables(num_vars):
    return {i: Bool(f"x{i}") for i in range(1, num_vars + 1)}


def z3_clause(clause, vars_by_id):
    exprs = []

    for lit in clause:
        v = vars_by_id[abs(lit)]
        exprs.append(v if lit > 0 else Not(v))

    return Or(*exprs)


def check_formula_sat_status(clauses, num_vars):
    vars_by_id = z3_variables(num_vars)
    solver = Solver()

    for clause in clauses:
        solver.add(z3_clause(clause, vars_by_id))

    result = solver.check()

    if result == sat:
        return "SAT"
    if result == unsat:
        return "UNSAT"
    return str(result).upper()


def check_assignment_extendable(clauses, num_vars, assignment):
    """
    Checks whether the extracted assignment is at least consistent with
    some satisfying assignment of the full formula.
    """
    vars_by_id = z3_variables(num_vars)
    solver = Solver()

    for clause in clauses:
        solver.add(z3_clause(clause, vars_by_id))

    for var_id, value in assignment.items():
        if 1 <= var_id <= num_vars:
            solver.add(vars_by_id[var_id] == bool(value))

    return solver.check() == sat


def check_unsat_core(clauses, num_vars, indices):
    """
    Checks whether the selected subset of clauses is itself UNSAT.
    """
    vars_by_id = z3_variables(num_vars)
    solver = Solver()

    for idx in indices:
        if idx < 0 or idx >= len(clauses):
            return False
        solver.add(z3_clause(clauses[idx], vars_by_id))

    return solver.check() == unsat


def evaluate_assignment_on_clauses(clauses, assignment):
    """
    Strict SAT-certificate check.

    A SAT certificate is valid only if the extracted assignment explicitly
    makes every clause true. If a clause has no true literal and contains
    unassigned variables, it is marked undetermined.
    """
    satisfied = 0
    undetermined = 0
    false_count = 0
    false_indices = []
    undetermined_indices = []

    for i, clause in enumerate(clauses):
        clause_satisfied = False
        clause_has_unassigned = False

        for lit in clause:
            var_id = abs(lit)

            if var_id not in assignment:
                clause_has_unassigned = True
                continue

            value = assignment[var_id]
            literal_value = value if lit > 0 else (not value)

            if literal_value:
                clause_satisfied = True
                break

        if clause_satisfied:
            satisfied += 1
        elif clause_has_unassigned:
            undetermined += 1
            undetermined_indices.append(i)
        else:
            false_count += 1
            false_indices.append(i)

    all_satisfied = false_count == 0 and undetermined == 0

    return {
        "all_satisfied": all_satisfied,
        "satisfied": satisfied,
        "undetermined": undetermined,
        "false_count": false_count,
        "false_indices": false_indices,
        "undetermined_indices": undetermined_indices,
    }


# ============================================================
# Extract SAT assignment from model response
# ============================================================

def flatten_assignment(obj):
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
            if t in {"true", "t", "yes", "1"}:
                flat.append(True)
            elif t in {"false", "f", "no", "0"}:
                flat.append(False)

    rec(obj)
    return flat


def extract_balanced_bracket(text, start):
    i = text.find("[", start)
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


def extract_structured_assignment(text, num_vars):
    """
    Looks for:
      Assignment: [[0, 1], [1, 0]]
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


def extract_variable_assignments(text, dims, num_vars):
    """
    Extracts assignments like:
      x(1, 2) = True
      x(1, 2) = \\text{False}
      x7 = 1
    """
    assignment = {}

    bool_pat = r"(?:\\text\{\s*)?(true|false|1|0|yes|no)(?:\s*\})?"

    struct_pat = re.compile(
        r"x\s*\(([^)]*)\)\s*(?:=|:|is|→|->)\s*" + bool_pat,
        flags=re.IGNORECASE,
    )

    for m in struct_pat.finditer(text):
        idx_text = m.group(1)
        val_text = m.group(2).lower()

        try:
            indices = [int(p.strip()) for p in idx_text.split(",") if p.strip() != ""]
        except Exception:
            continue

        var_id = struct_indices_to_dimacs(indices, dims)

        if var_id is not None and 1 <= var_id <= num_vars:
            assignment[var_id] = val_text in {"true", "1", "yes"}

    flat_pat = re.compile(
        r"\bx[_\s]*(\d+)\s*(?:=|:|is|→|->)\s*" + bool_pat,
        flags=re.IGNORECASE,
    )

    for m in flat_pat.finditer(text):
        var_id = int(m.group(1))
        val_text = m.group(2).lower()

        if 1 <= var_id <= num_vars:
            assignment[var_id] = val_text in {"true", "1", "yes"}

    structured = extract_structured_assignment(text, num_vars)
    assignment.update(structured)

    return assignment


# ============================================================
# Extract UNSAT core from model response
# ============================================================

def parse_ints(text):
    return [int(x) for x in re.findall(r"-?\d+", text)]


def extract_unsat_core_candidates(text):
    """
    Returns candidates as:
      (basis, numbers)

    basis:
      condition_1_based: condition numbers from natural-language conditions
      raw_0_based: clauses[i] indices
      raw_try_both: ambiguous clause numbers
    """
    candidates = []

    # clauses[0], clauses[2], ...
    bracket_indices = [
        int(x)
        for x in re.findall(r"clauses?\s*\[\s*(\d+)\s*\]", text, flags=re.IGNORECASE)
    ]

    if bracket_indices:
        candidates.append(("raw_0_based", sorted(set(bracket_indices))))

    # UNSAT core: [1, 3, 4], condition numbers: [1, 3, 4], clause indices: [0, 2]
    core_pat = re.compile(
        r"(unsat\s+core|conflicting\s+conditions?|condition\s+numbers?|conflicting\s+clauses?|clause\s+indices?)"
        r"[^\n\[]*\[([^\]]+)\]",
        flags=re.IGNORECASE,
    )

    for m in core_pat.finditer(text):
        label = m.group(1).lower()
        nums = parse_ints(m.group(2))

        if not nums:
            continue

        if "condition" in label:
            candidates.append(("condition_1_based", nums))
        elif "indices" in label:
            candidates.append(("raw_try_both", nums))
        else:
            candidates.append(("condition_1_based", nums))

    # Conditions 1, 3, and 4
    condition_pat = re.compile(
        r"conditions?\s+((?:\d+\s*(?:,|and|&)?\s*){1,10})",
        flags=re.IGNORECASE,
    )

    for m in condition_pat.finditer(text):
        nums = parse_ints(m.group(1))
        if nums:
            candidates.append(("condition_1_based", nums))

    # Clause 1, clause 2, ...
    clause_mentions = [
        int(x)
        for x in re.findall(r"\bclause\s+(\d+)\b", text, flags=re.IGNORECASE)
    ]

    if clause_mentions:
        candidates.append(("raw_try_both", sorted(set(clause_mentions))))

    # Deduplicate
    seen = set()
    unique = []

    for basis, nums in candidates:
        key = (basis, tuple(nums))
        if key not in seen:
            seen.add(key)
            unique.append((basis, nums))

    return unique


def validate_unsat_core_from_response(response, raw_clauses, readable_clauses, num_vars):
    candidates = extract_unsat_core_candidates(response)

    for basis, nums in candidates:
        if basis == "condition_1_based" and readable_clauses:
            indices = [n - 1 for n in nums]
            if check_unsat_core(readable_clauses, num_vars, indices):
                return True, "condition_1_based", indices

        elif basis == "raw_0_based":
            indices = nums
            if check_unsat_core(raw_clauses, num_vars, indices):
                return True, "raw_0_based", indices

        elif basis == "raw_try_both":
            # Try raw 0-based
            indices = nums
            if check_unsat_core(raw_clauses, num_vars, indices):
                return True, "raw_0_based", indices

            # Try raw 1-based
            indices = [n - 1 for n in nums]
            if check_unsat_core(raw_clauses, num_vars, indices):
                return True, "raw_1_based", indices

            # Try condition 1-based
            if readable_clauses and check_unsat_core(readable_clauses, num_vars, indices):
                return True, "condition_1_based", indices

    return False, None, []


# ============================================================
# Row-level validation
# ============================================================

def validate_row(row):
    out = {
        "dataset_id": row.get("dataset_id"),
        "row_id": row.get("row_id"),
        "eval_split": row.get("eval_split"),
        "gold_label": row.get("gold_label"),
        "predicted_label": row.get("predicted_label"),
        "correct": row.get("correct"),
        "certificate_kind": None,
        "certificate_extracted": False,
        "certificate_valid": False,
        "parse_warning": "",
        "validation_error": "",
    }

    try:
        fields = parse_prompt_fields(row.get("prompt_text") or "")

        dims = fields["dims"]
        num_vars = fields["num_vars"]
        raw_clauses = fields["clauses"]
        readable_clauses = fields["readable_clauses"]

        out["num_vars"] = num_vars
        out["num_clauses"] = len(raw_clauses)
        out["z3_formula_result"] = check_formula_sat_status(raw_clauses, num_vars)

        response = row.get("model_response") or ""
        pred = str(row.get("predicted_label") or "").upper()

        if pred == "SAT":
            out["certificate_kind"] = "sat_assignment"

            assignment = extract_variable_assignments(response, dims, num_vars)
            out["certificate_extracted"] = len(assignment) > 0
            out["sat_assignment_num_vars"] = len(assignment)
            out["extracted_assignment"] = {str(k): v for k, v in sorted(assignment.items())}

            clause_eval = evaluate_assignment_on_clauses(raw_clauses, assignment)
            extendable = check_assignment_extendable(raw_clauses, num_vars, assignment) if assignment else False

            out["sat_all_clauses_satisfied"] = clause_eval["all_satisfied"]
            out["sat_assignment_extendable"] = extendable
            out["sat_num_satisfied_clauses"] = clause_eval["satisfied"]
            out["sat_num_undetermined_clauses"] = clause_eval["undetermined"]
            out["sat_num_false_clauses"] = clause_eval["false_count"]
            out["sat_false_clause_indices"] = clause_eval["false_indices"]
            out["sat_undetermined_clause_indices"] = clause_eval["undetermined_indices"]

            # Strict SAT certificate validity:
            # the extracted assignment must explicitly satisfy every clause.
            out["certificate_valid"] = clause_eval["all_satisfied"]

            if not assignment:
                out["parse_warning"] = "No SAT assignment extracted."
            elif not clause_eval["all_satisfied"] and extendable:
                out["parse_warning"] = (
                    "Assignment is Z3-extendable, but it does not explicitly satisfy every clause."
                )

        elif pred == "UNSAT":
            out["certificate_kind"] = "unsat_core"

            valid, basis, indices = validate_unsat_core_from_response(
                response=response,
                raw_clauses=raw_clauses,
                readable_clauses=readable_clauses,
                num_vars=num_vars,
            )

            out["unsat_formula_is_unsat"] = out["z3_formula_result"] == "UNSAT"
            out["unsat_core_valid"] = valid
            out["unsat_core_basis"] = basis
            out["unsat_core_indices"] = indices
            out["unsat_core_size"] = len(indices)
            out["certificate_extracted"] = len(indices) > 0
            out["certificate_valid"] = valid

            if not valid:
                out["parse_warning"] = "No valid UNSAT core extracted from response."

        else:
            out["certificate_kind"] = "none"
            out["parse_warning"] = "predicted_label is not SAT or UNSAT."

    except Exception as e:
        out["validation_error"] = repr(e)
        out["certificate_valid"] = False

    return out


def make_summary(results):
    def frac(num, den):
        return num / den if den else None

    total = len(results)
    correct = [r for r in results if r.get("correct") is True]
    correct_and_valid = [
        r for r in results
        if r.get("correct") is True and r.get("certificate_valid") is True
    ]

    pred_sat = [r for r in results if r.get("predicted_label") == "SAT"]
    pred_unsat = [r for r in results if r.get("predicted_label") == "UNSAT"]

    correct_sat = [r for r in pred_sat if r.get("correct") is True]
    correct_unsat = [r for r in pred_unsat if r.get("correct") is True]

    return {
        "num_rows": total,
        "predicted_label_counts": dict(Counter(str(r.get("predicted_label") or "NONE") for r in results)),
        "gold_label_counts": dict(Counter(str(r.get("gold_label") or "NONE") for r in results)),

        "num_correct_predictions": len(correct),
        "prediction_accuracy": frac(len(correct), total),

        "num_valid_certificates_all_predictions": sum(r.get("certificate_valid") is True for r in results),
        "certificate_validity_all_predictions": frac(
            sum(r.get("certificate_valid") is True for r in results),
            total,
        ),

        "num_correct_and_valid_certificates": len(correct_and_valid),
        "overall_correct_and_valid_rate": frac(len(correct_and_valid), total),

        "sat_certificate_validity_among_predicted_sat": frac(
            sum(r.get("certificate_valid") is True for r in pred_sat),
            len(pred_sat),
        ),
        "unsat_certificate_validity_among_predicted_unsat": frac(
            sum(r.get("certificate_valid") is True for r in pred_unsat),
            len(pred_unsat),
        ),

        "sat_certificate_validity_among_correct_sat_predictions": frac(
            sum(r.get("certificate_valid") is True for r in correct_sat),
            len(correct_sat),
        ),
        "unsat_certificate_validity_among_correct_unsat_predictions": frac(
            sum(r.get("certificate_valid") is True for r in correct_unsat),
            len(correct_unsat),
        ),

        "num_parse_warnings": sum(bool(r.get("parse_warning")) for r in results),
        "num_validation_errors": sum(bool(r.get("validation_error")) for r in results),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/eval_sft_qwen35_08b_full_on_test/qwen35_08b_base_test_predictions_corrected.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/eval_sft_qwen35_08b_full_on_test/z3_certificate_validation"),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.predictions)
    results = []

    for row in tqdm(rows, desc="Validating generated certificates with Z3"):
        results.append(validate_row(row))

    summary = make_summary(results)

    out_jsonl = args.output_dir / "z3_certificate_validation_results.jsonl"
    out_csv = args.output_dir / "z3_certificate_validation_results.csv"
    out_summary = args.output_dir / "z3_certificate_validation_summary.json"

    write_jsonl(results, out_jsonl)
    write_csv(results, out_csv)

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSummary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\nSaved files:")
    print(out_jsonl)
    print(out_csv)
    print(out_summary)


if __name__ == "__main__":
    main()