#!/usr/bin/env python3
"""Verify SAT assignments and UNSAT cores in an enriched SATBench JSONL file.

Example:
    python satbench_verify_certificates.py \
        --input satbench_with_certificates_50.jsonl

Optional:
    python satbench_verify_certificates.py \
        --input satbench_with_certificates_50.jsonl \
        --output satbench_with_certificates_50_verified.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from z3 import Bool, Not, Or, Solver, sat, unsat


def lit_to_z3(lit: int, vars_: list[Any]) -> Any:
    idx = abs(lit) - 1
    atom = vars_[idx]
    return atom if lit > 0 else Not(atom)


def clause_to_z3(clause: list[int], vars_: list[Any]) -> Any:
    return Or(*[lit_to_z3(lit, vars_) for lit in clause])


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes"}:
            return True
        if v in {"false", "0", "no"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    raise ValueError(f"Cannot interpret assignment value as bool: {value!r}")


def clause_satisfied(clause: list[int], assignment: dict[str, Any]) -> bool:
    for lit in clause:
        var_idx = str(abs(lit))
        if var_idx not in assignment:
            raise KeyError(f"Missing assignment for variable {var_idx}")
        value = normalize_bool(assignment[var_idx])
        if (lit > 0 and value) or (lit < 0 and not value):
            return True
    return False


def verify_sat_assignment(entry: dict[str, Any]) -> tuple[bool, str]:
    clauses = entry["clauses"]
    assignment = entry.get("sat_assignment")

    if not isinstance(assignment, dict) or not assignment:
        return False, "Missing or empty sat_assignment"

    try:
        for i, clause in enumerate(clauses):
            if not clause_satisfied(clause, assignment):
                return False, f"Clause {i} is not satisfied by sat_assignment"
    except (KeyError, ValueError) as e:
        return False, str(e)

    return True, "SAT assignment satisfies all clauses"


def verify_unsat_core(entry: dict[str, Any]) -> tuple[bool, str]:
    num_vars = int(entry["num_vars"])
    clauses = entry["clauses"]
    core_indices = entry.get("unsat_core_clause_indices")

    if not isinstance(core_indices, list) or not core_indices:
        return False, "Missing or empty unsat_core_clause_indices"

    if any(not isinstance(i, int) for i in core_indices):
        return False, "UNSAT core contains non-integer indices"

    if any(i < 0 or i >= len(clauses) for i in core_indices):
        return False, "UNSAT core contains out-of-range clause indices"

    vars_ = [Bool(f"x{i+1}") for i in range(num_vars)]
    solver = Solver()
    for idx in core_indices:
        solver.add(clause_to_z3(clauses[idx], vars_))

    status = solver.check()
    if status == unsat:
        return True, f"UNSAT core is unsatisfiable ({len(core_indices)} clauses)"
    if status == sat:
        return False, "Claimed UNSAT core is actually satisfiable"
    return False, f"Unexpected solver result: {status}"


def verify_entry(entry: dict[str, Any]) -> dict[str, Any]:
    out = dict(entry)
    cert_type = entry.get("certificate_type")
    gold_satisfiable = entry.get("satisfiable")

    expected_type = None
    if gold_satisfiable is True:
        expected_type = "sat_assignment"
    elif gold_satisfiable is False:
        expected_type = "unsat_core"

    out["certificate_matches_gold_type"] = cert_type == expected_type

    if cert_type == "sat_assignment":
        valid, reason = verify_sat_assignment(entry)
        out["certificate_valid"] = valid
        out["certificate_validation_reason"] = reason
        out["verified_label"] = "SAT"
    elif cert_type == "unsat_core":
        valid, reason = verify_unsat_core(entry)
        out["certificate_valid"] = valid
        out["certificate_validation_reason"] = reason
        out["verified_label"] = "UNSAT"
    else:
        out["certificate_valid"] = False
        out["certificate_validation_reason"] = f"Unsupported certificate_type: {cert_type!r}"
        out["verified_label"] = None

    if gold_satisfiable is True:
        out["verified_label_matches_gold"] = out["verified_label"] == "SAT"
    elif gold_satisfiable is False:
        out["verified_label_matches_gold"] = out["verified_label"] == "UNSAT"
    else:
        out["verified_label_matches_gold"] = None

    return out


def print_summary(rows: list[dict[str, Any]]) -> None:
    total = len(rows)
    sat_count = sum(r.get("certificate_type") == "sat_assignment" for r in rows)
    unsat_count = sum(r.get("certificate_type") == "unsat_core" for r in rows)
    unknown_count = total - sat_count - unsat_count
    valid_count = sum(bool(r.get("certificate_valid")) for r in rows)
    type_match_count = sum(bool(r.get("certificate_matches_gold_type")) for r in rows)
    label_match_count = sum(bool(r.get("verified_label_matches_gold")) for r in rows)

    unsat_core_sizes = [
        len(r.get("unsat_core_clause_indices", []))
        for r in rows
        if r.get("certificate_type") == "unsat_core" and isinstance(r.get("unsat_core_clause_indices"), list)
    ]
    avg_core_size = (sum(unsat_core_sizes) / len(unsat_core_sizes)) if unsat_core_sizes else 0.0

    print(f"Total examples: {total}")
    print(f"SAT certificates: {sat_count}")
    print(f"UNSAT certificates: {unsat_count}")
    print(f"Unknown/other certificates: {unknown_count}")
    print(f"Certificate valid: {valid_count}/{total} ({(100*valid_count/total if total else 0):.1f}%)")
    print(f"Certificate type matches gold label: {type_match_count}/{total} ({(100*type_match_count/total if total else 0):.1f}%)")
    print(f"Verified label matches gold label: {label_match_count}/{total} ({(100*label_match_count/total if total else 0):.1f}%)")
    if unsat_core_sizes:
        print(f"Average UNSAT core size: {avg_core_size:.2f}")

    invalid_rows = [
        (i, r.get("certificate_validation_reason"))
        for i, r in enumerate(rows)
        if not r.get("certificate_valid")
    ]
    if invalid_rows:
        print("\nFirst invalid rows:")
        for i, reason in invalid_rows[:10]:
            print(f"  Row {i}: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL produced by satbench_z3_certificates.py")
    parser.add_argument("--output", default=None, help="Optional output JSONL with verification fields added")
    args = parser.parse_args()

    input_path = Path(args.input)
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e
            rows.append(verify_entry(entry))

    print_summary(rows)

    if args.output:
        output_path = Path(args.output)
        with output_path.open("w", encoding="utf-8") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nWrote verified rows to: {output_path}")


if __name__ == "__main__":
    main()
