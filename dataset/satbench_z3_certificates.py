#!/usr/bin/env python3
"""Attach SAT assignments or UNSAT cores to the released SATBench dataset.

Example:
    python satbench_z3_certificates.py \
        --output satbench_with_certificates.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from z3 import And, Bool, Not, Or, Solver, unsat, sat


def lit_to_z3(lit: int, vars_: list) -> Any:
    idx = abs(lit) - 1
    atom = vars_[idx]
    return atom if lit > 0 else Not(atom)


def clause_to_z3(clause: list[int], vars_: list) -> Any:
    return Or(*[lit_to_z3(lit, vars_) for lit in clause])


def solve_with_certificate(entry: dict[str, Any]) -> dict[str, Any]:
    num_vars = int(entry["num_vars"])
    clauses = entry["clauses"]
    vars_ = [Bool(f"x{i+1}") for i in range(num_vars)]

    solver = Solver()
    tracked_names = []
    for i, clause in enumerate(clauses):
        name = f"c{i}"
        tracked_names.append(name)
        solver.assert_and_track(clause_to_z3(clause, vars_), name)

    status = solver.check()
    out = dict(entry)

    if status == sat:
        model = solver.model()
        assignment = {}
        for i, var in enumerate(vars_, start=1):
            value = model.eval(var, model_completion=True)
            assignment[str(i)] = bool(value)
        out["certificate_type"] = "sat_assignment"
        out["sat_assignment"] = assignment
        out["unsat_core_clause_indices"] = None
    elif status == unsat:
        core = solver.unsat_core()
        core_names = {str(c) for c in core}
        core_indices = [i for i, name in enumerate(tracked_names) if name in core_names]
        out["certificate_type"] = "unsat_core"
        out["sat_assignment"] = None
        out["unsat_core_clause_indices"] = core_indices
    else:
        out["certificate_type"] = "unknown"
        out["sat_assignment"] = None
        out["unsat_core_clause_indices"] = None

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ds = load_dataset("LLM4Code/SATBench", split=args.split)
    entries = [dict(x) for x in ds]
    if args.limit is not None:
        entries = entries[: args.limit]

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as fout:
        for entry in entries:
            result = solve_with_certificate(entry)
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
