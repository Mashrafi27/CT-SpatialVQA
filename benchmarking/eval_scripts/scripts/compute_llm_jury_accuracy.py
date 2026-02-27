#!/usr/bin/env python3
"""Compute LLM-as-jury accuracy from correctness_matrix_avg3.csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute LLM-as-jury accuracy from correctness matrix.")
    p.add_argument(
        "--matrix-csv",
        type=Path,
        default=Path("3D_VLM_Spatial/reports/correctness_matrix_avg3.csv"),
        help="Correctness matrix CSV (question_id + model columns).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.66,
        help="Threshold to binarize jury average (default: 0.66).",
    )
    p.add_argument(
        "--missing",
        choices=["skip", "zero"],
        default="skip",
        help="How to handle missing values (default: skip).",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("3D_VLM_Spatial/reports/llm_jury_accuracy.json"),
        help="Output JSON path.",
    )
    return p.parse_args()


def load_matrix(path: Path) -> Tuple[List[str], List[List[str]]]:
    with path.open() as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit("Matrix CSV is empty.")
    header = rows[0]
    if not header or header[0] != "question_id":
        raise SystemExit("Matrix CSV must start with question_id column.")
    models = header[1:]
    return models, rows[1:]


def main() -> None:
    args = parse_args()
    models, rows = load_matrix(args.matrix_csv)

    sums: Dict[str, float] = {m: 0.0 for m in models}
    counts: Dict[str, int] = {m: 0 for m in models}

    for row in rows:
        for model, raw in zip(models, row[1:]):
            if raw == "" or raw is None:
                if args.missing == "zero":
                    sums[model] += 0.0
                    counts[model] += 1
                continue
            try:
                val = float(raw)
            except Exception:
                if args.missing == "zero":
                    sums[model] += 0.0
                    counts[model] += 1
                continue
            jury = 1.0 if val >= args.threshold else 0.0
            sums[model] += jury
            counts[model] += 1

    results = []
    for model in models:
        count = counts[model]
        acc = sums[model] / count if count else 0.0
        results.append({"model": model, "count": count, "llm_jury_accuracy": acc})

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
