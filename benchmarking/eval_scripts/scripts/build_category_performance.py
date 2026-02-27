#!/usr/bin/env python3
"""Aggregate per-category correctness by model."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Category-wise correctness summary.")
    p.add_argument(
        "--categories-json",
        type=Path,
        default=Path("3D_VLM_Spatial/Spatial Understanding - Assign Categories/spatial_qa_filtered_full_with_categories.json"),
        help="JSON with qa_pairs and category field.",
    )
    p.add_argument(
        "--matrix-csv",
        type=Path,
        default=Path("3D_VLM_Spatial/reports/correctness_matrix_avg3.csv"),
        help="Correctness matrix CSV (question_id + model columns).",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("3D_VLM_Spatial/reports/category_performance.csv"),
        help="Output CSV path.",
    )
    p.add_argument(
        "--missing",
        choices=["zero", "skip"],
        default="zero",
        help="How to handle missing model scores for a question (default: zero).",
    )
    p.add_argument(
        "--binarize-avg3",
        action="store_true",
        help="Convert averaged scores to binary: >=0.66 -> 1, else 0.",
    )
    return p.parse_args()


def load_categories(path: Path) -> Dict[str, List[str]]:
    data = json.loads(path.read_text())
    qid_to_cats: Dict[str, List[str]] = {}
    for image_id, entry in data.items():
        qa_pairs = entry.get("qa_pairs", [])
        for idx, qa in enumerate(qa_pairs):
            qid = f"{image_id}::{idx}"
            cats = qa.get("category") or []
            # Normalize to list of strings
            if isinstance(cats, str):
                cats = [cats]
            cats = [c.strip() for c in cats if isinstance(c, str) and c.strip()]
            # Normalize known duplicates
            normalized = []
            for c in cats:
                if c == "Medial-Lateral Orientation":
                    normalized.append("Medial-Lateral Orientation (Centricity)")
                else:
                    normalized.append(c)
            cats = normalized
            qid_to_cats[qid] = cats
    return qid_to_cats


def load_matrix(path: Path) -> Tuple[List[str], Dict[str, Dict[str, float | None]]]:
    with path.open() as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit("Matrix CSV is empty.")
    header = rows[0]
    if not header or header[0] != "question_id":
        raise SystemExit("Matrix CSV must start with question_id column.")
    models = header[1:]
    matrix: Dict[str, Dict[str, float | None]] = {}
    for row in rows[1:]:
        if not row:
            continue
        qid = row[0]
        scores: Dict[str, float | None] = {}
        for model, val in zip(models, row[1:]):
            if val == "" or val is None:
                scores[model] = None
            else:
                try:
                    scores[model] = float(val)
                except Exception:
                    scores[model] = None
        matrix[qid] = scores
    return models, matrix


def main() -> None:
    args = parse_args()
    qid_to_cats = load_categories(args.categories_json)
    models, matrix = load_matrix(args.matrix_csv)

    # category -> count
    cat_counts: Dict[str, int] = defaultdict(int)
    # category -> model -> sum score
    cat_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    # category -> model -> count of contributing questions (when missing=skip)
    cat_denoms: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for qid, scores in matrix.items():
        cats = qid_to_cats.get(qid, [])
        if not cats:
            continue
        for cat in cats:
            cat_counts[cat] += 1
            for model in models:
                val = scores.get(model)
                if val is None:
                    if args.missing == "zero":
                        cat_scores[cat][model] += 0.0
                        cat_denoms[cat][model] += 1
                    else:
                        # skip
                        continue
                else:
                    if args.binarize_avg3:
                        val = 1.0 if float(val) >= 0.66 else 0.0
                    cat_scores[cat][model] += float(val)
                    cat_denoms[cat][model] += 1

    # Write CSV
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "count"] + models)
        for cat in sorted(cat_counts.keys()):
            row = [cat, cat_counts[cat]]
            for model in models:
                denom = cat_denoms[cat][model]
                if denom == 0:
                    row.append("")
                else:
                    pct = (cat_scores[cat][model] / denom) * 100.0
                    row.append(f"{pct:.2f}")
            writer.writerow(row)

    print(f"Wrote {len(cat_counts)} categories to {args.output_csv}")


if __name__ == "__main__":
    main()
