#!/usr/bin/env python3
"""Analyze CT-SpatialVQA dataset statistics and (optionally) prediction lengths."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional


WORD_RE = re.compile(r"\b\w+\b")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze CT-SpatialVQA dataset statistics.")
    p.add_argument(
        "--dataset-json",
        type=Path,
        default=Path("3D_VLM_Spatial/qa_generation_v2/spatial_qa_filtered_full.json"),
        help="CT-SpatialVQA JSON (image_id -> qa_pairs).",
    )
    p.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional predictions JSONL to analyze prediction length.",
    )
    p.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Optional directory to analyze all prediction files.",
    )
    p.add_argument(
        "--pred-glob",
        default="*_predictions_full.jsonl",
        help="Glob pattern within predictions-dir (default: *_predictions_full.jsonl).",
    )
    p.add_argument(
        "--all-predictions",
        action="store_true",
        help="Analyze all predictions in 3D_VLM_Spatial/reports using --pred-glob.",
    )
    p.add_argument(
        "--prediction-field",
        default="prediction",
        help="Field name for model output in predictions JSONL.",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("3D_VLM_Spatial/reports/ct_spatialvqa_analysis.json"),
        help="Output JSON path.",
    )
    return p.parse_args()


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(WORD_RE.findall(text))


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def summary_stats(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def histogram(values: List[int], bins: List[int]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not values:
        return counts
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        label = f"{lo + 1}-{hi}"
        counts[label] = 0
    counts[f">{bins[-1]}"] = 0
    for v in values:
        placed = False
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            if lo < v <= hi:
                counts[f"{lo + 1}-{hi}"] += 1
                placed = True
                break
        if not placed:
            if v > bins[-1]:
                counts[f">{bins[-1]}"] += 1
            else:
                # v == 0
                counts[f"0"] = counts.get("0", 0) + 1
    return counts


def main() -> None:
    args = parse_args()
    data = json.loads(args.dataset_json.read_text())

    report_count = len(data)
    qa_counts = []
    question_lens = []
    answer_lens = []

    for entry in data.values():
        qa_pairs = entry.get("qa_pairs", [])
        qa_counts.append(len(qa_pairs))
        for qa in qa_pairs:
            question_lens.append(word_count(qa.get("question", "")))
            answer_lens.append(word_count(qa.get("answer", "")))

    qa_stats = summary_stats(qa_counts)
    qlen_stats = summary_stats(question_lens)
    alen_stats = summary_stats(answer_lens)

    bins = [0, 5, 10, 20, 30, 40, 50, 100]

    output = {
        "reports": report_count,
        "total_qa_pairs": sum(qa_counts),
        "qa_pairs_per_report": qa_stats,
        "question_length_words": {
            "stats": qlen_stats,
            "histogram": histogram(question_lens, bins),
        },
        "answer_length_words": {
            "stats": alen_stats,
            "histogram": histogram(answer_lens, bins),
        },
    }

    if args.all_predictions or args.predictions_dir:
        pred_dir = args.predictions_dir or Path("3D_VLM_Spatial/reports")
        files = sorted(pred_dir.glob(args.pred_glob))
        by_model: Dict[str, Dict[str, Dict[str, float | Dict[str, int]]]] = {}
        for path in files:
            model_name = path.stem.replace("_predictions_full", "").replace("_predictions", "")
            pred_lens = [word_count(rec.get(args.prediction_field, "")) for rec in iter_jsonl(path)]
            by_model[model_name] = {
                "stats": summary_stats(pred_lens),
                "histogram": histogram(pred_lens, bins),
            }
        output["prediction_length_words_by_model"] = by_model
    elif args.predictions and args.predictions.exists():
        pred_lens = [word_count(rec.get(args.prediction_field, "")) for rec in iter_jsonl(args.predictions)]
        output["prediction_length_words"] = {
            "stats": summary_stats(pred_lens),
            "histogram": histogram(pred_lens, bins),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
