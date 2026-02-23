#!/usr/bin/env python3
"""Evaluate predictions with AlignScore (text-to-text factual consistency)."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import List

try:
    from alignscore import AlignScore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install alignscore to use this script.") from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AlignScore-based evaluation of QA predictions")
    p.add_argument("--predictions", type=Path, required=True, help="Predictions JSONL (question/answer/prediction)")
    p.add_argument("--output", type=Path, required=True, help="Where to write AlignScore results JSON")
    p.add_argument("--ckpt-path", type=Path, required=True, help="Path to AlignScore checkpoint (.ckpt)")
    p.add_argument("--model", default="roberta-base", help="AlignScore backbone (e.g., roberta-base or roberta-large)")
    p.add_argument("--device", default="cuda:0", help="cuda, cuda:0, or cpu")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for scoring")
    p.add_argument("--evaluation-mode", default="nli_sp", help="AlignScore evaluation mode")
    p.add_argument("--prediction-field", default="prediction", help="Field to read model output from")
    p.add_argument("--context-mode", choices=["answer", "qa", "custom"], default="qa",
                   help="Context to score against: answer only, question+answer, or custom template")
    p.add_argument("--context-template", default="Question: {question}\nAnswer: {answer}",
                   help="Template used when context-mode=custom")
    p.add_argument("--limit", type=int, default=None, help="Optional limit on number of records")
    return p.parse_args()


def load_predictions(path: Path, limit: int | None) -> List[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def build_context(record: dict, mode: str, template: str) -> str:
    question = record.get("question", "") or ""
    answer = record.get("answer", "") or ""
    if mode == "answer":
        return answer
    if mode == "qa":
        return f"Question: {question}\nAnswer: {answer}"
    return template.format(question=question, answer=answer, prediction=record.get("prediction", ""))


def main() -> None:
    args = parse_args()
    records = load_predictions(args.predictions, args.limit)
    if not records:
        raise SystemExit("No prediction records found.")

    try:
        scorer = AlignScore(
            model=args.model,
            ckpt_path=str(args.ckpt_path),
            device=args.device,
            batch_size=args.batch_size,
            evaluation_mode=args.evaluation_mode,
        )
    except TypeError as exc:
        raise SystemExit(
            "Failed to initialize AlignScore. Check the required arguments for your version."
        ) from exc

    contexts = []
    claims = []
    for record in records:
        contexts.append(build_context(record, args.context_mode, args.context_template))
        claims.append(record.get(args.prediction_field, "") or "")

    scores = scorer.score(contexts=contexts, claims=claims)

    items = []
    for record, score in zip(records, scores):
        items.append(
            {
                "case_id": record.get("case_id"),
                "question": record.get("question"),
                "answer": record.get("answer"),
                "prediction": record.get(args.prediction_field),
                "alignscore": float(score),
            }
        )

    summary = {
        "count": len(items),
        "mean": float(statistics.mean(scores)) if scores else 0.0,
        "median": float(statistics.median(scores)) if scores else 0.0,
        "min": float(min(scores)) if scores else 0.0,
        "max": float(max(scores)) if scores else 0.0,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "items": items}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
