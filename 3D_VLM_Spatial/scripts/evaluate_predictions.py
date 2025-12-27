#!/usr/bin/env python3
"""Simple exact-match scorer for Med3DVLM predictions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare predictions JSONL vs. ground-truth answers")
    parser.add_argument("--predictions", type=Path, required=True, help="JSONL with prediction + answer")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON file to write summary stats")
    parser.add_argument("--show-mismatches", action="store_true", help="Print mismatched pairs to stdout")
    return parser.parse_args()


def normalize(text: str | None) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    args = parse_args()
    total = 0
    correct = 0
    mismatches = []

    for record in iter_jsonl(args.predictions):
        answer = normalize(record.get("answer"))
        pred = normalize(record.get("prediction"))
        if not answer:
            continue  # skip entries without GT
        total += 1
        if pred == answer:
            correct += 1
        elif args.show_mismatches:
            mismatches.append(
                {
                    "question": record.get("question"),
                    "answer": record.get("answer"),
                    "prediction": record.get("prediction"),
                }
            )

    accuracy = correct / total if total else 0.0
    summary = {"total": total, "correct": correct, "accuracy": accuracy}
    print(json.dumps(summary, indent=2))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2))

    if args.show_mismatches and mismatches:
        print("\nSample mismatches:")
        for entry in mismatches[:20]:
            print(f"Q: {entry['question']}")
            print(f"GT: {entry['answer']}")
            print(f"PR: {entry['prediction']}\n")


if __name__ == "__main__":
    main()
