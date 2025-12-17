#!/usr/bin/env python3
"""
Compute yes/no statistics for the spatial and relevant answers in progress.json.

Usage:
    python progress_yes_no_stats.py --progress progress.json --output reports/qa_label_stats.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize spatial/relevant label counts from progress.json.")
    parser.add_argument("--progress", default="progress.json", type=Path, help="Path to progress.json")
    parser.add_argument(
        "--output",
        default=Path("reports/qa_label_stats.json"),
        type=Path,
        help="Where to write the aggregated statistics (JSON).",
    )
    return parser.parse_args()


def tally_answers(progress: Dict[str, dict]) -> Dict[str, object]:
    per_user = {}
    overall_counter = {"spatial": Counter(), "relevant": Counter()}
    for user, cases in progress.items():
        user_counter = {"spatial": Counter(), "relevant": Counter()}
        for case_data in cases.values():
            answers = case_data.get("answers", [])
            for entry in answers:
                if not isinstance(entry, dict):
                    continue
                add_entry_counts(entry, user_counter, overall_counter)
        per_user[user] = {
            "spatial": counter_to_dict(user_counter["spatial"]),
            "relevant": counter_to_dict(user_counter["relevant"]),
        }
    overall = {
        "spatial": counter_to_dict(overall_counter["spatial"]),
        "relevant": counter_to_dict(overall_counter["relevant"]),
    }
    return {"overall": overall, "by_user": per_user}


def add_entry_counts(entry: dict, user_counter: Dict[str, Counter], overall_counter: Dict[str, Counter]) -> None:
    for field in ("spatial", "relevant"):
        value = entry.get(field)
        normalized = normalize_value(value)
        user_counter[field][normalized] += 1
        overall_counter[field][normalized] += 1


def normalize_value(value) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unanswered"


def counter_to_dict(counter: Counter) -> Dict[str, int]:
    return {"yes": counter.get("yes", 0), "no": counter.get("no", 0), "unanswered": counter.get("unanswered", 0)}


def main() -> None:
    args = parse_args()
    progress_data = json.loads(Path(args.progress).read_text(encoding="utf-8"))
    stats = tally_answers(progress_data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(f"Wrote statistics to {args.output}")


if __name__ == "__main__":
    main()
