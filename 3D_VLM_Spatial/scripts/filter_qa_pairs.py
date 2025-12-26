#!/usr/bin/env python3
"""Filter QA pairs based on Gemini judgments (spatial & relevance)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter QA pairs by Gemini judgments")
    parser.add_argument("--qa-json", type=Path, required=True, help="Path to spatial_qa_output.json")
    parser.add_argument("--judgments", type=Path, required=True, help="Path to gemini_judgments.json")
    parser.add_argument("--output", type=Path, required=True, help="Path to filtered QA JSON")
    parser.add_argument("--keep-nonspatial", action="store_true", help="Keep entries even if non-spatial")
    parser.add_argument("--keep-nonrelevant", action="store_true", help="Keep entries even if non-relevant")
    return parser.parse_args()


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    qa_data = load_json(args.qa_json)
    judgments = load_json(args.judgments)

    filtered = {}
    removed_counts = {"non_spatial": 0, "non_relevant": 0}

    for case_id, case_payload in qa_data.items():
        qa_pairs = case_payload.get("qa_pairs", [])
        judge_entries = judgments.get(case_id, [])
        keep_pairs = []
        for idx, qa_pair in enumerate(qa_pairs, start=1):
            match = next((entry for entry in judge_entries if entry.get("index") == idx), None)
            if not match:
                keep_pairs.append(qa_pair)
                continue
            spatial_ok = match.get("is_spatial", True)
            relevant_ok = match.get("is_relevant", True)
            if not spatial_ok and not args.keep_nonspatial:
                removed_counts["non_spatial"] += 1
                continue
            if not relevant_ok and not args.keep_nonrelevant:
                removed_counts["non_relevant"] += 1
                continue
            keep_pairs.append(qa_pair)
        if keep_pairs:
            filtered[case_id] = {"qa_pairs": keep_pairs}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(filtered, indent=2))
    kept = sum(len(v["qa_pairs"]) for v in filtered.values())
    print(
        f"Filtered {sum(removed_counts.values())} QA pairs "
        f"(removed {removed_counts['non_spatial']} non-spatial, {removed_counts['non_relevant']} non-relevant). "
        f"Kept {kept} pairs across {len(filtered)} cases."
    )


if __name__ == "__main__":
    main()
