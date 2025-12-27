#!/usr/bin/env python3
"""Convert nested QA JSON (case -> list) into flat JSONL records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert nested spatial QA JSON to JSONL")
    parser.add_argument("--input", type=Path, required=True, help="Input JSON (dict keyed by case)")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional root directory to prepend to case IDs for image paths",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.output.open("w") as out:
        for case_id, payload in data.items():
            image_path = Path(case_id)
            if args.image_root and not image_path.is_absolute():
                image_path = args.image_root / image_path
            for qa in payload.get("qa_pairs", []):
                record = {
                    "case_id": case_id,
                    "image_path": str(image_path),
                    "question": qa.get("question"),
                }
                if "answer" in qa:
                    record["answer"] = qa["answer"]
                out.write(json.dumps(record) + "\n")
                count += 1

    print(f"Wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()
