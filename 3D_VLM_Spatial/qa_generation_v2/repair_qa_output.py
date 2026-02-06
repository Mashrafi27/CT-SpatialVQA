#!/usr/bin/env python3
"""Repair qa_generation output by parsing raw_output strings into valid QA lists."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List


def sanitize(content: str):
    cleaned = content.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1]
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    cleaned = cleaned.replace("\r\n", "\n")
    cleaned = cleaned.replace("\n", "\\n")
    return cleaned


def try_parse(raw: str):
    cleaned = sanitize(raw)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except Exception:
        return None
    return None


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    obj = json.loads(inp.read_text())
    repaired = 0
    total = 0
    for k, v in obj.items():
        total += 1
        qa = v.get("qa_pairs", [])
        if qa and isinstance(qa, list) and isinstance(qa[0], dict) and qa[0].get("error"):
            raw = qa[0].get("raw_output", "")
            parsed = try_parse(raw)
            if parsed is not None:
                v["qa_pairs"] = parsed
                repaired += 1
    out.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    print(f"Repaired {repaired}/{total} entries -> {out}")


if __name__ == "__main__":
    main()
