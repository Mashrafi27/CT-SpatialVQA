#!/usr/bin/env python3
"""Repair qa_generation output by parsing raw_output strings into valid QA lists."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List


def sanitize(content: str) -> str:
    cleaned = content.strip()
    # Remove fenced code blocks if present
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1]
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    return cleaned


def extract_json_list(text: str) -> str | None:
    """Try to extract the first JSON list from a blob of text."""
    if not text:
        return None
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def try_parse(raw: str):
    if not raw:
        return None
    # 1) Try raw directly
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # 2) Sanitize and retry
    cleaned = sanitize(raw)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # 3) Extract bracketed list and retry
    extracted = extract_json_list(cleaned)
    if extracted:
        try:
            data = json.loads(extracted)
            if isinstance(data, list):
                return data
        except Exception:
            pass

    # 4) Replace literal newlines with escaped newlines and retry
    cleaned_nl = cleaned.replace("\r\n", "\n").replace("\n", "\\n")
    extracted_nl = extract_json_list(cleaned_nl) or cleaned_nl
    try:
        data = json.loads(extracted_nl)
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
