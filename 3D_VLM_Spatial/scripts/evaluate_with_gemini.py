#!/usr/bin/env python3
"""Use Gemini to judge predictions vs. ground-truth answers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

try:
    import google.generativeai as genai
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install google-generativeai to use this script.") from exc

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install tqdm to use this script.") from exc

PROMPT_TEMPLATE = """You are a medical QA evaluator.
For each item determine if the model prediction semantically matches the ground-truth answer.
Consider synonyms and equivalent medical phrasing as correct, but require the same clinical fact.
Respond strictly as JSON list where each element is:
{{"index": <int>, "is_correct": true/false, "reasoning": "short justification"}}

Batch:
{batch}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemini-based evaluation of QA predictions")
    parser.add_argument("--predictions", type=Path, required=True, help="Predictions JSONL (with question/answer/prediction)")
    parser.add_argument("--output", type=Path, required=True, help="Where to write Gemini judgments JSON")
    parser.add_argument("--model", default="models/gemini-2.0-flash", help="Gemini model name")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of QA pairs per Gemini call")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional delay between calls")
    return parser.parse_args()


def load_predictions(path: Path) -> List[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_text(response) -> str:
    if getattr(response, "text", None):
        return response.text
    parts = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts)


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`\n")
        if text.startswith("json\n"):
            text = text[5:]
    return text


def sanitize_json(text: str) -> str:
    # Gemini occasionally emits trailing commas inside arrays; strip them.
    text = text.replace(",\n]", "\n]")
    text = text.replace(", ]", "]")
    text = text.replace(",]", "]")
    # Also ensure each JSON object entry ends with a comma if needed.
    text = text.replace('"}  {', '"}, {')
    text = text.replace('}\n  {', '},\n  {')
    return text


def main() -> None:
    args = parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY before running.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    preds = load_predictions(args.predictions)
    judgments = []
    correct = 0
    total = 0

    for idx in range(0, len(preds), args.batch_size):
        batch = preds[idx : idx + args.batch_size]
        prompt_entries = []
        for offset, record in enumerate(batch, start=1):
            prompt_entries.append(
                f"{offset}. Question: {record.get('question')}\n"
                f"   Answer: {record.get('answer')}\n"
                f"   Prediction: {record.get('prediction')}"
            )
        prompt = PROMPT_TEMPLATE.format(batch="\n".join(prompt_entries))

        response = model.generate_content(prompt)
        text = strip_code_fence(extract_text(response))
        text = sanitize_json(text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            print("[WARN] Failed to parse Gemini output:\n", text)
            raise exc
        for judgment, record in zip(parsed, batch):
            item = {
                "question": record.get("question"),
                "answer": record.get("answer"),
                "prediction": record.get("prediction"),
                "is_correct": judgment.get("is_correct"),
                "reasoning": judgment.get("reasoning"),
            }
            judgments.append(item)
            if item["is_correct"]:
                correct += 1
            total += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(judgments, indent=2))
    accuracy = correct / total if total else 0.0
    print(json.dumps({"total": total, "correct": correct, "accuracy": accuracy}, indent=2))


if __name__ == "__main__":
    main()
