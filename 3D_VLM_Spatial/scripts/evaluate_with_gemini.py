#!/usr/bin/env python3
"""Use Gemini to judge predictions vs. ground-truth answers."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Optional
import time

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
    parser.add_argument("--models", nargs="+", default=None, help="Optional list of Gemini models for jury mode")
    parser.add_argument("--jury", action="store_true", help="Use multiple models as a jury (majority vote)")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of QA pairs per Gemini call")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional delay between calls")
    parser.add_argument("--prediction-field", default="prediction", help="Field to read model output from")
    parser.add_argument("--max-retries", type=int, default=8, help="Max retries for rate limit errors")
    parser.add_argument("--initial-backoff", type=float, default=5.0, help="Initial backoff seconds")
    parser.add_argument("--max-backoff", type=float, default=120.0, help="Maximum backoff seconds")
    parser.add_argument("--backoff-mult", type=float, default=2.0, help="Backoff multiplier")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
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


def load_existing_output(path: Path) -> Optional[List[dict]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "jury" in data:
            return None
    except Exception:
        pass
    # Fallback: try JSONL
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records or None


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


def _should_retry(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    if "resourceexhausted" in name or "quota" in msg or "rate" in msg or "429" in msg:
        return True
    return False


def _generate_with_retry(model, prompt: str, max_retries: int, initial_backoff: float, max_backoff: float, backoff_mult: float):
    attempt = 0
    backoff = initial_backoff
    while True:
        try:
            return model.generate_content(prompt)
        except Exception as exc:  # noqa: BLE001
            attempt += 1
            if attempt > max_retries or not _should_retry(exc):
                raise
            jitter = random.uniform(0, 0.25 * backoff)
            sleep_for = min(max_backoff, backoff + jitter)
            print(f"[WARN] Rate limited, retrying in {sleep_for:.1f}s (attempt {attempt}/{max_retries})")
            time.sleep(sleep_for)
            backoff = min(max_backoff, backoff * backoff_mult)


def judge_with_model(
    model_name: str,
    preds: List[dict],
    batch_size: int,
    sleep: float,
    prediction_field: str,
    max_retries: int,
    initial_backoff: float,
    max_backoff: float,
    backoff_mult: float,
):
    model = genai.GenerativeModel(model_name)
    judgments = []
    correct = 0
    total = 0

    for idx in tqdm(range(0, len(preds), batch_size), desc=f"{model_name} evaluating", unit="batch"):
        batch = preds[idx : idx + batch_size]
        prompt_entries = []
        for offset, record in enumerate(batch, start=1):
            prompt_entries.append(
                f"{offset}. Question: {record.get('question')}\n"
                f"   Answer: {record.get('answer')}\n"
                f"   Prediction: {record.get(prediction_field)}"
            )
        prompt = PROMPT_TEMPLATE.format(batch="\n".join(prompt_entries))

        try:
            response = _generate_with_retry(
                model,
                prompt,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                backoff_mult=backoff_mult,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Gemini call failed: {exc}")
            for record in batch:
                item = {
                    "case_id": record.get("case_id"),
                    "question": record.get("question"),
                    "answer": record.get("answer"),
                    "prediction": record.get(prediction_field),
                    "is_correct": None,
                    "reasoning": f"REQUEST_ERROR: {exc}",
                }
                judgments.append(item)
            if sleep:
                time.sleep(sleep)
            continue
        text = strip_code_fence(extract_text(response))
        text = sanitize_json(text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            print(f"[WARN] Failed to parse Gemini output for {model_name}:\n", text)
            for record in batch:
                item = {
                    "case_id": record.get("case_id"),
                    "question": record.get("question"),
                    "answer": record.get("answer"),
                    "prediction": record.get(prediction_field),
                    "is_correct": None,
                    "reasoning": f"PARSE_ERROR: {text[:200]}...",
                }
                judgments.append(item)
            continue
        for judgment, record in zip(parsed, batch):
            item = {
                "case_id": record.get("case_id"),
                "question": record.get("question"),
                "answer": record.get("answer"),
                "prediction": record.get(prediction_field),
                "is_correct": judgment.get("is_correct"),
                "reasoning": judgment.get("reasoning"),
            }
            judgments.append(item)
            if item["is_correct"] is not None:
                total += 1
                if item["is_correct"]:
                    correct += 1
        if sleep:
            import time
            time.sleep(sleep)

    accuracy = correct / total if total else 0.0
    return judgments, {"total": total, "correct": correct, "accuracy": accuracy}


def main() -> None:
    args = parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY before running.")
    genai.configure(api_key=api_key)

    preds = load_predictions(args.predictions)
    existing = None
    if args.resume:
        existing = load_existing_output(args.output)
        if existing is None:
            print("[WARN] No valid existing output found; starting from scratch.")

    if existing is not None and args.jury:
        print("[WARN] --resume with --jury is not supported yet; rerun without --resume.")
        existing = None
    model_list = args.models if args.models else [args.model]

    if args.jury and len(model_list) > 1:
        jury_records = []
        per_model_stats = {}
        per_model_judgments = {}
        for model_name in model_list:
            judgments, stats = judge_with_model(
                model_name,
                preds,
                args.batch_size,
                args.sleep,
                args.prediction_field,
                args.max_retries,
                args.initial_backoff,
                args.max_backoff,
                args.backoff_mult,
            )
            per_model_judgments[model_name] = judgments
            per_model_stats[model_name] = stats

        total = 0
        correct = 0
        for i, record in enumerate(preds):
            votes = []
            reasons = {}
            for model_name in model_list:
                j = per_model_judgments[model_name][i]
                votes.append(j.get("is_correct"))
                reasons[model_name] = j.get("reasoning")
            true_votes = sum(1 for v in votes if v is True)
            false_votes = sum(1 for v in votes if v is False)
            jury_decision = None
            if true_votes + false_votes > 0:
                jury_decision = true_votes >= false_votes
                total += 1
                if jury_decision:
                    correct += 1

            jury_records.append(
                {
                    "question": record.get("question"),
                    "answer": record.get("answer"),
                    "prediction": record.get(args.prediction_field),
                    "jury_is_correct": jury_decision,
                    "jury_votes": {"true": true_votes, "false": false_votes, "total": len(model_list)},
                    "per_model": {m: per_model_judgments[m][i] for m in model_list},
                }
            )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"jury": jury_records, "per_model_stats": per_model_stats}, indent=2))
        accuracy = correct / total if total else 0.0
        print(json.dumps({"total": total, "correct": correct, "accuracy": accuracy}, indent=2))
    else:
        start_idx = 0
        if isinstance(existing, list):
            start_idx = len(existing)
        if start_idx:
            print(f"[INFO] Resuming from item {start_idx}/{len(preds)}")
        judgments, stats = judge_with_model(
            model_list[0],
            preds[start_idx:],
            args.batch_size,
            args.sleep,
            args.prediction_field,
            args.max_retries,
            args.initial_backoff,
            args.max_backoff,
            args.backoff_mult,
        )
        combined = (existing or []) + judgments
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(combined, indent=2))
        # Recompute stats on combined list
        total = sum(1 for j in combined if j.get("is_correct") is not None)
        correct = sum(1 for j in combined if j.get("is_correct") is True)
        accuracy = correct / total if total else 0.0
        print(json.dumps({"total": total, "correct": correct, "accuracy": accuracy}, indent=2))


if __name__ == "__main__":
    main()
