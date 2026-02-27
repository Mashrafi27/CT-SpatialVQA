#!/usr/bin/env python3
"""Compute text similarity metrics for QA predictions.

Metrics:
- Sentence-BERT cosine similarity
- BLEU (sacrebleu)
- ROUGE (rouge1/rouge2/rougeL)
- METEOR (nltk)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text metrics for QA predictions")
    parser.add_argument("--predictions", type=Path, required=True, help="JSONL with answer/prediction")
    parser.add_argument("--output", type=Path, required=True, help="JSON output with metrics")
    parser.add_argument("--answer-field", default="answer", help="Field name for ground truth")
    parser.add_argument("--prediction-field", default="prediction", help="Field name for model output")
    parser.add_argument("--skip-empty", action="store_true", help="Skip rows with empty answers or predictions")
    parser.add_argument("--lower", action="store_true", help="Lowercase both sides")
    parser.add_argument("--per-item", type=Path, default=None, help="Optional JSONL with per-item metrics")

    # SBERT
    parser.add_argument("--sbert-model", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--no-sbert", action="store_true", help="Disable SBERT cosine")
    parser.add_argument("--no-bleu", action="store_true", help="Disable BLEU")
    parser.add_argument("--no-rouge", action="store_true", help="Disable ROUGE")
    parser.add_argument("--no-meteor", action="store_true", help="Disable METEOR")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def normalize(text: str, lower: bool) -> str:
    if text is None:
        return ""
    text = text.strip()
    if lower:
        text = text.lower()
    return text


def compute_sbert(preds: List[str], refs: List[str], model_name: str) -> Dict[str, float]:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer(model_name)
    pred_emb = model.encode(preds, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    ref_emb = model.encode(refs, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    sims = (pred_emb * ref_emb).sum(axis=1)
    return {"sbert_cosine": float(np.mean(sims))}


def compute_bleu(preds: List[str], refs: List[str]) -> Dict[str, float]:
    import sacrebleu
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return {"bleu": float(bleu.score)}


def compute_rouge(preds: List[str], refs: List[str]) -> Dict[str, float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1, rouge2, rougel = [], [], []
    for p, r in zip(preds, refs):
        scores = scorer.score(r, p)
        rouge1.append(scores["rouge1"].fmeasure)
        rouge2.append(scores["rouge2"].fmeasure)
        rougel.append(scores["rougeL"].fmeasure)
    return {
        "rouge1_f": float(np.mean(rouge1)),
        "rouge2_f": float(np.mean(rouge2)),
        "rougeL_f": float(np.mean(rougel)),
    }


def compute_meteor(preds: List[str], refs: List[str]) -> Dict[str, float]:
    from nltk.translate.meteor_score import meteor_score
    scores = [meteor_score([r], p) for p, r in zip(preds, refs)]
    return {"meteor": float(np.mean(scores))}


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.predictions)

    preds, refs, kept_records = [], [], []
    for r in records:
        pred = normalize(r.get(args.prediction_field, ""), args.lower)
        ref = normalize(r.get(args.answer_field, ""), args.lower)
        if args.skip_empty and (not pred or not ref):
            continue
        if not ref:
            continue
        preds.append(pred)
        refs.append(ref)
        kept_records.append(r)

    metrics = {"count": len(preds)}
    if len(preds) == 0:
        args.output.write_text(json.dumps(metrics, indent=2))
        return

    if not args.no_sbert:
        metrics.update(compute_sbert(preds, refs, args.sbert_model))
    if not args.no_bleu:
        metrics.update(compute_bleu(preds, refs))
    if not args.no_rouge:
        metrics.update(compute_rouge(preds, refs))
    if not args.no_meteor:
        metrics.update(compute_meteor(preds, refs))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))

    if args.per_item:
        args.per_item.parent.mkdir(parents=True, exist_ok=True)
        with args.per_item.open("w") as f:
            for r, p, ref in zip(kept_records, preds, refs):
                item = {
                    "question": r.get("question"),
                    "answer": r.get(args.answer_field),
                    "prediction": r.get(args.prediction_field),
                }
                f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
