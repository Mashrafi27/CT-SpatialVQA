#!/usr/bin/env python3
"""Run non-API text metrics for all prediction JSONL files in one shot.

Metrics:
- Sentence-BERT cosine similarity
- BERTScore (P/R/F1)
- BLEU (sacrebleu)
- ROUGE (rouge1/rouge2/rougeL)
- METEOR (nltk)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch text metrics for multiple prediction files")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("3D_VLM_Spatial/reports"),
        help="Directory containing *_predictions.jsonl files",
    )
    parser.add_argument(
        "--glob",
        default="*_predictions.jsonl",
        help="Glob pattern within reports-dir",
    )
    parser.add_argument(
        "--predictions",
        nargs="*",
        type=Path,
        default=None,
        help="Optional explicit list of prediction files (overrides --reports-dir/--glob)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("3D_VLM_Spatial/reports/text_metrics_summary.csv"),
        help="CSV summary output path",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("3D_VLM_Spatial/reports/text_metrics_summary.json"),
        help="JSON summary output path",
    )
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--skip-empty", action="store_true")
    parser.add_argument("--lower", action="store_true")

    parser.add_argument("--sbert-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--bertscore-model", default="roberta-base")
    parser.add_argument("--no-sbert", action="store_true")
    parser.add_argument("--no-bertscore", action="store_true")
    parser.add_argument("--no-bleu", action="store_true")
    parser.add_argument("--no-rouge", action="store_true")
    parser.add_argument("--no-meteor", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from existing summary outputs")
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


def normalize(text: Optional[str], lower: bool) -> str:
    if text is None:
        return ""
    text = text.strip()
    if lower:
        text = text.lower()
    return text


def compute_sbert(preds: List[str], refs: List[str], model, batch_size: int = 64) -> Dict[str, float]:
    pred_embs = []
    ref_embs = []
    for start in tqdm(range(0, len(preds), batch_size), desc="SBERT batches", leave=False):
        end = start + batch_size
        pred_embs.append(
            model.encode(
                preds[start:end],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )
        ref_embs.append(
            model.encode(
                refs[start:end],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )
    pred_emb = np.concatenate(pred_embs, axis=0)
    ref_emb = np.concatenate(ref_embs, axis=0)
    sims = (pred_emb * ref_emb).sum(axis=1)
    return {"sbert_cosine": float(np.mean(sims))}


def compute_bertscore(preds: List[str], refs: List[str], scorer, batch_size: int = 32) -> Dict[str, float]:
    p_all, r_all, f1_all = [], [], []
    for start in tqdm(range(0, len(preds), batch_size), desc="BERTScore batches", leave=False):
        end = start + batch_size
        p, r, f1 = scorer.score(preds[start:end], refs[start:end])
        p_all.append(p)
        r_all.append(r)
        f1_all.append(f1)
    p = torch.cat(p_all)
    r = torch.cat(r_all)
    f1 = torch.cat(f1_all)
    return {
        "bertscore_precision": float(p.mean()),
        "bertscore_recall": float(r.mean()),
        "bertscore_f1": float(f1.mean()),
    }


def compute_bleu(preds: List[str], refs: List[str]) -> Dict[str, float]:
    import sacrebleu
    for _ in tqdm(range(1), desc="BLEU", leave=False):
        bleu = sacrebleu.corpus_bleu(preds, [refs])
    return {"bleu": float(bleu.score)}


def compute_rouge(preds: List[str], refs: List[str], scorer, show_progress: bool) -> Dict[str, float]:
    rouge1, rouge2, rougel = [], [], []
    iterator = zip(preds, refs)
    if show_progress:
        iterator = tqdm(iterator, total=len(preds), desc="ROUGE per-item", leave=False)
    for p, r in iterator:
        scores = scorer.score(r, p)
        rouge1.append(scores["rouge1"].fmeasure)
        rouge2.append(scores["rouge2"].fmeasure)
        rougel.append(scores["rougeL"].fmeasure)
    return {
        "rouge1_f": float(np.mean(rouge1)),
        "rouge2_f": float(np.mean(rouge2)),
        "rougeL_f": float(np.mean(rougel)),
    }


def compute_meteor(preds: List[str], refs: List[str], show_progress: bool) -> Dict[str, float]:
    from nltk.translate.meteor_score import meteor_score
    iterator = zip(preds, refs)
    if show_progress:
        iterator = tqdm(iterator, total=len(preds), desc="METEOR per-item", leave=False)
    scores = []
    for p, r in iterator:
        # METEOR expects tokenized input
        p_tok = p.split()
        r_tok = r.split()
        scores.append(meteor_score([r_tok], p_tok))
    return {"meteor": float(np.mean(scores))}


def main() -> None:
    args = parse_args()

    if args.predictions:
        files = [p for p in args.predictions if p.exists()]
    else:
        files = sorted(args.reports_dir.glob(args.glob))

    if not files:
        raise SystemExit("No prediction files found.")

    # Init heavy models once
    sbert_model = None
    if not args.no_sbert:
        from sentence_transformers import SentenceTransformer
        sbert_model = SentenceTransformer(args.sbert_model)

    bert_scorer = None
    if not args.no_bertscore:
        from bert_score import BERTScorer
        bert_scorer = BERTScorer(model_type=args.bertscore_model, lang="en", rescale_with_baseline=False)

    rouge_scorer = None
    if not args.no_rouge:
        from rouge_score import rouge_scorer as rouge_scorer_mod
        rouge_scorer = rouge_scorer_mod.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    results = []
    existing_by_model = {}
    if args.resume and args.output_json.exists():
        try:
            existing = json.loads(args.output_json.read_text())
            if isinstance(existing, list):
                existing_by_model = {r.get("model"): r for r in existing if r.get("model")}
                results = list(existing)
        except Exception:
            existing_by_model = {}
    for path in tqdm(files, desc="Evaluating files"):
        records = load_jsonl(path)
        preds, refs = [], []
        for r in records:
            pred = normalize(r.get(args.prediction_field, ""), args.lower)
            ref = normalize(r.get(args.answer_field, ""), args.lower)
            if args.skip_empty and (not pred or not ref):
                continue
            if not ref:
                continue
            preds.append(pred)
            refs.append(ref)

        model_name = path.stem.replace("_predictions", "")
        if args.resume and model_name in existing_by_model:
            continue
        row = {"model": model_name, "count": len(preds)}
        if len(preds) == 0:
            results.append(row)
            # Persist progress after each file
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(results, indent=2))
            fieldnames = sorted({k for r in results for k in r.keys()})
            args.output_csv.parent.mkdir(parents=True, exist_ok=True)
            with args.output_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            continue

        if sbert_model is not None:
            row.update(compute_sbert(preds, refs, sbert_model))
        if bert_scorer is not None:
            row.update(compute_bertscore(preds, refs, bert_scorer))
        if not args.no_bleu:
            row.update(compute_bleu(preds, refs))
        if rouge_scorer is not None:
            row.update(compute_rouge(preds, refs, rouge_scorer, show_progress=True))
        if not args.no_meteor:
            row.update(compute_meteor(preds, refs, show_progress=True))

        results.append(row)

        # Persist progress after each file
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2))
        fieldnames = sorted({k for r in results for k in r.keys()})
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
