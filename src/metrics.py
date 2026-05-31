from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
import torch
from torch.nn import MSELoss


try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None


try:
    import evaluate
except ImportError:
    evaluate = None


cos = torch.nn.CosineSimilarity(dim=1)
mse_loss = MSELoss()


def eval_embeddings(x: torch.Tensor, y: torch.Tensor):
    return cos(x, y).mean(), mse_loss(x, y)


def _f1(overlap: int, pred_total: int, ref_total: int) -> float:
    if overlap == 0 or pred_total == 0 or ref_total == 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / ref_total
    return 2 * precision * recall / (precision + recall)


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def _lcs(left: List[str], right: List[str]) -> int:
    prev = [0] * (len(right) + 1)
    for left_token in left:
        cur = [0]
        for idx, right_token in enumerate(right, start=1):
            if left_token == right_token:
                cur.append(prev[idx - 1] + 1)
            else:
                cur.append(max(prev[idx], cur[-1]))
        prev = cur
    return prev[-1]


def fallback_rouge(predictions: List[str], references: List[str]) -> Dict[str, List[float]]:
    results = defaultdict(list)
    for prediction, reference in zip(predictions, references):
        pred_tokens = prediction.split()
        ref_tokens = reference.split()

        pred_uni = _ngrams(pred_tokens, 1)
        ref_uni = _ngrams(ref_tokens, 1)
        results["rouge1_f"].append(
            _f1(sum((pred_uni & ref_uni).values()), len(pred_tokens), len(ref_tokens))
        )

        pred_bi = _ngrams(pred_tokens, 2)
        ref_bi = _ngrams(ref_tokens, 2)
        results["rouge2_f"].append(
            _f1(
                sum((pred_bi & ref_bi).values()),
                max(0, len(pred_tokens) - 1),
                max(0, len(ref_tokens) - 1),
            )
        )

        results["rougeL_f"].append(
            _f1(_lcs(pred_tokens, ref_tokens), len(pred_tokens), len(ref_tokens))
        )
    return results


def rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, List[float]]:
    if rouge_scorer is None:
        return fallback_rouge(predictions, references)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results = defaultdict(list)
    for prediction, reference in zip(predictions, references):
        scores = scorer.score(reference, prediction)
        for metric, score in scores.items():
            results[f"{metric}_f"].append(score.fmeasure)
    return results


def generation_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    rouge = rouge_scores(predictions, references)
    exact = np.array(predictions) == np.array(references)

    metrics = {
        "rougeL": round(float(np.mean(rouge["rougeL_f"])), 4),
        "rouge1": round(float(np.mean(rouge["rouge1_f"])), 4),
        "rouge2": round(float(np.mean(rouge["rouge2_f"])), 4),
        "exact_match": round(float(exact.mean()), 4),
        "bleu": -1.0,
        "bleu1": -1.0,
        "bleu2": -1.0,
        "bleu3": -1.0,
        "bleu4": -1.0,
    }

    if evaluate is not None:
        try:
            bleu_metric = evaluate.load("sacrebleu")
            bleu = bleu_metric.compute(predictions=predictions, references=references)
            metrics.update(
                {
                    "bleu": round(float(bleu["score"]), 2),
                    "bleu1": round(float(bleu["precisions"][0]), 2),
                    "bleu2": round(float(bleu["precisions"][1]), 2),
                    "bleu3": round(float(bleu["precisions"][2]), 2),
                    "bleu4": round(float(bleu["precisions"][3]), 2),
                }
            )
        except Exception as exc:
            print(f"BLEU unavailable: {exc}")

    return metrics
