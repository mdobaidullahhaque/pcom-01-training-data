"""Turn the chosen score into the 5 target counts. numpy and the standard library only.

Strategies:
  S0  all zeros
  S1  score buckets; each bucket predicts the rounded mean of each target column in training
  S2  threshold: score >= T predicts 1 in one severity, else all zeros

Fitting reads models/pcom_entry_1/oof_predictions.csv and writes the strategy settings and the
evaluation table into model_meta.json, so predict.py can switch strategy without retraining.

Usage: python -u -m src.model.counts
"""
import csv
import json
import math

import numpy as np

import config

SEVERITIES = config.SEVERITIES  # low, moderate, high, critical, unknown
TARGET_COLUMNS = [f"target_ghsa_{severity}_count_90d" for severity in SEVERITIES]
STRATEGIES = ("S0", "S1", "S2")
BUCKET_EDGES = [0.0, 1.0, 3.0, 10.0]  # score lower bounds; score = ghsa_total + 0.5 * probability
MIN_BUCKET_ROWS = 30
FALLBACK_SEVERITY = "moderate"
DEFAULT_STRATEGY = "S2"
CV_FOLDS = 5


def clean_scores(scores):
    """Scores as floats; missing or unusable scores count as 0, which predicts no advisories."""
    values = np.asarray(scores, dtype=float).reshape(-1)
    return np.where(np.isfinite(values), values, 0.0)


def bucket_index(scores, edges):
    return np.clip(np.searchsorted(np.asarray(edges, dtype=float), clean_scores(scores), side="right") - 1, 0, len(edges) - 1)


def merge_small_buckets(scores, edges, min_rows=MIN_BUCKET_ROWS):
    """Drop a bucket's lower edge, joining it to the bucket below, until every bucket has min_rows."""
    edges = list(edges)
    while len(edges) > 1:
        sizes = np.bincount(bucket_index(scores, edges), minlength=len(edges))
        small = [i for i in range(1, len(edges)) if sizes[i] < min_rows]
        if not small:
            break
        del edges[small[-1]]
    return edges


def fit_buckets(scores, counts):
    edges = merge_small_buckets(scores, BUCKET_EDGES)
    index = bucket_index(scores, edges)
    means = [counts[index == bucket].mean(axis=0).tolist() for bucket in range(len(edges))]
    predictions = [[int(math.floor(value + 0.5)) for value in row] for row in means]
    for row in predictions:
        row[SEVERITIES.index("unknown")] = 0
    sizes = np.bincount(index, minlength=len(edges)).tolist()
    return {"edges": edges, "rows": sizes, "means": means, "predictions": predictions}


def any_f1(predicted_any, actual_any):
    hits = int((predicted_any & actual_any).sum())
    precision = hits / predicted_any.sum() if predicted_any.sum() else 0.0
    recall = hits / actual_any.sum() if actual_any.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def fit_threshold(scores, counts):
    """Choose T to maximise F1 of "any advisory"; ties go to the higher T."""
    scores = clean_scores(scores)
    actual = counts.sum(axis=1) > 0
    best_threshold, best_f1 = math.inf, -1.0
    for threshold in np.unique(scores):
        f1 = any_f1(scores >= threshold, actual)[2]
        if f1 > best_f1 or (f1 == best_f1 and threshold > best_threshold):
            best_threshold, best_f1 = float(threshold), f1
    above = actual & (scores >= best_threshold)
    totals = counts[above].sum(axis=0)
    known = {severity: int(totals[i]) for i, severity in enumerate(SEVERITIES) if severity != "unknown"}
    severity = max(known, key=lambda name: (known[name], name == FALLBACK_SEVERITY)) if any(known.values()) else FALLBACK_SEVERITY
    return {"threshold": best_threshold, "severity": severity, "positives_above_by_severity": known}


def predict_counts(scores, settings, strategy):
    """Return an (n, 5) int array of non-negative counts; unknown is always 0. Never raises on bad scores."""
    scores = clean_scores(scores)
    result = np.zeros((len(scores), len(SEVERITIES)), dtype=int)
    if strategy == "S1":
        buckets = settings["S1"]
        result = np.asarray(buckets["predictions"], dtype=int)[bucket_index(scores, buckets["edges"])]
    elif strategy == "S2":
        rule = settings["S2"]
        result[scores >= rule["threshold"], SEVERITIES.index(rule["severity"])] = 1
    result = np.maximum(result, 0)
    result[:, SEVERITIES.index("unknown")] = 0
    return result


def evaluate(predicted, actual):
    errors = predicted - actual
    mae = np.abs(errors).mean(axis=0)
    rmse = np.sqrt((errors ** 2).mean(axis=0))
    precision, recall, f1 = any_f1(predicted.sum(axis=1) > 0, actual.sum(axis=1) > 0)
    return {
        **{f"mae_{severity}": float(value) for severity, value in zip(SEVERITIES, mae)},
        "mae_sum": float(mae.sum()),
        "rmse_sum": float(rmse.sum()),
        "exact_match": float((predicted == actual).all(axis=1).mean()),
        "any_precision": precision,
        "any_recall": recall,
        "any_f1": f1,
        "predicted_positive_rows": int((predicted.sum(axis=1) > 0).sum()),
    }


def cross_validated_threshold(scores, counts, seed=config.RANDOM_SEED):
    """Honest S2 check: choose T on 4 folds of the out-of-fold scores, apply it to the fifth."""
    actual_any = counts.sum(axis=1) > 0
    order = np.random.default_rng(seed).permutation(len(scores))
    folds = np.zeros(len(scores), dtype=int)
    for label in (False, True):  # stratify by label so every fold gets positives
        members = order[actual_any[order] == label]
        folds[members] = np.arange(len(members)) % CV_FOLDS
    predicted = np.zeros_like(counts)
    for fold in range(CV_FOLDS):
        train, test = folds != fold, folds == fold
        settings = {"S2": fit_threshold(scores[train], counts[train])}
        predicted[test] = predict_counts(scores[test], settings, "S2")
    return evaluate(predicted, counts)


def load_oof(path=config.MODEL_DIR / "oof_predictions.csv"):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    scores = np.array([float(row["score_chosen"]) for row in rows])
    counts = np.array([[int(row[column]) for column in TARGET_COLUMNS] for row in rows])
    return scores, counts


def fit_all(scores, counts):
    settings = {"S0": {}, "S1": fit_buckets(scores, counts), "S2": fit_threshold(scores, counts)}
    table = {strategy: evaluate(predict_counts(scores, settings, strategy), counts) for strategy in STRATEGIES}
    return settings, table


def main():
    scores, counts = load_oof()
    settings, table = fit_all(scores, counts)
    honest = cross_validated_threshold(scores, counts)
    meta_path = config.MODEL_DIR / "model_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["counts"] = {
        "default_strategy": DEFAULT_STRATEGY,
        "strategies": settings,
        "evaluation_on_oof": table,
        "s2_cross_validated": honest,
        "note": "S1 and S2 are fitted and evaluated on the same out-of-fold scores, so their table "
                "values are optimistic; s2_cross_validated chooses T on 4/5 and tests on 1/5.",
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"{len(scores)} rows, {int((counts.sum(axis=1) > 0).sum())} positive")
    print("S1 buckets:", list(zip(settings["S1"]["edges"], settings["S1"]["rows"], settings["S1"]["predictions"])))
    print("S1 means:", [[round(v, 3) for v in row] for row in settings["S1"]["means"]])
    print("S2:", settings["S2"])
    metrics = ["mae_low", "mae_moderate", "mae_high", "mae_critical", "mae_unknown", "mae_sum", "rmse_sum",
               "exact_match", "any_precision", "any_recall", "any_f1", "predicted_positive_rows"]
    print(f"{'metric':24}" + "".join(f"{s:>10}" for s in STRATEGIES) + f"{'S2 (CV)':>10}")
    for metric in metrics:
        print(f"{metric:24}" + "".join(f"{table[s][metric]:10.4f}" for s in STRATEGIES) + f"{honest[metric]:10.4f}")
    print(f"Saved to {meta_path} (default {DEFAULT_STRATEGY})")


if __name__ == "__main__":
    main()
