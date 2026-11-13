"""Evaluate baselines and LightGBM with repeated stratified CV, then train and save the final model.

Target: 1 if any target_ghsa_*_count_90d > 0. Every model is scored on exactly the same folds.
Metrics are computed on each repeat's out-of-fold predictions; the table shows mean and standard
deviation over the repeats.

Usage: python -u -m src.model.train
"""
import csv
import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

import config
from src.model import features as ft

SEED = config.RANDOM_SEED
N_SPLITS, N_REPEATS = 5, 5
TOP_K = 50
TARGET_COLUMNS = [f"target_ghsa_{severity}_count_90d" for severity in config.SEVERITIES]
BASE_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": 1,
    "verbose": -1,
}
# At most 4 small settings. With max_depth 3 a tree has at most 8 leaves.
SETTINGS = {
    "lgbm_4leaves_100": {"num_leaves": 4, "n_estimators": 100},
    "lgbm_4leaves_300": {"num_leaves": 4, "n_estimators": 300},
    "lgbm_8leaves_150": {"num_leaves": 8, "n_estimators": 150},
    "lgbm_8leaves_300": {"num_leaves": 8, "n_estimators": 300},
}


def load_dataset(path=config.OUTPUT_CSV):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts = np.array([[int(row[column]) for column in TARGET_COLUMNS] for row in rows])
    y = (counts.sum(axis=1) > 0).astype(int)
    return rows, y, counts


def feature_sets(rows):
    """Return {"with": (X, names), "without": (X, names)} for the reference-date features."""
    references, _ = ft.resolve_reference_dates(rows)
    return {
        "with": ft.build_features(rows, references, use_reference_features=True),
        "without": ft.build_features(rows, references, use_reference_features=False),
    }


def lgbm_params(setting, y_train):
    positives = int(y_train.sum())
    return {**BASE_PARAMS, **SETTINGS[setting], "scale_pos_weight": (len(y_train) - positives) / positives}


def fit_lgbm(setting, X, y, names="auto"):
    model = lgb.LGBMClassifier(**lgbm_params(setting, y))
    model.fit(X, y, feature_name=names)
    return model


def fit_logistic(X, y):
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED),
    )
    return model.fit(X, y)


def rule_score(X, names):
    """Past GHSA total, ties broken by GHSAs in the last 12 months."""
    total = np.nan_to_num(X[:, names.index("ghsa_total")])
    recent = np.nan_to_num(X[:, names.index("ghsa_12m")])
    return total + recent / (recent.max() + 1)


def top_k(scores, y, k=TOP_K):
    """Precision and recall in the k highest scores. Ties keep row order, so results are stable."""
    order = np.argsort(-scores, kind="stable")[:k]
    hits = y[order].sum()
    return hits / k, hits / y.sum()


def metrics(scores, y):
    precision, recall = top_k(scores, y)
    return {
        "pr_auc": average_precision_score(y, scores),
        "roc_auc": roc_auc_score(y, scores),
        "precision_at_50": precision,
        "recall_at_50": recall,
    }


def model_names():
    names = ["rule_ghsa_total"]
    for feature_set in ("with", "without"):
        names.append(f"logreg_{feature_set}")
        names += [f"{setting}_{feature_set}" for setting in SETTINGS]
    return names


def cross_validate(sets, y):
    """Return ({model: [metrics per repeat]}, {model: mean out-of-fold score per row}, [out-of-fold per repeat])."""
    splitter = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    splits = list(splitter.split(np.zeros(len(y)), y))
    results = {name: [] for name in model_names()}
    averaged = {name: np.zeros(len(y)) for name in model_names()}
    per_repeat = []
    X_with, names_with = sets["with"]
    for repeat in range(N_REPEATS):
        oof = {name: np.zeros(len(y)) for name in model_names()}
        for train, test in splits[repeat * N_SPLITS:(repeat + 1) * N_SPLITS]:
            oof["rule_ghsa_total"][test] = rule_score(X_with[test], names_with)
            for feature_set, (X, _) in sets.items():
                oof[f"logreg_{feature_set}"][test] = fit_logistic(X[train], y[train]).predict_proba(X[test])[:, 1]
                for setting in SETTINGS:
                    model = fit_lgbm(setting, X[train], y[train])
                    oof[f"{setting}_{feature_set}"][test] = model.predict_proba(X[test])[:, 1]
        for name, scores in oof.items():
            results[name].append(metrics(scores, y))
            averaged[name] += scores / N_REPEATS
        per_repeat.append(oof)
        print(f"  repeat {repeat + 1}/{N_REPEATS} done")
    return results, averaged, per_repeat


def rank_average(rule, probability):
    """Mean of the two normalised ranks (0 to 1, ties share their average rank)."""
    return (rankdata(rule) / len(rule) + rankdata(probability) / len(probability)) / 2


def lexicographic(ghsa_total, probability):
    """Order by past GHSA total; the LightGBM probability (0 to 1) only breaks ties within a total."""
    return np.nan_to_num(ghsa_total) + 0.5 * probability


def combined_scores(oof, X_with, names_with, lgbm_name):
    return {
        "rank_average": rank_average(oof["rule_ghsa_total"], oof[lgbm_name]),
        "lexicographic": lexicographic(X_with[:, names_with.index("ghsa_total")], oof[lgbm_name]),
    }


def choose_scorer(summary):
    """The combination with the best PR-AUC if it beats the rule alone, otherwise the rule."""
    best = max(("rank_average", "lexicographic"), key=lambda name: summary[name]["pr_auc"][0])
    return best if summary[best]["pr_auc"][0] > summary["rule_ghsa_total"]["pr_auc"][0] else "rule"


def summarize(results):
    return {
        name: {metric: (float(np.mean([r[metric] for r in runs])), float(np.std([r[metric] for r in runs])))
               for metric in runs[0]}
        for name, runs in results.items()
    }


def choose(summary):
    """Best LightGBM setting per feature set by mean PR-AUC; keep 'without' unless 'with' is clearly better."""
    best = {}
    for feature_set in ("with", "without"):
        candidates = [f"{setting}_{feature_set}" for setting in SETTINGS]
        best[feature_set] = max(candidates, key=lambda name: summary[name]["pr_auc"][0])
    with_mean, with_std = summary[best["with"]]["pr_auc"]
    without_mean, without_std = summary[best["without"]]["pr_auc"]
    margin = max(with_std, without_std)
    use_reference = with_mean - without_mean > margin
    chosen = best["with"] if use_reference else best["without"]
    return chosen, use_reference, best, with_mean - without_mean, margin


def train_final(setting, X, y, names):
    """Fit on all rows; the feature names are stored in the model file so predict.py can check them."""
    return fit_lgbm(setting, X, y, names)


def explain(booster, X, names, top=10):
    gain = booster.feature_importance(importance_type="gain")
    contributions = booster.predict(X, pred_contrib=True)[:, :-1]
    order = np.argsort(-gain)[:top]
    return [
        {"feature": names[i], "gain": float(gain[i]),
         "mean_abs_shap": float(np.abs(contributions[:, i]).mean()),
         "mean_shap": float(contributions[:, i].mean())}
        for i in order
    ]


def save(model_dir, booster, meta, oof_rows):
    model_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_dir / "model.txt"))
    (model_dir / "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    with open(model_dir / "oof_predictions.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(oof_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(oof_rows)


def main():
    started = time.perf_counter()
    rows, y, counts = load_dataset()
    sets = feature_sets(rows)
    print(f"{len(y)} rows, {y.sum()} positive ({y.mean():.2%}); features with {len(sets['with'][1])}, without {len(sets['without'][1])}")
    print(f"CV: {N_REPEATS} x stratified {N_SPLITS}-fold, seed {SEED}")
    results, averaged, per_repeat = cross_validate(sets, y)
    chosen, use_reference, best, difference, margin = choose(summarize(results))
    X_with, names_with = sets["with"]
    for name in ("rank_average", "lexicographic"):
        results[name] = []
        averaged[name] = np.zeros(len(y))
    for oof in per_repeat:
        for name, scores in combined_scores(oof, X_with, names_with, chosen).items():
            results[name].append(metrics(scores, y))
            averaged[name] += scores / N_REPEATS
    cv_seconds = time.perf_counter() - started
    summary = summarize(results)
    scorer = choose_scorer(summary)
    scorer_model = {"rule": "rule_ghsa_total", "rank_average": "rank_average", "lexicographic": "lexicographic"}[scorer]

    print(f"\nRandom baseline PR-AUC (positive rate): {y.mean():.4f}")
    print(f"{'model':28} {'PR-AUC':>17} {'ROC-AUC':>17} {'prec@50':>17} {'recall@50':>17}")
    for name, values in summary.items():
        cells = "".join(f"{values[m][0]:9.3f} ± {values[m][1]:5.3f}" for m in ("pr_auc", "roc_auc", "precision_at_50", "recall_at_50"))
        print(f"{name:28} {cells}")

    setting = chosen.rsplit("_", 1)[0]
    print(f"\nBest LightGBM with: {best['with']}, without: {best['without']}; "
          f"PR-AUC difference {difference:+.3f}, needed > {margin:.3f}")
    print(f"Chosen LightGBM: {chosen} (reference-date features {'used' if use_reference else 'not used'})")
    print(f"Chosen scorer: {scorer} (rule PR-AUC {summary['rule_ghsa_total']['pr_auc'][0]:.3f}, "
          f"rank_average {summary['rank_average']['pr_auc'][0]:.3f}, lexicographic {summary['lexicographic']['pr_auc'][0]:.3f})")

    X, names = sets["with" if use_reference else "without"]
    fit_started = time.perf_counter()
    model = train_final(setting, X, y, names)
    fit_seconds = time.perf_counter() - fit_started
    booster = model.booster_
    explanation = explain(booster, X, names)

    meta = {
        "scorer": {
            "type": scorer,
            "uses_lightgbm": scorer != "rule",
            "description": {
                "rule": "ghsa_total, ties broken by ghsa_12m",
                "rank_average": "mean of the normalised ranks of the rule and the LightGBM probability",
                "lexicographic": "ghsa_total + 0.5 * LightGBM probability: total first, probability breaks ties",
            }[scorer],
            "cv_pr_auc": summary[scorer_model]["pr_auc"][0],
        },
        "model": "LightGBM binary classifier, class-weighted",
        "target": "1 if any target_ghsa_*_count_90d > 0",
        "feature_names": names,
        "use_reference_features": use_reference,
        "setting": setting,
        "params": lgbm_params(setting, y),
        "training_t0": config.PREDICTION_DATES[0],
        "training_rows": int(len(y)),
        "training_positives": int(y.sum()),
        "m0_is_most_recent": config.M0_IS_MOST_RECENT,
        "cv": {"scheme": f"{N_REPEATS} x stratified {N_SPLITS}-fold", "seed": SEED, "top_k": TOP_K,
               "random_pr_auc": float(y.mean()),
               "results": {name: {m: {"mean": v[0], "std": v[1]} for m, v in values.items()} for name, values in summary.items()},
               "chosen": chosen, "pr_auc_difference_with_minus_without": difference, "margin": margin},
        "top_features": explanation,
        "versions": {"python": platform.python_version(), "lightgbm": lgb.__version__,
                     "numpy": np.__version__, "scikit_learn": sklearn.__version__},
    }
    oof_rows = [
        {"purl": row["purl"], "y": int(label),
         **{column: int(value) for column, value in zip(TARGET_COLUMNS, count_row)},
         **{f"p_{name}": round(float(averaged[name][index]), 6) for name in [*model_names(), "rank_average", "lexicographic"]},
         "score_chosen": round(float(averaged[scorer_model][index]), 6)}
        for index, (row, label, count_row) in enumerate(zip(rows, y, counts))
    ]
    save(config.MODEL_DIR, booster, meta, oof_rows)

    print("\nTop 10 features by gain (mean |SHAP| and mean SHAP from pred_contrib, log-odds):")
    for item in explanation:
        print(f"  {item['feature']:30} gain {item['gain']:9.1f}  mean|SHAP| {item['mean_abs_shap']:.3f}  mean SHAP {item['mean_shap']:+.3f}")
    print(f"\nTimes: CV {cv_seconds:.1f} s, final fit {fit_seconds:.2f} s, total {time.perf_counter() - started:.1f} s")
    print(f"Saved to {config.MODEL_DIR}")


if __name__ == "__main__":
    main()
