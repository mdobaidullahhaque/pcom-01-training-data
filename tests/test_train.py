import json

import lightgbm as lgb
import numpy as np
import pytest

import config
from src.model import features as ft
from src.model import train as tr


@pytest.fixture(scope="module")
def data():
    rows, y, counts = tr.load_dataset()
    return rows, y, counts, tr.feature_sets(rows)


def test_target_is_any_positive_count(data):
    _, y, counts, _ = data
    assert y.sum() == 26
    assert ((counts.sum(axis=1) > 0) == (y == 1)).all()


def test_training_is_reproducible_with_the_same_seed(data):
    _, y, _, sets = data
    X, names = sets["without"]
    first = tr.train_final("lgbm_4leaves_100", X, y, names).predict_proba(X)[:, 1]
    second = tr.train_final("lgbm_4leaves_100", X, y, names).predict_proba(X)[:, 1]
    assert np.array_equal(first, second)


def test_saved_model_gives_the_same_predictions(data, tmp_path):
    _, y, _, sets = data
    X, names = sets["without"]
    booster = tr.train_final("lgbm_4leaves_100", X, y, names).booster_
    oof = [{"purl": "pkg:npm/a@1.0.0", "y": 0}]
    tr.save(tmp_path, booster, {"feature_names": names}, oof)
    loaded = lgb.Booster(model_file=str(tmp_path / "model.txt"))
    assert np.allclose(loaded.predict(X), booster.predict(X), rtol=0, atol=1e-12)
    assert loaded.feature_name() == names


def test_scale_pos_weight_follows_the_fold():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    assert tr.lgbm_params("lgbm_4leaves_100", y)["scale_pos_weight"] == 4


def test_top_k_and_rule():
    scores = np.array([0.9, 0.8, 0.8, 0.1])
    y = np.array([1, 0, 1, 0])
    assert tr.top_k(scores, y, k=2) == (0.5, 0.5)
    names = ["ghsa_total", "ghsa_12m"]
    X = np.array([[2, 0], [2, 1], [np.nan, 0]], dtype=float)
    ranked = tr.rule_score(X, names)
    assert ranked[1] > ranked[0] > ranked[2]


def test_choice_keeps_without_unless_with_is_clearly_better():
    def summary(with_mean, without_mean, std):
        result = {}
        for setting in tr.SETTINGS:
            result[f"{setting}_with"] = {"pr_auc": (with_mean, std)}
            result[f"{setting}_without"] = {"pr_auc": (without_mean, std)}
        return result

    assert tr.choose(summary(0.30, 0.25, 0.06))[1] is False
    assert tr.choose(summary(0.40, 0.25, 0.06))[1] is True


def test_lexicographic_orders_by_total_then_probability():
    totals = np.array([0, 2, 2, 1, np.nan])
    probability = np.array([0.99, 0.1, 0.9, 1.0, 0.5])
    order = np.argsort(-tr.lexicographic(totals, probability), kind="stable")
    assert list(order) == [2, 1, 3, 0, 4]


def test_rank_average_is_between_zero_and_one_and_shares_ties():
    scores = tr.rank_average(np.array([0, 0, 5, 1]), np.array([0.1, 0.2, 0.4, 0.3]))
    assert ((scores > 0) & (scores <= 1)).all()
    assert list(scores) == [0.3125, 0.4375, 1.0, 0.75]


@pytest.mark.parametrize(
    "rule, rank, lex, expected",
    [(0.28, 0.30, 0.29, "rank_average"), (0.28, 0.29, 0.31, "lexicographic"), (0.28, 0.27, 0.28, "rule")],
)
def test_scorer_choice(rule, rank, lex, expected):
    summary = {"rule_ghsa_total": {"pr_auc": (rule, 0)}, "rank_average": {"pr_auc": (rank, 0.01)},
               "lexicographic": {"pr_auc": (lex, 0.01)}}
    assert tr.choose_scorer(summary) == expected


@pytest.mark.skipif(not (config.MODEL_DIR / "model_meta.json").exists(), reason="model not trained yet")
def test_saved_meta_matches_features_py():
    meta = json.loads((config.MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))
    _, names = ft.build_features([{}], use_reference_features=meta["use_reference_features"])
    assert meta["feature_names"] == names
    booster = lgb.Booster(model_file=str(config.MODEL_DIR / "model.txt"))
    assert booster.feature_name() == names
