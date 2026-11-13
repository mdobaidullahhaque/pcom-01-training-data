import json

import numpy as np
import pytest

import config
from src.model import counts as ct

META_PATH = config.MODEL_DIR / "model_meta.json"
SETTINGS = {
    "S0": {},
    "S1": {"edges": [0.0, 1.0, 3.0], "predictions": [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 2, 1, 0, 3]]},
    "S2": {"threshold": 6.0, "severity": "high"},
}
ODD_SCORES = [np.nan, np.inf, -np.inf, -5.0, 0.0, 0.2, 1.5, 3.0, 6.0, 1e12]


@pytest.mark.parametrize("strategy", ct.STRATEGIES)
def test_outputs_are_non_negative_integers_with_unknown_zero(strategy):
    result = ct.predict_counts(ODD_SCORES, SETTINGS, strategy)
    assert result.shape == (len(ODD_SCORES), 5)
    assert result.dtype.kind == "i"
    assert (result >= 0).all()
    assert (result[:, ct.SEVERITIES.index("unknown")] == 0).all()


@pytest.mark.parametrize("strategy", ct.STRATEGIES)
def test_very_low_or_missing_scores_give_all_zeros(strategy):
    result = ct.predict_counts([0.0, 0.3, -1.0, np.nan], SETTINGS, strategy)
    assert (result == 0).all()


def test_threshold_predicts_one_in_its_severity():
    result = ct.predict_counts([5.9, 6.0, 40.0], SETTINGS, "S2")
    assert result.tolist() == [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]


def test_buckets_apply_their_rounded_means():
    result = ct.predict_counts([0.5, 1.0, 2.9, 3.0, 99.0], SETTINGS, "S1")
    assert result.tolist() == [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 2, 1, 0, 0], [1, 2, 1, 0, 0]]


def test_small_buckets_are_merged_downwards():
    scores = np.array([0.1] * 40 + [1.5] * 40 + [12.0] * 5)
    assert ct.merge_small_buckets(scores, [0.0, 1.0, 3.0, 10.0]) == [0.0, 1.0]


def test_threshold_maximises_f1_and_severity_falls_back_to_moderate():
    scores = np.array([0.1, 0.2, 2.0, 3.0, 4.0])
    counts = np.zeros((5, 5), dtype=int)
    counts[3, 2] = 1
    counts[4, 2] = 2
    rule = ct.fit_threshold(scores, counts)
    assert rule["threshold"] == 3.0 and rule["severity"] == "high"
    assert ct.fit_threshold(scores, np.zeros((5, 5), dtype=int))["severity"] == "moderate"


def test_evaluate_on_a_small_case():
    predicted = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]])
    actual = np.array([[0, 1, 0, 0, 0], [0, 0, 2, 0, 0]])
    result = ct.evaluate(predicted, actual)
    assert result["mae_high"] == 1.0 and result["mae_sum"] == 1.0
    assert result["exact_match"] == 0.5
    assert (result["any_precision"], result["any_recall"]) == (1.0, 0.5)


@pytest.mark.skipif(not META_PATH.exists(), reason="model not trained yet")
def test_saved_settings_reproduce_the_table():
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))["counts"]
    scores, counts = ct.load_oof()
    for strategy in ct.STRATEGIES:
        result = ct.evaluate(ct.predict_counts(scores, meta["strategies"], strategy), counts)
        assert result == pytest.approx(meta["evaluation_on_oof"][strategy])
    assert meta["default_strategy"] in ct.STRATEGIES
