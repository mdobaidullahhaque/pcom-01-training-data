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


EV_SETTINGS = {"EV": {"edges": [0.0, 1.0, 6.0], "means": [[0.0, 0.0, 0.002, 0.0, 0.5], [0.001, 0.004, 0.0, 0.0, 0.0], [0.06, 0.36, 0.12, 0.03, 0.0]]}}


def test_expected_values_are_non_negative_floats_with_unknown_zero():
    result = ct.predict_expected(ODD_SCORES, EV_SETTINGS)
    assert result.shape == (len(ODD_SCORES), 5) and result.dtype.kind == "f"
    assert (result >= 0).all()
    assert (result[:, ct.SEVERITIES.index("unknown")] == 0.0).all()
    assert result[ODD_SCORES.index(1e12)].tolist() == [0.06, 0.36, 0.12, 0.03, 0.0]
    assert result[0].tolist() == [0.0, 0.0, 0.002, 0.0, 0.0]  # NaN score: lowest bucket


def test_expected_buckets_keep_the_threshold_as_top_edge_when_large_enough():
    scores = np.array([0.1] * 50 + [1.5] * 50 + [4.0] * 40 + [8.0] * 35)
    counts = np.zeros((len(scores), 5), dtype=int)
    counts[-5:, 1] = 1
    rule = ct.fit_expected(scores, counts, threshold=6.26)
    assert rule["edges"] == [0.0, 1.0, 3.0, 6.26] and rule["rows"] == [50, 50, 40, 35]
    assert rule["means"][-1][1] == pytest.approx(5 / 35)
    small = ct.fit_expected(np.array([0.1] * 50 + [1.5] * 50 + [4.0] * 40 + [8.0] * 10), counts[:150], threshold=6.26)
    assert small["edges"] == [0.0, 1.0, 3.0] and small["rows"][-1] == 50


@pytest.mark.skipif(not META_PATH.exists(), reason="model not trained yet")
def test_saved_ev_settings_reproduce_the_table():
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))["counts"]
    scores, counts = ct.load_oof()
    result = ct.evaluate(ct.predict_expected(scores, meta["strategies"]), counts)
    assert result == pytest.approx(meta["evaluation_on_oof"]["EV"])
    assert meta["strategies"]["EV"]["edges"][-1] == pytest.approx(meta["strategies"]["S2"]["threshold"])
    assert all(rows >= ct.MIN_BUCKET_ROWS for rows in meta["strategies"]["EV"]["rows"])


@pytest.mark.skipif(not META_PATH.exists(), reason="model not trained yet")
def test_saved_settings_reproduce_the_table():
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))["counts"]
    scores, counts = ct.load_oof()
    for strategy in ct.STRATEGIES:
        result = ct.evaluate(ct.predict_counts(scores, meta["strategies"], strategy), counts)
        assert result == pytest.approx(meta["evaluation_on_oof"][strategy])
    assert meta["default_strategy"] in ct.STRATEGIES


# --- Homework 7: half-rate float strategies (HR, HRMAX) ---

def hand_made_row():
    """3 low in M0, 5 moderate over M0/M2/M11, 1 high in M11, nothing critical, 2 unknown in M5;
    plus a GHSA in a month outside M0..M11 and junk cells that must be ignored."""
    row = {f"ghsa_{s}_count_M{m}": "0" for s in ct.SEVERITIES for m in range(12)}
    row.update({"ghsa_low_count_M0": "3", "ghsa_moderate_count_M0": "2", "ghsa_moderate_count_M2": "1.0",
                "ghsa_moderate_count_M11": "2", "ghsa_high_count_M11": "1", "ghsa_unknown_count_M5": "2",
                "ghsa_critical_count_M12": "9", "ghsa_critical_count_M3": "", "ghsa_high_count_M4": "junk",
                "ghsa_low_count_M7": "-4", "ghsa_moderate_count": "40"})
    return row


def test_half_rate_math_on_a_hand_made_row():
    result = ct.half_rate([hand_made_row()])
    assert result.shape == (1, 5)
    assert result[0].tolist() == [3 / 8, 5 / 8, 1 / 8, 0.0, 2 / 8]
    assert ct.half_rate([hand_made_row()], divisor=4)[0].tolist() == [0.75, 1.25, 0.25, 0.0, 0.5]


def test_half_rate_is_zero_without_ghsa_columns():
    assert ct.half_rate([{}, None, {"ghsa_low_count_M0": None}]).tolist() == [[0.0] * 5] * 3


def test_hrmax_is_at_least_ev_and_hr_per_severity():
    settings = {"EV": {"edges": [0.0, 1.0, 6.0], "means": [[0, 0, 0.001, 0, 0], [0.01, 0.02, 0, 0, 0], [0.06, 0.36, 0.12, 0.03, 0]]}}
    rows = [hand_made_row(), {}, hand_made_row(), {"ghsa_critical_count_M1": "8"}]
    scores = [26.0, 0.2, np.nan, 7.0]
    ev = ct.predict_float(scores, rows, settings, "EV")
    hr = ct.predict_float(scores, rows, settings, "HR")
    hrmax = ct.predict_float(scores, rows, settings, "HRMAX")
    assert (hrmax >= ev).all() and (hrmax >= hr).all() and (hrmax >= 0).all()
    assert np.array_equal(hrmax, np.maximum(ev, hr))
    assert hrmax[1].tolist() == ev[1].tolist()  # no recent GHSAs: keeps the small EV values
    assert hrmax[3, 3] == 1.0  # 8 critical / 8


def test_unknown_float_strategy_raises():
    with pytest.raises(ValueError):
        ct.predict_float([1.0], [{}], {"EV": {"edges": [0.0], "means": [[0] * 5]}}, "XX")


@pytest.mark.skipif(not META_PATH.exists(), reason="model not trained yet")
def test_hrmax_is_at_least_ev_on_all_training_rows():
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))["counts"]
    scores, _, purls = ct.load_oof(with_purls=True)
    training = ct.load_training_rows()
    rows = [training[purl] for purl in purls]
    ev = ct.predict_float(scores, rows, meta["strategies"], "EV")
    assert (ct.predict_float(scores, rows, meta["strategies"], "HRMAX") >= ev).all()


@pytest.mark.skipif(not META_PATH.exists(), reason="model not trained yet")
def test_saved_float_settings_reproduce_the_float_table():
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))["counts"]
    assert meta["default_float_strategy"] in ct.FLOAT_STRATEGIES
    assert meta["strategies"]["HR"]["divisor"] == ct.HALF_RATE_DIVISOR
    scores, counts, purls = ct.load_oof(with_purls=True)
    training = ct.load_training_rows()
    table = ct.float_evaluation(scores, counts, [training[purl] for purl in purls])
    flat = lambda t: {(s, g, k): v for s, groups in t.items() for g, values in groups.items() for k, v in values.items()}
    assert flat(table) == pytest.approx(flat(meta["float_evaluation"]))
    # EV cross-validated here is the same as in cross_validated()
    assert table["EV"]["all"]["mse_sum"] == pytest.approx(meta["cross_validated"]["EV"]["mse_sum"])
