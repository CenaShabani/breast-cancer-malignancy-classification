"""Tests for the threshold logic — the part of this project that is a decision.

The metrics themselves come from scikit-learn and are not retested here. What
is tested is the reasoning built on top of them, because that reasoning is the
project's contribution and a subtle error in it would be invisible in the
output.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import evaluation as E


def test_confusion_rates_at_a_known_point():
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    p = [0.1, 0.2, 0.6, 0.3, 0.9, 0.8, 0.4, 0.2]
    r = E.confusion_rates(y, p, threshold=0.5)
    assert (r["tp"], r["fp"], r["tn"], r["fn"]) == (2, 1, 3, 2)
    assert r["recall"] == pytest.approx(0.5)
    assert r["specificity"] == pytest.approx(0.75)
    assert r["precision"] == pytest.approx(2 / 3)


def test_lowering_the_threshold_cannot_reduce_recall():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = rng.random(300)
    sweep = E.threshold_sweep(y, p[None, :], np.linspace(0.05, 0.95, 19))
    recalls = sweep.sort_values("threshold", ascending=False)["recall"].to_numpy()
    # Monotone non-decreasing as the threshold falls. This is the property the
    # whole sensitivity argument depends on.
    assert np.all(np.diff(recalls) >= -1e-12)


def test_threshold_for_sensitivity_picks_the_highest_qualifying_cut():
    """The bug this test exists to prevent.

    Every threshold below a qualifying one also qualifies, so taking the first
    row of an ascending sweep returns the *loosest* possible model — target
    sensitivity met, specificity destroyed. The correct answer is the highest
    threshold that still clears the bar.
    """
    y = np.array([0] * 50 + [1] * 50)
    # Malignant scores 0.30-0.79, benign 0.00-0.49: overlapping, so the choice
    # of threshold genuinely trades the two rates off.
    p = np.concatenate([np.linspace(0.00, 0.49, 50), np.linspace(0.30, 0.79, 50)])
    sweep = E.threshold_sweep(y, p[None, :], np.round(np.arange(0.01, 0.80, 0.01), 2))

    chosen = E.threshold_for_sensitivity(sweep, 0.90)
    assert chosen["recall"] >= 0.90

    # Nothing stricter would have worked...
    stricter = sweep[sweep["threshold"] > chosen["threshold"]]
    assert (stricter["recall"] < 0.90).all()
    # ...and the chosen cut keeps more specificity than the loosest one.
    assert chosen["specificity"] > sweep.sort_values("threshold").iloc[0]["specificity"]


def test_unreachable_sensitivity_returns_none_rather_than_guessing():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.0])
    sweep = E.threshold_sweep(y, p[None, :], np.round(np.arange(0.01, 0.5, 0.01), 2))
    # One malignant case scores 0.0 and is never flagged above threshold 0.01,
    # so 100% sensitivity is genuinely impossible on this grid.
    assert E.threshold_for_sensitivity(sweep, 1.0) is None


def test_sensitivity_table_marks_unreachable_targets():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.0, 0.9])
    sweep = E.threshold_sweep(y, p[None, :], np.round(np.arange(0.01, 0.5, 0.01), 2))
    table = E.sensitivity_table(sweep, targets=(0.5, 1.0))
    assert table.set_index("target_recall").loc[0.5, "reachable"]
    assert not table.set_index("target_recall").loc[1.0, "reachable"]


def test_repeated_oof_probabilities_shape_and_range():
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    X, y = make_classification(n_samples=120, n_features=6, random_state=0)
    probabilities = E.repeated_oof_probabilities(
        LogisticRegression(), X, y, n_splits=5, n_repeats=3
    )
    assert probabilities.shape == (3, 120)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    # Different fold shufflings must actually produce different predictions,
    # otherwise averaging over repeats is buying nothing.
    assert not np.allclose(probabilities[0], probabilities[1])


@pytest.mark.parametrize(
    "successes,total",
    [(42, 42), (0, 30), (39, 42)],
)
def test_wilson_interval_stays_in_bounds_and_contains_estimate(successes, total):
    low, high = E.wilson_interval(successes, total)
    assert 0.0 <= low <= high <= 1.0
    assert low <= successes / total <= high


def test_wilson_interval_is_asymmetric_at_the_boundary():
    """Why Wilson and not the normal approximation.

    At 42/42 the textbook interval is [1.0, 1.0] — zero uncertainty from 42
    observations, which is nonsense. Wilson keeps a lower bound below 1.
    """
    low, high = E.wilson_interval(42, 42)
    assert high == pytest.approx(1.0)
    assert low < 0.94
