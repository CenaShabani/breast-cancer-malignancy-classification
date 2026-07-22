"""Tests that the modelling setup does not leak and the profiling maths is right."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src import data as D
from src import models as M
from src import profiling as P


def test_scaling_lives_inside_the_pipeline():
    """No candidate may be handed a pre-scaled matrix.

    If the scaler sat outside cross-validation, each training fold would be
    standardised using statistics computed from its own validation fold. That
    is the most common silent leak in a project shaped like this one.
    """
    for name, estimator in M.candidates().items():
        steps = dict(estimator.named_steps)
        if "scale" in steps:
            assert isinstance(steps["scale"], StandardScaler)
        # Every candidate must be a Pipeline, so cross_validate refits the
        # whole chain per fold rather than reusing a global transform.
        assert hasattr(estimator, "named_steps"), name


def test_final_model_is_a_pipeline_with_scaling():
    model = M.final_model()
    assert list(model.named_steps) == ["scale", "model"]


def test_final_model_predicts_probabilities_in_range():
    X_train, X_test, y_train, _ = D.split()
    fitted = M.final_model().fit(X_train, y_train)
    p = fitted.predict_proba(X_test)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
    assert len(p) == 114


def test_benjamini_hochberg_against_a_known_case():
    # Worked by hand: p * n / rank, then enforced monotone from the largest down.
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    adjusted = P._benjamini_hochberg(p)
    assert adjusted == pytest.approx([0.05, 0.05, 0.05, 0.05, 0.05])
    assert np.all(adjusted >= p)


def test_benjamini_hochberg_is_monotone():
    rng = np.random.default_rng(3)
    p = rng.random(40)
    adjusted = P._benjamini_hochberg(p)
    order = np.argsort(p)
    assert np.all(np.diff(adjusted[order]) >= -1e-12)
    assert np.all(adjusted <= 1.0)


def test_univariate_separation_auc_matches_roc_auc_score():
    """The rank-based AUC computed from Mann-Whitney U must equal sklearn's."""
    from sklearn.metrics import roc_auc_score

    frame = D.load()
    separation = P.univariate_separation(frame).set_index("feature")
    y = frame["diagnosis"].eq("M").astype(int)
    for column in ["perimeter_worst", "symmetry_se", "texture_mean"]:
        assert separation.loc[column, "auc"] == pytest.approx(
            roc_auc_score(y, frame[column]), abs=1e-9
        )


def test_collinearity_finds_the_geometric_identities():
    """radius, perimeter and area are one measurement, not three.

    area = pi*r^2 and perimeter = 2*pi*r are definitions, so if the profiling
    code reports these correlations as anything other than ~1 it is broken.
    """
    result = P.collinearity(D.load())
    for statistic in ("mean", "worst"):
        identities = result["geometric_identities"][statistic]
        assert identities["area_vs_pi_r_squared"] > 0.99
        assert identities["perimeter_vs_2_pi_r"] > 0.99
    assert result["vif_above_10"] >= 20


def test_quality_report_flags_a_duplicated_patient():
    """The guard has to actually fire, not just pass on clean data."""
    frame = D.load()
    corrupted = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    assert P.quality_report(corrupted)["duplicate_ids"] == 1
