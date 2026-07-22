"""Tests that pin the dataset's shape and the split's discipline.

These are not tests of pandas. They are the assumptions every number in the
README rests on, written down so a change upstream fails loudly instead of
quietly shifting a result.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import data as D


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return D.load()


def test_shape_and_classes(frame):
    assert frame.shape == (569, 32)
    assert frame["diagnosis"].value_counts().to_dict() == {"B": 357, "M": 212}


def test_no_missing_and_no_duplicate_patients(frame):
    assert frame.isna().sum().sum() == 0
    # 569 distinct IDs is what makes "569 patients" true rather than
    # "569 rows". Without it the CV folds could share a patient.
    assert frame["id"].nunique() == 569


def test_feature_columns_are_the_thirty_measurements(frame):
    features = D.feature_columns(frame)
    assert len(features) == 30
    assert "id" not in features and "diagnosis" not in features
    # Ten base measurements x three statistics, no more and no less.
    assert set(features) == {
        f"{base}_{stat}" for base in D.BASE_FEATURES for stat in ("mean", "se", "worst")
    }


def test_blocks_partition_the_features(frame):
    blocks = [D.block(frame, s) for s in ("mean", "se", "worst")]
    assert all(len(b) == 10 for b in blocks)
    combined = [c for b in blocks for c in b]
    assert sorted(combined) == sorted(D.feature_columns(frame))


def test_block_rejects_unknown_statistic(frame):
    with pytest.raises(ValueError):
        D.block(frame, "median")


def test_target_encodes_malignant_as_positive(frame):
    _, y = D.xy(frame)
    # Recall must mean "cancers caught". If this ever flips, every headline
    # number in the README silently inverts.
    assert y.sum() == 212
    assert y.loc[frame["diagnosis"].eq("M")].eq(1).all()


def test_split_is_stratified_and_disjoint():
    X_train, X_test, y_train, y_test = D.split()
    assert len(X_train) == 455 and len(X_test) == 114
    assert set(X_train.index).isdisjoint(X_test.index)
    # Stratification holds the malignant share to within a point of the 37.3%
    # population rate, in both halves.
    assert y_train.mean() == pytest.approx(0.3726, abs=0.01)
    assert y_test.mean() == pytest.approx(0.3726, abs=0.01)


def test_split_is_deterministic():
    first = D.split()[0].index.tolist()
    second = D.split()[0].index.tolist()
    assert first == second


def test_zero_concavity_rows_are_all_or_nothing(frame):
    """The 13 zeros are a structural floor, not scattered missing values.

    If a row could be zero on some concavity columns but not others, the
    "perfectly convex nucleus" reading would be wrong and imputation would be
    back on the table.
    """
    zeros = frame[D.ZERO_CONCAVITY_COLUMNS] == 0
    assert (zeros.all(axis=1) == zeros.any(axis=1)).all()
    assert zeros.all(axis=1).sum() == 13
    # All 13 are benign, which is why deleting or imputing them would remove
    # real signal rather than clean up noise.
    assert frame.loc[zeros.all(axis=1), "diagnosis"].eq("B").all()
