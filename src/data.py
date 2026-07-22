"""Loading, validation, and the single canonical train/test split.

Every notebook and script imports the split from here rather than calling
``train_test_split`` itself. One definition means the held-out set is the same
114 patients everywhere, so "the test set was touched once" is a property of
the code rather than a claim in the README.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PARQUET = PROJECT_ROOT / "data" / "raw" / "wdbc_raw.parquet"

# Fixed for reproducibility. Changing it changes every number in the README.
RANDOM_STATE = 42
TEST_SIZE = 0.2

BASE_FEATURES = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
]

# The label is encoded malignant=1 deliberately: malignant is the event we are
# trying not to miss, so "recall" and "positive" refer to cancer without the
# reader having to re-check which class is which.
POSITIVE_LABEL = "M"

# The 13 rows whose six concavity columns are all exactly zero. They are real
# measurements (perfectly convex nuclei in very small benign masses), not a
# missing-data code — see notebooks/02_eda.ipynb. Named here so the claim is
# checkable rather than buried in prose.
ZERO_CONCAVITY_COLUMNS = [
    "concavity_mean",
    "concave_points_mean",
    "concavity_se",
    "concave_points_se",
    "concavity_worst",
    "concave_points_worst",
]


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """The 30 measurement columns, in file order."""
    return [c for c in frame.columns if c not in ("id", "diagnosis")]


def block(frame: pd.DataFrame, statistic: str) -> list[str]:
    """The 10 columns for one statistic block: ``mean``, ``se``, or ``worst``."""
    if statistic not in ("mean", "se", "worst"):
        raise ValueError(f"statistic must be mean/se/worst, got {statistic!r}")
    return [f"{base}_{statistic}" for base in BASE_FEATURES]


def load() -> pd.DataFrame:
    """Read the acquired parquet and assert the shape the analysis assumes."""
    if not RAW_PARQUET.exists():
        raise FileNotFoundError(
            f"{RAW_PARQUET} not found. Run `python src/acquisition.py` first."
        )
    frame = pd.read_parquet(RAW_PARQUET)

    # These are the assumptions every downstream number rests on. If the
    # upstream file ever changes, this fails here rather than quietly shifting
    # a result.
    if frame.shape != (569, 32):
        raise ValueError(f"Expected (569, 32), got {frame.shape}")
    if frame.isna().any().any():
        raise ValueError("Unexpected missing values; UCI documents none")
    if frame["id"].duplicated().any():
        raise ValueError("Duplicate patient IDs — rows are not independent")

    return frame


def xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split into the feature matrix and a 1=malignant target."""
    y = (frame["diagnosis"] == POSITIVE_LABEL).astype(int)
    y.name = "malignant"
    return frame[feature_columns(frame)], y


def split(frame: pd.DataFrame | None = None):
    """The canonical stratified 80/20 split: 455 train, 114 held out.

    Stratified because the classes are 37/63 and an unstratified 20% draw would
    let the malignant share in the test set wander by several points, which at
    n=114 is enough to move every metric reported against it.
    """
    frame = load() if frame is None else frame
    X, y = xy(frame)
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
