"""Metrics, threshold selection, and the cost argument behind them.

The organising idea of this module: **a classifier is a probability model plus
a threshold, and only the first half is learned.** Nearly every published
version of this dataset reports accuracy at the default 0.5 cut, which silently
assumes a false negative and a false positive cost the same. In breast cancer
screening they do not — a false positive costs a biopsy, a false negative can
cost a life. So the threshold is chosen here explicitly, against a stated
sensitivity target, and the specificity it costs is reported alongside it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Reported alongside every model. Ranking metrics (AUC, AP) judge the
# probability ordering and are threshold-free; Brier and log loss judge whether
# the probabilities are *calibrated*, which is what makes a chosen threshold
# mean the same thing on new data.
RANKING_METRICS = ["roc_auc", "average_precision"]
CALIBRATION_METRICS = ["neg_brier_score", "neg_log_loss"]


def confusion_rates(y_true, y_prob, threshold: float) -> dict[str, float]:
    """Confusion matrix and the rates derived from it at one threshold."""
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        # Sensitivity: of the cancers present, the share the model flags.
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        # Specificity: of the benign masses, the share correctly left alone.
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        # Precision: of the flagged masses, the share truly malignant.
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "accuracy": (tp + tn) / len(y_true),
    }


def repeated_oof_probabilities(
    estimator, X, y, n_splits: int = 10, n_repeats: int = 20, random_state: int = 42
) -> np.ndarray:
    """Out-of-fold probabilities, repeated over ``n_repeats`` fold shufflings.

    Returns an array of shape ``(n_repeats, n_samples)``.

    Repeating matters for threshold choice specifically. A threshold read off a
    single 10-fold split is fitted to that split's accidents; at n=455 a shift
    of one or two patients moves the chosen cut noticeably. Averaging the
    trade-off curve over 20 shufflings gives a threshold that is a property of
    the data rather than of one random seed.
    """
    return np.array(
        [
            cross_val_predict(
                estimator,
                X,
                y,
                cv=StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=random_state + r
                ),
                method="predict_proba",
                n_jobs=-1,
            )[:, 1]
            for r in range(n_repeats)
        ]
    )


def threshold_sweep(y_true, prob_matrix: np.ndarray, thresholds) -> pd.DataFrame:
    """Average the confusion rates over repeats at each candidate threshold."""
    prob_matrix = np.atleast_2d(prob_matrix)
    rows = []
    for t in thresholds:
        per_repeat = [confusion_rates(y_true, p, t) for p in prob_matrix]
        frame = pd.DataFrame(per_repeat)
        row = frame.mean(numeric_only=True).to_dict()
        row["threshold"] = t
        row["recall_sd"] = frame["recall"].std()
        row["specificity_sd"] = frame["specificity"].std()
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_for_sensitivity(sweep: pd.DataFrame, target: float) -> pd.Series | None:
    """The **highest** threshold that still meets a sensitivity target.

    Highest, not lowest: any threshold below the target-meeting one also meets
    the target, but throws away specificity for nothing. Taking the loosest
    qualifying cut is the whole point — it buys the required sensitivity at the
    smallest false-positive cost. Returns ``None`` when the target is
    unreachable, which is a real answer and not an error: some malignant cases
    in this data are scored benign at every threshold.
    """
    qualifying = sweep[sweep["recall"] >= target]
    if qualifying.empty:
        return None
    return qualifying.sort_values("threshold").iloc[-1]


def sensitivity_table(sweep: pd.DataFrame, targets=(0.95, 0.97, 0.98, 0.99)) -> pd.DataFrame:
    """What each sensitivity target costs in specificity and false alarms."""
    rows = []
    for target in targets:
        chosen = threshold_for_sensitivity(sweep, target)
        if chosen is None:
            rows.append({"target_recall": target, "reachable": False})
            continue
        rows.append(
            {
                "target_recall": target,
                "reachable": True,
                "threshold": chosen["threshold"],
                "recall": chosen["recall"],
                "specificity": chosen["specificity"],
                "precision": chosen["precision"],
                "accuracy": chosen["accuracy"],
                "false_neg": chosen["fn"],
                "false_pos": chosen["fp"],
            }
        )
    return pd.DataFrame(rows)


def summarise(y_true, y_prob, threshold: float) -> dict[str, float]:
    """Threshold-free quality, calibration quality, and operating point."""
    out = {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, np.clip(y_prob, 1e-12, 1 - 1e-12)),
    }
    out.update(confusion_rates(y_true, y_prob, threshold))
    return out


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Used on the 114-patient test set, where the normal approximation is poor
    near the boundary and every metric sits above 0.95. Reporting a point
    estimate alone there would imply a precision the sample size cannot support:
    one patient is 0.9 percentage points.
    """
    if total == 0:
        return (np.nan, np.nan)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))
