"""Data-quality checks and the redundancy structure of the 30 features.

Separated from the EDA notebook so the claims the README makes about this
dataset are executable assertions rather than remembered observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .data import BASE_FEATURES, ZERO_CONCAVITY_COLUMNS, feature_columns


def quality_report(frame: pd.DataFrame) -> dict:
    """The checks worth running before trusting a single downstream number."""
    features = feature_columns(frame)
    zero_rows = (frame[ZERO_CONCAVITY_COLUMNS] == 0).all(axis=1)
    any_zero = (frame[ZERO_CONCAVITY_COLUMNS] == 0).any(axis=1)

    return {
        "rows": len(frame),
        "features": len(features),
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_ids": int(frame["id"].duplicated().sum()),
        "duplicate_feature_rows": int(frame[features].duplicated().sum()),
        "class_counts": frame["diagnosis"].value_counts().to_dict(),
        "malignant_rate": float(frame["diagnosis"].eq("M").mean()),
        "zero_concavity_rows": int(zero_rows.sum()),
        # If these two disagree, the zeros are scattered and the
        # "structurally convex nucleus" explanation is wrong.
        "zero_concavity_is_all_or_nothing": bool((zero_rows == any_zero).all()),
        "zero_concavity_diagnoses": frame.loc[zero_rows, "diagnosis"]
        .value_counts()
        .to_dict(),
        "scale_ratio": float(frame[features].std().max() / frame[features].std().min()),
    }


def zero_concavity_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    """Are the 13 all-zero concavity rows missing data, or very small tumours?

    The reflex is to impute zeros as missing. The test is whether they look
    like a coding artefact (scattered across the size distribution) or like a
    real biological floor (concentrated among the smallest masses, where a
    nucleus contour genuinely has no concave portions).
    """
    zero_rows = (frame[ZERO_CONCAVITY_COLUMNS] == 0).all(axis=1)
    size = ["radius_mean", "area_mean", "perimeter_mean"]

    return pd.DataFrame(
        {
            "zero_concavity": frame.loc[zero_rows, size].mean(),
            "all_others": frame.loc[~zero_rows, size].mean(),
            "benign_others": frame.loc[~zero_rows & frame["diagnosis"].eq("B"), size].mean(),
            "zero_conc_mean_percentile": [
                float(frame[c].rank(pct=True)[zero_rows].mean()) for c in size
            ],
        }
    )


def univariate_separation(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-feature class separation, ranked.

    AUC rather than a t-statistic as the headline: several of these features
    are strongly right-skewed, and AUC is the rank-based quantity that survives
    that without assuming normality. Mann-Whitney supplies the p-value for the
    same reason. Cohen's d is reported too, but as an effect size to read
    against the AUC, not as the basis for the ranking.
    """
    features = feature_columns(frame)
    malignant = frame["diagnosis"].eq("M")
    rows = []

    for column in features:
        m = frame.loc[malignant, column]
        b = frame.loc[~malignant, column]
        u, p = stats.mannwhitneyu(m, b, alternative="two-sided")
        pooled_sd = np.sqrt(
            ((len(m) - 1) * m.var() + (len(b) - 1) * b.var()) / (len(m) + len(b) - 2)
        )
        rows.append(
            {
                "feature": column,
                "auc": u / (len(m) * len(b)),
                "cohens_d": (m.mean() - b.mean()) / pooled_sd,
                "p_value": p,
                "statistic": column.rsplit("_", 1)[-1],
            }
        )

    result = pd.DataFrame(rows).sort_values("auc", ascending=False)
    # Benjamini-Hochberg: 30 correlated tests on one dataset, so the raw
    # p-values overstate how surprising the smallest of them is.
    result["p_adjusted"] = _benjamini_hochberg(result["p_value"].to_numpy())
    return result.reset_index(drop=True)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """BH step-up adjusted p-values, controlling the false discovery rate."""
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n)
    running_min = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        running_min = min(running_min, p_values[idx] * n / (n - rank + 1))
        adjusted[idx] = running_min
    return adjusted


def collinearity(frame: pd.DataFrame) -> dict:
    """How much independent information the 30 columns actually carry."""
    features = feature_columns(frame)
    X = frame[features]
    corr = X.corr()
    upper = corr.abs().where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()

    scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=features)
    vif = pd.Series(
        [variance_inflation_factor(scaled.values, i) for i in range(len(features))],
        index=features,
    ).sort_values(ascending=False)

    # radius, perimeter and area are not three measurements of a nucleus; they
    # are one measurement expressed three ways. Quantified rather than asserted.
    geometric = {}
    for statistic in ("mean", "worst"):
        r = frame[f"radius_{statistic}"]
        geometric[statistic] = {
            "area_vs_pi_r_squared": float(np.corrcoef(frame[f"area_{statistic}"], np.pi * r**2)[0, 1]),
            "perimeter_vs_2_pi_r": float(np.corrcoef(frame[f"perimeter_{statistic}"], 2 * np.pi * r)[0, 1]),
        }

    return {
        "top_pairs": upper.sort_values(ascending=False).head(10),
        "pairs_above_0.9": int((upper > 0.9).sum()),
        "pairs_above_0.8": int((upper > 0.8).sum()),
        "total_pairs": int(len(upper)),
        "vif": vif,
        "vif_above_10": int((vif > 10).sum()),
        "vif_above_5": int((vif > 5).sum()),
        "geometric_identities": geometric,
        "block_correlations": pd.DataFrame(
            [
                {
                    "feature": base,
                    "mean~worst": corr.loc[f"{base}_mean", f"{base}_worst"],
                    "mean~se": corr.loc[f"{base}_mean", f"{base}_se"],
                    "se~worst": corr.loc[f"{base}_se", f"{base}_worst"],
                }
                for base in BASE_FEATURES
            ]
        ).set_index("feature"),
    }
