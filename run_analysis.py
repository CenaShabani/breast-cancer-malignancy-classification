"""Run the whole analysis and write every artifact the README cites.

    python run_analysis.py

Order matters and is the point: the operating threshold is chosen from
out-of-fold predictions on the **training** split, and only then applied to the
held-out test set, once, at the end. Nothing before the final section reads
``y_test``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from src import data as D
from src import evaluation as E
from src import figures as F
from src import models as M
from src import profiling as P

warnings.filterwarnings("ignore", category=FutureWarning)

RESULTS_DIR = Path(__file__).resolve().parent / "outputs" / "results"
SEED = 42

# The sensitivity the tool is asked to hit. 0.98 rather than 0.95, because at
# 0.95 the model misses roughly one malignant case in twenty and the default
# threshold already delivers that by accident; and not 0.99, because reaching
# it costs 13 further points of specificity (see the trade-off table). This is a
# judgement, not a result — it is exposed here so it can be argued with.
TARGET_SENSITIVITY = 0.98


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = D.load()
    X_train, X_test, y_train, y_test = D.split(frame)
    features = D.feature_columns(frame)

    section("1. DATA QUALITY")
    quality = P.quality_report(frame)
    for key, value in quality.items():
        print(f"  {key:34s} {value}")
    (RESULTS_DIR / "quality_report.json").write_text(json.dumps(quality, indent=2, default=str))

    print("\n  Are the 13 zero-concavity rows missing data or small tumours?")
    print(P.zero_concavity_evidence(frame).round(3).to_string())

    section("2. FEATURE STRUCTURE")
    separation = P.univariate_separation(frame)
    separation.to_csv(RESULTS_DIR / "univariate_separation.csv", index=False)
    print("  Strongest single separators:")
    print(separation.head(6).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n  Weakest — indistinguishable from noise:")
    print(separation.tail(4).to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    collin = P.collinearity(frame)
    print(f"\n  Correlated pairs: {collin['pairs_above_0.9']} above r=0.9, "
          f"{collin['pairs_above_0.8']} above r=0.8, of {collin['total_pairs']} total")
    print(f"  VIF above 10: {collin['vif_above_10']} of 30 features "
          f"(max {collin['vif'].iloc[0]:.0f}, {collin['vif'].index[0]})")
    for statistic, identities in collin["geometric_identities"].items():
        print(f"  Geometric redundancy ({statistic}): "
              f"area~pi*r^2 r={identities['area_vs_pi_r_squared']:.4f}, "
              f"perimeter~2*pi*r r={identities['perimeter_vs_2_pi_r']:.4f}")
    collin["vif"].to_csv(RESULTS_DIR / "vif.csv", header=["vif"])

    section("3. MODEL COMPARISON (10x10 repeated stratified CV, train only)")
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=SEED)
    scoring = ["roc_auc", "average_precision", "recall", "precision", "accuracy",
               "neg_brier_score"]
    rows = []
    for name, estimator in M.candidates().items():
        scores = cross_validate(estimator, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        row = {"model": name}
        for metric in scoring:
            row[metric] = scores[f"test_{metric}"].mean()
            row[f"{metric}_sd"] = scores[f"test_{metric}"].std()
        row["brier"] = -row.pop("neg_brier_score")
        row["brier_sd"] = row.pop("neg_brier_score_sd")
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print(comparison[["model", "roc_auc", "recall", "precision", "accuracy", "brier"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    section("4. FEATURE SETS (logistic regression held fixed)")
    # The 1993 authors reported 97.5% accuracy from these three features alone,
    # using a linear-programming decision tree. It is the benchmark this project
    # has to beat to have earned its 30 columns.
    authors_1993 = ["area_worst", "smoothness_worst", "texture_mean"]
    feature_sets = {
        "all 30": features,
        "10 'worst' only": D.block(frame, "worst"),
        "10 'mean' only": D.block(frame, "mean"),
        "10 'SE' only": D.block(frame, "se"),
        "authors' 3 (1993)": authors_1993,
    }
    rows = []
    for name, columns in feature_sets.items():
        scores = cross_validate(
            M.final_model(), X_train[columns], y_train, cv=cv,
            scoring=["roc_auc", "recall", "accuracy"], n_jobs=-1,
        )
        rows.append({
            "feature_set": name,
            "n_features": len(columns),
            "roc_auc": scores["test_roc_auc"].mean(),
            "recall": scores["test_recall"].mean(),
            "accuracy": scores["test_accuracy"].mean(),
            "accuracy_sd": scores["test_accuracy"].std(),
        })
    feature_results = pd.DataFrame(rows)
    feature_results.to_csv(RESULTS_DIR / "feature_sets.csv", index=False)
    print(feature_results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    section("5. CALIBRATION AND THRESHOLD (out-of-fold on train, 20 repeats)")
    calibration_curves = {}
    oof = {}
    for name, (estimator, columns) in {
        "logistic regression": (M.final_model(), features),
        "SVM (RBF)": (M.candidates()["SVM (RBF)"], features),
        "random forest": (M.candidates()["random forest"], features),
    }.items():
        probabilities = E.repeated_oof_probabilities(
            estimator, X_train[columns], y_train, n_repeats=20, random_state=SEED
        )
        oof[name] = probabilities
        mean_probabilities = probabilities.mean(axis=0)
        brier = np.mean([
            ((p - y_train.to_numpy()) ** 2).mean() for p in probabilities
        ])
        calibration_curves[name] = (y_train.to_numpy(), mean_probabilities, brier)
        print(f"  {name:22s} out-of-fold Brier {brier:.4f}")

    probabilities = oof["logistic regression"]
    grid = np.round(np.arange(0.01, 0.71, 0.01), 3)
    sweep = E.threshold_sweep(y_train, probabilities, grid)
    sweep.to_csv(RESULTS_DIR / "threshold_sweep.csv", index=False)

    targets = E.sensitivity_table(sweep, targets=(0.95, 0.97, 0.98, 0.99))
    targets.to_csv(RESULTS_DIR / "sensitivity_targets.csv", index=False)
    print("\n  What each sensitivity target costs (per 455 training patients):")
    print(targets.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    chosen = E.threshold_for_sensitivity(sweep, TARGET_SENSITIVITY)
    if chosen is None:
        raise SystemExit(f"Sensitivity {TARGET_SENSITIVITY} unreachable on training data")
    operating_threshold = float(chosen["threshold"])
    print(f"\n  Operating threshold for {TARGET_SENSITIVITY:.0%} sensitivity: "
          f"{operating_threshold:.2f}")

    # The malignant cases no threshold rescues. Reported rather than buried:
    # they bound what this feature set can do.
    malignant = y_train.to_numpy() == 1
    never_flagged = int((probabilities[:, malignant] < 0.05).all(axis=0).sum())
    print(f"  Malignant cases scored below 0.05 in all 20 repeats: "
          f"{never_flagged} of {malignant.sum()}")

    section("6. HELD-OUT TEST SET — evaluated once")
    final = M.final_model().fit(X_train, y_train)
    test_probabilities = final.predict_proba(X_test)[:, 1]

    results = {}
    for label, threshold in {
        "default 0.50": 0.5,
        f"chosen {operating_threshold:.2f}": operating_threshold,
    }.items():
        summary = E.summarise(y_test, test_probabilities, threshold)
        low, high = E.wilson_interval(summary["tp"], summary["tp"] + summary["fn"])
        summary["recall_ci95"] = [round(low, 4), round(high, 4)]
        results[label] = summary
        print(f"\n  {label}:")
        for key, value in summary.items():
            print(f"    {key:20s} {value if isinstance(value, (int, list)) else f'{value:.4f}'}")

    (RESULTS_DIR / "test_results.json").write_text(
        json.dumps(
            {"operating_threshold": operating_threshold,
             "target_sensitivity": TARGET_SENSITIVITY,
             "results": results},
            indent=2, default=float,
        )
    )

    section("7. FIGURES")
    top_features = separation.head(5)["feature"].tolist() + ["concavity_mean"]
    paths = [
        F.class_balance_and_scale(frame),
        F.correlation_structure(frame),
        F.univariate_separation(separation),
        F.feature_distributions(frame, top_features),
        F.pca_projection(X_train, y_train),
        F.model_comparison(comparison),
        F.roc_and_pr(y_test, test_probabilities),
        F.calibration(calibration_curves),
        F.sensitivity_tradeoff(sweep, targets),
        F.coefficients(final, features),
        F.confusion_at_thresholds(
            y_test, test_probabilities,
            {"Default threshold": 0.5, "Chosen for 98% sensitivity": operating_threshold},
        ),
    ]
    for path in paths:
        print(f"  wrote {path.name}")

    print(f"\nDone. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
