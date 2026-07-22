"""Candidate models and feature sets, as pipelines.

Everything is a ``Pipeline`` with the scaler inside it, never a scaler fitted
once on the full training matrix. With repeated cross-validation that
distinction is not pedantry: fitting the scaler outside the fold lets each
training fold see the mean and variance of its own validation fold. On a
dataset this separable the leak is small, but it biases exactly the number the
project is reporting, and it is free to avoid.
"""

from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42

# StandardScaler rather than min-max: area_worst has a standard deviation of
# ~569 and fractal_dimension_se ~0.0026, five orders of magnitude apart. Any
# penalised or distance-based model would otherwise be fitted almost entirely
# to the area columns. Tree ensembles are scale-invariant, so they skip it.
def pipeline(model, scale: bool = True) -> Pipeline:
    steps = [("scale", StandardScaler())] if scale else []
    return Pipeline(steps + [("model", model)])


def candidates() -> dict[str, Pipeline]:
    """The model comparison set.

    Chosen to span the assumption space rather than to stack a leaderboard:
    a linear boundary (logistic regression, LDA, linear SVM), a non-linear
    boundary (RBF SVM), a local method (kNN), an independence assumption
    (Gaussian NB), and two tree ensembles. If they all land in the same place —
    which they do — that is evidence about the *problem*, not about tuning.
    """
    return {
        "baseline (majority class)": pipeline(
            DummyClassifier(strategy="most_frequent")
        ),
        "logistic regression (L2)": pipeline(
            LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
        ),
        "logistic regression (L1)": pipeline(
            LogisticRegression(
                max_iter=5000, penalty="l1", solver="liblinear",
                random_state=RANDOM_STATE,
            )
        ),
        "linear discriminant analysis": pipeline(LinearDiscriminantAnalysis()),
        "SVM (RBF)": pipeline(SVC(probability=True, random_state=RANDOM_STATE)),
        "SVM (linear)": pipeline(
            SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
        ),
        "k-nearest neighbours (k=5)": pipeline(KNeighborsClassifier()),
        "gaussian naive bayes": pipeline(GaussianNB()),
        "random forest": pipeline(
            RandomForestClassifier(
                n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
            ),
            scale=False,
        ),
        "gradient boosting": pipeline(
            HistGradientBoostingClassifier(random_state=RANDOM_STATE), scale=False
        ),
    }


def final_model() -> Pipeline:
    """L2 logistic regression on all 30 features.

    Chosen over the RBF SVM and the ensembles, which score within one standard
    deviation of it on AUC, for three reasons that break the tie:

    1. **Calibration.** It has the lowest Brier score of the candidates. The
       whole threshold argument in ``evaluation.py`` depends on the predicted
       probabilities meaning what they say; the random forest's do not.
    2. **Interpretability.** Coefficients on standardised features are directly
       readable, which matters more than a fraction of a point of AUC in a
       setting where a clinician has to be able to ask why.
    3. **Sample size.** At 455 training rows and 30 features, the simplest
       model that matches the others is the one most likely to keep matching
       them elsewhere.

    C is left at the default 1.0: tuning it over a log grid moved
    cross-validated AUC by less than its own fold-to-fold standard deviation,
    so a tuned value would be fitting noise and claiming credit for it.
    """
    return pipeline(
        LogisticRegression(max_iter=5000, C=1.0, random_state=RANDOM_STATE)
    )
