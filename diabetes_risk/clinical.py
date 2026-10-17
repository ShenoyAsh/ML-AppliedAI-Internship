"""Detailed check: Pima Indians Diabetes Dataset, uses lab readings (2-h glucose, insulin).

All preprocessing lives inside the pipeline so it is learned from training
folds only (no leakage into validation or test data).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import ARTIFACTS_DIR, DATA_DIR, RANDOM_STATE

NAME = "clinical"
TITLE = "Detailed check (Pima Indians Diabetes Dataset)"
DATA_PATH = DATA_DIR / "pima_diabetes.csv"
MODEL_PATH = ARTIFACTS_DIR / "clinical_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "clinical_metrics.json"
FIG_PREFIX = "clinical_"
CV_REPEATS = 5

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
TARGET = "Outcome"

# A reading of 0 is physiologically impossible for these columns; in the Pima
# dataset it means "not measured", so it is treated as missing.
ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# English labels for the report charts (the app uses locales/*.json).
LABELS = {
    "Pregnancies": "Pregnancies",
    "Glucose": "Glucose (2-h OGTT)",
    "BloodPressure": "Diastolic blood pressure",
    "SkinThickness": "Skin-fold thickness",
    "Insulin": "Insulin (2-h serum)",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "Family history score",
    "Age": "Age",
}


def load_data(path=DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Read the header-less CSV and check it has the expected shape and values."""
    df = pd.read_csv(path, header=None, names=FEATURES + [TARGET])
    if df.isna().any().any():
        raise ValueError(f"{path} contains blank cells or non-numeric values")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError(f"{TARGET} must only contain 0 and 1")
    if (df[FEATURES] < 0).any().any():
        raise ValueError("Feature values cannot be negative")
    return df[FEATURES], df[TARGET]


def missing_counts(X: pd.DataFrame) -> dict:
    return {c: int((X[c] == 0).sum()) for c in ZERO_MEANS_MISSING}


class ZeroToNaN(BaseEstimator, TransformerMixin):
    """Replace impossible zero readings with NaN so the imputer can fill them."""

    def __init__(self, columns=tuple(ZERO_MEANS_MISSING)):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=FEATURES).astype(float)
        cols = list(self.columns)
        X[cols] = X[cols].replace(0, np.nan)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURES, dtype=object)


def _preprocess():
    # add_indicator lets the model learn whether a reading was missing,
    # which is itself informative (e.g. insulin is often not tested).
    return [("zero_to_nan", ZeroToNaN()),
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler())]


def build_candidates() -> dict[str, Pipeline]:
    """Candidate models compared by cross-validation."""
    return {
        "Logistic Regression": Pipeline(_preprocess() + [
            ("clf", LogisticRegression(max_iter=2000, C=1.0))]),
        "Random Forest": Pipeline(_preprocess() + [
            ("clf", RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=3,
                                           random_state=RANDOM_STATE, n_jobs=-1))]),
        "Gradient Boosting": Pipeline(_preprocess() + [
            ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                   max_iter=200, l2_regularization=1.0,
                                                   random_state=RANDOM_STATE))]),
        "Support Vector Machine": Pipeline(_preprocess() + [
            ("clf", SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE))]),
    }


def reference_values(X: pd.DataFrame) -> dict:
    """The 'typical patient' used for explanations: medians of real readings."""
    return {f: float(X.loc[X[f] > 0, f].median()) if f in ZERO_MEANS_MISSING
            else float(X[f].median()) for f in FEATURES}


def to_frame(rows: list[dict]) -> pd.DataFrame:
    """Rows in model column order; None (not measured) becomes NaN for the imputer."""
    return pd.DataFrame([{f: np.nan if r.get(f) is None else float(r[f]) for f in FEATURES}
                         for r in rows], columns=FEATURES)


def predict_many(model, rows: list[dict], metrics: dict | None = None) -> np.ndarray:
    return model.predict_proba(to_frame(rows))[:, 1]
