"""Quick check: CDC BRFSS survey (253,680 US adults), uses only questions a person can
answer without a blood test. Source: UCI ML Repository, "CDC Diabetes Health Indicators".

US-specific questions (income bracket, education level, health insurance, cost of
doctor visits, cholesterol check in 5 years) and the two mental/physical-health
day counts are left out: they don't transfer to Indian users and together add
only ~0.004 ROC-AUC.
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ARTIFACTS_DIR, DATA_DIR, RANDOM_STATE

NAME = "survey"
TITLE = "Quick check (CDC BRFSS survey)"
DATA_PATH = DATA_DIR / "cdc_brfss_diabetes.csv.gz"
MODEL_PATH = ARTIFACTS_DIR / "survey_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "survey_metrics.json"
FIG_PREFIX = "survey_"
CV_REPEATS = 1  # 5 folds over 200k rows is already a very stable estimate

TARGET = "Diabetes_binary"  # 1 = diabetes or prediabetes
FEATURES = ["Age", "Sex", "BMI", "HighBP", "HighChol", "GenHlth", "PhysActivity", "Fruits",
            "Veggies", "Smoker", "HvyAlcoholConsump", "Stroke", "HeartDiseaseorAttack",
            "DiffWalk"]
BINARY = [f for f in FEATURES if f not in ("Age", "BMI", "GenHlth")]

# Questions many people (especially in rural India) have never had tested.
# Answering "never checked" averages the prediction over both answers.
MAY_BE_UNKNOWN = ["HighBP", "HighChol"]

LABELS = {
    "Age": "Age", "Sex": "Sex (male)", "BMI": "BMI", "HighBP": "High blood pressure",
    "HighChol": "High cholesterol", "GenHlth": "General health (poor)",
    "PhysActivity": "Physically active", "Fruits": "Fruit daily", "Veggies": "Vegetables daily",
    "Smoker": "Smoked 100+ cigarettes", "HvyAlcoholConsump": "Heavy drinking",
    "Stroke": "Had a stroke", "HeartDiseaseorAttack": "Heart disease", "DiffWalk": "Difficulty walking",
}


def age_to_category(age_years: float) -> int:
    """BRFSS 13-level age code: 1 = 18-24, 2 = 25-29, ... 12 = 75-79, 13 = 80+."""
    if age_years < 25:
        return 1
    return int(min(13, (age_years - 25) // 5 + 2))


def bmi(height_cm: float, weight_kg: float) -> float:
    return weight_kg / (height_cm / 100) ** 2


def load_data(path=DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError(f"{path} contains blank cells")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError(f"{TARGET} must only contain 0 and 1")
    if not df[BINARY].isin([0, 1]).all(axis=None):
        raise ValueError("Yes/no questions must be coded 0 or 1")
    return df[FEATURES].astype(float), df[TARGET].astype(int)


def missing_counts(X: pd.DataFrame) -> dict:
    return {}


def build_candidates() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([("scale", StandardScaler()),
                                         ("clf", LogisticRegression(max_iter=2000))]),
        "Gradient Boosting": Pipeline([("clf", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
            random_state=RANDOM_STATE))]),
    }


def reference_values(X: pd.DataFrame) -> dict:
    """The 'typical respondent': median of each answer (the most common one for yes/no)."""
    return {f: float(X[f].median()) for f in FEATURES}


def unknown_priors(X: pd.DataFrame) -> dict:
    """Share answering 'yes' to each maybe-unknown question, per age group."""
    return {f: {str(int(a)): float(p) for a, p in X.groupby("Age")[f].mean().items()}
            for f in MAY_BE_UNKNOWN}


def predict_many(model, rows: list[dict], metrics: dict) -> np.ndarray:
    """Probability per row; a None in MAY_BE_UNKNOWN is averaged over yes/no,
    weighted by how common 'yes' is at that person's age."""
    priors = metrics["unknown_priors"]
    expanded, owner, weight = [], [], []
    for i, row in enumerate(rows):
        unknown = [f for f in MAY_BE_UNKNOWN if row.get(f) is None]
        for combo in itertools.product([0.0, 1.0], repeat=len(unknown)):
            filled = {**row, **dict(zip(unknown, combo))}
            w = 1.0
            for f, v in zip(unknown, combo):
                p = priors[f][str(int(row["Age"]))]
                w *= p if v else 1 - p
            expanded.append(filled)
            owner.append(i)
            weight.append(w)
    X = pd.DataFrame([{f: float(r[f]) for f in FEATURES} for r in expanded], columns=FEATURES)
    p = model.predict_proba(X)[:, 1] * np.array(weight)
    return np.bincount(owner, weights=p, minlength=len(rows))
