"""Turn a model probability into a risk band, an explanation and health tips."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FEATURES, LABELS


@dataclass(frozen=True)
class RiskBand:
    level: str        # "Low" | "Moderate" | "High"
    message: str


def risk_band(prob: float, screening_threshold: float, high_threshold: float) -> RiskBand:
    if prob >= high_threshold:
        return RiskBand("High", "High risk — please see a doctor soon for an HbA1c or "
                                "fasting blood glucose test.")
    if prob >= screening_threshold:
        return RiskBand("Moderate", "Moderate risk — a confirmatory blood test is recommended, "
                                    "along with the lifestyle steps below.")
    return RiskBand("Low", "Low risk — keep up a healthy lifestyle and re-check every few years.")


def to_frame(values: dict) -> pd.DataFrame:
    """Single-row DataFrame in model column order; None becomes NaN (unknown)."""
    row = {f: (np.nan if values.get(f) is None else float(values[f])) for f in FEATURES}
    return pd.DataFrame([row], columns=FEATURES)


def factor_contributions(model, values: dict, reference: dict) -> pd.Series:
    """How much each reading moves the risk compared with a typical person.

    For every known feature we swap the user's value for the population median
    and measure the change in predicted probability. Positive = raises risk.
    Model-agnostic and easy to explain to non-experts.
    """
    base = to_frame(values)
    p_user = model.predict_proba(base)[0, 1]
    known = [f for f in FEATURES if values.get(f) is not None]
    variants = pd.concat([base.assign(**{f: reference[f]}) for f in known], ignore_index=True)
    p_ref = model.predict_proba(variants)[:, 1]
    return pd.Series(p_user - p_ref, index=[LABELS[f] for f in known]).sort_values(
        key=np.abs, ascending=False)


def health_tips(values: dict) -> list[str]:
    """Plain-language guidance based on widely used clinical cut-offs."""
    tips = []
    g = values.get("Glucose")
    if g is not None:
        if g >= 200:
            tips.append("Your 2-hour glucose is in the **diabetes range (≥ 200 mg/dL)**. "
                        "Please see a doctor promptly.")
        elif g >= 140:
            tips.append("Your 2-hour glucose suggests **impaired glucose tolerance "
                        "(140–199 mg/dL, prediabetes)** — this is reversible with diet and activity.")
    bmi = values.get("BMI")
    if bmi is not None:
        if bmi >= 30:
            tips.append("BMI ≥ 30 is in the obese range. Losing even **5–7 % of body weight** "
                        "cuts diabetes risk by more than half in prevention trials.")
        elif bmi >= 23:
            tips.append("BMI ≥ 23 already raises diabetes risk for people of South Asian descent. "
                        "Aim for **150 minutes of brisk activity per week**.")
    bp = values.get("BloodPressure")
    if bp is not None and bp >= 80:
        tips.append("A diastolic pressure ≥ 80 mm Hg is classed as high blood pressure — "
                    "reduce salt and get it checked, as it compounds diabetes risk.")
    age = values.get("Age")
    if age is not None and age >= 35:
        tips.append("Guidelines recommend **routine diabetes screening for all adults 35+**, "
                    "even without symptoms.")
    dpf = values.get("DiabetesPedigreeFunction")
    if dpf is not None and dpf >= 0.5:
        tips.append("Your family-history score is above average — tell your doctor which "
                    "relatives have diabetes.")
    if not tips:
        tips.append("Your readings look healthy. Keep eating whole foods, staying active and "
                    "sleeping well.")
    return tips
