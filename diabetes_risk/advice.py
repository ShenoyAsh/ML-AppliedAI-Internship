"""Turn a model probability into a risk band, an explanation and health tips.

Tips are returned as (locale key, format params) so the app can show them in
any language; see locales/*.json.
"""
from typing import Callable

import numpy as np
import pandas as pd

Tip = tuple[str, dict]


def risk_band(prob: float, screening_threshold: float, high_threshold: float) -> str:
    """'low' | 'moderate' | 'high' — also the locale key suffix for the message."""
    if prob >= high_threshold:
        return "high"
    if prob >= screening_threshold:
        return "moderate"
    return "low"


def factor_contributions(predict_many: Callable[[list[dict]], np.ndarray], values: dict,
                         reference: dict, features: list[str]) -> pd.Series:
    """How much each answer moves the risk compared with a typical person.

    For every known feature we swap the user's value for the population's typical
    value and measure the change in predicted probability. Positive = raises risk.
    Model-agnostic and easy to explain to non-experts. One batched prediction.
    """
    known = [f for f in features if values.get(f) is not None and values[f] != reference[f]]
    probs = predict_many([values] + [{**values, f: reference[f]} for f in known])
    return pd.Series(probs[0] - probs[1:], index=known, dtype=float).sort_values(
        key=np.abs, ascending=False)


def clinical_tips(v: dict) -> list[Tip]:
    """Guidance for the detailed (lab readings) check, from common clinical cut-offs."""
    tips = []
    g = v.get("Glucose")
    if g is not None:
        if g >= 200:
            tips.append(("tip.glucose_diabetes", {}))
        elif g >= 140:
            tips.append(("tip.glucose_igt", {}))
    tips += _bmi_tips(v.get("BMI"))
    bp = v.get("BloodPressure")
    if bp is not None and bp >= 80:
        tips.append(("tip.diastolic_high", {}))
    tips += _age_tips(v.get("Age"))
    dpf = v.get("DiabetesPedigreeFunction")
    if dpf is not None and dpf >= 0.45:
        tips.append(("tip.family_history", {}))
    return tips or [("tip.all_good", {})]


def survey_tips(v: dict, age_years: float) -> list[Tip]:
    """Guidance for the quick (questionnaire) check."""
    tips = []
    tips += _bmi_tips(v.get("BMI"))
    if v.get("HighBP") is None or v.get("HighChol") is None:
        tips.append(("tip.get_checked", {}))
    if v.get("HighBP") == 1:
        tips.append(("tip.bp_known", {}))
    if v.get("PhysActivity") == 0:
        tips.append(("tip.activity", {}))
    if v.get("Fruits") == 0 or v.get("Veggies") == 0:
        tips.append(("tip.diet", {}))
    if v.get("Smoker") == 1:
        tips.append(("tip.tobacco", {}))
    if v.get("HvyAlcoholConsump") == 1:
        tips.append(("tip.alcohol", {}))
    if v.get("GenHlth", 0) >= 4:
        tips.append(("tip.general_health", {}))
    tips += _age_tips(age_years)
    return tips or [("tip.all_good", {})]


def _bmi_tips(bmi) -> list[Tip]:
    if bmi is None:
        return []
    if bmi >= 30:
        return [("tip.bmi_obese", {"bmi": f"{bmi:.1f}"})]
    if bmi >= 23:
        return [("tip.bmi_overweight", {"bmi": f"{bmi:.1f}"})]
    return []


def _age_tips(age) -> list[Tip]:
    if age is not None and age >= 30:
        return [("tip.ncd_screening", {})]
    return []
