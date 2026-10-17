import numpy as np
import pandas as pd
import pytest

from diabetes_risk.advice import factor_contributions, health_tips, risk_band, to_frame
from diabetes_risk.config import FEATURES
from diabetes_risk.data import load_data, split_xy
from diabetes_risk.modeling import ZeroToNaN, build_candidates, threshold_for_sensitivity

HEALTHY = {"Pregnancies": 0, "Glucose": 90, "BloodPressure": 68, "SkinThickness": None,
           "Insulin": None, "BMI": 21.5, "DiabetesPedigreeFunction": 0.2, "Age": 24}
AT_RISK = {"Pregnancies": 6, "Glucose": 185, "BloodPressure": 88, "SkinThickness": 40,
           "Insulin": None, "BMI": 38.0, "DiabetesPedigreeFunction": 1.1, "Age": 55}


@pytest.fixture(scope="module")
def data():
    return split_xy(load_data())


@pytest.fixture(scope="module")
def model(data):
    X, y = data
    return build_candidates()["Logistic Regression"].fit(X, y)


def test_dataset_shape(data):
    X, y = data
    assert X.shape == (768, 8)
    assert list(X.columns) == FEATURES
    assert y.sum() == 268


def test_load_rejects_bad_labels(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("1,85,66,29,0,26.6,0.351,31,2\n")
    with pytest.raises(ValueError):
        load_data(bad)


def test_zero_to_nan_only_touches_medical_columns():
    row = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0.5, 30]], columns=FEATURES)
    out = ZeroToNaN().transform(row)
    assert out["Pregnancies"].iloc[0] == 0          # zero pregnancies is valid
    assert out[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].isna().all(axis=None)


def test_all_candidates_handle_missing_values(data):
    X, y = data
    sample = to_frame(HEALTHY)
    for name, pipe in build_candidates().items():
        pipe.fit(X, y)
        assert pipe.predict(sample).shape == (1,), name


def test_risk_is_monotonic_in_obvious_cases(model):
    p_healthy = model.predict_proba(to_frame(HEALTHY))[0, 1]
    p_risk = model.predict_proba(to_frame(AT_RISK))[0, 1]
    assert p_healthy < 0.2 < 0.6 < p_risk


def test_threshold_reaches_target_sensitivity():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    scores = y * 0.3 + rng.random(500)
    t = threshold_for_sensitivity(y, scores, 0.8)
    assert ((scores >= t) & (y == 1)).sum() / y.sum() >= 0.8


@pytest.mark.parametrize("prob,level", [(0.1, "Low"), (0.35, "Moderate"), (0.7, "High")])
def test_risk_band(prob, level):
    assert risk_band(prob, screening_threshold=0.3, high_threshold=0.5).level == level


def test_contributions_skip_unknown_and_point_the_right_way(model, data):
    X, _ = data
    reference = X.replace(0, np.nan).median().to_dict()
    contrib = factor_contributions(model, AT_RISK, reference)
    assert "Insulin (2-h serum)" not in contrib.index
    assert contrib["Glucose (2-h OGTT)"] > 0


def test_health_tips():
    assert any("diabetes range" in t for t in health_tips({**AT_RISK, "Glucose": 210}))
    assert any("healthy" in t for t in health_tips(HEALTHY))
