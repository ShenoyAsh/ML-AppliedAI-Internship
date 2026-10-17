import numpy as np
import pandas as pd
import pytest

from diabetes_risk import clinical
from diabetes_risk.advice import clinical_tips, factor_contributions
from diabetes_risk.training import threshold_for_sensitivity

HEALTHY = {"Pregnancies": 0, "Glucose": 90, "BloodPressure": 68, "SkinThickness": None,
           "Insulin": None, "BMI": 21.5, "DiabetesPedigreeFunction": 0.2, "Age": 24}
AT_RISK = {"Pregnancies": 6, "Glucose": 185, "BloodPressure": 88, "SkinThickness": 40,
           "Insulin": None, "BMI": 38.0, "DiabetesPedigreeFunction": 1.1, "Age": 55}


@pytest.fixture(scope="module")
def data():
    return clinical.load_data()


@pytest.fixture(scope="module")
def model(data):
    return clinical.build_candidates()["Logistic Regression"].fit(*data)


def test_dataset_shape(data):
    X, y = data
    assert X.shape == (768, 8)
    assert list(X.columns) == clinical.FEATURES
    assert y.sum() == 268


def test_load_rejects_bad_labels(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("1,85,66,29,0,26.6,0.351,31,2\n")
    with pytest.raises(ValueError):
        clinical.load_data(bad)


def test_zero_to_nan_only_touches_medical_columns():
    row = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0.5, 30]], columns=clinical.FEATURES)
    out = clinical.ZeroToNaN().transform(row)
    assert out["Pregnancies"].iloc[0] == 0          # zero pregnancies is valid
    assert out[clinical.ZERO_MEANS_MISSING].isna().all(axis=None)


def test_all_candidates_handle_missing_values(data):
    for name, pipe in clinical.build_candidates().items():
        pipe.fit(*data)
        # Raw candidates (the SVM has no predict_proba until calibrated) -> use predict.
        assert pipe.predict(clinical.to_frame([HEALTHY])).shape == (1,), name


def test_risk_is_monotonic_in_obvious_cases(model):
    p_healthy, p_risk = clinical.predict_many(model, [HEALTHY, AT_RISK])
    assert p_healthy < 0.2 < 0.6 < p_risk


def test_threshold_reaches_target_sensitivity():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    scores = y * 0.3 + rng.random(500)
    t = threshold_for_sensitivity(y, scores, 0.8)
    assert ((scores >= t) & (y == 1)).sum() / y.sum() >= 0.8


def test_contributions_skip_unknown_and_point_the_right_way(model, data):
    reference = clinical.reference_values(data[0])
    contrib = factor_contributions(lambda rows: clinical.predict_many(model, rows),
                                   AT_RISK, reference, clinical.FEATURES)
    assert "Insulin" not in contrib.index
    assert contrib["Glucose"] > 0


def test_clinical_tips():
    keys = [k for k, _ in clinical_tips({**AT_RISK, "Glucose": 210})]
    assert "tip.glucose_diabetes" in keys and "tip.bmi_obese" in keys
    assert clinical_tips({**HEALTHY, "Age": 24}) == [("tip.all_good", {})]
