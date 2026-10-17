import pytest

from diabetes_risk import survey
from diabetes_risk.advice import factor_contributions, risk_band, survey_tips

HEALTHY = dict(Age=survey.age_to_category(28), Sex=0, BMI=22, HighBP=0, HighChol=0, GenHlth=2,
               PhysActivity=1, Fruits=1, Veggies=1, Smoker=0, HvyAlcoholConsump=0, Stroke=0,
               HeartDiseaseorAttack=0, DiffWalk=0)
AT_RISK = {**HEALTHY, "Age": survey.age_to_category(62), "Sex": 1, "BMI": 34, "HighBP": 1,
           "HighChol": 1, "GenHlth": 4, "PhysActivity": 0, "DiffWalk": 1}


@pytest.fixture(scope="module")
def data():
    return survey.load_data()


@pytest.fixture(scope="module")
def fitted(data):
    X, y = data
    X, y = X.iloc[:40000], y.iloc[:40000]          # a slice keeps the test fast
    model = survey.build_candidates()["Logistic Regression"].fit(X, y)
    return model, {"unknown_priors": survey.unknown_priors(X)}, survey.reference_values(X)


def test_dataset(data):
    X, y = data
    assert X.shape == (253680, 14)
    assert 0.13 < y.mean() < 0.15


@pytest.mark.parametrize("age,cat", [(18, 1), (24, 1), (25, 2), (29, 2), (30, 3), (62, 9),
                                     (79, 12), (80, 13), (99, 13)])
def test_age_to_category(age, cat):
    assert survey.age_to_category(age) == cat


def test_bmi():
    assert survey.bmi(170, 72.25) == pytest.approx(25.0)


def test_risk_ordering(fitted):
    model, metrics, _ = fitted
    low, high = survey.predict_many(model, [HEALTHY, AT_RISK], metrics)
    assert low < 0.05 < 0.3 < high


def test_never_checked_lies_between_no_and_yes(fitted):
    model, metrics, _ = fitted
    mid = {**HEALTHY, "Age": survey.age_to_category(48), "BMI": 28}
    no, unknown, yes = survey.predict_many(
        model, [mid, {**mid, "HighBP": None, "HighChol": None},
                {**mid, "HighBP": 1, "HighChol": 1}], metrics)
    assert no < unknown < yes


def test_contributions(fitted):
    model, metrics, reference = fitted
    contrib = factor_contributions(lambda rows: survey.predict_many(model, rows, metrics),
                                   {**AT_RISK, "HighChol": None}, reference, survey.FEATURES)
    assert "HighChol" not in contrib.index
    assert contrib["HighBP"] > 0 and contrib["BMI"] > 0


def test_survey_tips():
    keys = [k for k, _ in survey_tips({**AT_RISK, "HighChol": None, "Smoker": 1}, 62)]
    for k in ("tip.bmi_obese", "tip.get_checked", "tip.bp_known", "tip.activity",
              "tip.tobacco", "tip.general_health", "tip.ncd_screening"):
        assert k in keys
    assert survey_tips(HEALTHY, 28) == [("tip.all_good", {})]


@pytest.mark.parametrize("prob,band", [(0.1, "low"), (0.35, "moderate"), (0.7, "high")])
def test_risk_band(prob, band):
    assert risk_band(prob, screening_threshold=0.3, high_threshold=0.5) == band
