"""End-to-end check of the Streamlit app (needs a trained model: python train.py)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from diabetes_risk.config import MODEL_PATH

APP = str(Path(__file__).resolve().parent.parent / "app.py")
pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run python train.py first")


def run_app():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    return at


def number(at, label_start):
    return next(n for n in at.number_input if n.label.startswith(label_start))


def test_app_loads_with_three_tabs():
    at = run_app()
    assert at.title[0].value.endswith("Diabetes Risk Checker")
    assert len(at.tabs) == 3


@pytest.mark.parametrize("glucose,bmi,expected", [(95, 21.0, "Low"), (190, 40.0, "High")])
def test_prediction_flow(glucose, bmi, expected):
    at = run_app()
    number(at, "Glucose").set_value(glucose)
    number(at, "BMI (kg").set_value(bmi)
    if expected == "High":
        number(at, "Age").set_value(55)
        at.selectbox[0].set_value("Two or more close relatives")
    at.button[0].click().run()
    assert not at.exception
    assert any(expected in m.value for m in at.metric)
