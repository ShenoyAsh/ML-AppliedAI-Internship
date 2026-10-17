"""End-to-end check of the Streamlit app (needs trained models: python train.py)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from diabetes_risk import clinical, survey
from diabetes_risk.i18n import t

APP = str(Path(__file__).resolve().parent.parent / "app.py")
pytestmark = pytest.mark.skipif(
    not (clinical.MODEL_PATH.exists() and survey.MODEL_PATH.exists()),
    reason="run python train.py first")


def run_app(lang="en"):
    at = AppTest.from_file(APP, default_timeout=90)
    at.query_params["lang"] = lang
    at.run()
    assert not at.exception, at.exception
    return at


def widget(elements, key):
    return next(e for e in elements if e.key == key)


@pytest.mark.parametrize("lang", ["en", "hi", "kn"])
def test_app_loads_in_each_language(lang):
    at = run_app(lang)
    assert at.title[0].value.endswith(t(lang, "app.title"))
    assert [tab.label for tab in at.tabs] == [
        t(lang, k) for k in ("tab.quick", "tab.detailed", "tab.models", "tab.about")]


@pytest.mark.parametrize("lang", ["en", "kn"])
def test_quick_check_flow(lang):
    at = run_app(lang)
    widget(at.number_input, "s_age").set_value(62)
    widget(at.number_input, "s_w").set_value(95)
    widget(at.radio, "s_bp").set_value(1)
    widget(at.radio, "s_chol").set_value(None)
    widget(at.select_slider, "s_gen").set_value(4)
    at.button[0].click().run()
    assert not at.exception, at.exception
    levels = " ".join(m.value for m in at.metric)
    assert t(lang, "band.moderate") in levels or t(lang, "band.high") in levels


@pytest.mark.parametrize("glucose,weight,expected", [(95, 55, "low"), (190, 105, "high")])
def test_detailed_check_flow(glucose, weight, expected):
    at = run_app()
    widget(at.number_input, "c_glu").set_value(glucose)
    widget(at.number_input, "c_w").set_value(weight)
    if expected == "high":
        widget(at.number_input, "c_age").set_value(55)
        widget(at.selectbox, "c_fam").set_value("fam.two")
    at.button[1].click().run()
    assert not at.exception, at.exception
    assert any(t("en", f"band.{expected}") in m.value for m in at.metric)
