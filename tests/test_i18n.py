"""Every language must define every key with the same {placeholders},
and every key the code asks for must exist."""
import re
import string

import pytest

from diabetes_risk import clinical, survey
from diabetes_risk.config import ROOT
from diabetes_risk.i18n import LANGUAGES, load, t


def placeholders(text):
    return {f for _, f, _, _ in string.Formatter().parse(text) if f}


def used_keys():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    advice = (ROOT / "diabetes_risk" / "advice.py").read_text(encoding="utf-8")
    keys = set(re.findall(r'T\(\s*"([a-z_.0-9]+)"', app))
    keys |= set(re.findall(r'"((?:tip|fam)\.[a-z_]+)"', app + advice))
    # Keys built at runtime from a prefix.
    keys |= {f"band.{b}" for b in ("low", "moderate", "high")}
    keys |= {f"band.{b}.msg" for b in ("low", "moderate", "high")}
    keys |= {f"opt.gen{i}" for i in range(1, 6)}
    keys |= {f"feat.{f}" for f in survey.FEATURES + clinical.FEATURES}
    keys |= {f"fig.{n}" for n in ("feature_importance", "roc_curve", "confusion_matrix",
                                  "calibration_curve", "model_comparison")}
    keys |= {f"models.{m}_{s}" for m in ("survey", "clinical") for s in ("heading", "data")}
    return keys


@pytest.mark.parametrize("lang", [code for code in LANGUAGES if code != "en"])
def test_same_keys_and_placeholders_as_english(lang):
    en, other = load("en"), load(lang)
    assert set(other) == set(en), (f"{lang}: missing {set(en) - set(other)}, "
                                   f"extra {set(other) - set(en)}")
    for key, text in en.items():
        assert placeholders(other[key]) == placeholders(text), key


@pytest.mark.parametrize("lang", list(LANGUAGES))
def test_no_empty_strings(lang):
    for key, text in load(lang).items():
        if key != "app.translation_note":   # intentionally empty in English
            assert text.strip(), f"{lang}:{key} is empty"


def test_every_used_key_exists():
    missing = used_keys() - set(load("en"))
    assert not missing, f"keys used in code but missing from en.json: {sorted(missing)}"


def test_fallback_and_formatting():
    assert t("kn", "no.such.key") == "no.such.key"
    assert "{" not in t("hi", "res.vs_average", pts="+5")
