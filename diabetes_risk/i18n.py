"""Tiny translation helper. Strings live in locales/<code>.json (flat key -> text).

Missing keys fall back to English, so a partial translation never breaks the app.
Text may contain {placeholders} filled with str.format.
"""
import json
from functools import lru_cache

from .config import LOCALES_DIR

LANGUAGES = {"en": "English", "hi": "हिन्दी", "kn": "ಕನ್ನಡ"}
DEFAULT = "en"


@lru_cache(maxsize=None)
def load(lang: str) -> dict[str, str]:
    return json.loads((LOCALES_DIR / f"{lang}.json").read_text(encoding="utf-8"))


def t(lang: str, key: str, **params) -> str:
    text = load(lang).get(key)
    if text is None:
        text = load(DEFAULT).get(key, key)
    return text.format(**params) if params else text
