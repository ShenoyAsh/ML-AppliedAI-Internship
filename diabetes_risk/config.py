"""Settings shared by both models."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
FIGURES_DIR = ROOT / "reports" / "figures"
LOCALES_DIR = ROOT / "locales"

RANDOM_STATE = 42

# Each model's screening threshold is tuned during training so that at least
# this share of people with diabetes are flagged (missing a case is costlier
# than a false alarm, which only leads to a cheap confirmatory blood test).
TARGET_SENSITIVITY = 0.80
