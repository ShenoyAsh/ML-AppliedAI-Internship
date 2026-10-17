"""
SDG 3 - Good Health & Well-being
Diabetes Risk Prediction using Machine Learning — two models:

  survey    Quick check, no blood test needed. CDC BRFSS survey, 253,680 adults,
            14 self-reported questions.
  clinical  Detailed check with lab readings. Pima Indians Diabetes Dataset,
            768 patients, 8 health measurements.

Run:  python train.py                    (both models, a few minutes)
      python train.py --model clinical   (just one)
      python train.py --fast             (fewer CV folds, for quick experiments)
"""
import argparse

from diabetes_risk import clinical, survey
from diabetes_risk.training import train

MODELS = {"survey": survey, "clinical": clinical}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the diabetes risk models.")
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--fast", action="store_true", help="3 CV folds, no repeats")
    args = parser.parse_args()
    for name, spec in MODELS.items():
        if args.model in (name, "all"):
            train(spec, fast=args.fast)
