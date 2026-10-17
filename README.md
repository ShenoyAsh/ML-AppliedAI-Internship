# Diabetes Risk Predictor — SDG 3: Good Health & Well-being

A machine-learning screening tool that estimates a person's risk of diabetes from
simple health readings, so high-risk people can be identified early and referred
for a blood test.

## Problem statement
Many people with diabetes, especially in rural and low-income areas, are diagnosed
late because regular check-ups are expensive or unavailable. Undiagnosed diabetes
leads to preventable kidney failure, blindness, amputations and heart disease.
A quick screen run by a health worker (for example an ASHA worker) can decide who
needs a confirmatory test first. This directly supports **SDG target 3.4**:
reduce premature deaths from non-communicable diseases.

## Results
Evaluated **once** on 154 held-out patients that were never used for training,
model selection or threshold tuning:

| Metric | Screening threshold (0.30) | Default 0.50 cut-off |
|---|---|---|
| **Sensitivity** (diabetics caught) | **79.6 %** | 53.7 % |
| Specificity (healthy cleared) | 70.0 % | 82.0 % |
| Accuracy | 73.4 % | 72.1 % |
| ROC-AUC | 0.818 | 0.818 |

*Baseline: always guessing "no diabetes" scores 64.9 % accuracy.*

For screening, a missed case is far costlier than a false alarm, which only costs
one blood test. The threshold is therefore tuned to catch ~80 % of diabetics,
**cutting missed cases from 25 to 11 out of 54** compared with the usual 0.5 cut-off.

Cross-validated comparison (5-fold × 5 repeats, training set):

| Model | ROC-AUC | Accuracy |
|---|---|---|
| **Logistic Regression (chosen)** | **0.836 ± 0.036** | 0.783 |
| Random Forest | 0.834 ± 0.035 | 0.758 |
| Support Vector Machine | 0.827 ± 0.028 | 0.764 |
| Gradient Boosting | 0.821 ± 0.029 | 0.762 |

Glucose is by far the strongest predictor, followed by BMI and number of pregnancies
(see `reports/figures/feature_importance.png`). The simplest model performed best,
which is common on small tabular datasets. It is also the easiest to explain.

| | |
|---|---|
| ![ROC](reports/figures/roc_curve.png) | ![Confusion matrix](reports/figures/confusion_matrix.png) |
| ![Importance](reports/figures/feature_importance.png) | ![Calibration](reports/figures/calibration_curve.png) |

## How it works
1. **Data**: Pima Indians Diabetes Dataset, 768 women aged 21+, 8 features and 35 % prevalence.
2. **Cleaning inside the pipeline**: zeros that are physiologically impossible
   (glucose, blood pressure, skin thickness, insulin, BMI) are treated as missing,
   median-imputed and flagged with a "was missing" indicator. Because this happens
   inside the scikit-learn `Pipeline`, it is learned from training folds only:
   **no data leakage**.
3. **Model selection**: four models compared with repeated stratified
   cross-validation on ROC-AUC. The test set is not touched.
4. **Calibration**: probabilities are calibrated (Platt scaling), so a displayed
   "30 % risk" really means roughly 30 out of 100 similar people.
5. **Threshold tuning**: the cut-off is chosen on out-of-fold predictions to reach
   80 % sensitivity.
6. **Final model**: refit on all 768 patients and saved to `artifacts/`.

## The app
- Fields people rarely know (insulin, skin-fold thickness) can be left as
  **not measured**, and the model imputes them.
- Family history is asked in plain language instead of the obscure "pedigree function".
- A built-in BMI calculator.
- **Explains each result**: a chart shows which readings raise or lower *your* risk.
- Personalised next steps based on clinical cut-offs (prediabetes range, BMI ≥ 23
  for South Asians, blood pressure, age ≥ 35 screening).
- A downloadable summary to show a doctor.
- A "How the model works" tab with live metrics and charts.

## How to run
```bash
pip install -r requirements.txt
python train.py          # ~1 min: trains, evaluates, saves model + charts + metrics.json
streamlit run app.py     # opens the web app in your browser
pytest                   # 14 tests: data, pipeline, advice logic and the app end-to-end
```

## Project structure
```
diabetes_risk/      reusable package
  config.py         features, paths, target sensitivity
  data.py           loading + validation
  modeling.py       leakage-free pipelines, threshold tuning
  advice.py         risk bands, per-person explanations, health tips
train.py            training & evaluation script
app.py              Streamlit web app
tests/              pytest suite (unit + Streamlit AppTest)
artifacts/          trained model + metrics.json
reports/figures/    evaluation charts
```

## Limitations
- Trained only on women of Pima heritage, so performance on men and on Indian
  populations is unknown. **Re-train on local data before any real deployment.**
- The glucose feature is a 2-hour oral glucose tolerance test, not fasting glucose
  or HbA1c, which are more common today.
- 768 patients is small; metrics have wide confidence intervals (± ~0.035 AUC).

## Future scope
- Re-train on data from Indian hospitals, using HbA1c / fasting glucose and waist circumference
- Regional-language interface and an offline mobile app for ASHA / health workers
- Extend to other NCDs (heart disease, hypertension)

*Disclaimer: a screening and awareness tool, not a medical diagnosis.*
