"""
Diabetes Risk Checker — SDG 3: Good Health & Well-being
Run with:  streamlit run app.py   (train first with:  python train.py)
"""
import json

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from diabetes_risk.advice import factor_contributions, health_tips, risk_band, to_frame
from diabetes_risk.config import FIGURES_DIR, METRICS_PATH, MODEL_PATH

st.set_page_config(page_title="Diabetes Risk Checker", page_icon="🩺", layout="centered")

# Approximate Diabetes Pedigree Function values for people who don't know their
# score (25th, ~60th, ~85th and ~95th percentiles of the dataset).
FAMILY_HISTORY = {
    "No known relatives with diabetes": 0.24,
    "One parent or sibling": 0.45,
    "Two or more close relatives": 0.75,
    "Many relatives, some diagnosed young": 1.10,
}
BAND_STYLE = {"Low": ("success", "🟢"), "Moderate": ("warning", "🟠"), "High": ("error", "🔴")}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text())


st.title("🩺 Diabetes Risk Checker")
st.caption("SDG 3: Good Health & Well-being — free, early screening with machine learning")

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.error("No trained model found. Run `python train.py` first, then reload this page.")
    st.stop()

model, metrics = load_model(), load_metrics()
screen_t, high_t = metrics["screening_threshold"], metrics["high_threshold"]

tab_check, tab_model, tab_about = st.tabs(["Check my risk", "How the model works", "About"])

# --------------------------------------------------------------------------- check
with tab_check:
    st.write("Enter your most recent readings. Fields you haven't had measured can be "
             "left as **not measured** — the model fills them in from similar patients.")

    with st.form("readings"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", 21, 100, 35,
                                  help="The model was trained on adults aged 21–81.")
            pregnancies = st.number_input("Number of pregnancies", 0, 20, 1,
                                          help="Enter 0 if never pregnant.")
            glucose = st.number_input(
                "Glucose — 2 hours after glucose drink (mg/dL)", 40, 300, 110,
                help="Plasma glucose 2 hours into an oral glucose tolerance test (OGTT). "
                     "Normal < 140, prediabetes 140–199, diabetes ≥ 200. "
                     "A fasting reading is lower and will under-estimate risk.")
            bp = st.number_input("Diastolic blood pressure (mm Hg)", 30, 140, 72,
                                 help="The lower number, e.g. 72 in 120/72.")
        with c2:
            bmi = st.number_input("BMI (kg/m²)", 10.0, 70.0, 26.0, step=0.1, format="%.1f",
                                  help="Weight in kg ÷ (height in m)². "
                                       "Not sure? Use the calculator below the form.")
            family = st.selectbox("Family history of diabetes", list(FAMILY_HISTORY))
            insulin_known = st.checkbox("I know my 2-hour insulin level")
            insulin = st.number_input("2-hour serum insulin (µU/mL)", 10, 900, 120)
            skin_known = st.checkbox("I know my triceps skin-fold thickness")
            skin = st.number_input("Triceps skin-fold thickness (mm)", 5, 100, 29)
        submitted = st.form_submit_button("Check my risk", type="primary", width="stretch")

    with st.expander("BMI calculator"):
        h = st.number_input("Height (cm)", 100.0, 230.0, 160.0, step=0.5)
        w = st.number_input("Weight (kg)", 25.0, 250.0, 65.0, step=0.5)
        st.write(f"Your BMI is **{w / (h / 100) ** 2:.1f}** — enter it in the form above.")

    if submitted:
        values = {
            "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": bp,
            "SkinThickness": skin if skin_known else None,
            "Insulin": insulin if insulin_known else None, "BMI": bmi,
            "DiabetesPedigreeFunction": FAMILY_HISTORY[family], "Age": age,
        }
        prob = float(model.predict_proba(to_frame(values))[0, 1])
        band = risk_band(prob, screen_t, high_t)
        kind, dot = BAND_STYLE[band.level]

        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Estimated diabetes risk", f"{prob:.0%}",
                  delta=f"{(prob - metrics['prevalence']) * 100:+.0f} pts vs. average",
                  delta_color="inverse")
        m2.metric("Risk level", f"{dot} {band.level}")
        st.progress(prob)
        getattr(st, kind)(band.message)

        if age > 81 or glucose > 199 or (insulin_known and insulin > 846):
            st.caption("Some readings are outside the range the model was trained on, so "
                       "the estimate is less certain — but they are a reason to see a doctor.")

        st.subheader("What is driving your result")
        contrib = factor_contributions(model, values, metrics["reference_values"])
        chart_df = pd.DataFrame({"factor": contrib.index, "change": (contrib.values * 100).round(1)})
        chart_df["direction"] = chart_df["change"].map(
            lambda v: "Raises risk" if v > 0 else "Lowers risk")
        st.altair_chart(
            alt.Chart(chart_df).mark_bar(cornerRadius=3, height=16).encode(
                x=alt.X("change:Q", title="Change in risk vs. a typical patient (% points)"),
                y=alt.Y("factor:N", sort=None, title=None),
                color=alt.Color("direction:N", title=None,
                                scale=alt.Scale(domain=["Raises risk", "Lowers risk"],
                                                range=["#eb6834", "#2a78d6"]),
                                legend=alt.Legend(orient="top")),
                tooltip=[alt.Tooltip("factor:N", title="Factor"),
                         alt.Tooltip("change:Q", title="Change (pts)", format="+.1f")],
            ).properties(height=36 * len(chart_df) + 20),
            width="stretch")
        st.caption("Each bar shows how your risk would change if that one reading were "
                   "replaced by the typical value in the dataset. Unmeasured fields are skipped.")

        st.subheader("Next steps")
        for tip in health_tips(values):
            st.markdown(f"- {tip}")

        report = "\n".join(
            [f"Diabetes risk screening — {pd.Timestamp.now():%d %b %Y}",
             f"Estimated risk: {prob:.0%} ({band.level})", "", "Readings:"]
            + [f"  {k}: {'not measured' if v is None else v}" for k, v in values.items()]
            + ["", "Next steps:"] + [f"  - {t.replace('**', '')}" for t in health_tips(values)]
            + ["", "This is a screening estimate, not a diagnosis. Confirm with an HbA1c or "
                   "fasting glucose test."])
        st.download_button("Download summary to show your doctor", report,
                           file_name="diabetes_risk_summary.txt")

    st.info("⚠️ This is a screening tool for awareness, not a medical diagnosis. "
            "Only a blood test (HbA1c, fasting glucose or OGTT) can diagnose diabetes.")

# --------------------------------------------------------------------------- model
with tab_model:
    test, op = metrics["test"], metrics["test"]["at_screening_threshold"]
    st.write(f"**Model:** calibrated {metrics['model_name']}, chosen from four candidates by "
             f"repeated 5-fold cross-validation. Evaluated once on {metrics['n_test']} "
             f"held-out patients it never saw during training or tuning.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ROC-AUC", f"{test['roc_auc']:.2f}")
    k2.metric("Sensitivity", f"{op['sensitivity']:.0%}", help="Share of diabetics flagged.")
    k3.metric("Specificity", f"{op['specificity']:.0%}",
              help="Share of healthy people correctly cleared.")
    k4.metric("Accuracy", f"{op['accuracy']:.0%}",
              help=f"Always guessing 'no diabetes' scores {test['baseline_accuracy']:.0%}.")
    st.write(f"The risk cut-off is set at **{screen_t:.0%}** rather than 50 %, so that about "
             f"{metrics['target_sensitivity']:.0%} of people with diabetes are flagged. At "
             f"50 % the model would miss {1 - test['at_0.5']['sensitivity']:.0%} of them. For "
             "screening, a false alarm costs one blood test; a missed case can cost years.")

    cv = pd.DataFrame(metrics["cv"]).T[["roc_auc_mean", "accuracy_mean", "recall_mean"]]
    cv.columns = ["ROC-AUC", "Accuracy", "Sensitivity (at 0.5)"]
    st.dataframe(cv.sort_values("ROC-AUC", ascending=False).style.format("{:.3f}"),
                 width="stretch")

    figs = [("feature_importance.png", "Which factors matter most"),
            ("roc_curve.png", "ROC curve"), ("confusion_matrix.png", "Confusion matrix"),
            ("calibration_curve.png", "Calibration"), ("model_comparison.png", "Model comparison")]
    for name, caption in figs:
        if (FIGURES_DIR / name).exists():
            st.image(str(FIGURES_DIR / name), caption=caption)

# --------------------------------------------------------------------------- about
with tab_about:
    st.markdown(f"""
**Why this matters.** Many people with diabetes, especially in rural and low-income areas,
are diagnosed late because check-ups are expensive or unavailable. A quick screen that
uses simple readings can help health workers decide who needs a blood test first.

**Data.** Pima Indians Diabetes Dataset — {metrics['n_patients']} women aged 21+ of Pima
heritage, {metrics['prevalence']:.0%} of whom had diabetes.

**Limitations.**
- Trained only on women from one population; accuracy for men and other groups is unknown.
- Uses a 2-hour glucose tolerance reading, not the more common fasting glucose or HbA1c.
- A small dataset from the 1980s–90s; it should be re-trained on local data before real use.

*Model trained {metrics['trained_at'][:10]} with scikit-learn {metrics['sklearn_version']}.*
""")
