"""
Diabetes Risk Checker — SDG 3: Good Health & Well-being
Run with:  streamlit run app.py   (train first with:  python train.py)
Languages: English, Hindi, Kannada — share a link with ?lang=hi or ?lang=kn
"""
import json

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from diabetes_risk import clinical, survey
from diabetes_risk.advice import clinical_tips, factor_contributions, risk_band, survey_tips
from diabetes_risk.config import FIGURES_DIR
from diabetes_risk.i18n import DEFAULT, LANGUAGES, t

st.set_page_config(page_title="Diabetes Risk Checker", page_icon="🩺", layout="centered")

# Approximate Diabetes Pedigree Function values for people who don't know their
# score (25th, ~60th, ~85th and ~95th percentiles of the Pima dataset).
FAMILY_HISTORY = {"fam.none": 0.24, "fam.one": 0.45, "fam.two": 0.75, "fam.many": 1.10}
BAND_STYLE = {"low": ("success", "🟢"), "moderate": ("warning", "🟠"), "high": ("error", "🔴")}

# ----------------------------------------------------------------------------- language
if "lang" not in st.session_state:
    q = st.query_params.get("lang", DEFAULT)
    st.session_state.lang = q if q in LANGUAGES else DEFAULT
lang = st.radio("Language / भाषा / ಭಾಷೆ", list(LANGUAGES), format_func=LANGUAGES.get,
                horizontal=True, key="lang", label_visibility="collapsed")
st.query_params["lang"] = lang


def T(key, **kw):
    return t(lang, key, **kw)


@st.cache_resource
def load_model(spec_name):
    spec = {"survey": survey, "clinical": clinical}[spec_name]
    if not spec.MODEL_PATH.exists() or not spec.METRICS_PATH.exists():
        return None, None
    return joblib.load(spec.MODEL_PATH), json.loads(spec.METRICS_PATH.read_text())


st.title("🩺 " + T("app.title"))
st.caption(T("app.subtitle"))
if lang != "en":
    st.caption(T("app.translation_note"))


# ----------------------------------------------------------------------------- widgets
def yes_no(label, key, help=None, default=0):
    return st.radio(label, [1, 0], index=0 if default else 1, horizontal=True, key=key, help=help,
                    format_func=lambda v: T("opt.yes") if v else T("opt.no"))


def yes_no_unknown(label, key, help=None):
    return st.radio(label, [1, 0, None], index=1, horizontal=True, key=key, help=help,
                    format_func=lambda v: {1: T("opt.yes"), 0: T("opt.no"),
                                           None: T("opt.never_checked")}[v])


def answer_text(v):
    return {1: T("opt.yes"), 0: T("opt.no"), None: T("opt.never_checked")}.get(v, v)


def show_result(spec, model, metrics, values, tips, answers, risk_label):
    """Shared result panel: headline, band, explanation chart, tips, download."""
    predict = lambda rows: spec.predict_many(model, rows, metrics)  # noqa: E731
    prob = float(predict([values])[0])
    band = risk_band(prob, metrics["screening_threshold"], metrics["high_threshold"])
    kind, dot = BAND_STYLE[band]

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric(T("res.risk"), f"{prob:.0%}", help=risk_label,
              delta=T("res.vs_average", pts=f"{(prob - metrics['prevalence']) * 100:+.0f}"),
              delta_color="inverse")
    m2.metric(T("res.level"), f"{dot} {T('band.' + band)}")
    st.progress(min(prob, 1.0))
    st.caption(risk_label)
    getattr(st, kind)(T(f"band.{band}.msg"))

    st.subheader(T("res.drivers"))
    contrib = factor_contributions(predict, values, metrics["reference_values"], spec.FEATURES)
    contrib = contrib[contrib.abs() >= 0.001]
    if contrib.empty:
        st.write(T("res.no_drivers"))
    else:
        raises, lowers = T("res.raises"), T("res.lowers")
        df = pd.DataFrame({"factor": [T(f"feat.{f}") for f in contrib.index],
                           "change": (contrib.values * 100).round(1)})
        df["direction"] = [raises if v > 0 else lowers for v in df["change"]]
        st.altair_chart(
            alt.Chart(df).mark_bar(cornerRadius=3).encode(
                x=alt.X("change:Q", title=T("res.axis")),
                y=alt.Y("factor:N", sort=None, title=None,
                          axis=alt.Axis(labelLimit=220),
                          scale=alt.Scale(paddingInner=0.35)),
                color=alt.Color("direction:N", title=None,
                                scale=alt.Scale(domain=[raises, lowers],
                                                range=["#eb6834", "#2a78d6"]),
                                legend=alt.Legend(orient="top")),
                tooltip=[alt.Tooltip("factor:N", title=T("res.factor")),
                         alt.Tooltip("change:Q", title=T("res.change"), format="+.1f")],
            ).properties(height=alt.Step(34)),
            width="stretch")
        st.caption(T("res.drivers_caption"))

    st.subheader(T("res.next_steps"))
    tip_texts = [T(k, **p) for k, p in tips]
    for tip in tip_texts:
        st.markdown(f"- {tip}")

    report = "\n".join(
        [T("sum.title", date=f"{pd.Timestamp.now():%d-%m-%Y}"),
         f"{risk_label}: {prob:.0%} ({T('band.' + band)})", "", T("sum.answers")]
        + [f"  {q}: {a}" for q, a in answers]
        + ["", T("res.next_steps")] + [f"  - {s.replace('**', '')}" for s in tip_texts]
        + ["", T("app.disclaimer")])
    st.download_button(T("res.download"), report.encode("utf-8"),
                       file_name=f"diabetes_risk_summary_{lang}.txt", mime="text/plain")


def need_model(spec):
    model, metrics = load_model(spec.NAME)
    if model is None:
        st.error(T("app.no_model", cmd=f"python train.py --model {spec.NAME}"))
    return model, metrics


tab_quick, tab_detailed, tab_models, tab_about = st.tabs(
    [T("tab.quick"), T("tab.detailed"), T("tab.models"), T("tab.about")])

# ----------------------------------------------------------------------------- quick check
with tab_quick:
    st.write(T("quick.intro"))
    model, metrics = need_model(survey)
    if model is not None:
        with st.form("quick"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input(T("q.age"), 18, 100, 40, key="s_age")
                height = st.number_input(T("q.height"), 100, 220, 160, key="s_h")
            with c2:
                sex = st.radio(T("q.sex"), [0, 1], horizontal=True, key="s_sex",
                               format_func=lambda v: T("opt.male") if v else T("opt.female"))
                weight = st.number_input(T("q.weight"), 25.0, 250.0, 65.0, step=0.5, format="%.1f",
                                         key="s_w")
            genhlth = st.select_slider(T("q.genhlth"), [1, 2, 3, 4, 5], value=3, key="s_gen",
                                       format_func=lambda v: T(f"opt.gen{v}"))
            high_bp = yes_no_unknown(T("q.highbp"), "s_bp")
            high_chol = yes_no_unknown(T("q.highchol"), "s_chol")
            c3, c4 = st.columns(2)
            with c3:
                # Defaults are the most common answers, so an untouched form is neutral.
                active = yes_no(T("q.activity"), "s_act", default=1)
                fruits = yes_no(T("q.fruits"), "s_fruit", default=1)
                veggies = yes_no(T("q.veggies"), "s_veg", default=1)
                smoker = yes_no(T("q.smoker"), "s_smoke")
            with c4:
                alcohol = yes_no(T("q.alcohol"), "s_alc", help=T("q.alcohol_help"))
                stroke = yes_no(T("q.stroke"), "s_stroke")
                heart = yes_no(T("q.heart"), "s_heart")
                diffwalk = yes_no(T("q.diffwalk"), "s_walk")
            submitted = st.form_submit_button(T("btn.check"), type="primary", width="stretch")

        if submitted:
            bmi = survey.bmi(height, weight)
            values = {"Age": survey.age_to_category(age), "Sex": sex, "BMI": round(bmi),
                      "HighBP": high_bp, "HighChol": high_chol, "GenHlth": genhlth,
                      "PhysActivity": active, "Fruits": fruits, "Veggies": veggies,
                      "Smoker": smoker, "HvyAlcoholConsump": alcohol, "Stroke": stroke,
                      "HeartDiseaseorAttack": heart, "DiffWalk": diffwalk}
            answers = [(T("q.age"), age), (T("q.sex"), T("opt.male") if sex else T("opt.female")),
                       (T("sum.bmi"), f"{bmi:.1f}"), (T("q.genhlth"), T(f"opt.gen{genhlth}")),
                       (T("q.highbp"), answer_text(high_bp)),
                       (T("q.highchol"), answer_text(high_chol)),
                       (T("q.activity"), answer_text(active)), (T("q.fruits"), answer_text(fruits)),
                       (T("q.veggies"), answer_text(veggies)), (T("q.smoker"), answer_text(smoker)),
                       (T("q.alcohol"), answer_text(alcohol)), (T("q.stroke"), answer_text(stroke)),
                       (T("q.heart"), answer_text(heart)), (T("q.diffwalk"), answer_text(diffwalk))]
            show_result(survey, model, metrics, values,
                        survey_tips({**values, "BMI": bmi}, age), answers, T("res.risk_survey"))
    st.info(T("app.disclaimer"))

# ----------------------------------------------------------------------------- detailed check
with tab_detailed:
    st.write(T("detailed.intro"))
    model, metrics = need_model(clinical)
    if model is not None:
        with st.form("detailed"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input(T("q.age"), 21, 100, 35, key="c_age", help=T("c.age_help"))
                pregnancies = st.number_input(T("c.pregnancies"), 0, 20, 1, key="c_preg",
                                              help=T("c.pregnancies_help"))
                glucose = st.number_input(T("c.glucose"), 40, 300, 110, key="c_glu",
                                          help=T("c.glucose_help"))
                bp = st.number_input(T("c.bp"), 30, 140, 72, key="c_bp", help=T("c.bp_help"))
            with c2:
                height = st.number_input(T("q.height"), 100, 220, 160, key="c_h")
                weight = st.number_input(T("q.weight"), 25.0, 250.0, 65.0, step=0.5, format="%.1f",
                                         key="c_w")
                family = st.selectbox(T("c.family"), list(FAMILY_HISTORY), format_func=T,
                                      key="c_fam")
            insulin_known = st.checkbox(T("c.insulin_known"), key="c_ins_k")
            insulin = st.number_input(T("c.insulin"), 10, 900, 120, key="c_ins")
            skin_known = st.checkbox(T("c.skin_known"), key="c_skin_k")
            skin = st.number_input(T("c.skin"), 5, 100, 29, key="c_skin")
            submitted = st.form_submit_button(T("btn.check"), type="primary", width="stretch")

        if submitted:
            bmi = survey.bmi(height, weight)
            values = {"Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": bp,
                      "SkinThickness": skin if skin_known else None,
                      "Insulin": insulin if insulin_known else None, "BMI": bmi,
                      "DiabetesPedigreeFunction": FAMILY_HISTORY[family], "Age": age}
            not_measured = T("opt.not_measured")
            answers = [(T("q.age"), age), (T("c.pregnancies"), pregnancies),
                       (T("c.glucose"), glucose), (T("c.bp"), bp), (T("sum.bmi"), f"{bmi:.1f}"),
                       (T("c.family"), T(family)),
                       (T("c.insulin"), insulin if insulin_known else not_measured),
                       (T("c.skin"), skin if skin_known else not_measured)]
            if age > 81 or glucose > 199 or (insulin_known and insulin > 846):
                st.caption(T("c.out_of_range"))
            show_result(clinical, model, metrics, values, clinical_tips(values), answers,
                        T("res.risk_clinical"))
    st.info(T("app.disclaimer"))

# ----------------------------------------------------------------------------- models
with tab_models:
    st.write(T("models.intro"))
    for spec in (survey, clinical):
        model, metrics = load_model(spec.NAME)
        if model is None:
            continue
        test, op = metrics["test"], metrics["test"]["at_screening_threshold"]
        st.subheader(T(f"models.{spec.NAME}_heading"))
        st.write(T(f"models.{spec.NAME}_data", n=f"{metrics['n_patients']:,}",
                   prev=f"{metrics['prevalence']:.0%}"))
        st.write(T("models.method", model=metrics["model_name"], n_test=f"{metrics['n_test']:,}"))
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("ROC-AUC", f"{test['roc_auc']:.2f}")
        k2.metric(T("models.sensitivity"), f"{op['sensitivity']:.0%}",
                  help=T("models.sensitivity_help"))
        k3.metric(T("models.specificity"), f"{op['specificity']:.0%}",
                  help=T("models.specificity_help"))
        k4.metric(T("models.accuracy"), f"{op['accuracy']:.0%}",
                  help=T("models.accuracy_help", base=f"{test['baseline_accuracy']:.0%}"))
        st.write(T("models.threshold", t=f"{metrics['screening_threshold']:.0%}",
                   target=f"{metrics['target_sensitivity']:.0%}",
                   missed=f"{1 - test['at_0.5']['sensitivity']:.0%}"))
        cv = pd.DataFrame(metrics["cv"]).T[["roc_auc_mean", "accuracy_mean", "recall_mean"]]
        cv.columns = ["ROC-AUC", T("models.accuracy"), T("models.sensitivity_at_half")]
        st.dataframe(cv.sort_values("ROC-AUC", ascending=False).style.format("{:.3f}"),
                     width="stretch")
        with st.expander(T("models.charts")):
            for name in ("feature_importance", "roc_curve", "confusion_matrix",
                         "calibration_curve", "model_comparison"):
                path = FIGURES_DIR / f"{spec.FIG_PREFIX}{name}.png"
                if path.exists():
                    st.image(str(path), caption=T(f"fig.{name}"))

# ----------------------------------------------------------------------------- about
with tab_about:
    st.markdown(T("about.body"))
