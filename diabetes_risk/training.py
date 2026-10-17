"""One training & evaluation routine shared by both models.

  1. Hold out 20 % as an untouched test set.
  2. Compare candidate pipelines with stratified k-fold CV on the training set.
  3. Pick the best by ROC-AUC, calibrate its probabilities, and tune a screening
     threshold on out-of-fold predictions to reach the target sensitivity.
  4. Report everything once on the test set, save charts + metrics JSON,
     then refit on all data and save the model for the app.
"""
import json
import sys
from datetime import datetime, timezone

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,  # noqa: E402
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,  # noqa: E402
                                     cross_val_predict, cross_validate, train_test_split)

from .config import ARTIFACTS_DIR, FIGURES_DIR, RANDOM_STATE, TARGET_SENSITIVITY  # noqa: E402

# Chart styling (validated reference palette; text never wears series colour).
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#a3a29c"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 10, "axes.axisbelow": True,
})

SCORING = {"roc_auc": "roc_auc", "accuracy": "accuracy", "recall": "recall",
           "precision": "precision", "f1": "f1"}


def threshold_for_sensitivity(y_true, scores, target: float) -> float:
    """Highest probability cut-off that still flags `target` share of true cases."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    ok = np.where(tpr >= target)[0]
    return float(min(thresholds[ok[0]], 1.0))


def evaluate(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": accuracy_score(y_true, pred),
        "sensitivity": recall_score(y_true, pred),
        "specificity": tn / (tn + fp),
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


# ----------------------------------------------------------------------------- charts
def _save(fig, name):
    fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(cv_table, best, title, path):
    t = cv_table.sort_values("roc_auc_mean")
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(t) + 1.0))
    colors = [ORANGE if n == best else BLUE for n in t.index]
    ax.barh(t.index, t["roc_auc_mean"], xerr=t["roc_auc_std"], color=colors, height=0.55,
            error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
    for i, (m, s) in enumerate(zip(t["roc_auc_mean"], t["roc_auc_std"])):
        ax.text(m + s + 0.005, i, f"{m:.3f}", va="center", color=INK_2)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Cross-validated ROC-AUC (mean ± std)")
    ax.set_title(f"{title}: model comparison — selected in orange")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def plot_roc(y_test, prob, op, title, path):
    fpr, tpr, _ = roc_curve(y_test, prob)
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1, label="Random guess")
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"Model (AUC = {roc_auc_score(y_test, prob):.3f})")
    ax.scatter([1 - op["specificity"]], [op["sensitivity"]], s=70, color=ORANGE, zorder=3,
               edgecolor=SURFACE, linewidth=2, label="Screening threshold")
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(f"{title}: ROC curve, test set")
    ax.legend(loc="lower right", frameon=False)
    _save(fig, path)


def plot_confusion(cm, threshold, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#cde2fb", "#256abf"]))
    labels = ["No diabetes", "Diabetes"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=15,
                    fontweight="bold", color="white" if cm[i, j] > cm.max() / 2 else INK)
    ax.set_title(f"{title}: threshold {threshold:.2f}")
    _save(fig, path)


def plot_calibration(y_test, prob, title, path):
    frac, mean_pred = calibration_curve(y_test, prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    lim = max(mean_pred.max(), frac.max()) * 1.08
    ax.plot([0, lim], [0, lim], ls="--", color=GREY, lw=1, label="Perfect calibration")
    ax.plot(mean_pred, frac, color=BLUE, lw=2, marker="o", ms=7, mec=SURFACE, mew=2, label="Model")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Observed share with diabetes")
    ax.set_title(f"{title}: calibration")
    ax.legend(loc="upper left", frameon=False)
    _save(fig, path)


def plot_importance(imp, labels, title, path):
    imp = imp.sort_values("mean")
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(imp) + 1.2))
    ax.barh([labels[f] for f in imp.index], imp["mean"], xerr=imp["std"], color=BLUE,
            height=0.55, error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
    ax.set_xlabel("Drop in test ROC-AUC when the feature is shuffled")
    ax.set_title(f"{title}: which factors drive the prediction?")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


# ----------------------------------------------------------------------------- training
def train(spec, fast: bool = False) -> dict:
    """Train, evaluate and save one model. `spec` is diabetes_risk.clinical or .survey."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    short = spec.NAME.capitalize()
    print(f"\n=== {spec.TITLE} ===")

    # 1. Data
    X, y = spec.load_data()
    print(f"Dataset: {len(X):,} people, {y.mean():.1%} with diabetes")
    if spec.missing_counts(X):
        print("Missing (recorded as 0):", spec.missing_counts(X))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # 2. Compare candidates with stratified CV (training data only)
    cv = RepeatedStratifiedKFold(n_splits=3 if fast else 5,
                                 n_repeats=1 if fast else spec.CV_REPEATS,
                                 random_state=RANDOM_STATE)
    rows = {}
    for name, pipe in spec.build_candidates().items():
        s = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        rows[name] = {f"{k}_{agg}": float(getattr(np, agg)(s[f"test_{k}"]))
                      for k in SCORING for agg in ("mean", "std")}
        print(f"  {name:<24} ROC-AUC {rows[name]['roc_auc_mean']:.3f} "
              f"± {rows[name]['roc_auc_std']:.3f}   accuracy {rows[name]['accuracy_mean']:.3f}")
    cv_table = pd.DataFrame(rows).T
    best_name = cv_table["roc_auc_mean"].idxmax()
    print(f"Selected: {best_name}")

    # 3. Calibrate probabilities and tune the screening threshold out-of-fold
    def make_model():
        return CalibratedClassifierCV(clone(spec.build_candidates()[best_name]),
                                      method="sigmoid", cv=3 if fast else 5)

    inner = StratifiedKFold(3 if fast else 5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(make_model(), X_train, y_train, cv=inner,
                            method="predict_proba", n_jobs=-1)[:, 1]
    screen_t = threshold_for_sensitivity(y_train, oof, TARGET_SENSITIVITY)
    high_t = max(0.5, screen_t)
    print(f"Screening threshold for ≥{TARGET_SENSITIVITY:.0%} sensitivity: {screen_t:.3f}")

    # 4. Single, final evaluation on the untouched test set
    model = make_model().fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    at_screen = evaluate(y_test, prob, screen_t)
    at_half = evaluate(y_test, prob, 0.5)
    test = {
        "roc_auc": roc_auc_score(y_test, prob),
        "pr_auc": average_precision_score(y_test, prob),
        "brier": brier_score_loss(y_test, prob),
        "baseline_accuracy": float(1 - y_test.mean()),
        "at_screening_threshold": at_screen,
        "at_0.5": at_half,
    }
    print(f"Test ROC-AUC {test['roc_auc']:.3f} | PR-AUC {test['pr_auc']:.3f} "
          f"| Brier {test['brier']:.3f}")
    for label, m in [("Screening", at_screen), ("0.5 cut-off", at_half)]:
        print(f"  {label:<12} t={m['threshold']:.2f}  accuracy {m['accuracy']:.3f}  "
              f"sensitivity {m['sensitivity']:.3f}  specificity {m['specificity']:.3f}")

    big = len(X_test) > 5000
    pi = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                                n_repeats=5 if big else 30, max_samples=20000 if big else 1.0,
                                random_state=RANDOM_STATE, n_jobs=-1)
    imp = pd.DataFrame({"mean": pi.importances_mean, "std": pi.importances_std},
                       index=spec.FEATURES)

    p = spec.FIG_PREFIX
    plot_model_comparison(cv_table, best_name, short, f"{p}model_comparison.png")
    plot_roc(y_test, prob, at_screen, short, f"{p}roc_curve.png")
    plot_confusion(at_screen["confusion_matrix"], screen_t, short, f"{p}confusion_matrix.png")
    plot_calibration(y_test, prob, short, f"{p}calibration_curve.png")
    plot_importance(imp, spec.LABELS, short, f"{p}feature_importance.png")

    # 5. Refit on all data for deployment (threshold stays as tuned)
    final = make_model().fit(X, y)
    joblib.dump(final, spec.MODEL_PATH, compress=3)

    metrics = {
        "model": spec.NAME,
        "title": spec.TITLE,
        "model_name": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "features": spec.FEATURES,
        "n_patients": len(X),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "prevalence": float(y.mean()),
        "target_sensitivity": TARGET_SENSITIVITY,
        "screening_threshold": screen_t,
        "high_threshold": high_t,
        "cv": cv_table.round(4).to_dict(orient="index"),
        "test": test,
        "permutation_importance": imp["mean"].round(4).to_dict(),
        "reference_values": spec.reference_values(X),
    }
    if hasattr(spec, "unknown_priors"):
        metrics["unknown_priors"] = spec.unknown_priors(X)
    spec.METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=float))
    print(f"Saved {spec.MODEL_PATH.name}, {spec.METRICS_PATH.name} and {p}*.png charts")
    return metrics
