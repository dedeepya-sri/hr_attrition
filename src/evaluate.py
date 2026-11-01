import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    recall_score, precision_score, fbeta_score,
    ConfusionMatrixDisplay, RocCurveDisplay,
)

from features import get_split, get_cv, SEED

warnings.filterwarnings("ignore")


def main():
    X_train, X_test, y_train, y_test = get_split()
    cv = get_cv()

    # ---- 1. Pick the winner using CV results only (test set not involved) ----
    tuned = joblib.load("models/tuned_models.joblib")
    summary = pd.read_csv("results/tuning_results.csv")
    winner = summary.sort_values("best_cv_roc_auc", ascending=False).iloc[0]["model"]
    model = tuned[winner]
    print("Selected model:", winner)

    # ---- 2. Honest probabilities for each training row (out-of-fold) ----
    oof = cross_val_predict(
        model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]

    # ---- 3. Choose the threshold that maximises F2 on those probabilities ----
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    f2_scores = [fbeta_score(y_train, oof >= t, beta=2) for t in thresholds]
    thr = float(thresholds[int(np.argmax(f2_scores))])
    print(f"Chosen threshold (max F2 on training CV): {thr:.2f}")

    print("\nTrade-off on training CV (what each cutoff would do):")
    print(f"{'cutoff':>7} {'recall':>8} {'precision':>10} {'% flagged':>10}")
    for t in [0.2, 0.3, 0.4, 0.5, 0.6]:
        p = oof >= t
        print(f"{t:>7.2f} {recall_score(y_train, p):>8.3f} "
              f"{precision_score(y_train, p, zero_division=0):>10.3f} "
              f"{p.mean() * 100:>9.1f}%")

    # ---- 4. Final exam: the locked test set, evaluated once ----
    proba = model.predict_proba(X_test)[:, 1]
    lines = [f"Model: {winner}", f"Chosen threshold: {thr:.2f}", ""]
    for label, t in [("default 0.50", 0.5), (f"chosen {thr:.2f}", thr)]:
        pred = (proba >= t).astype(int)
        lines.append(f"=== Test set @ cutoff {label} ===")
        lines.append(classification_report(
            y_test, pred, target_names=["Stayed", "Left"], zero_division=0))
        lines.append("Confusion matrix  [[TN FP], [FN TP]]:")
        lines.append(str(confusion_matrix(y_test, pred)))
        lines.append("")
    lines.append(f"Test ROC-AUC: {roc_auc_score(y_test, proba):.3f}")

    # ---- 5. Bootstrap: how uncertain are these numbers? ----
    rng = np.random.default_rng(SEED)
    y_arr, pred = y_test.to_numpy(), (proba >= thr).astype(int)
    boots = {"Recall": [], "Precision": [], "ROC-AUC": []}
    for _ in range(1000):
        i = rng.integers(0, len(y_arr), len(y_arr))
        if y_arr[i].sum() == 0 or y_arr[i].sum() == len(i):
            continue
        boots["Recall"].append(recall_score(y_arr[i], pred[i], zero_division=0))
        boots["Precision"].append(precision_score(y_arr[i], pred[i], zero_division=0))
        boots["ROC-AUC"].append(roc_auc_score(y_arr[i], proba[i]))
    lines.append("\n95% bootstrap intervals (chosen cutoff):")
    for k, v in boots.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        lines.append(f"  {k:<10} {np.mean(v):.3f}  [{lo:.3f} - {hi:.3f}]")

    report = "\n".join(lines)
    print("\n" + report)
    with open("results/test_metrics.txt", "w") as f:
        f.write(report)

    # ---- 6. Figures ----
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Stayed", "Left"], cmap="Blues", ax=ax)
    ax.set_title(f"Confusion matrix (test, cutoff {thr:.2f})")
    fig.tight_layout(); fig.savefig("results/confusion_matrix.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, proba, name=winner, ax=ax)
    ax.set_title("ROC curve (test set)")
    fig.tight_layout(); fig.savefig("results/roc_curve.png", dpi=200); plt.close(fig)

    rec = [recall_score(y_train, oof >= t) for t in thresholds]
    prec = [precision_score(y_train, oof >= t, zero_division=0) for t in thresholds]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(thresholds, rec, label="Recall")
    ax.plot(thresholds, prec, label="Precision")
    ax.axvline(thr, color="gray", linestyle="--", label=f"Chosen cutoff {thr:.2f}")
    ax.set_xlabel("Decision cutoff"); ax.set_ylabel("Score")
    ax.set_title("Recall vs precision as the cutoff moves"); ax.legend()
    fig.tight_layout(); fig.savefig("results/threshold_tradeoff.png", dpi=200); plt.close(fig)

    # ---- 7. Remember the choice for later steps ----
    with open("models/chosen.json", "w") as f:
        json.dump({"model": winner, "threshold": thr}, f)
    print("\nSaved: results/test_metrics.txt, confusion_matrix.png, roc_curve.png, "
          "threshold_tradeoff.png, models/chosen.json")


if __name__ == "__main__":
    main()