import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from features import get_split, load_data, SEED

warnings.filterwarnings("ignore")


def main():
    X_train, X_test, y_train, y_test = get_split()
    tuned = joblib.load("models/tuned_models.joblib")
    name = json.load(open("models/chosen.json"))["model"]
    model = tuned[name]
    print("Explaining:", name)

    # ------------------------------------------------------------------
    # 1. Permutation importance (works for any model)
    #    Shuffle one ORIGINAL column at a time; measure the drop in ROC-AUC.
    # ------------------------------------------------------------------
    pi = permutation_importance(
        model, X_test, y_test, scoring="roc_auc",
        n_repeats=30, random_state=SEED, n_jobs=-1,
    )
    imp = pd.DataFrame({
        "feature": X_test.columns,
        "importance": pi.importances_mean,
        "std": pi.importances_std,
    }).sort_values("importance", ascending=False)
    imp.to_csv("results/permutation_importance.csv", index=False)

    print("\n=== Top 10 features by permutation importance (drop in ROC-AUC) ===")
    print(imp.head(10).round(4).to_string(index=False))

    top = imp.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#2E8B8B")
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    ax.set_title(f"Top 10 features: permutation importance ({name})")
    fig.tight_layout()
    fig.savefig("results/feature_importance.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Coefficients and odds ratios (linear models only: gives DIRECTION)
    # ------------------------------------------------------------------
    clf = model.named_steps["clf"]
    if hasattr(clf, "coef_"):
        names = model.named_steps["prep"].get_feature_names_out()
        names = [n.split("__", 1)[1] for n in names]       # drop "num__" etc.
        coefs = pd.DataFrame({"feature": names, "coef": clf.coef_[0]})
        coefs["odds_ratio"] = np.exp(coefs["coef"])
        coefs = coefs.sort_values("coef", ascending=False)
        coefs.to_csv("results/coefficients.csv", index=False)

        print("\n=== Raise the risk of leaving (odds ratio > 1) ===")
        print(coefs.head(8).round(3).to_string(index=False))
        print("\n=== Lower the risk of leaving (odds ratio < 1) ===")
        print(coefs.tail(8).iloc[::-1].round(3).to_string(index=False))

        show = pd.concat([coefs.head(8), coefs.tail(8)]).iloc[::-1]
        colors = ["#B04040" if c > 0 else "#1F3864" for c in show["coef"]]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(show["feature"], show["coef"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Coefficient  (red = more likely to leave, blue = less likely)")
        ax.set_title("Logistic regression: strongest drivers")
        fig.tight_layout()
        fig.savefig("results/coefficients.png", dpi=200)
        plt.close(fig)
    else:
        print("\n(Selected model has no coefficients; skipping the coefficient chart.)")

    # ------------------------------------------------------------------
    # 3. Raw attrition rates by group (plain percentages, no model involved)
    # ------------------------------------------------------------------
    X, y = load_data()
    df = X.copy()
    df["left"] = y
    overall = df["left"].mean() * 100

    groups = {
        "OverTime": df["OverTime"],
        "BusinessTravel": df["BusinessTravel"],
        "MaritalStatus": df["MaritalStatus"],
        "JobSatisfaction (1=low)": df["JobSatisfaction"],
        "Years at company": pd.cut(df["YearsAtCompany"], [-1, 2, 5, 10, 100],
                                   labels=["0-2", "3-5", "6-10", "11+"]),
        "Monthly income quartile": pd.qcut(df["MonthlyIncome"], 4,
                                           labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]),
        "Distance from home (km)": pd.cut(df["DistanceFromHome"], [0, 5, 10, 20, 100],
                                          labels=["1-5", "6-10", "11-20", "21+"]),
    }

    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, (title, grp) in zip(axes.flat, groups.items()):
        t = df.groupby(grp, observed=True)["left"].agg(["mean", "count"])
        t["mean"] *= 100
        print(f"\n--- Attrition rate by {title} ---")
        for g, r in t.iterrows():
            print(f"  {str(g):<16} {r['mean']:5.1f}%   (n = {int(r['count'])})")
            rows.append({"variable": title, "group": str(g),
                         "attrition_rate_pct": round(r["mean"], 1),
                         "employees": int(r["count"])})
        ax.bar([str(i) for i in t.index], t["mean"], color="#2E8B8B")
        ax.axhline(overall, color="gray", linestyle="--", linewidth=1)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Attrition %")
        ax.tick_params(axis="x", labelsize=8)
    for ax in list(axes.flat)[len(groups):]:
        ax.axis("off")                      # hide unused panel
    fig.suptitle(f"Attrition rate by group (dashed line = overall {overall:.1f}%)")
    fig.tight_layout()
    fig.savefig("results/attrition_by_group.png", dpi=200)
    plt.close(fig)
    pd.DataFrame(rows).to_csv("results/attrition_by_group.csv", index=False)

    print("\nSaved to results/: permutation_importance.csv, feature_importance.png, "
          "coefficients.csv, coefficients.png, attrition_by_group.csv, attrition_by_group.png")


if __name__ == "__main__":
    main()