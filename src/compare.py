import warnings
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import make_scorer, precision_score, f1_score
from sklearn.model_selection import cross_validate

from features import get_split, get_cv, SEED
from pipeline import build_pipeline

warnings.filterwarnings("ignore")

# Metrics reported for every fold (zero_division=0 stops warnings for the Dummy model)
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": make_scorer(f1_score, zero_division=0),
    "f1_macro": "f1_macro",
    "roc_auc": "roc_auc",
}


def get_models():
    """name -> (model, needs_scaling). Trees don't need scaled inputs."""
    return {
        "Dummy (always 'stayed')": (
            DummyClassifier(strategy="most_frequent"), True),
        "Logistic Regression": (
            LogisticRegression(class_weight="balanced", max_iter=2000), True),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=SEED), False),
        "Gradient Boosting": (
            HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED), False),
    }


def main():
    X_train, X_test, y_train, y_test = get_split()
    cv = get_cv()

    print(f"Training rows: {len(X_train)} | Test rows (locked away): {len(X_test)}")
    print(f"Leaver rate  train: {y_train.mean():.3f} | test: {y_test.mean():.3f}\n")

    rows = []
    for name, (model, scale) in get_models().items():
        print("Running:", name)
        pipe = build_pipeline(model, X_train, scale=scale)
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1
        )
        row = {"model": name}
        for metric in SCORING:
            vals = scores[f"test_{metric}"]
            row[metric] = f"{vals.mean():.3f} +/- {vals.std():.3f}"
        rows.append(row)

    table = pd.DataFrame(rows)
    print("\n=== 5-fold stratified CV results (mean +/- std) ===")
    print(table.to_string(index=False))

    table.to_csv("results/cv_results.csv", index=False)
    print("\nSaved to results/cv_results.csv")


if __name__ == "__main__":
    main()