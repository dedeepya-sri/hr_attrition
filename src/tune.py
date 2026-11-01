import warnings
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV

from features import get_split, get_cv
from pipeline import build_pipeline
from compare import get_models

warnings.filterwarnings("ignore")

# Parameter names start with "clf__" because the classifier is the
# pipeline step called "clf" (see pipeline.py)
GRIDS = {
    "Logistic Regression": {
        "clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3],
    },
    "Random Forest": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [3, 5, 8, None],
        "clf__min_samples_leaf": [1, 3, 5, 10],
    },
    "Gradient Boosting": {
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__max_depth": [2, 3, 4],
        "clf__max_iter": [100, 200, 300],
    },
}


def main():
    X_train, X_test, y_train, y_test = get_split()
    cv = get_cv()
    models = get_models()

    tuned, rows = {}, []
    for name, grid in GRIDS.items():
        print(f"Tuning: {name} ...")
        model, scale = models[name]
        pipe = build_pipeline(model, X_train, scale=scale)

        search = GridSearchCV(
            pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True
        )
        search.fit(X_train, y_train)

        tuned[name] = search.best_estimator_   # already refit on all training rows
        rows.append({
            "model": name,
            "best_cv_roc_auc": round(search.best_score_, 4),
            "best_params": str({k.replace("clf__", ""): v
                                for k, v in search.best_params_.items()}),
        })
        print(f"   best ROC-AUC: {search.best_score_:.3f}")
        print(f"   best params : {search.best_params_}")

    table = pd.DataFrame(rows)
    print("\n=== Tuning summary ===")
    print(table.to_string(index=False))

    table.to_csv("results/tuning_results.csv", index=False)
    joblib.dump(tuned, "models/tuned_models.joblib")
    print("\nSaved results/tuning_results.csv and models/tuned_models.joblib")


if __name__ == "__main__":
    main()