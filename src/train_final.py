import json
import warnings
from datetime import date

import joblib
import pandas as pd
from sklearn.base import clone

from features import load_data, ORDINAL_COLS, NOMINAL_COLS, BINARY_COLS

warnings.filterwarnings("ignore")


def build_schema(X):
    """What the app needs to know about each input: options/ranges and a 'typical' value."""
    schema = {}
    for col in X.columns:
        if col in ORDINAL_COLS:
            schema[col] = {"type": "category",
                           "options": list(ORDINAL_COLS[col]),
                           "typical": str(X[col].mode()[0])}
        elif col in NOMINAL_COLS + BINARY_COLS:
            schema[col] = {"type": "category",
                           "options": sorted(X[col].unique().tolist()),
                           "typical": str(X[col].mode()[0])}
        else:
            schema[col] = {"type": "number",
                           "min": int(X[col].min()),
                           "max": int(X[col].max()),
                           "typical": int(X[col].median())}
    return schema


def main():
    X, y = load_data()
    tuned = joblib.load("models/tuned_models.joblib")
    chosen = json.load(open("models/chosen.json"))
    name, threshold = chosen["model"], float(chosen["threshold"])

    # Same tuned settings, fresh copy, trained on ALL 1,470 employees
    model = clone(tuned[name])
    model.fit(X, y)

    schema = build_schema(X)
    bundle = {
        "model": model,
        "model_name": name,
        "threshold": threshold,
        "columns": list(X.columns),
        "schema": schema,
        "base_rate": float(y.mean()),
        "n_train": int(len(X)),
        "trained_on": str(date.today()),
    }
    joblib.dump(bundle, "models/final_model.joblib")

    # Sanity check: score a 'typical' employee (median / most common value everywhere)
    typical = pd.DataFrame([{c: schema[c]["typical"] for c in X.columns}])[list(X.columns)]
    score = model.predict_proba(typical)[0, 1]
    print(f"Final model : {name}")
    print(f"Trained on  : {len(X)} employees (settings tuned earlier)")
    print(f"Cutoff      : {threshold:.2f}")
    print(f"Typical employee risk score: {score:.2f}  (flagged: {score >= threshold})")
    print("Saved models/final_model.joblib")


if __name__ == "__main__":
    main()