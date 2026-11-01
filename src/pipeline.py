from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

from features import ORDINAL_COLS, NOMINAL_COLS, BINARY_COLS, get_numeric_cols


def make_preprocessor(X, scale=True):
    """Build the ColumnTransformer that prepares every column type."""
    numeric_cols = get_numeric_cols(X)

    # Numeric: fill gaps with the median, then (optionally) standardise
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))

    return ColumnTransformer([
        # Numbers
        ("num", Pipeline(num_steps), numeric_cols),

        # Ordered categories -> 0, 1, 2 in the order we defined
        ("ordinal", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(categories=list(ORDINAL_COLS.values()))),
        ]), list(ORDINAL_COLS)),

        # Unordered categories -> one 0/1 column per category
        ("nominal", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), NOMINAL_COLS),

        # Two-category columns -> a single 0/1 column
        ("binary", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
        ]), BINARY_COLS),
    ])


def build_pipeline(model, X, scale=True):
    """Preprocessing + classifier as ONE object (this is what prevents leakage)."""
    return Pipeline([
        ("prep", make_preprocessor(X, scale=scale)),
        ("clf", model),
    ])


if __name__ == "__main__":
    # Self-check: run this file directly to test the preprocessing
    import numpy as np
    from sklearn.model_selection import train_test_split
    from features import load_data

    X, y = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    prep = make_preprocessor(X_tr)
    Xt = prep.fit_transform(X_tr)        # fit ONLY on training data
    Xe = prep.transform(X_te)            # test data is only transformed

    print("Training matrix shape:", Xt.shape)
    print("Test matrix shape    :", Xe.shape)
    print("Any NaN left?        :", bool(np.isnan(Xt).any()))

    names = prep.get_feature_names_out()
    print(f"\nFeature names after encoding ({len(names)}):")
    for n in names:
        print("  ", n)

    # Leakage demo: the scaler learned its mean from training rows only
    scaler = prep.named_transformers_["num"].named_steps["scale"]
    print("\nAge mean learned by scaler (train only):",
          round(float(scaler.mean_[0]), 3))
    print("Age mean of the FULL dataset            :",
          round(float(X["Age"].mean()), 3))