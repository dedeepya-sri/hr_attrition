import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

SEED = 42
# Paths are relative to attrition_project/ (where you run commands from)
DATA_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

# Columns with no predictive value: constant values or just an ID
DROP_COLS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]

# Ordered categories: the order carries meaning, so encode as 0, 1, 2
ORDINAL_COLS = {
    "BusinessTravel": ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
}

# Unordered categories: one-hot encode (one 0/1 column per category)
NOMINAL_COLS = ["Department", "EducationField", "JobRole", "MaritalStatus"]

# Two-category columns: one-hot encode with one column kept (0/1)
BINARY_COLS = ["Gender", "OverTime"]


def load_data():
    """Return X (features) and y (target: 1 = left, 0 = stayed)."""
    df = pd.read_csv(DATA_PATH)
    y = (df["Attrition"] == "Yes").astype(int)
    X = df.drop(columns=["Attrition"] + DROP_COLS)
    return X, y


def get_numeric_cols(X):
    """Every column that is not one of the categorical groups above."""
    categorical = list(ORDINAL_COLS) + NOMINAL_COLS + BINARY_COLS
    return [c for c in X.columns if c not in categorical]

def get_split(test_size=0.2):
    """Stratified train/test split. Same result every time (fixed seed)."""
    X, y = load_data()
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )


def get_cv():
    """The 5 stratified folds used for every comparison and tuning step."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

if __name__ == "__main__":
    # Quick self-check when you run this file directly
    X, y = load_data()
    numeric_cols = get_numeric_cols(X)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Leavers (y = 1):", y.sum(), "| Stayers (y = 0):", (y == 0).sum())
    print("\nOrdinal columns :", list(ORDINAL_COLS))
    print("Nominal columns :", NOMINAL_COLS)
    print("Binary columns  :", BINARY_COLS)
    print(f"\nNumeric columns ({len(numeric_cols)}):")
    print(numeric_cols)

    # Safety check: every column must be in exactly one group
    all_cols = (numeric_cols + list(ORDINAL_COLS) + NOMINAL_COLS + BINARY_COLS)
    assert sorted(all_cols) == sorted(X.columns), "A column is missing or duplicated!"
    print("\nAll columns accounted for.")