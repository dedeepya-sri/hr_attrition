import pandas as pd

# Path is relative to attrition_project/ (where you run the command from)
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# 1. Size of the data
print("Shape (rows, columns):", df.shape)

# 2. Column names and data types
print("\n--- Column types ---")
print(df.dtypes)

# 3. Missing values
print("\nTotal missing values:", df.isna().sum().sum())

# 4. Target balance
print("\n--- Attrition counts ---")
print(df["Attrition"].value_counts())
print("\n--- Attrition proportions ---")
print(df["Attrition"].value_counts(normalize=True).round(3))

# 5. Columns that never change (useless for prediction)
nunique = df.nunique()
print("\nConstant columns:", list(nunique[nunique == 1].index))

# 6. Columns that are just IDs
print("ID-like columns:", [c for c in df.columns if nunique[c] == len(df)])

# 7. Text (categorical) columns and their categories
print("\n--- Categorical columns ---")
for col in df.select_dtypes(include="object").columns:
    print(col, "->", df[col].unique().tolist())