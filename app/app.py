from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Works no matter which folder you launch it from
ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="Employee Attrition Risk", page_icon="📊", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(ROOT / "models" / "final_model.joblib")


bundle = load_bundle()
model = bundle["model"]
threshold = bundle["threshold"]
columns = bundle["columns"]
schema = bundle["schema"]

# How the 30 inputs are arranged on screen
GROUPS = {
    "Personal": ["Age", "Gender", "MaritalStatus", "DistanceFromHome",
                 "Education", "EducationField"],
    "Job": ["Department", "JobRole", "JobLevel", "BusinessTravel", "OverTime",
            "JobInvolvement", "PerformanceRating"],
    "Pay": ["MonthlyIncome", "DailyRate", "HourlyRate", "MonthlyRate",
            "PercentSalaryHike", "StockOptionLevel"],
    "Satisfaction": ["JobSatisfaction", "EnvironmentSatisfaction",
                     "RelationshipSatisfaction", "WorkLifeBalance",
                     "TrainingTimesLastYear"],
    "Tenure": ["TotalWorkingYears", "NumCompaniesWorked", "YearsAtCompany",
               "YearsInCurrentRole", "YearsSinceLastPromotion",
               "YearsWithCurrManager"],
}
GROUPS = {g: [c for c in cols if c in columns] for g, cols in GROUPS.items()}
leftover = [c for c in columns if c not in sum(GROUPS.values(), [])]
if leftover:
    GROUPS["Other"] = leftover


def make_input(col, container):
    """Dropdown for categories, slider/number box for numbers; defaults to a typical value."""
    s = schema[col]
    if s["type"] == "category":
        opts = s["options"]
        return container.selectbox(col, opts, index=opts.index(s["typical"]), key=col)
    lo, hi, typ = s["min"], s["max"], s["typical"]
    if hi - lo <= 50:
        return container.slider(col, lo, hi, typ, key=col)
    return container.number_input(col, min_value=lo, max_value=hi, value=typ,
                                  step=1, key=col)


def driver_effects(row):
    """For each feature: how many score points change if it is set to a typical value.
    Positive = this feature is pushing this employee's risk UP."""
    base = model.predict_proba(row)[0, 1]
    variants = pd.concat([row] * len(columns), ignore_index=True)
    for i, col in enumerate(columns):
        variants.loc[i, col] = schema[col]["typical"]
    alt = model.predict_proba(variants)[:, 1]
    return pd.Series((base - alt) * 100, index=columns)


tab_predict, tab_about = st.tabs(["Predict", "About the model"])

# ----------------------------------------------------------------------------
with tab_predict:
    st.title("Employee attrition risk")
    st.caption("Decision support for HR retention conversations. "
               "Change the details on the left; the result updates instantly.")

    left, right = st.columns([3, 2], gap="large")

    values = {}
    with left:
        st.subheader("Employee details")
        group_tabs = st.tabs(list(GROUPS))
        for gt, cols in zip(group_tabs, GROUPS.values()):
            with gt:
                grid = st.columns(2)
                for i, col in enumerate(cols):
                    values[col] = make_input(col, grid[i % 2])

        if (values["YearsAtCompany"] > values["TotalWorkingYears"]
                or values["YearsInCurrentRole"] > values["YearsAtCompany"]):
            st.warning("Check the Tenure tab: years in role cannot exceed years at the "
                       "company, and years at the company cannot exceed total working years.")

    row = pd.DataFrame([values])[columns]
    score = float(model.predict_proba(row)[0, 1])
    flagged = score >= threshold

    with right:
        st.subheader("Result")
        st.metric("Risk score (0-100)", f"{score * 100:.0f}")
        st.progress(min(max(score, 0.0), 1.0))
        st.caption(f"Employees scoring {threshold * 100:.0f} or above are flagged.")
        if flagged:
            st.error("FLAGGED: above the cutoff. Consider a retention conversation.")
        else:
            st.success("Not flagged: below the cutoff.")

        st.subheader("What is driving this score?")
        effects = driver_effects(row)
        up = effects[effects > 0.5].nlargest(5)
        down = effects[effects < -0.5].nsmallest(5)
        shown = pd.concat([up, down]).sort_values()
        if len(shown) == 0:
            st.info("This employee looks close to the typical profile, "
                    "so no single feature stands out.")
        else:
            fig, ax = plt.subplots(figsize=(5, 0.45 * len(shown) + 1.2))
            ax.barh([f"{c} = {values[c]}" for c in shown.index], shown.values,
                    color=["#B04040" if v > 0 else "#1F3864" for v in shown.values])
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Score points added (red) or removed (blue)")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        st.caption("Each bar shows how many points this employee's score would change "
                   "if that one feature were set to a typical value (median, or most "
                   "common category). It is an approximate, one-feature-at-a-time view.")

# ----------------------------------------------------------------------------
with tab_about:
    st.subheader("How this works")
    st.markdown(
        f"""
- **Model:** {bundle['model_name']}, trained on {bundle['n_train']:,} employees
  ({bundle['trained_on']}), with settings tuned by cross-validation.
- **Flag cutoff:** {threshold * 100:.0f} on the risk score, chosen from training
  cross-validation to favour catching leavers over avoiding false alarms.
- **The risk score is a ranking score, not a literal probability.** The model was
  trained with class weights to handle the 84:16 imbalance, which inflates the raw
  numbers. Use it to compare and prioritise employees, not to say "this person has
  a 40% chance of leaving".
- **Limitations:** the data is a public sample dataset; patterns are associations,
  not causes; the hold-out test set contained only 47 leavers, so the metrics below
  have wide uncertainty; and the score should support a conversation, never be the
  basis of an employment decision on its own.
"""
    )

    st.subheader("Evaluation on the hold-out test set")
    figures = [
        ("confusion_matrix.png", "Confusion matrix"),
        ("roc_curve.png", "ROC curve"),
        ("threshold_tradeoff.png", "Recall vs precision as the cutoff moves"),
        ("feature_importance.png", "Permutation importance"),
        ("coefficients.png", "Model coefficients"),
        ("attrition_by_group.png", "Raw attrition rate by group"),
    ]
    available = [(f, c) for f, c in figures if (ROOT / "results" / f).exists()]
    if not available:
        st.info("Run the evaluate and explain scripts to see charts here.")
    for i in range(0, len(available), 2):
        cols = st.columns(2)
        for col, (fname, caption) in zip(cols, available[i:i + 2]):
            col.image(str(ROOT / "results" / fname), caption=caption)

    report = ROOT / "results" / "test_metrics.txt"
    if report.exists():
        with st.expander("Full test-set report"):
            st.code(report.read_text())