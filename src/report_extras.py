import matplotlib
matplotlib.use("Agg")                      # save to files, no pop-up windows
import matplotlib.pyplot as plt
from features import load_data

COLS = ["Age", "MonthlyIncome", "DistanceFromHome", "TotalWorkingYears",
        "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
        "PercentSalaryHike"]


def main():
    X, y = load_data()

    # 1. Class distribution chart
    counts = y.map({0: "Stayed", 1: "Left"}).value_counts().reindex(["Stayed", "Left"])
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(counts.index, counts.values, color=["#1F3864", "#B04040"])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 15,
                f"{v}\n({v / len(y) * 100:.1f}%)", ha="center", fontsize=9)
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_ylabel("Number of employees")
    ax.set_title(f"Attrition class distribution (n = {len(y):,})")
    fig.tight_layout()
    fig.savefig("results/class_distribution.png", dpi=200)
    plt.close(fig)

    # 2. Correlation heatmap of numeric features
    corr = X[COLS].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels(COLS, rotation=45, ha="right")
    ax.set_yticks(range(len(COLS)))
    ax.set_yticklabels(COLS)
    for i in range(len(COLS)):
        for j in range(len(COLS)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, label="Correlation coefficient")
    ax.set_title("Correlation heatmap of numeric features")
    fig.tight_layout()
    fig.savefig("results/correlation_heatmap.png", dpi=200)
    plt.close(fig)

    # 3. Summary statistics table
    stats = X[COLS].describe().T[["count", "mean", "std", "min", "max"]].round(2)
    stats.to_csv("results/summary_stats.csv")
    print(stats.to_string())
    print("\nSaved: results/class_distribution.png, correlation_heatmap.png, summary_stats.csv")


if __name__ == "__main__":
    main()