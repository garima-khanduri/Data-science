"""
Task 3 — Exploratory Data Analysis (EDA)

This script:
1. Loads a self-contained CSV dataset.
2. Performs data-quality checks.
3. Produces descriptive statistics.
4. Examines target/class distribution.
5. Creates distribution plots, boxplots, scatter plot and correlation heatmap.
6. Calculates and saves the strongest feature correlations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "breast_cancer.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def main():
    # -----------------------------
    # 1. Load data
    # -----------------------------
    df = pd.read_csv(DATA_PATH)

    print("=" * 70)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    print("\nDataset shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    # -----------------------------
    # 2. Data-quality checks
    # -----------------------------
    missing = df.isnull().sum()
    duplicate_count = int(df.duplicated().sum())

    print("\nTotal missing values:", int(missing.sum()))
    print("Duplicate rows:", duplicate_count)

    # Save missing-value summary.
    missing_summary = (
        missing.rename("missing_values")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    missing_summary.to_csv(OUTPUT_DIR / "missing_values.csv", index=False)

    # -----------------------------
    # 3. Descriptive statistics
    # -----------------------------
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    summary = df[numeric_cols].describe().T
    summary.to_csv(OUTPUT_DIR / "summary_statistics.csv")

    print("\nDescriptive statistics:")
    print(summary.round(3))

    # -----------------------------
    # 4. Class distribution
    # -----------------------------
    class_counts = df["target"].value_counts().sort_index()
    class_percent = df["target"].value_counts(normalize=True).sort_index() * 100

    print("\nClass distribution:")
    for class_value in class_counts.index:
        label = "Malignant" if class_value == 0 else "Benign"
        print(
            f"{label}: {class_counts[class_value]} "
            f"({class_percent[class_value]:.2f}%)"
        )

    plt.figure(figsize=(7, 5))
    labels = ["Malignant", "Benign"]
    values = [class_counts.get(0, 0), class_counts.get(1, 0)]
    plt.bar(labels, values)
    plt.title("Target Class Distribution")
    plt.xlabel("Diagnosis")
    plt.ylabel("Number of observations")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=200)
    plt.close()

    # -----------------------------
    # 5. Feature distributions
    # -----------------------------
    selected_features = [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "mean smoothness",
        "mean compactness",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, feature in zip(axes.flat, selected_features):
        sns.histplot(
            data=df,
            x=feature,
            hue="target",
            kde=True,
            element="step",
            stat="density",
            common_norm=False,
            ax=ax,
        )
        ax.set_title(feature.title())
        ax.set_xlabel(feature.title())
        ax.set_ylabel("Density")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_distributions.png", dpi=200)
    plt.close()

    # -----------------------------
    # 6. Boxplots by target
    # -----------------------------
    box_features = [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
    ]

    long_df = df[["target"] + box_features].copy()
    long_df["Diagnosis"] = long_df["target"].map(
        {0: "Malignant", 1: "Benign"}
    )
    melted = long_df.melt(
        id_vars="Diagnosis",
        value_vars=box_features,
        var_name="Feature",
        value_name="Value",
    )

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted, x="Feature", y="Value", hue="Diagnosis")
    plt.title("Selected Feature Distributions by Diagnosis")
    plt.xlabel("Feature")
    plt.ylabel("Value")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_boxplots.png", dpi=200)
    plt.close()

    # -----------------------------
    # 7. Scatter plot
    # -----------------------------
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="mean radius",
        y="mean texture",
        hue=df["target"].map({0: "Malignant", 1: "Benign"}),
        alpha=0.75,
    )
    plt.title("Mean Radius vs Mean Texture")
    plt.xlabel("Mean Radius")
    plt.ylabel("Mean Texture")
    plt.legend(title="Diagnosis")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "radius_texture_scatter.png", dpi=200)
    plt.close()

    # -----------------------------
    # 8. Correlation analysis
    # -----------------------------
    feature_df = df.drop(columns=["target"])
    corr_matrix = feature_df.corr()

    plt.figure(figsize=(16, 13))
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        center=0,
        square=False,
        linewidths=0.2,
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()

    # Get unique feature pairs and rank by absolute correlation.
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    top_corr = (
        upper.stack()
        .reset_index()
        .rename(
            columns={
                "level_0": "feature_1",
                "level_1": "feature_2",
                0: "correlation",
            }
        )
    )
    top_corr["absolute_correlation"] = top_corr["correlation"].abs()
    top_corr = top_corr.sort_values(
        "absolute_correlation", ascending=False
    )

    top_corr.to_csv(OUTPUT_DIR / "top_correlations.csv", index=False)

    print("\nTop 10 feature correlations:")
    print(top_corr.head(10).round(4).to_string(index=False))

    # -----------------------------
    # 9. Class-wise feature means
    # -----------------------------
    class_means = df.groupby("target")[selected_features].mean().T
    class_means.columns = ["Malignant", "Benign"]
    class_means.to_csv(OUTPUT_DIR / "classwise_feature_means.csv")

    print("\nClass-wise means for selected features:")
    print(class_means.round(3))

    print("\nEDA completed successfully.")
    print("Generated files:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.is_file():
            print("-", path.name)


if __name__ == "__main__":
    main()
