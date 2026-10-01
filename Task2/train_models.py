"""
Predictive Modeling Using Machine Learning
Internship Task: Train and evaluate supervised classification models.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "breast_cancer.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # -----------------------------
    # 1. Load and inspect the data
    # -----------------------------
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())
    print("\nMissing values:", int(df.isnull().sum().sum()))

    # Separate features and target.
    X = df.drop(columns=["target"])
    y = df["target"]

    # -----------------------------
    # 2. Train-test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # -----------------------------
    # 3. Define models
    # -----------------------------
    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    predictions = {}
    probabilities = {}

    # -----------------------------
    # 4. Train and evaluate models
    # -----------------------------
    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
        }
        results.append(metrics)
        predictions[name] = y_pred
        probabilities[name] = y_prob

        print(f"\n{'=' * 60}")
        print(name)
        print("=" * 60)
        print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))

    results_df = pd.DataFrame(results).sort_values(
        by="ROC-AUC", ascending=False
    )
    results_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    # -----------------------------
    # 5. Confusion matrices
    # -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Malignant", "Benign"],
            yticklabels=["Malignant", "Benign"],
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=200, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # 6. ROC curves
    # -----------------------------
    plt.figure(figsize=(8, 6))

    for name, y_prob in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curves.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("\nFiles saved in:", OUTPUT_DIR)
    print("- model_metrics.csv")
    print("- confusion_matrices.png")
    print("- roc_curves.png")


if __name__ == "__main__":
    main()
