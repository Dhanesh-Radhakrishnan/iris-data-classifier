

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
# Resolve outputs/ relative to the project root, regardless of the
# directory the script is launched from.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def load_data() -> pd.DataFrame:
    """Step 1: Load the dataset and return it as a labeled DataFrame."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = [iris.target_names[i] for i in iris.target]
    return df


def explore_data(df: pd.DataFrame) -> None:
    """Step 1b: Print a quick summary so we understand what we're working with."""
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    print("First 5 rows:")
    print(df.head(), "\n")
    print("Class balance:")
    print(df["species"].value_counts(), "\n")
    print("Summary statistics:")
    print(df.describe(), "\n")


def split_data(df: pd.DataFrame):
    """Step 2: Split features/target into training and testing sets."""
    X = df.drop(columns=["species"])
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,          # keeps class proportions even across splits
    )

    print("=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}\n")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train) -> DecisionTreeClassifier:
    """Step 3: Train a simple, explainable classifier."""
    model = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> None:
    """Step 4: Evaluate performance and save visual outputs."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Accuracy: {acc:.2%}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    # --- Confusion matrix plot ---
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    # --- Feature importance plot ---
    importances = pd.Series(model.feature_importances_, index=model.feature_names_in_)
    importances = importances.sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    importances.plot(kind="barh", color="#4C72B0", ax=ax)
    ax.set_title("Feature Importance (Decision Tree)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150)
    plt.close()

    print(f"Saved: {OUTPUT_DIR}/confusion_matrix.png")
    print(f"Saved: {OUTPUT_DIR}/feature_importance.png")


def explore_pairplot(df: pd.DataFrame) -> None:
    """Bonus: Visualize how the classes separate across feature pairs."""
    sns.pairplot(df, hue="species", palette="Set2", corner=True)
    plt.savefig(f"{OUTPUT_DIR}/pairplot.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/pairplot.png\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()
    explore_data(df)
    explore_pairplot(df)

    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    print("=" * 60)
    print("DONE — check the outputs/ folder for saved charts.")
    print("=" * 60)


if __name__ == "__main__":
    main()
