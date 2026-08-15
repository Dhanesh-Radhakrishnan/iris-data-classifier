# Iris Species Classifier

A basic supervised learning project built for the **Decode Labs** assignment:
*"Build a basic classification model using a small dataset."*

## Overview

This project loads the classic **Iris flower dataset** (150 samples, 4 features,
3 species), splits it into training/testing sets, trains a **Decision Tree
Classifier**, and evaluates how well it predicts the species of a flower from
its measurements.

| | |
|---|---|
| **Dataset** | Iris (built into scikit-learn, no download needed) |
| **Samples** | 150 rows, 4 numeric features |
| **Classes** | Setosa, Versicolor, Virginica |
| **Algorithm** | Decision Tree Classifier |
| **Split** | 80% train / 20% test (stratified) |

## Project Structure

```
iris-classifier/
├── src/
│   └── classify.py        # Main script — run this
├── outputs/                # Generated charts appear here after running
├── requirements.txt
└── README.md
```

## How to Run

1. Open this folder in VS Code.
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:
   ```bash
   python src/classify.py
   ```

## What It Does

1. **Load & understand the data** — prints shape, class balance, and summary
   statistics; saves a pairplot showing how species separate across features.
2. **Split the data** — 80/20 train/test split, stratified so each class is
   proportionally represented in both sets.
3. **Train a model** — fits a `DecisionTreeClassifier` on the training data.
4. **Evaluate** — prints accuracy and a full classification report, and saves
   a confusion matrix and feature importance chart to `outputs/`.

## Sample Output

```
Accuracy: 93.33%

              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        10
  versicolor       0.90      0.90      0.90        10
   virginica       0.90      0.90      0.90        10
```

Generated charts:
- `outputs/pairplot.png` — feature relationships colored by species
- `outputs/confusion_matrix.png` — where the model got predictions right/wrong
- `outputs/feature_importance.png` — which measurements mattered most

## Why This Approach

- **Decision Tree** was chosen over more complex models because it's easy to
  explain and interpret — ideal for demonstrating the classification workflow
  clearly rather than chasing maximum accuracy.
- **Stratified split** ensures the small dataset doesn't end up with an
  imbalanced test set purely by chance.
- The code is modular (`load_data`, `split_data`, `train_model`,
  `evaluate_model`) so any step can be swapped — e.g. try `KNeighborsClassifier`
  or `LogisticRegression` in place of the Decision Tree with one line changed.

## Possible Extensions

- Swap in a different algorithm (KNN, Logistic Regression, Random Forest) and
  compare accuracy.
- Use `cross_val_score` for more robust evaluation on a small dataset.
- Replace the Iris dataset with any custom CSV by editing `load_data()`.
