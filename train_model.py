"""
train_model.py
---------------
STEP 2 of the pipeline: train a classifier on the landmark dataset
produced by data_collection.py.

Usage:
    python train_model.py

What it does:
- Loads data/landmarks.csv
- Splits into train/test sets
- Trains a Random Forest classifier (fast, robust, no GPU needed, and
  works well on small/medium tabular landmark datasets)
- Prints accuracy + a classification report
- Saves the trained model to model/sign_classifier.joblib
- Saves the label list to model/labels.joblib

Swap in a different scikit-learn classifier (SVM, MLPClassifier, etc.)
easily by changing the `clf = ...` line if you want to compare options
for your report.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "sign_classifier.joblib")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.joblib")


def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run data_collection.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        print("ERROR: dataset is empty. Collect some samples first.")
        return

    print(f"Loaded {len(df)} samples across {df['label'].nunique()} classes:")
    print(df["label"].value_counts())

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(sorted(df["label"].unique().tolist()), LABELS_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Labels saved to {LABELS_PATH}")


if __name__ == "__main__":
    main()
