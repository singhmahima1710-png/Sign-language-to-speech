"""
evaluate_ablation.py
---------------------
Report experiment: how much does each feature group contribute?

Trains the same Random Forest on three feature sets from data/landmarks.csv
and compares them with 5-fold cross-validation plus a held-out test split:

    1. Landmarks only      (MediaPipe, first 63 columns)
    2. Classical CV only   (skin-contour shape features, last 13 columns)
    3. Hybrid              (all 76 columns)

Usage:
    python evaluate_ablation.py

Results are printed and saved to results/ablation.csv.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

CSV_PATH = os.path.join("data", "landmarks.csv")
N_LANDMARK = 63


def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Collect data first.")
        return
    df = pd.read_csv(CSV_PATH)
    X_all = df.drop(columns=["label"]).values
    y = df["label"].values
    min_class = pd.Series(y).value_counts().min()
    if min_class < 5:
        print("ERROR: need at least 5 samples per class for 5-fold CV.")
        return

    sets = {
        "Landmarks only (MediaPipe)": X_all[:, :N_LANDMARK],
        "Classical CV only (contour features)": X_all[:, N_LANDMARK:],
        "Hybrid (both)": X_all,
    }

    rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"{len(df)} samples, {len(set(y))} classes\n")
    for name, X in sets.items():
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        cv_scores = cross_val_score(clf, X, y, cv=cv)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        clf.fit(Xtr, ytr)
        test_acc = accuracy_score(yte, clf.predict(Xte))
        rows.append({
            "feature_set": name,
            "num_features": X.shape[1],
            "cv_mean_acc": round(cv_scores.mean(), 4),
            "cv_std": round(cv_scores.std(), 4),
            "test_acc": round(test_acc, 4),
        })
        print(f"{name:40s} features={X.shape[1]:3d}  "
              f"CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}  test={test_acc:.3f}")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join("results", "ablation.csv"), index=False)
    print("\nSaved results/ablation.csv")


if __name__ == "__main__":
    main()
