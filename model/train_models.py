"""
Standalone training script
--------------------------
Run from the project root:
    python model/train_models.py

Outputs:
  - model/*.pkl       (5 trained models + scaler)
  - model/results.csv (metric comparison table)
  - test_data.csv     (held-out test set for upload demo)
"""
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
)

SEED        = 42
MODEL_DIR   = os.path.dirname(os.path.abspath(__file__))   # .../model/
PROJECT_ROOT = os.path.dirname(MODEL_DIR)                  # project root


def train():
    # ── Dataset ──────────────────────────────────────────────────────────────
    data         = load_breast_cancer()
    feature_names = list(data.feature_names)
    X = pd.DataFrame(data.data, columns=feature_names)
    y = pd.Series(data.target, name="target")   # 0=Malignant, 1=Benign

    print(f"\nDataset : Breast Cancer Wisconsin (UCI)")
    print(f"Features: {X.shape[1]}  |  Instances: {X.shape[0]}")
    print(f"Classes : {list(data.target_names)}  (0=Malignant, 1=Benign)\n")

    # ── Split & Scale ────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Models ───────────────────────────────────────────────────────────────
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Decision_Tree":       DecisionTreeClassifier(random_state=SEED),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
        "Naive_Bayes":         GaussianNB(),
        "Random_Forest":       RandomForestClassifier(n_estimators=100, random_state=SEED),
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        results[name] = {
            "Accuracy":  accuracy_score(y_test, y_pred),
            "AUC Score": roc_auc_score(y_test, y_prob),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall":    recall_score(y_test, y_pred, zero_division=0),
            "F1 Score":  f1_score(y_test, y_pred, zero_division=0),
            "MCC Score": matthews_corrcoef(y_test, y_pred),
        }

        artifact = os.path.join(MODEL_DIR, f"{name.lower()}.pkl")
        joblib.dump(model, artifact)
        print(f"[✓] {name:<25} saved → {artifact}")

    # ── Save scaler ──────────────────────────────────────────────────────────
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[✓] {'Scaler':<25} saved → {scaler_path}")

    # ── Print & save results ─────────────────────────────────────────────────
    results_df = pd.DataFrame(results).T.round(4)
    print("\n── Evaluation Metrics ──────────────────────────────────────────")
    print(results_df.to_string())

    results_csv = os.path.join(MODEL_DIR, "results.csv")
    results_df.to_csv(results_csv)
    print(f"\n[✓] results.csv saved → {results_csv}")

    chart_path = os.path.join(MODEL_DIR, "comparison_chart.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    results_df[["Accuracy", "F1 Score", "AUC Score"]].plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] comparison chart saved → {chart_path}")

    # ── Save test_data.csv ───────────────────────────────────────────────────
    test_df = X_test.copy()
    test_df["target"] = y_test.values
    test_csv = os.path.join(PROJECT_ROOT, "test_data.csv")
    test_df.to_csv(test_csv, index=False)
    print(f"[✓] test_data.csv  saved → {test_csv}")

    winner = results_df["F1 Score"].idxmax()
    print(f"\n🏆 Winner: {winner}  (F1 = {results_df.loc[winner, 'F1 Score']:.4f})")
    return results_df


if __name__ == "__main__":
    train()
