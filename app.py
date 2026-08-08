import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    confusion_matrix, classification_report,
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🤖",
    layout="wide",
)

MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
]

CLASS_LABELS = ["Malignant", "Benign"]  # 0 = Malignant, 1 = Benign (sklearn convention)


# ─── Train & Cache All Models ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models — please wait...")
def train_all_models():
    data = load_breast_cancer()
    feature_names = list(data.feature_names)
    X = pd.DataFrame(data.data, columns=feature_names)
    y = pd.Series(data.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model_map = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes":         GaussianNB(),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    }

    trained_models, metrics_dict = {}, {}

    for name, model in model_map.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        trained_models[name] = model
        metrics_dict[name] = {
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "AUC Score": round(roc_auc_score(y_test, y_prob), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1 Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
            "MCC Score": round(matthews_corrcoef(y_test, y_pred), 4),
        }

    return trained_models, metrics_dict, scaler, X_test, y_test, feature_names


# ─── Helpers ──────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    return fig


def plot_metrics_bar(metrics_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(13, 5))
    metrics_df.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="black", width=0.75)
    ax.set_title("All Models — Metric Comparison", fontsize=13)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xticklabels(metrics_df.index, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    st.title("🤖 ML Classification Dashboard")
    st.markdown(
        "**Dataset:** Breast Cancer Wisconsin (UCI) &nbsp;|&nbsp; "
        "**Task:** Binary Classification — Malignant vs Benign &nbsp;|&nbsp; "
        "**Features:** 30 &nbsp;|&nbsp; **Instances:** 569"
    )
    st.divider()

    trained_models, metrics_dict, scaler, X_test_default, y_test_default, feature_names = (
        train_all_models()
    )

    metrics_df = pd.DataFrame(metrics_dict).T  # rows = models, cols = metrics

    tab1, tab2 = st.tabs(["📊 Model Overview", "🔬 Test with Custom Data"])

    # ── Tab 1: Overview ───────────────────────────────────────────────────────
    with tab1:
        st.subheader("Evaluation Metrics  (20% stratified hold-out test set)")
        styled = metrics_df.style.highlight_max(axis=0, color="#90EE90").format("{:.4f}")
        st.dataframe(styled, use_container_width=True)

        winner = metrics_df["F1 Score"].idxmax()
        st.success(
            f"🏆 **Overall Winner:** {winner} &nbsp;|&nbsp; "
            f"F1 Score: {metrics_df.loc[winner, 'F1 Score']:.4f} &nbsp;|&nbsp; "
            f"AUC: {metrics_df.loc[winner, 'AUC Score']:.4f}"
        )

        st.subheader("Visual Comparison")
        st.pyplot(plot_metrics_bar(metrics_df))

        st.subheader("Confusion Matrix Viewer")
        cm_model = st.selectbox("Select model:", MODEL_NAMES, key="cm_select")
        model = trained_models[cm_model]
        X_test_s = scaler.transform(X_test_default)
        y_pred_cm = model.predict(X_test_s)

        col_cm, col_rep = st.columns(2)
        with col_cm:
            st.pyplot(plot_confusion_matrix(y_test_default, y_pred_cm, cm_model))
        with col_rep:
            st.subheader("Classification Report")
            st.text(classification_report(y_test_default, y_pred_cm, target_names=CLASS_LABELS))

    # ── Tab 2: Custom CSV Test ─────────────────────────────────────────────────
    with tab2:
        st.subheader("Upload CSV Test Data")
        st.info(
            "Upload a CSV with 30 breast cancer feature columns. "
            "Include a `target` column (0 = Malignant, 1 = Benign) to see evaluation metrics. "
            "Use `test_data.csv` from the repo as a reference."
        )

        col_up, col_sel = st.columns([2, 1])
        with col_up:
            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        with col_sel:
            selected_model = st.selectbox("Select Model", MODEL_NAMES, key="model_select")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            has_target = "target" in df.columns
            y_true     = df["target"] if has_target else None
            X_upload   = df.drop(columns=["target"], errors="ignore")

            missing = [c for c in feature_names if c not in X_upload.columns]
            if missing:
                st.error(f"Missing columns in uploaded file: {missing}")
                return

            X_scaled = scaler.transform(X_upload[feature_names])
            model    = trained_models[selected_model]
            y_pred   = model.predict(X_scaled)
            y_prob   = model.predict_proba(X_scaled)[:, 1]

            st.success(f"Predictions generated using **{selected_model}** on {len(df)} samples.")

            pred_df = X_upload[feature_names].copy()
            pred_df["Predicted Label"]      = [CLASS_LABELS[p] for p in y_pred]
            pred_df["Probability (Benign)"] = np.round(y_prob, 4)
            if has_target:
                pred_df["Actual Label"] = [CLASS_LABELS[int(t)] for t in y_true]

            st.dataframe(pred_df, use_container_width=True)

            if has_target:
                st.divider()
                c1, c2, c3 = st.columns(3)
                live_metrics = {
                    "Accuracy":  accuracy_score(y_true, y_pred),
                    "AUC Score": roc_auc_score(y_true, y_prob),
                    "Precision": precision_score(y_true, y_pred, zero_division=0),
                    "Recall":    recall_score(y_true, y_pred, zero_division=0),
                    "F1 Score":  f1_score(y_true, y_pred, zero_division=0),
                    "MCC Score": matthews_corrcoef(y_true, y_pred),
                }
                items = list(live_metrics.items())
                for col, chunk in zip([c1, c2, c3], [items[:2], items[2:4], items[4:]]):
                    with col:
                        for k, v in chunk:
                            st.metric(k, f"{v:.4f}")

                col_cm2, col_rep2 = st.columns(2)
                with col_cm2:
                    st.pyplot(plot_confusion_matrix(y_true, y_pred, selected_model))
                with col_rep2:
                    st.subheader("Classification Report")
                    st.text(classification_report(y_true, y_pred, target_names=CLASS_LABELS))


if __name__ == "__main__":
    main()
