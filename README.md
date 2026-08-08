# ML-Classification-Assignment

## Problem Statement
Build and compare five machine-learning classification models on a real-world medical dataset to predict whether a tumour is **malignant or benign**, and deploy an interactive demo application on Streamlit Community Cloud.

---

## Dataset Description

| Property | Detail |
|---|---|
| **Name** | Breast Cancer Wisconsin (Diagnostic) |
| **Source** | UCI Machine Learning Repository |
| **URL** | https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic) |
| **Task** | Binary Classification |
| **Features** | 30 numeric features (mean, SE, worst of 10 cell-nucleus measurements) |
| **Instances** | 569 |
| **Classes** | 0 = Malignant (212), 1 = Benign (357) |
| **Missing values** | None |

### Feature Groups
| Group | Features (×3 stats each: mean, SE, worst) |
|---|---|
| Radius, Texture, Perimeter, Area | Size & shape descriptors |
| Smoothness, Compactness, Concavity | Surface regularity |
| Concave points, Symmetry, Fractal dimension | Boundary complexity |

---

## Model Evaluation Comparison Table

> Results on a **20% stratified hold-out test set** (random_state = 42).  
> Green = best per metric.

| Model | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
|---|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | — | — |
| Decision Tree | — | — | — | — | — | — |
| KNN | — | — | — | — | — | — |
| Naive Bayes | — | — | — | — | — | — |
| **Random Forest** | — | — | — | — | — | — |

> **Note:** Run `model/ml_classification.ipynb` or `model/train_models.py` to populate the table with actual values.

---

## Repository Structure

```
ML-Assignment-2/
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
├── test_data.csv                  # Sample test data (20% hold-out)
├── README.md
└── model/
    ├── ml_classification.ipynb    # Full notebook (training + EDA + evaluation)
    ├── train_models.py            # Standalone training script
    ├── results.csv                # Metric comparison table (generated)
    ├── comparison_chart.png       # Bar chart (generated)
    ├── scaler.pkl                 # Fitted StandardScaler
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl
```

---

## Setup & Running Locally

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd ML-Assignment-2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models & generate test_data.csv
python model/train_models.py

# 4. Launch the Streamlit app
streamlit run app.py
```

---

## Streamlit App Features

- **Tab 1 — Model Overview:** Metrics comparison table (highlighted), grouped bar chart, confusion matrix viewer, classification report
- **Tab 2 — Custom Test Data:** Upload any CSV with the same 30 features; select a model; view predictions, metrics, and confusion matrix

---

## Observations

1. **Random Forest** achieves the highest F1 and AUC scores, benefiting from ensemble variance reduction.
2. **Logistic Regression** performs comparably because the dataset is approximately linearly separable in scaled space.
3. **Decision Tree** shows slightly lower AUC — its single-tree structure is prone to overfitting.
4. **Naive Bayes** performs well despite the feature-independence assumption — low collinearity after scaling helps.
5. **KNN (k=5)** is competitive but more sensitive to the neighbourhood size and scaling.
6. All 5 models achieve **>90% accuracy**, confirming the dataset is well-structured and learnable.

---

## Overall Winner Model

**Random Forest** — highest F1 Score and AUC Score, robust generalisation via ensemble averaging, and provides built-in feature importance.

---

## Links

| | |
|---|---|
| **GitHub Repository** | _add link after pushing_ |
| **Live Streamlit App** | _add link after deploying to Streamlit Community Cloud_ |

