# ML Classification Assignment

## a. Problem Statement
Build and compare multiple machine learning classification models on the Breast Cancer Wisconsin (Diagnostic) dataset to predict whether a tumour is malignant or benign. The project also includes an interactive Streamlit web application for model comparison and prediction using uploaded test data.

---

## b. Dataset Description

| Property | Detail |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) Dataset |
| Source | UCI Machine Learning Repository |
| Task | Binary Classification |
| Features | 30 numeric features |
| Instances | 569 |
| Classes | 0 = Malignant, 1 = Benign |
| Missing Values | None |

This dataset contains measurements of cell nuclei characteristics and is widely used for binary classification tasks in medical diagnosis.

---

## Streamlit App Link
https://ml-classification-assignment-fthkfv8chrrsvhyvnvsski.streamlit.app/

## c. GitHub Repository Link
https://github.com/bala1123/ML-Classification-Assignment

---

## d. Models Used and Evaluation Metrics

The following models were trained and evaluated on the same test data:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest | 0.9561 | 0.9939 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved the best overall performance with very high accuracy, AUC, precision, recall, F1, and MCC. |
| Decision Tree | Performed reasonably well but showed weaker generalisation compared with the stronger models. |
| KNN | Performed well and produced competitive results, though slightly lower than Logistic Regression. |
| Naive Bayes | Achieved good performance, especially in AUC, but was less balanced than Logistic Regression. |
| Random Forest | Delivered very strong and balanced results, making it one of the top-performing models. |

### Overall Winner
Logistic Regression performed best overall on this dataset based on the evaluation metrics.

---


