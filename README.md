# Customer Churn Prediction Using Machine Learning

## 1. Problem Statement

Customer churn is an important challenge for telecommunications companies. Identifying customers who are likely to leave helps companies improve customer retention.

The objective of this project is to develop and compare five machine learning classification algorithms to predict whether a customer will churn or stay.

## 2. Dataset Description

**Dataset:** IBM Telco Customer Churn Dataset

The dataset contains 7,043 records and 21 columns, including customer demographics, account information, subscribed services, contract details, monthly charges, total charges, and churn status.

The `customerID` column was removed because it is an identifier and does not provide useful predictive information.

Target variable:
- No = 0
- Yes = 1

## 3. Machine Learning Models

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 4. Evaluation Metrics

The models were evaluated using:
- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## 5. Results and Model Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8055 | 0.8419 | 0.6572 | 0.5588 | 0.6040 | 0.4790 |
| Decision Tree | 0.7289 | 0.6573 | 0.4896 | 0.5053 | 0.4974 | 0.3119 |
| KNN | 0.7637 | 0.7884 | 0.5527 | 0.5749 | 0.5636 | 0.4018 |
| Naive Bayes | 0.6948 | 0.8074 | 0.4589 | 0.8369 | 0.5928 | 0.4245 |
| SVM | 0.7913 | 0.7905 | 0.6418 | 0.4840 | 0.5518 | 0.4264 |

## 6. Observations

### Logistic Regression
Achieved the highest accuracy, AUC, precision, F1 score, and MCC among the five models. It provided the strongest overall performance.

### Decision Tree
Achieved the lowest overall performance across most evaluation metrics. It may require further tuning to improve its generalization.

### K-Nearest Neighbors
Provided moderate performance. Its predictions depend on distances between customer records.

### Gaussian Naive Bayes
Achieved the highest recall, identifying the largest proportion of actual churn customers. However, its precision was comparatively low.

### Support Vector Machine
Provided good accuracy and precision, but its recall was lower than Logistic Regression and KNN.

## 7. Overall Conclusion

Logistic Regression performed best overall for this dataset, achieving an accuracy of 80.55% and an AUC of 0.8419.

Naive Bayes achieved the highest recall of 83.69%, making it useful when identifying as many potential churn customers as possible is a priority.

Therefore, Logistic Regression is selected as the best overall model based on the combined evaluation metrics.

## 8. Project Structure

```text
Assignment2/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── preprocessor.pkl
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   └── svm.pkl
├── outputs/
│   └── model_comparison.csv
└── src/
    ├── data_loader.py
    ├── preprocess.py
    └── train_models.py
```

## 9. Links

- **GitHub Repository:** To be added
- **Streamlit Application:** To be added
