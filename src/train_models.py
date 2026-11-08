
import sys
import joblib
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD PREPROCESSED DATA
# ==========================================

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import prepare_data


# ==========================================
# 3. INITIALIZE MODELS
# ==========================================

def get_models():

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),

        "Naive Bayes": GaussianNB(),

        "SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        )

    }

    return models


# ==========================================
# 4. EVALUATION FUNCTION
# ==========================================

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {

        "Accuracy": accuracy_score(y_test, y_pred),

        "AUC": roc_auc_score(y_test, y_prob),

        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),

        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),

        "F1 Score": f1_score(
            y_test, y_pred, zero_division=0
        ),

        "MCC": matthews_corrcoef(y_test, y_pred)

    }

    return metrics, y_pred


# ==========================================
# 5. TRAIN ALL MODELS
# ==========================================

def train_all_models():

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_data()

    models = get_models()

    results = []

    for name, model in models.items():

        print("\n" + "=" * 55)
        print("Training:", name)
        print("=" * 55)

        # GaussianNB requires dense arrays
        if name == "Naive Bayes":
            X_train_model = (
                X_train.toarray()
                if hasattr(X_train, "toarray")
                else X_train
            )

            X_test_model = (
                X_test.toarray()
                if hasattr(X_test, "toarray")
                else X_test
            )

        else:
            X_train_model = X_train
            X_test_model = X_test

        # Train model
        model.fit(X_train_model, y_train)

        # Evaluate model
        metrics, y_pred = evaluate_model(
            model,
            X_test_model,
            y_test
        )

        # Display results
        print("\nEvaluation Metrics:")

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=["No Churn", "Churn"],
                zero_division=0
            )
        )

        # Save trained model
        model_filename = (
            name.lower()
            .replace(" ", "_")
            + ".pkl"
        )

        joblib.dump(
            model,
            MODEL_DIR / model_filename
        )

        print("\nModel saved:", model_filename)

        # Store metrics
        results.append({
            "Model": name,
            **metrics
        })

    # ======================================
    # 6. SAVE COMPARISON TABLE
    # ======================================

    results_df = pd.DataFrame(results)

    results_df = results_df.round(4)

    results_df.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("FINAL MODEL COMPARISON")
    print("=" * 65)

    print(results_df.to_string(index=False))

    print("\nAll five models trained successfully!")

    print("\nComparison saved to outputs/model_comparison.csv")


# ==========================================
# 7. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    train_all_models()
