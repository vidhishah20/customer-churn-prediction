
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
from pathlib import Path

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

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(
            135deg, #f5f7ff 0%, #eef2ff 100%
        );
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        color: #243b80;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
    }

    h2, h3 {
        color: #344b91 !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e0e7ff;
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0 4px 12px rgba(50, 70, 130, 0.08);
    }

    [data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #4338ca;
        font-weight: 800;
    }

    [data-testid="stFileUploader"],
    [data-testid="stSelectbox"] {
        background: white;
        border-radius: 12px;
    }

    .stButton > button {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: #3730a3;
        color: white;
        border: none;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #e0e7ff;
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border-color: #dbe3ff;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD SAVED MODELS AND PREPROCESSOR
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

preprocessor = joblib.load(
    MODEL_DIR / "preprocessor.pkl"
)

models = {
    "Logistic Regression": joblib.load(
        MODEL_DIR / "logistic_regression.pkl"
    ),
    "Decision Tree": joblib.load(
        MODEL_DIR / "decision_tree.pkl"
    ),
    "KNN": joblib.load(
        MODEL_DIR / "knn.pkl"
    ),
    "Naive Bayes": joblib.load(
        MODEL_DIR / "naive_bayes.pkl"
    ),
    "SVM": joblib.load(
        MODEL_DIR / "svm.pkl"
    )
}


# --------------------------------------------------
# APPLICATION TITLE
# --------------------------------------------------

st.title("📊 Customer Churn Prediction")

st.markdown("""
<p style="font-size:17px; color:#64748b;">
An intelligent machine learning dashboard to analyze customer
churn using five classification algorithms.
</p>
""", unsafe_allow_html=True)

st.divider()


# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

st.subheader("⚙️ Model Configuration")

selected_model = st.selectbox(
    "Choose a model:",
    list(models.keys())
)

# Retrieve selected trained model
model = models[selected_model]


# --------------------------------------------------
# CSV UPLOAD
# --------------------------------------------------

st.subheader("📁 Upload Customer Dataset")

uploaded_file = st.file_uploader(
    "Upload your test_data.csv file",
    type=["csv"]
)


if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.write("### Dataset Preview")
    st.dataframe(test_df.head())

    if len(test_df) == 0:
        st.error("The uploaded dataset is empty.")

    elif "Churn" not in test_df.columns:
        st.error(
            "The uploaded CSV must contain the Churn column."
        )

    else:

        # ------------------------------------------
        # SEPARATE FEATURES AND TARGET
        # ------------------------------------------

        X_test = test_df.drop(columns=["Churn"])

        y_test = test_df["Churn"].replace({
            "No": 0,
            "Yes": 1
        })

        y_test = pd.to_numeric(
            y_test,
            errors="coerce"
        )

        if (
            y_test.isnull().any()
            or not y_test.isin([0, 1]).all()
        ):
            st.error(
                "Churn must contain only Yes/No or 0/1 values."
            )

        elif y_test.nunique() < 2:
            st.error(
                "The uploaded dataset must contain both churn classes "
                "to calculate AUC."
            )

        else:

            # --------------------------------------
            # PREPROCESS DATA
            # --------------------------------------

            try:
                X_test_processed = preprocessor.transform(X_test)

            except Exception as e:
                st.error(
                    f"Error preprocessing the uploaded dataset: {e}"
                )
                st.stop()


            # --------------------------------------
            # INDIVIDUAL CUSTOMER PREDICTION
            # --------------------------------------

            st.divider()

            st.subheader("🔮 Customer Churn Prediction")

            st.write(
                "Select a customer from the uploaded test dataset."
            )

            customer_index = st.number_input(
                "Customer row number",
                min_value=0,
                max_value=len(test_df) - 1,
                value=0,
                step=1
            )

            # Keep the selected row two-dimensional
            customer_data = X_test_processed[
                int(customer_index):int(customer_index) + 1
            ]

            # Predict individual customer
            prediction = model.predict(customer_data)[0]

            probability = model.predict_proba(
                customer_data
            )[0][1]

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Prediction",
                    "Will Churn" if prediction == 1 else "Will Stay"
                )

            with col2:
                st.metric(
                    "Churn Probability",
                    f"{probability * 100:.2f}%"
                )

            if prediction == 1:
                st.warning(
                    "This customer is predicted to leave the company."
                )
            else:
                st.success(
                    "This customer is predicted to stay with the company."
                )


            # --------------------------------------
            # PREDICTIONS FOR COMPLETE TEST DATASET
            # --------------------------------------

            y_pred = model.predict(X_test_processed)

            y_prob = model.predict_proba(
                X_test_processed
            )[:, 1]


            # --------------------------------------
            # EVALUATION METRICS
            # --------------------------------------

            accuracy = accuracy_score(y_test, y_pred)

            auc = roc_auc_score(y_test, y_prob)

            precision = precision_score(
                y_test, y_pred, zero_division=0
            )

            recall = recall_score(
                y_test, y_pred, zero_division=0
            )

            f1 = f1_score(
                y_test, y_pred, zero_division=0
            )

            mcc = matthews_corrcoef(y_test, y_pred)


            st.divider()
            st.subheader("📈 Model Evaluation Metrics")

            metric_options = [
                "All Metrics",
                "Accuracy",
                "AUC",
                "Precision",
                "Recall",
                "F1 Score",
                "MCC"
            ]

            selected_metric = st.selectbox(
                "Choose an evaluation metric:",
                metric_options
            )

            metrics = {
                "Accuracy": accuracy,
                "AUC": auc,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "MCC": mcc
            }

            if selected_metric == "All Metrics":

                cols = st.columns(3)

                for i, (name, value) in enumerate(metrics.items()):
                    cols[i % 3].metric(
                        name,
                        f"{value:.4f}"
                    )

            else:

                st.metric(
                    selected_metric,
                    f"{metrics[selected_metric]:.4f}"
                )


            # --------------------------------------
            # COMPARE ALL FIVE MODELS
            # --------------------------------------

            st.divider()
            st.subheader("🏆 Model Comparison")

            comparison_results = []

            for model_name, comparison_model in models.items():

                comparison_pred = comparison_model.predict(
                    X_test_processed
                )

                comparison_prob = comparison_model.predict_proba(
                    X_test_processed
                )[:, 1]

                comparison_results.append({

                    "Model": model_name,

                    "Accuracy": accuracy_score(
                        y_test, comparison_pred
                    ),

                    "AUC": roc_auc_score(
                        y_test, comparison_prob
                    ),

                    "Precision": precision_score(
                        y_test,
                        comparison_pred,
                        zero_division=0
                    ),

                    "Recall": recall_score(
                        y_test,
                        comparison_pred,
                        zero_division=0
                    ),

                    "F1 Score": f1_score(
                        y_test,
                        comparison_pred,
                        zero_division=0
                    ),

                    "MCC": matthews_corrcoef(
                        y_test, comparison_pred
                    )
                })

            comparison_df = pd.DataFrame(
                comparison_results
            )

            comparison_metric = st.selectbox(
                "Choose metric for comparison:",
                [
                    "Accuracy",
                    "AUC",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "MCC"
                ]
            )

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.bar(
                comparison_df["Model"],
                comparison_df[comparison_metric]
            )

            ax.set_title(
                f"Model Comparison - {comparison_metric}"
            )

            ax.set_ylabel(comparison_metric)
            ax.set_xlabel("Machine Learning Models")
            ax.set_ylim(0, 1)

            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            st.write("### Complete Model Comparison")

            st.dataframe(
                comparison_df.style.format({
                    "Accuracy": "{:.4f}",
                    "AUC": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1 Score": "{:.4f}",
                    "MCC": "{:.4f}"
                }),
                use_container_width=True
            )


            # --------------------------------------
            # CONFUSION MATRIX
            # --------------------------------------

            st.divider()
            st.subheader("📊 Confusion Matrix")

            cm = confusion_matrix(
                y_test,
                y_pred,
                labels=[0, 1]
            )

            fig, ax = plt.subplots(figsize=(7, 4))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Purples",
                linewidths=1,
                linecolor="white",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"],
                ax=ax
            )

            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("Actual Label")

            ax.set_title(
                f"{selected_model} - Confusion Matrix"
            )

            st.pyplot(fig)
            plt.close(fig)


            # --------------------------------------
            # CLASSIFICATION REPORT
            # --------------------------------------

            st.divider()
            st.subheader("📋 Detailed Classification Report")

            report = classification_report(
                y_test,
                y_pred,
                labels=[0, 1],
                target_names=["No Churn", "Churn"],
                output_dict=True,
                zero_division=0
            )

            st.dataframe(
                pd.DataFrame(report).transpose()
            )

else:

    st.info(
        "Please upload your test_data.csv file to start "
        "customer churn prediction and model evaluation."
    )

    
# --------------------------------------------------
# NEW CUSTOMER PREDICTION
# --------------------------------------------------

st.divider()
st.header("🧑‍💼 Predict Churn for a New Customer")

st.write(
    "Enter the customer's information below to predict "
    "whether they are likely to leave the company."
)

with st.form("new_customer_form"):

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1]
        )

        partner = st.selectbox(
            "Has Partner?", ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Has Dependents?", ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No phone service", "No", "Yes"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with col2:
        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=1000.0,
            value=70.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=100000.0,
            value=840.0
        )

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        use_container_width=True
    )


if submitted:

    new_customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    # Apply the saved preprocessing pipeline
    new_customer_processed = preprocessor.transform(
        new_customer
    )

    # Predict using the selected trained model
    new_prediction = model.predict(
        new_customer_processed
    )[0]

    new_probability = model.predict_proba(
        new_customer_processed
    )[0][1]

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Customer Prediction",
            "Will Churn" if new_prediction == 1 else "Will Stay"
        )

    with col2:
        st.metric(
            "Churn Probability",
            f"{new_probability * 100:.2f}%"
        )

    if new_prediction == 1:
        st.warning(
            "The model predicts that this customer is likely to churn."
        )
    else:
        st.success(
            "The model predicts that this customer is likely to stay."
        )

