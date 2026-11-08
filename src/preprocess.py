
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD AND CLEAN DATA
# ==========================================

def load_and_clean_data():

    df = pd.read_csv(DATA_PATH)

    # Remove customer identifier
    df = df.drop(columns=["customerID"])

    # Convert TotalCharges into numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Convert target into binary values
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    print("\nMissing values after conversion:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    return df


# ==========================================
# 3. PREPARE FEATURES AND TARGET
# ==========================================

def prepare_data():

    df = load_and_clean_data()

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Identify numerical and categorical columns
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()

    print("\nNumerical features:")
    print(numerical_features)

    print("\nCategorical features:")
    print(categorical_features)

    # ======================================
    # 4. NUMERICAL PREPROCESSING
    # ======================================

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # ======================================
    # 5. CATEGORICAL PREPROCESSING
    # ======================================

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # ======================================
    # 6. COMBINE PREPROCESSING
    # ======================================

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    # ======================================
    # 7. TRAIN TEST SPLIT
    # ======================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Fit only on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data without fitting again
    X_test_processed = preprocessor.transform(X_test)

    # ======================================
    # 8. SAVE PREPROCESSOR
    # ======================================

    joblib.dump(
        preprocessor,
        MODEL_DIR / "preprocessor.pkl"
    )

    # Save untouched test features and target
    test_data = X_test.copy()
    test_data["Churn"] = y_test

    test_data.to_csv(
        BASE_DIR / "test_data.csv",
        index=False
    )

    print("\n========== PREPROCESSING SUMMARY ==========")

    print("Original dataset shape:", df.shape)

    print("Training data shape:", X_train_processed.shape)
    print("Testing data shape:", X_test_processed.shape)

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())

    print("\nPreprocessor saved successfully!")
    print("Test data saved successfully!")

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )


# ==========================================
# 9. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    prepare_data()

