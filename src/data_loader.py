
import pandas as pd
from pathlib import Path


# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"


def load_data():
    """Load the IBM Telco Customer Churn dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    return df


def inspect_data(df):
    """Display basic information about the dataset."""

    print("\n========== DATASET OVERVIEW ==========")

    print("\nDataset Shape:")
    print(df.shape)

    print("\nFirst 5 Records:")
    print(df.head())

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Records:")
    print(df.duplicated().sum())

    print("\nTarget Variable Distribution:")
    print(df["Churn"].value_counts())

    print("\nTarget Variable Percentage:")
    print(df["Churn"].value_counts(normalize=True) * 100)


if __name__ == "__main__":

    data = load_data()

    inspect_data(data)

    print("\nDataset loaded successfully!")
