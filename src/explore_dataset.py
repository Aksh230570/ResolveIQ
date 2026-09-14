
from pathlib import Path
import pandas as pd

# Location of our dataset
DATA_PATH = Path("data/raw/twcs.csv")


def explore_dataset():
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        return

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("\n--- Dataset Shape ---")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\n--- Column Names ---")
    print(df.columns.tolist())

    print("\n--- First 5 Rows ---")
    print(df.head().to_string())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Data Types ---")
    print(df.dtypes)


if __name__ == "__main__":
    explore_dataset()