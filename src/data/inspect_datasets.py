
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

LABEL_CANDIDATES = [
    "label",
    "class",
    "attack",
    "attack_cat",
    "category",
    "attack_type",
    "target"
]


def inspect_dataset(file_path):

    print("\n" + "=" * 75)
    print(f"DATASET: {file_path.name}")
    print("=" * 75)

    df = pd.read_csv(file_path, low_memory=False)

    # 1. Dataset dimensions
    print("\n1. SHAPE")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    # 2. Column names
    print("\n2. COLUMN NAMES")
    for col in df.columns:
        print("-", repr(col))

    # 3. Data types
    print("\n3. DATA TYPES")
    print(df.dtypes.to_string())

    # 4. Missing values
    print("\n4. MISSING VALUES")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    if missing.sum() == 0:
        print("No missing values found.")

    # 5. Duplicate rows
    print("\n5. DUPLICATES")
    print("Duplicate rows:", df.duplicated().sum())

    # 6. Infinite values
    print("\n6. INFINITE VALUES")
    numeric = df.select_dtypes(include=np.number)
    if len(numeric.columns):
        print(np.isinf(numeric).sum().sum())
    else:
        print("No numerical columns.")

    # 7. Potential label columns
    print("\n7. POTENTIAL LABEL DISTRIBUTIONS")

    normalized_columns = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    found = False

    for candidate in LABEL_CANDIDATES:
        if candidate in normalized_columns:

            actual_col = normalized_columns[candidate]

            print(f"\nColumn: {actual_col}")
            print(df[actual_col].value_counts(dropna=False))

            found = True

    if not found:
        print("No standard label column detected.")

    # 8. Categorical features
    print("\n8. CATEGORICAL COLUMNS")

    categorical = df.select_dtypes(
        include=["object", "category"]
    )

    print("Categorical columns:", list(categorical.columns))

    for col in categorical.columns:
        print(
            f"{col}: {df[col].nunique(dropna=False)} unique values"
        )

    # 9. Numerical summary
    print("\n9. NUMERICAL SUMMARY")
    print(numeric.describe().to_string())


def main():

    print("HFL-DSE-DP DATASET AUDIT")
    print("Project root:", ROOT)

    files = sorted(RAW_DIR.rglob("*.csv"))

    files = [
        f for f in files
        if "features" not in f.name.lower()
    ]

    if not files:
        print("No CSV datasets found!")
        return

    for file_path in files:
        try:
            inspect_dataset(file_path)
        except Exception as error:
            print(f"Could not inspect {file_path}: {error}")


if __name__ == "__main__":
    main()
