
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports" / "tables"

REPORTS.mkdir(parents=True, exist_ok=True)


def audit_file(path, dataset_name):

    print("\n" + "=" * 80)
    print(f"FEATURE AUDIT: {dataset_name}")
    print("=" * 80)

    df = pd.read_csv(path, low_memory=False)

    records = []

    for col in df.columns:

        unique_count = df[col].nunique(dropna=False)

        records.append({
            "feature": col,
            "dtype": str(df[col].dtype),
            "unique_values": unique_count,
            "missing_values": df[col].isna().sum(),
            "is_constant": unique_count <= 1,
            "example_values": str(
                df[col].dropna().unique()[:5].tolist()
            )
        })

    feature_report = pd.DataFrame(records)

    print("\n1. FEATURE SUMMARY")
    print(feature_report.to_string(index=False))

    feature_report.to_csv(
        REPORTS / f"{dataset_name}_feature_summary.csv",
        index=False
    )

    # Exact duplicate column investigation
    print("\n2. IDENTICAL COLUMNS")

    checked = set()
    identical_pairs = []

    for i, col1 in enumerate(df.columns):

        for col2 in df.columns[i + 1:]:

            if (col1, col2) in checked:
                continue

            if df[col1].equals(df[col2]):

                identical_pairs.append((col1, col2))
                checked.add((col1, col2))

    if identical_pairs:
        for pair in identical_pairs:
            print(pair)
    else:
        print("No identical columns found.")

    # Identifier investigation
    print("\n3. IDENTIFIER INVESTIGATION")

    for col in ["FlowID", "id", "SrcAddr", "DstAddr"]:

        if col in df.columns:

            print(f"\n{col}")

            print("Unique values:", df[col].nunique())
            print("First five:", df[col].head().tolist())

            if col in ["FlowID", "id"]:

                sequential = (
                    df[col].reset_index(drop=True)
                    .equals(
                        pd.Series(
                            np.arange(1, len(df) + 1),
                            name=col
                        )
                    )
                )

                print("Sequential 1..N:", sequential)

    # Correlation investigation
    print("\n4. HIGH NUMERICAL CORRELATIONS")

    numeric = df.select_dtypes(include=[np.number])

    corr = numeric.corr().abs()

    upper = corr.where(
        np.triu(
            np.ones(corr.shape),
            k=1
        ).astype(bool)
    )

    pairs = (
        upper.stack()
        .sort_values(ascending=False)
    )

    if not pairs.empty:
        print(pairs.head(20).to_string())

    # Label distributions
    print("\n5. TARGET DISTRIBUTION")

    for label_col in ["label", "attack_cat"]:

        if label_col in df.columns:

            print(f"\n{label_col}")
            print(df[label_col].value_counts(dropna=False))

    print("\n6. REPORT SAVED:")
    print(REPORTS / f"{dataset_name}_feature_summary.csv")


def main():

    uav_path = RAW / "uav_ids" / "UAVIDS-2025_0 (1).csv"

    unsw_train = RAW / "unsw_nb15" / "UNSW_NB15_training-set.csv"

    unsw_test = RAW / "unsw_nb15" / "UNSW_NB15_testing-set.csv"

    audit_file(uav_path, "uav")

    audit_file(unsw_train, "unsw_train")

    audit_file(unsw_test, "unsw_test")


if __name__ == "__main__":
    main()
