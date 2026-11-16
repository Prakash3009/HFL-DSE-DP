
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed" / "unsw"
RESULTS_DIR = ROOT / "experiments" / "results"


def load_split(name):

    data = np.load(DATA_DIR / f"{name}.npz")

    X = data["X"]
    y = data["y"].astype(int)

    return X, y


def main():

    X_val, y_val = load_split("validation")
    X_test, y_test = load_split("test")

    feature_names_path = (
        DATA_DIR / "feature_names.json"
    )

    import json

    with open(feature_names_path, "r") as f:
        feature_names = json.load(f)

    if len(feature_names) != X_val.shape[1]:
        raise ValueError("Feature-name count does not match X.")

    print("\nUNSW VALIDATION VS TEST DISTRIBUTION")
    print("=" * 65)

    for split_name, y in [
        ("Validation", y_val),
        ("Test", y_test)
    ]:
        unique, counts = np.unique(y, return_counts=True)

        print(f"\n{split_name} class distribution:")

        for label, count in zip(unique, counts):
            print(
                f"Class {label}: {count} "
                f"({count / len(y) * 100:.2f}%)"
            )

    rows = []

    # Compare each class separately.
    for label, class_name in [
        (0, "normal"),
        (1, "attack")
    ]:

        val_data = X_val[y_val == label]
        test_data = X_test[y_test == label]

        for i, feature in enumerate(feature_names):

            val_feature = val_data[:, i]
            test_feature = test_data[:, i]

            ks = ks_2samp(
                val_feature,
                test_feature
            )

            rows.append({
                "class": class_name,
                "feature": feature,
                "validation_mean": np.mean(val_feature),
                "test_mean": np.mean(test_feature),
                "validation_median": np.median(val_feature),
                "test_median": np.median(test_feature),
                "ks_statistic": ks.statistic,
                "ks_pvalue": ks.pvalue
            })

    results = pd.DataFrame(rows)

    for class_name in ["normal", "attack"]:

        print("\n" + "=" * 65)
        print(f"TOP 20 DRIFTING FEATURES: {class_name.upper()}")
        print("=" * 65)

        subset = results[
            results["class"] == class_name
        ].sort_values(
            "ks_statistic",
            ascending=False
        )

        print(
            subset.head(20).to_string(index=False)
        )

    output_path = (
        RESULTS_DIR / "unsw_feature_drift.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\nFull feature drift report saved to:")
    print(output_path)


if __name__ == "__main__":
    main()
