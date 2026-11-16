
from pathlib import Path
import numpy as np
import json

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

def validate_dataset(domain):

    folder = PROCESSED / domain

    print(f"\n========== {domain.upper()} VALIDATION ==========")

    expected_splits = ["train", "validation", "test"]
    feature_count = None

    for split in expected_splits:

        path = folder / f"{split}.npz"

        data = np.load(path)

        X = data["X"]
        y = data["y"]
        original_labels = data["original_labels"]

        print(f"\n{split.upper()}")
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("Original labels shape:", original_labels.shape)

        # Check dimensions
        assert X.ndim == 2, "Features must be a 2D matrix."
        assert y.ndim == 1, "Targets must be a 1D array."
        assert len(X) == len(y), "Feature/target row mismatch."

        # Check feature dimensions remain constant.
        if feature_count is None:
            feature_count = X.shape[1]

        assert X.shape[1] == feature_count, (
            "Feature count differs between splits."
        )

        # Check numerical validity.
        assert np.isfinite(X).all(), "NaN or infinity detected."
        assert np.isfinite(y).all(), "Invalid target values detected."

        # Check binary targets.
        assert set(np.unique(y)).issubset({0, 1}), (
            "Unexpected target values."
        )

        # Check original labels are preserved.
        assert len(original_labels) == len(y), (
            "Original labels do not align with targets."
        )

        print("Class counts:", np.unique(y, return_counts=True))
        print("Finite values: PASS")
        print("Binary labels: PASS")
        print("Row alignment: PASS")

    with open(folder / "feature_names.json") as f:
        feature_names = json.load(f)

    assert len(feature_names) == feature_count

    print("\nFeature names:", len(feature_names))
    print("All checks passed for", domain)


if __name__ == "__main__":

    validate_dataset("uav")
    validate_dataset("unsw")

    print("\nALL PROCESSED DATA VALIDATION CHECKS PASSED.")
