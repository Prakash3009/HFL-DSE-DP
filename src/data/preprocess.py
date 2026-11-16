
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

UAV_RAW = RAW / "uav_ids" / "UAVIDS-2025_0 (1).csv"

UNSW_TRAIN_RAW = RAW / "unsw_nb15" / "UNSW_NB15_training-set.csv"
UNSW_TEST_RAW = RAW / "unsw_nb15" / "UNSW_NB15_testing-set.csv"

UAV_OUT = PROCESSED / "uav"
UNSW_OUT = PROCESSED / "unsw"

UAV_OUT.mkdir(parents=True, exist_ok=True)
UNSW_OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def save_dataset(output_dir, prefix, X, y, original_labels=None):
    """
    Save features, binary targets and optional original labels.
    Features are stored as float32 to reduce memory usage.
    """

    np.savez_compressed(
        output_dir / f"{prefix}.npz",
        X=X.astype(np.float32),
        y=y.astype(np.int64),
        original_labels=(
            np.asarray(original_labels, dtype=str)
            if original_labels is not None
            else np.array([], dtype=str)
        )
    )


def print_split_summary(name, X, y):
    print(f"\n{name}")
    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)
    print("Class distribution:")
    print(pd.Series(y).value_counts().sort_index())


# ============================================================
# 3. UAV PREPROCESSING
# ============================================================

def preprocess_uav():

    print("\n" + "=" * 60)
    print("UAV DATASET PREPROCESSING")
    print("=" * 60)

    df = pd.read_csv(UAV_RAW)

    print("Original dataset shape:", df.shape)

    # Remove identifiers, addresses, constant/redundant columns.
    drop_columns = [
        "FlowID",
        "SrcAddr",
        "DstAddr",
        "Protocol",
        "DstPort",
        "SrcPort",
        "RxByteRate/s",
    ]

    df = df.drop(columns=drop_columns)

    # Keep original multiclass labels for later analysis.
    original_labels = df["label"].copy()

    # Convert multiclass labels into binary targets.
    # Normal Traffic = 0
    # Every attack category = 1
    y = (df["label"] != "Normal Traffic").astype(np.int64)

    X = df.drop(columns=["label"])

    # All remaining UAV predictors should be numeric.
    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        raise ValueError("Unexpected non-numeric UAV feature detected.")

    # Split: 70% train, 15% validation, 15% test.
    X_train, X_temp, y_train, y_temp, labels_train, labels_temp = (
        train_test_split(
            X,
            y,
            original_labels,
            test_size=0.30,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    X_val, X_test, y_val, y_test, labels_val, labels_test = (
        train_test_split(
            X_temp,
            y_temp,
            labels_temp,
            test_size=0.50,
            random_state=RANDOM_STATE,
            stratify=y_temp
        )
    )

    # Fit scaler ONLY on training data.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), X.columns.tolist())
        ],
        remainder="drop"
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Convert to float32 for efficient model training.
    X_train_processed = np.asarray(X_train_processed, dtype=np.float32)
    X_val_processed = np.asarray(X_val_processed, dtype=np.float32)
    X_test_processed = np.asarray(X_test_processed, dtype=np.float32)

    # Save processed datasets.
    save_dataset(
        UAV_OUT, "train",
        X_train_processed, y_train, labels_train
    )

    save_dataset(
        UAV_OUT, "validation",
        X_val_processed, y_val, labels_val
    )

    save_dataset(
        UAV_OUT, "test",
        X_test_processed, y_test, labels_test
    )

    # Save fitted preprocessing object and feature names.
    joblib.dump(preprocessor, UAV_OUT / "preprocessor.joblib")

    feature_names = preprocessor.get_feature_names_out().tolist()

    with open(UAV_OUT / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=4)

    print("\nUAV preprocessing completed.")

    print_split_summary("UAV TRAIN", X_train_processed, y_train)
    print_split_summary("UAV VALIDATION", X_val_processed, y_val)
    print_split_summary("UAV TEST", X_test_processed, y_test)

    print("\nNumber of UAV features:", len(feature_names))


# ============================================================
# 4. UNSW-NB15 PREPROCESSING
# ============================================================

def preprocess_unsw():

    print("\n" + "=" * 60)
    print("UNSW-NB15 DATASET PREPROCESSING")
    print("=" * 60)

    train_df = pd.read_csv(UNSW_TRAIN_RAW)
    test_df = pd.read_csv(UNSW_TEST_RAW)

    print("Official training shape:", train_df.shape)
    print("Official testing shape:", test_df.shape)

    # Remove identifiers and redundant/target-related columns.
    drop_columns = [
        "id",
        "attack_cat",
        "ct_ftp_cmd",
    ]

    train_df = train_df.drop(columns=drop_columns)
    test_df = test_df.drop(columns=drop_columns)

    # Separate binary targets.
    y_train_full = train_df["label"].astype(np.int64)
    y_test = test_df["label"].astype(np.int64)

    # Preserve attack categories separately for later analysis.
    original_train_labels = pd.read_csv(
        UNSW_TRAIN_RAW,
        usecols=["attack_cat"]
    )["attack_cat"]

    original_test_labels = pd.read_csv(
        UNSW_TEST_RAW,
        usecols=["attack_cat"]
    )["attack_cat"]

    X_train_full = train_df.drop(columns=["label"])
    X_test = test_df.drop(columns=["label"])

    # Create validation set from official training data.
    (
        X_train,
        X_val,
        y_train,
        y_val,
        labels_train,
        labels_val
    ) = train_test_split(
        X_train_full,
        y_train_full,
        original_train_labels,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )

    # Identify categorical and numerical columns.
    categorical_columns = ["proto", "service", "state"]

    numerical_columns = [
        col for col in X_train.columns
        if col not in categorical_columns
    ]

    # Numerical features: standardization.
    # Categorical features: one-hot encoding.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numerical_columns
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32
                ),
                categorical_columns
            )
        ],
        remainder="drop"
    )

    # Fit ONLY on the training partition.
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform validation and official test partitions.
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    X_train_processed = np.asarray(X_train_processed, dtype=np.float32)
    X_val_processed = np.asarray(X_val_processed, dtype=np.float32)
    X_test_processed = np.asarray(X_test_processed, dtype=np.float32)

    # Save datasets.
    save_dataset(
        UNSW_OUT, "train",
        X_train_processed, y_train, labels_train
    )

    save_dataset(
        UNSW_OUT, "validation",
        X_val_processed, y_val, labels_val
    )

    save_dataset(
        UNSW_OUT, "test",
        X_test_processed, y_test, original_test_labels
    )

    # Save preprocessing object and feature names.
    joblib.dump(preprocessor, UNSW_OUT / "preprocessor.joblib")

    feature_names = preprocessor.get_feature_names_out().tolist()

    with open(UNSW_OUT / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=4)

    print("\nUNSW preprocessing completed.")

    print_split_summary("UNSW TRAIN", X_train_processed, y_train)
    print_split_summary("UNSW VALIDATION", X_val_processed, y_val)
    print_split_summary("UNSW TEST", X_test_processed, y_test)

    print("\nNumber of UNSW features:", len(feature_names))


# ============================================================
# 5. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    preprocess_uav()
    preprocess_unsw()

    print("\n" + "=" * 60)
    print("ALL PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)
