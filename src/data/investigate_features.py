
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

uav = pd.read_csv(
    RAW / "uav_ids" / "UAVIDS-2025_0 (1).csv"
)

train = pd.read_csv(
    RAW / "unsw_nb15" / "UNSW_NB15_training-set.csv"
)

test = pd.read_csv(
    RAW / "unsw_nb15" / "UNSW_NB15_testing-set.csv"
)

print("\n========== UAV INVESTIGATION ==========")

print("\n1. SrcPort and DstPort:")
print("Equal:", uav["SrcPort"].equals(uav["DstPort"]))

print("\n2. RxByteRate and Throughput:")
print(
    "Equal:",
    np.allclose(
        uav["RxByteRate/s"],
        uav["Throughput/Kbps"],
        rtol=1e-5,
        atol=1e-8
    )
)

print("\n3. FlowID vs label correlation:")
print(uav.groupby("label")["FlowID"].agg(["min", "max", "mean"]))

print("\n4. Source address distribution:")
print(uav.groupby("SrcAddr")["label"].nunique().value_counts())

print("\n========== UNSW INVESTIGATION ==========")

print("\n5. FTP feature relationship:")

for name, df in [("TRAIN", train), ("TEST", test)]:

    print(f"\n{name}")

    print(
        pd.crosstab(
            df["is_ftp_login"],
            df["ct_ftp_cmd"]
        )
    )

    print(
        "Exact equality:",
        df["is_ftp_login"].equals(df["ct_ftp_cmd"])
    )

print("\n6. Training/testing categorical values:")

for col in ["proto", "service", "state"]:

    train_values = set(train[col].unique())
    test_values = set(test[col].unique())

    print(f"\n{col}")
    print("Train categories:", len(train_values))
    print("Test categories:", len(test_values))
    print("Test-only categories:", sorted(test_values - train_values))

print("\n7. Target distribution by partition:")

for name, df in [("TRAIN", train), ("TEST", test)]:

    print(f"\n{name}")
    print(df["label"].value_counts(normalize=True))

print("\nInvestigation completed.")
