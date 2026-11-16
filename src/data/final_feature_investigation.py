
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

uav = pd.read_csv(
    RAW / "uav_ids" / "UAVIDS-2025_0 (1).csv"
)

unsw_train = pd.read_csv(
    RAW / "unsw_nb15" / "UNSW_NB15_training-set.csv"
)

unsw_test = pd.read_csv(
    RAW / "unsw_nb15" / "UNSW_NB15_testing-set.csv"
)

# --------------------------------------------------
# 1. UAV: SrcPort relationship with attack labels
# --------------------------------------------------

print("\n========== UAV SRC PORT ANALYSIS ==========")

print("\nSrcPort value counts:")
print(uav["SrcPort"].value_counts())

print("\nSrcPort distribution by label:")
print(
    pd.crosstab(
        uav["label"],
        uav["SrcPort"],
        normalize="index"
    ).round(4)
)

print("\nLabel distribution by SrcPort:")
print(
    pd.crosstab(
        uav["SrcPort"],
        uav["label"],
        normalize="index"
    ).round(4)
)

# --------------------------------------------------
# 2. UAV: Compare byte rate and throughput
# --------------------------------------------------

print("\n========== UAV RATE VS THROUGHPUT ==========")

rate = uav["RxByteRate/s"]
throughput = uav["Throughput/Kbps"]

difference = rate - throughput

print("\nCorrelation:")
print(rate.corr(throughput))

print("\nDifference statistics:")
print(difference.describe())

print("\nAre values approximately equal?")
print(np.allclose(rate, throughput, rtol=1e-5, atol=1e-8))

print("\nSample comparison:")
print(
    pd.DataFrame({
        "RxByteRate": rate.head(10),
        "Throughput": throughput.head(10),
        "Difference": difference.head(10)
    })
)

# --------------------------------------------------
# 3. UNSW: Find inconsistent FTP rows in test data
# --------------------------------------------------

print("\n========== UNSW FTP INCONSISTENCY ==========")

mismatch = unsw_test[
    unsw_test["is_ftp_login"] != unsw_test["ct_ftp_cmd"]
]

print("\nNumber of mismatched test rows:", len(mismatch))

print("\nMismatched rows:")
print(
    mismatch[
        [
            "is_ftp_login",
            "ct_ftp_cmd",
            "proto",
            "service",
            "attack_cat",
            "label"
        ]
    ].to_string(index=False)
)

print("\nMismatch label distribution:")
print(mismatch["label"].value_counts())

print("\nMismatch attack categories:")
print(mismatch["attack_cat"].value_counts())

print("\nInvestigation completed.")
