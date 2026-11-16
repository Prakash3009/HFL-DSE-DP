
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

from src.models.domain_encoder import DomainEncoder
from src.models.shared_network import SharedNetwork


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"

DEVICE = torch.device("cpu")

THRESHOLDS = np.arange(0.05, 1.00, 0.05)


def load_dp_model():

    checkpoint = torch.load(
        RESULTS / "hfl_dp_global_model.pt",
        map_location=DEVICE,
        weights_only=True
    )

    uav_encoder = DomainEncoder(15)
    unsw_encoder = DomainEncoder(191)
    shared = SharedNetwork(32)

    uav_encoder.load_state_dict(checkpoint["uav_encoder"])
    unsw_encoder.load_state_dict(checkpoint["unsw_encoder"])
    shared.load_state_dict(checkpoint["shared"])

    for model in [uav_encoder, unsw_encoder, shared]:
        model.eval()

    return uav_encoder, unsw_encoder, shared


def get_probabilities(domain, models):

    uav_encoder, unsw_encoder, shared = models

    data = np.load(
        ROOT / "data" / "processed" /
        domain / "validation.npz"
    )

    X = torch.tensor(data["X"], dtype=torch.float32)
    y = data["y"].astype(int)

    encoder = uav_encoder if domain == "uav" else unsw_encoder

    with torch.no_grad():
        embeddings = encoder(X)
        logits = shared(embeddings).squeeze(1)
        probabilities = torch.sigmoid(logits).numpy()

    return y, probabilities


def analyze_thresholds(domain, y_true, probabilities):

    rows = []

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).ravel()

        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        rows.append({
            "domain": domain,
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, predictions),
            "precision": precision_score(
                y_true, predictions, zero_division=0
            ),
            "attack_recall": recall_score(
                y_true, predictions, zero_division=0
            ),
            "macro_f1": f1_score(
                y_true, predictions,
                average="macro",
                zero_division=0
            ),
            "mcc": matthews_corrcoef(y_true, predictions),
            "fpr": fpr,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        })

    return pd.DataFrame(rows)


def main():

    models = load_dp_model()

    all_results = []

    for domain in ["uav", "unsw"]:

        print(f"\nAnalyzing DP {domain.upper()} validation data")

        y_true, probabilities = get_probabilities(
            domain, models
        )

        results = analyze_thresholds(
            domain, y_true, probabilities
        )

        all_results.append(results)

        print("\nBest Macro-F1 threshold:")
        print(
            results.loc[
                results["macro_f1"].idxmax()
            ].to_string()
        )

        print("\nBest MCC threshold:")
        print(
            results.loc[
                results["mcc"].idxmax()
            ].to_string()
        )

        practical = results[
            (results["attack_recall"] >= 0.90) &
            (results["fpr"] <= 0.10)
        ]

        print("\nThresholds with recall >= 0.90 and FPR <= 0.10:")

        if practical.empty:
            print("No threshold satisfies both requirements.")
        else:
            print(practical.to_string(index=False))

        results.to_csv(
            RESULTS / f"dp_{domain}_validation_thresholds.csv",
            index=False
        )

    combined = pd.concat(all_results, ignore_index=True)

    combined.to_csv(
        RESULTS / "dp_validation_threshold_analysis.csv",
        index=False
    )

    print("\nThreshold analysis completed.")


if __name__ == "__main__":
    main()
