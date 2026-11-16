
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
)

from src.models.domain_encoder import DomainEncoder
from src.models.shared_network import SharedNetwork


ROOT = Path(__file__).resolve().parents[2]

DEVICE = torch.device("cpu")

RESULTS_DIR = ROOT / "experiments/results"

LOCAL_CHECKPOINT = (
    RESULTS_DIR / "local_baseline_models.pt"
)


def evaluate_client(domain, client_id, local_models):

    model_key = f"{domain}_client_{client_id}"

    # Load the domain validation dataset.
    data = np.load(
        ROOT / f"data/processed/{domain}/validation.npz",
        allow_pickle=True
    )

    X = torch.tensor(
        data["X"].astype(np.float32),
        dtype=torch.float32
    )

    y = data["y"].astype(np.int64)

    # Initialize the correct domain encoder.
    encoder = DomainEncoder(
        input_dim=X.shape[1]
    ).to(DEVICE)

    shared = SharedNetwork(
        embedding_dim=32
    ).to(DEVICE)

    # Load this client's independent model.
    encoder.load_state_dict(
        local_models[model_key]["encoder"]
    )

    shared.load_state_dict(
        local_models[model_key]["shared"]
    )

    encoder.eval()
    shared.eval()

    # Generate predictions.
    with torch.no_grad():

        embeddings = encoder(X)

        logits = shared(
            embeddings
        ).squeeze(1)

        probabilities = torch.sigmoid(
            logits
        ).numpy()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # Calculate metrics.
    accuracy = accuracy_score(y, predictions)

    precision = precision_score(
        y, predictions, zero_division=0
    )

    attack_recall = recall_score(
        y, predictions, zero_division=0
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    mcc = matthews_corrcoef(y, predictions)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    ).ravel()

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    return {
        "domain": domain.upper(),
        "client_id": client_id,
        "accuracy": accuracy,
        "precision": precision,
        "attack_recall": attack_recall,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "fpr": fpr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():

    print("\nLOCAL-ONLY VALIDATION EVALUATION")

    local_models = torch.load(
        LOCAL_CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    all_results = []

    for domain in ["uav", "unsw"]:

        print("\n" + "=" * 60)
        print(f"DOMAIN: {domain.upper()}")
        print("=" * 60)

        for client_id in range(1, 6):

            metrics = evaluate_client(
                domain,
                client_id,
                local_models
            )

            all_results.append(metrics)

            print(
                f"\nClient {client_id}: "
                f"Accuracy={metrics['accuracy']:.4f}, "
                f"Macro-F1={metrics['macro_f1']:.4f}, "
                f"MCC={metrics['mcc']:.4f}, "
                f"Recall={metrics['attack_recall']:.4f}, "
                f"FPR={metrics['fpr']:.4f}"
            )

    results_df = pd.DataFrame(all_results)

    # Save individual client metrics.
    results_df.to_csv(
        RESULTS_DIR / "local_baseline_validation.csv",
        index=False
    )

    # Calculate mean and standard deviation across clients.
    summary = (
        results_df
        .groupby("domain")[
            [
                "accuracy",
                "precision",
                "attack_recall",
                "macro_f1",
                "mcc",
                "fpr"
            ]
        ]
        .agg(["mean", "std"])
    )

    summary.to_csv(
        RESULTS_DIR / "local_baseline_validation_summary.csv"
    )

    print("\n" + "=" * 60)
    print("LOCAL-ONLY VALIDATION SUMMARY")
    print("=" * 60)

    print(summary.round(4).to_string())

    print("\nResults saved successfully.")


if __name__ == "__main__":
    main()
