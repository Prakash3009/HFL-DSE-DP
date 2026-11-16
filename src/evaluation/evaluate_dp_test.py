
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


# =====================================================
# 1. CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

DEVICE = torch.device("cpu")

RESULTS_DIR = ROOT / "experiments" / "results"

# Thresholds selected using noise-0.5 validation results
SELECTED_THRESHOLDS = {
    "uav": 0.85,
    "unsw": 0.90
}

DOMAINS = {
    "uav": 15,
    "unsw": 191
}

EXPERIMENT_NAME = "noise_1.5"


# =====================================================
# 2. LOAD DP CHECKPOINT
# =====================================================

def load_dp_model():

    checkpoint_path = (
        RESULTS_DIR / "hfl_dp_global_model.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"DP checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    uav_encoder = DomainEncoder(
        input_dim=DOMAINS["uav"]
    ).to(DEVICE)

    unsw_encoder = DomainEncoder(
        input_dim=DOMAINS["unsw"]
    ).to(DEVICE)

    shared = SharedNetwork(
        embedding_dim=32
    ).to(DEVICE)

    uav_encoder.load_state_dict(
        checkpoint["uav_encoder"]
    )

    unsw_encoder.load_state_dict(
        checkpoint["unsw_encoder"]
    )

    shared.load_state_dict(
        checkpoint["shared"]
    )

    uav_encoder.eval()
    unsw_encoder.eval()
    shared.eval()

    return uav_encoder, unsw_encoder, shared


# =====================================================
# 3. EVALUATE DOMAIN
# =====================================================

def evaluate_domain(domain, models):

    uav_encoder, unsw_encoder, shared = models

    data_path = (
        ROOT / "data" / "processed" /
        domain / "test.npz"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {data_path}"
        )

    data = np.load(data_path)

    X = torch.tensor(
        data["X"],
        dtype=torch.float32
    )

    y_true = data["y"].astype(int)

    encoder = (
        uav_encoder if domain == "uav"
        else unsw_encoder
    )

    # Generate attack probabilities
    with torch.no_grad():

        embeddings = encoder(X.to(DEVICE))

        logits = shared(embeddings).squeeze(1)

        probabilities = (
            torch.sigmoid(logits).cpu().numpy()
        )

    # Apply validation-selected threshold
    threshold = SELECTED_THRESHOLDS[domain]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # Calculate classification metrics
    accuracy = accuracy_score(
        y_true, predictions
    )

    precision = precision_score(
        y_true, predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true, predictions,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true, predictions,
        average="macro",
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_true, predictions
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    pr_auc = average_precision_score(
        y_true, probabilities
    )

    roc_auc = roc_auc_score(
        y_true, probabilities
    )

    return {
        "experiment": EXPERIMENT_NAME,
        "domain": domain,
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "attack_recall": recall,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "fpr": fpr,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


# =====================================================
# 4. MAIN
# =====================================================

def main():

    print("\nDP MODEL TEST EVALUATION")
    print("Experiment:", EXPERIMENT_NAME)
    print("Selected thresholds:", SELECTED_THRESHOLDS)

    models = load_dp_model()

    results = []

    for domain in DOMAINS:

        print("\n" + "=" * 55)
        print(f"EVALUATING {domain.upper()}")
        print("=" * 55)

        metrics = evaluate_domain(
            domain, models
        )

        results.append(metrics)

        for key, value in metrics.items():

            if isinstance(value, (float, np.floating)):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

    results_df = pd.DataFrame(results)

    output_path = (
    RESULTS_DIR /
    "hfl_dp_test_metrics_noise_1.5.csv"
)
    results_df.to_csv(
        output_path,
        index=False
    )

    print("\nFINAL DP TEST RESULTS")

    print(
        results_df.to_string(index=False)
    )

    print("\nSaved to:", output_path)


if __name__ == "__main__":
    main()
