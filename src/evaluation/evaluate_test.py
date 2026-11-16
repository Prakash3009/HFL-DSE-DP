
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
    roc_auc_score,
)

from src.models.domain_encoder import DomainEncoder
from src.models.shared_network import SharedNetwork


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DEVICE = torch.device("cpu")

CHECKPOINT_PATH = (
    ROOT / "experiments/results/hfl_global_model.pt"
)

RESULTS_DIR = ROOT / "experiments/results"

# Thresholds selected using validation data only.
SELECTED_THRESHOLDS = {
    "uav": 0.85,
    "unsw": 0.90
}


# --------------------------------------------------
# 2. EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_domain(domain):

    print("\n" + "=" * 60)
    print(f"EVALUATING {domain.upper()}")
    print("=" * 60)

    # Load the untouched test dataset.
    test_path = (
        ROOT / f"data/processed/{domain}/test.npz"
    )

    data = np.load(test_path, allow_pickle=True)

    X_test = torch.tensor(
        data["X"].astype(np.float32),
        dtype=torch.float32
    )

    y_test = data["y"].astype(np.int64)

    # Load global checkpoint.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    # Select the appropriate domain encoder.
    if domain == "uav":
        encoder_state = checkpoint["uav_encoder"]
    elif domain == "unsw":
        encoder_state = checkpoint["unsw_encoder"]
    else:
        raise ValueError(f"Unknown domain: {domain}")

    # Initialize models.
    encoder = DomainEncoder(
        input_dim=X_test.shape[1]
    )

    shared_model = SharedNetwork()

    # Load trained parameters.
    encoder.load_state_dict(encoder_state)

    shared_model.load_state_dict(
        checkpoint["shared"]
    )

    encoder.to(DEVICE)
    shared_model.to(DEVICE)

    encoder.eval()
    shared_model.eval()

    # Generate prediction probabilities.
    with torch.no_grad():

        embeddings = encoder(X_test)

        logits = shared_model(
            embeddings
        ).squeeze(1)

        probabilities = torch.sigmoid(
            logits
        ).cpu().numpy()

    # Apply validation-selected threshold.
    selected_threshold = SELECTED_THRESHOLDS[domain]

    predictions = (
        probabilities >= selected_threshold
    ).astype(int)

    # --------------------------------------------------
    # 3. CALCULATE EVALUATION METRICS
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test, predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    attack_recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test, predictions
    )

    # Confusion matrix.
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    # False Positive Rate.
    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    # Threshold-independent ranking metrics.
    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    # --------------------------------------------------
    # 4. STORE RESULTS
    # --------------------------------------------------

    metrics = {
        "domain": domain.upper(),
        "threshold": selected_threshold,
        "accuracy": accuracy,
        "precision": precision,
        "attack_recall": attack_recall,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "fpr": fpr,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    # Display results.
    for key, value in metrics.items():

        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    return metrics


# --------------------------------------------------
# 5. MAIN FUNCTION
# --------------------------------------------------

def main():

    all_metrics = []

    for domain in ["uav", "unsw"]:

        metrics = evaluate_domain(domain)

        all_metrics.append(metrics)

    results = pd.DataFrame(all_metrics)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save separately to preserve original threshold-0.5 results.
    output_path = (
        RESULTS_DIR /
        "hfl_test_metrics_selected_thresholds.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION COMPLETED")
    print("=" * 60)

    print(
        results.to_string(index=False)
    )

    print(
        f"\nResults saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
