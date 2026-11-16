
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

CHECKPOINT_PATH = ROOT / "experiments/results/hfl_global_model.pt"
RESULTS_DIR = ROOT / "experiments/results"

# Test a range of decision thresholds.
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)


def get_validation_probabilities(domain, checkpoint):
    """Load validation data and generate attack probabilities."""

    data_path = ROOT / f"data/processed/{domain}/validation.npz"
    data = np.load(data_path, allow_pickle=True)

    X_val = torch.tensor(
        data["X"].astype(np.float32),
        dtype=torch.float32
    )
    y_val = data["y"].astype(np.int64)

    # Load the correct domain-specific encoder.
    if domain == "uav":
        encoder_state = checkpoint["uav_encoder"]
    else:
        encoder_state = checkpoint["unsw_encoder"]

    encoder = DomainEncoder(input_dim=X_val.shape[1])
    shared_model = SharedNetwork()

    encoder.load_state_dict(encoder_state)
    shared_model.load_state_dict(checkpoint["shared"])

    encoder.to(DEVICE)
    shared_model.to(DEVICE)

    encoder.eval()
    shared_model.eval()

    with torch.no_grad():
        embeddings = encoder(X_val.to(DEVICE))
        logits = shared_model(embeddings).squeeze(1)
        probabilities = torch.sigmoid(logits).cpu().numpy()

    return y_val, probabilities


def calculate_metrics(y_true, probabilities, threshold):
    """Calculate classification metrics at a given threshold."""

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "attack_recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "macro_f1": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        ),
        "mcc": matthews_corrcoef(y_true, predictions),
        "fpr": fpr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_domain(domain, checkpoint):
    print("\n" + "=" * 65)
    print(f"THRESHOLD ANALYSIS: {domain.upper()}")
    print("=" * 65)

    y_val, probabilities = get_validation_probabilities(
        domain, checkpoint
    )

    results = []

    for threshold in THRESHOLDS:
        metrics = calculate_metrics(
            y_val,
            probabilities,
            threshold
        )
        results.append(metrics)

    results_df = pd.DataFrame(results)

    # Best threshold according to Macro-F1.
    best_f1 = results_df.loc[
        results_df["macro_f1"].idxmax()
    ]

    # Best threshold according to MCC.
    best_mcc = results_df.loc[
        results_df["mcc"].idxmax()
    ]

    # Practical operating point:
    # maintain at least 90% attack recall and FPR <= 10%.
    acceptable = results_df[
        (results_df["attack_recall"] >= 0.90) &
        (results_df["fpr"] <= 0.10)
    ]

    print("\nMetrics across thresholds:")
    print(
        results_df.round(4).to_string(index=False)
    )

    print("\nBest threshold by Macro-F1:")
    print(best_f1.round(4).to_string())

    print("\nBest threshold by MCC:")
    print(best_mcc.round(4).to_string())

    if not acceptable.empty:
        best_practical = acceptable.loc[
            acceptable["macro_f1"].idxmax()
        ]

        print(
            "\nBest practical threshold "
            "(recall >= 90%, FPR <= 10%):"
        )
        print(best_practical.round(4).to_string())
    else:
        best_practical = None
        print(
            "\nNo threshold satisfies both "
            "recall >= 90% and FPR <= 10%."
        )

    # Save every threshold result.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        RESULTS_DIR / f"{domain}_validation_thresholds.csv"
    )
    results_df.to_csv(output_path, index=False)

    return {
        "domain": domain.upper(),
        "best_f1_threshold": float(best_f1["threshold"]),
        "best_f1": float(best_f1["macro_f1"]),
        "best_mcc_threshold": float(best_mcc["threshold"]),
        "best_mcc": float(best_mcc["mcc"]),
        "practical_threshold": (
            float(best_practical["threshold"])
            if best_practical is not None else np.nan
        ),
        "practical_macro_f1": (
            float(best_practical["macro_f1"])
            if best_practical is not None else np.nan
        ),
    }


def main():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    recommendations = []

    for domain in ["uav", "unsw"]:
        result = evaluate_domain(domain, checkpoint)
        recommendations.append(result)

    summary = pd.DataFrame(recommendations)

    summary_path = (
        RESULTS_DIR / "threshold_recommendations.csv"
    )
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("THRESHOLD ANALYSIS COMPLETED")
    print("=" * 65)

    print(summary.round(4).to_string(index=False))
    print(f"\nRecommendations saved to: {summary_path}")


if __name__ == "__main__":
    main()
