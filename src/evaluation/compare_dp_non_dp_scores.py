
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.domain_encoder import DomainEncoder
from src.models.shared_network import SharedNetwork


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
DEVICE = torch.device("cpu")

DOMAINS = {
    "uav": 15,
    "unsw": 191
}

CHECKPOINTS = {
    "non_dp": "hfl_global_model_noniid.pt",
    "dp": "hfl_dp_global_model.pt"
}


def load_models(checkpoint_name, input_dim):

    checkpoint_path = RESULTS / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    encoder = DomainEncoder(input_dim)
    shared = SharedNetwork(32)

    encoder_key = (
        "uav_encoder"
        if input_dim == 15
        else "unsw_encoder"
    )

    encoder.load_state_dict(checkpoint[encoder_key])
    shared.load_state_dict(checkpoint["shared"])

    encoder.eval()
    shared.eval()

    return encoder, shared


def get_scores(domain, split, encoder, shared):

    data = np.load(
        ROOT / "data" / "processed" /
        domain / f"{split}.npz"
    )

    X = torch.tensor(data["X"], dtype=torch.float32)
    y = data["y"].astype(int)

    with torch.no_grad():

        embeddings = encoder(X)
        logits = shared(embeddings).squeeze(1)
        probabilities = torch.sigmoid(logits).numpy()

    return y, probabilities


def summarize_scores(model_name, domain, split, y, scores):

    rows = []

    for label, class_name in [(0, "normal"), (1, "attack")]:

        values = scores[y == label]

        rows.append({
            "model": model_name,
            "domain": domain,
            "split": split,
            "class": class_name,
            "count": len(values),
            "mean": np.mean(values),
            "median": np.median(values),
            "p10": np.percentile(values, 10),
            "p90": np.percentile(values, 90)
        })

    return rows


def main():

    all_rows = []

    for model_name, checkpoint_name in CHECKPOINTS.items():

        print("\n" + "=" * 70)
        print(f"MODEL: {model_name.upper()}")
        print("=" * 70)

        for domain, input_dim in DOMAINS.items():

            encoder, shared = load_models(
                checkpoint_name,
                input_dim
            )

            for split in ["validation", "test"]:

                y, scores = get_scores(
                    domain, split, encoder, shared
                )

                rows = summarize_scores(
                    model_name, domain, split, y, scores
                )

                all_rows.extend(rows)

                for row in rows:

                    print(
                        f"{row['model']} | "
                        f"{row['domain']} | "
                        f"{row['split']} | "
                        f"{row['class']} | "
                        f"mean={row['mean']:.4f} | "
                        f"median={row['median']:.4f} | "
                        f"p10={row['p10']:.4f} | "
                        f"p90={row['p90']:.4f}"
                    )

    results = pd.DataFrame(all_rows)

    output_path = (
        RESULTS / "dp_vs_non_dp_score_distribution.csv"
    )

    results.to_csv(output_path, index=False)

    print("\nComparison saved to:")
    print(output_path)


if __name__ == "__main__":
    main()
