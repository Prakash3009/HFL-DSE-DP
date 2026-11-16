
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.domain_encoder import DomainEncoder
from src.models.shared_network import SharedNetwork


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
DEVICE = torch.device("cpu")

DOMAINS = {"uav": 15, "unsw": 191}


def load_models():

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


def analyze(domain, split, models):

    uav_encoder, unsw_encoder, shared = models

    data = np.load(
        ROOT / "data" / "processed" /
        domain / f"{split}.npz"
    )

    X = torch.tensor(data["X"], dtype=torch.float32)
    y = data["y"].astype(int)

    encoder = uav_encoder if domain == "uav" else unsw_encoder

    with torch.no_grad():
        embeddings = encoder(X)
        logits = shared(embeddings).squeeze(1)
        probabilities = torch.sigmoid(logits).numpy()

    for label, class_name in [(0, "normal"), (1, "attack")]:

        scores = probabilities[y == label]

        print(
            f"{domain.upper()} | {split} | {class_name}: "
            f"count={len(scores)}, "
            f"mean={scores.mean():.4f}, "
            f"median={np.median(scores):.4f}, "
            f"p10={np.percentile(scores, 10):.4f}, "
            f"p90={np.percentile(scores, 90):.4f}"
        )


def main():

    models = load_models()

    for domain in DOMAINS:

        print("\n" + "=" * 65)
        print(f"{domain.upper()} DP SCORE DISTRIBUTION")
        print("=" * 65)

        for split in ["validation", "test"]:
            analyze(domain, split, models)


if __name__ == "__main__":
    main()
