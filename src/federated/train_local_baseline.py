
from pathlib import Path
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from src.models.domain_encoder import DomainEncoder
from src.models.shared_network import SharedNetwork


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DEVICE = torch.device("cpu")

N_CLIENTS = 5

LOCAL_EPOCHS = 2
BATCH_SIZE = 256
LEARNING_RATE = 0.001

PARTITION = "iid"

CLIENTS_DIR = (
    ROOT / "data/processed/clients" / PARTITION
)

RESULTS_DIR = ROOT / "experiments/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. MODEL INITIALIZATION
# ============================================================

def initialize_local_model(domain):

    if domain == "uav":
        input_dim = 15

    elif domain == "unsw":
        input_dim = 191

    else:
        raise ValueError(f"Unknown domain: {domain}")

    encoder = DomainEncoder(
        input_dim=input_dim
    ).to(DEVICE)

    shared_model = SharedNetwork(
        embedding_dim=32
    ).to(DEVICE)

    return encoder, shared_model


# ============================================================
# 3. LOAD CLIENT DATA
# ============================================================

def load_client_data(domain, client_id):

    path = (
        CLIENTS_DIR
        / domain
        / f"client_{client_id}.npz"
    )

    data = np.load(path)

    X = torch.tensor(
        data["X"],
        dtype=torch.float32
    )

    y = torch.tensor(
        data["y"],
        dtype=torch.float32
    )

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    return X, y, loader


# ============================================================
# 4. TRAIN ONE INDEPENDENT CLIENT
# ============================================================

def train_local_client(domain, client_id):

    X, y, loader = load_client_data(
        domain,
        client_id
    )

    # Initialize independent client models.
    encoder, shared_model = initialize_local_model(
        domain
    )

    encoder.train()
    shared_model.train()

    optimizer = torch.optim.Adam(
        list(encoder.parameters())
        + list(shared_model.parameters()),
        lr=LEARNING_RATE
    )

    criterion = nn.BCEWithLogitsLoss()

    epoch_losses = []

    # Train independently for two local epochs.
    for epoch in range(LOCAL_EPOCHS):

        total_loss = 0.0
        total_samples = 0

        for batch_X, batch_y in loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            # Forward propagation.
            embeddings = encoder(batch_X)

            logits = shared_model(
                embeddings
            ).squeeze(1)

            # Binary classification loss.
            loss = criterion(
                logits,
                batch_y
            )

            # Backpropagation.
            loss.backward()

            # Update local parameters.
            optimizer.step()

            batch_size = len(batch_X)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

        average_epoch_loss = (
            total_loss / total_samples
        )

        epoch_losses.append(average_epoch_loss)

        print(
            f"  Epoch {epoch + 1}/{LOCAL_EPOCHS}: "
            f"loss={average_epoch_loss:.4f}"
        )

    average_loss = float(
        np.mean(epoch_losses)
    )

    # Return training results and trained model weights.
    return {
        "domain": domain,
        "client_id": client_id,
        "samples": len(X),
        "average_loss": average_loss,

        "encoder_state": copy.deepcopy(
            encoder.state_dict()
        ),

        "shared_state": copy.deepcopy(
            shared_model.state_dict()
        )
    }


# ============================================================
# 5. MAIN EXPERIMENT
# ============================================================

def main():

    # Reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)

    results = []

    # Store all independently trained client models.
    local_models = {}

    print("\nLOCAL-ONLY BASELINE STARTED")
    print("Partition:", PARTITION)
    print("Clients per domain:", N_CLIENTS)
    print("Local epochs:", LOCAL_EPOCHS)

    for domain in ["uav", "unsw"]:

        print("\n" + "=" * 60)
        print(f"DOMAIN: {domain.upper()}")
        print("=" * 60)

        for client_id in range(
            1,
            N_CLIENTS + 1
        ):

            print(
                f"\nTraining {domain.upper()} "
                f"Client {client_id}"
            )

            result = train_local_client(
                domain,
                client_id
            )

            # Save this client's model weights.
            model_key = (
                f"{domain}_client_{client_id}"
            )

            local_models[model_key] = {
                "encoder": result["encoder_state"],
                "shared": result["shared_state"]
            }

            # Store client training metrics.
            results.append({
                "domain": result["domain"],
                "client_id": result["client_id"],
                "samples": result["samples"],
                "average_loss": result["average_loss"]
            })

            print(
                f"Completed Client {client_id}: "
                f"samples={result['samples']}, "
                f"average_loss={result['average_loss']:.4f}"
            )

    # ========================================================
    # 6. SAVE CLIENT TRAINING METRICS
    # ========================================================

    results_df = pd.DataFrame(results)

    metrics_path = (
        RESULTS_DIR / "local_baseline_training.csv"
    )

    results_df.to_csv(
        metrics_path,
        index=False
    )

    # ========================================================
    # 7. SAVE INDEPENDENT CLIENT MODELS
    # ========================================================

    checkpoint_path = (
        RESULTS_DIR / "local_baseline_models.pt"
    )

    torch.save(
        local_models,
        checkpoint_path
    )

    # ========================================================
    # 8. COMPLETION
    # ========================================================

    print("\n" + "=" * 60)
    print("LOCAL-ONLY BASELINE COMPLETED")
    print("=" * 60)

    print("\nClient training results:")
    print(results_df.to_string(index=False))

    print(
        f"\nTraining metrics saved to:\n{metrics_path}"
    )

    print(
        f"\nLocal client models saved to:\n{checkpoint_path}"
    )

    print(
        f"\nTotal independently trained models: "
        f"{len(local_models)}"
    )


if __name__ == "__main__":
    main()
