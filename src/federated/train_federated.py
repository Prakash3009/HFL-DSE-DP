
from pathlib import Path
import sys
import copy
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 1. IMPORT MODEL ARCHITECTURE
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src" / "models"))
from src.models.domain_encoder import DomainEncoder
from src.models.shared_network import SharedNetwork

# ============================================================
# 2. CONFIGURATION
# ============================================================

DEVICE = torch.device("cpu")

N_CLIENTS = 5
ROUNDS = 2
LOCAL_EPOCHS = 1

BATCH_SIZE = 256
LEARNING_RATE = 0.001

PARTITION = "noniid"

CLIENTS_DIR = (
    ROOT / "data" / "processed" / "clients" / PARTITION
)

PROCESSED_DIR = ROOT / "data" / "processed"

RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. INITIALIZE GLOBAL MODELS
# ============================================================

def initialize_global_models():

    global_models = {
        "uav_encoder": DomainEncoder(input_dim=15).to(DEVICE),
        "unsw_encoder": DomainEncoder(input_dim=191).to(DEVICE),
        "shared": SharedNetwork(embedding_dim=32).to(DEVICE)
    }

    return global_models


# ============================================================
# 4. LOAD CLIENT TRAINING DATA
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
# 5. LOCAL CLIENT TRAINING
# ============================================================

def train_local_client(domain, client_id, global_models):

    # Load this client's private local data.
    X, y, loader = load_client_data(domain, client_id)

    # Select the correct domain-specific encoder.
    encoder_name = f"{domain}_encoder"

    # Create local copies of the current global models.
    local_encoder = copy.deepcopy(
        global_models[encoder_name]
    )

    local_shared = copy.deepcopy(
        global_models["shared"]
    )

    local_encoder.train()
    local_shared.train()

    # Both encoder and shared network are trained locally.
    optimizer = torch.optim.Adam(
        list(local_encoder.parameters())
        + list(local_shared.parameters()),
        lr=LEARNING_RATE
    )

    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_samples = 0

    for epoch in range(LOCAL_EPOCHS):

        for batch_X, batch_y in loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            # Forward propagation.
            embeddings = local_encoder(batch_X)

            logits = local_shared(embeddings).squeeze(1)

            # Calculate local classification loss.
            loss = criterion(logits, batch_y)

            # Backpropagation.
            loss.backward()

            # Local parameter update.
            optimizer.step()

            total_loss += loss.item() * len(batch_X)
            total_samples += len(batch_X)

    average_loss = total_loss / total_samples

    # Return updated model parameters, not raw training data.
    return {
        "domain": domain,
        "client_id": client_id,
        "samples": len(X),
        "encoder_state": copy.deepcopy(
            local_encoder.state_dict()
        ),
        "shared_state": copy.deepcopy(
            local_shared.state_dict()
        ),
        "average_loss": average_loss
    }


# ============================================================
# 6. SAMPLE-WEIGHTED FEDAVG
# ============================================================

def fedavg(state_dicts, sample_counts):

    total_samples = sum(sample_counts)

    aggregated_state = {}

    for key in state_dicts[0].keys():

        # Weighted sum of client parameters.
        aggregated_parameter = torch.zeros_like(
            state_dicts[0][key]
        )

        for client_state, count in zip(
            state_dicts,
            sample_counts
        ):

            weight = count / total_samples

            aggregated_parameter += (
                client_state[key] * weight
            )

        aggregated_state[key] = aggregated_parameter

    return aggregated_state


# ============================================================
# 7. SERVER AGGREGATION
# ============================================================

def aggregate_client_updates(global_models, updates):

    # Aggregate UAV encoders among UAV clients only.
    uav_updates = [
        update for update in updates
        if update["domain"] == "uav"
    ]

    uav_encoder_state = fedavg(
        [u["encoder_state"] for u in uav_updates],
        [u["samples"] for u in uav_updates]
    )

    global_models["uav_encoder"].load_state_dict(
        uav_encoder_state
    )

    # Aggregate UNSW encoders among UNSW clients only.
    unsw_updates = [
        update for update in updates
        if update["domain"] == "unsw"
    ]

    unsw_encoder_state = fedavg(
        [u["encoder_state"] for u in unsw_updates],
        [u["samples"] for u in unsw_updates]
    )

    global_models["unsw_encoder"].load_state_dict(
        unsw_encoder_state
    )

    # Aggregate the shared network across ALL clients.
    shared_state = fedavg(
        [u["shared_state"] for u in updates],
        [u["samples"] for u in updates]
    )

    global_models["shared"].load_state_dict(
        shared_state
    )


# ============================================================
# 8. VALIDATION
# ============================================================

def evaluate_domain(domain, global_models):

    path = PROCESSED_DIR / domain / "validation.npz"

    data = np.load(path)

    X = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.float32)

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=1024,
        shuffle=False
    )

    encoder = global_models[f"{domain}_encoder"]
    shared = global_models["shared"]

    encoder.eval()
    shared.eval()

    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch_X, batch_y in loader:

            logits = shared(
                encoder(batch_X.to(DEVICE))
            ).squeeze(1)

            batch_y = batch_y.to(DEVICE)

            loss = criterion(logits, batch_y)

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= 0.5
            ).float()

            total_loss += loss.item() * len(batch_y)

            correct += (
                predictions == batch_y
            ).sum().item()

            total += len(batch_y)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total
    }


# ============================================================
# 9. FEDERATED TRAINING
# ============================================================

def main():

    torch.manual_seed(42)
    np.random.seed(42)

    global_models = initialize_global_models()

    history = []

    print("\nFEDERATED TRAINING STARTED")
    print("Partition:", PARTITION)
    print("Clients per domain:", N_CLIENTS)
    print("Global rounds:", ROUNDS)

    for round_number in range(1, ROUNDS + 1):

        print("\n" + "=" * 60)
        print(f"GLOBAL ROUND {round_number}")
        print("=" * 60)

        updates = []

        # All five UAV clients train locally.
        for client_id in range(1, N_CLIENTS + 1):

            update = train_local_client(
                "uav",
                client_id,
                global_models
            )

            updates.append(update)

            print(
                f"UAV Client {client_id}: "
                f"samples={update['samples']}, "
                f"loss={update['average_loss']:.4f}"
            )

        # All five UNSW clients train locally.
        for client_id in range(1, N_CLIENTS + 1):

            update = train_local_client(
                "unsw",
                client_id,
                global_models
            )

            updates.append(update)

            print(
                f"UNSW Client {client_id}: "
                f"samples={update['samples']}, "
                f"loss={update['average_loss']:.4f}"
            )

        # Server aggregates all local updates.
        aggregate_client_updates(
            global_models,
            updates
        )

        print("\nSERVER AGGREGATION COMPLETED")

        # Validate each domain using its global encoder
        # and the common global shared network.
        for domain in ["uav", "unsw"]:

            metrics = evaluate_domain(
                domain,
                global_models
            )

            print(
                f"{domain.upper()} validation: "
                f"loss={metrics['loss']:.4f}, "
                f"accuracy={metrics['accuracy']:.4f}"
            )

            history.append({
                "round": round_number,
                "domain": domain,
                "validation_loss": metrics["loss"],
                "validation_accuracy": metrics["accuracy"]
            })

    # Save global models.
    torch.save(
        {
            "uav_encoder": global_models["uav_encoder"].state_dict(),
            "unsw_encoder": global_models["unsw_encoder"].state_dict(),
            "shared": global_models["shared"].state_dict()
        },
        RESULTS_DIR / "hfl_global_model.pt"
    )

    # Save validation history.
    import pandas as pd

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        RESULTS_DIR / "hfl_training_history.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("FEDERATED TRAINING COMPLETED")
    print("=" * 60)

    print("\nSaved global model:")
    print(RESULTS_DIR / "hfl_global_model.pt")

    print("\nSaved training history:")
    print(RESULTS_DIR / "hfl_training_history.csv")


if __name__ == "__main__":
    main()
