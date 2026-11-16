
from pathlib import Path
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from opacus import PrivacyEngine
from opacus.accountants import PRVAccountant

from src.models.domain_encoder import DomainEncoder
from src.models.shared_network import SharedNetwork


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DEVICE = torch.device("cpu")

N_CLIENTS = 5
ROUNDS = 2
LOCAL_EPOCHS = 1

BATCH_SIZE = 256
LEARNING_RATE = 0.001

PARTITION = "noniid"

NOISE_MULTIPLIER = 1.5
MAX_GRAD_NORM = 1.0
DELTA = 1e-5

CLIENTS_DIR = (
    ROOT / "data" / "processed" / "clients" / PARTITION
)

PROCESSED_DIR = ROOT / "data" / "processed"

RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. INITIALIZE GLOBAL MODELS
# ============================================================

def initialize_global_models():

    return {
        "uav_encoder": DomainEncoder(input_dim=15).to(DEVICE),
        "unsw_encoder": DomainEncoder(input_dim=191).to(DEVICE),
        "shared": SharedNetwork(embedding_dim=32).to(DEVICE)
    }


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
        shuffle=True,
        drop_last=False
    )

    return X, y, loader


# ============================================================
# 4. LOCAL DIFFERENTIALLY PRIVATE TRAINING
# ============================================================

def train_local_client(
    domain,
    client_id,
    global_models,
    client_accountants
):

    # Load private client data.
    X, y, loader = load_client_data(domain, client_id)

    encoder_name = f"{domain}_encoder"

    # Copy the latest global models.
    local_encoder = copy.deepcopy(
        global_models[encoder_name]
    )

    local_shared = copy.deepcopy(
        global_models["shared"]
    )

    # Combine encoder and shared classifier.
    local_model = nn.Sequential(
        local_encoder,
        local_shared
    ).to(DEVICE)

    local_model.train()

    # Keep references to the original model components.
    # Opacus wraps local_model but trains the same underlying
    # model, so these references contain updated parameters.
    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.BCEWithLogitsLoss()

    # Attach Opacus DP-SGD.
    privacy_engine = PrivacyEngine(
        secure_mode=False
    )

    (
        private_model,
        private_optimizer,
        private_loader
    ) = privacy_engine.make_private(
        module=local_model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=NOISE_MULTIPLIER,
        max_grad_norm=MAX_GRAD_NORM,
        poisson_sampling=True
    )

    total_loss = 0.0
    total_steps = 0

    # Local DP training.
    for epoch in range(LOCAL_EPOCHS):

        for batch_X, batch_y in private_loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            private_optimizer.zero_grad()

            logits = private_model(
                batch_X
            ).squeeze(1)

            loss = criterion(
                logits,
                batch_y
            )

            loss.backward()

            private_optimizer.step()

            total_loss += loss.item()
            total_steps += 1

    # --------------------------------------------------------
    # COMPOSE PRIVACY ACCOUNTING ACROSS ROUNDS
    # --------------------------------------------------------

    round_accountant = privacy_engine.accountant

    persistent_accountant = client_accountants[
        (domain, client_id)
    ]

    # Opacus stores compressed accounting events as:
    # (noise_multiplier, sample_rate, number_of_steps)
    for event in round_accountant.history:

        event_noise, event_rate, event_steps = event

        for _ in range(event_steps):

            persistent_accountant.step(
                noise_multiplier=event_noise,
                sample_rate=event_rate
            )

    # Calculate cumulative epsilon for this client.
    epsilon = persistent_accountant.get_epsilon(
        delta=DELTA
    )

    # --------------------------------------------------------
    # EXTRACT UPDATED MODEL PARAMETERS
    # --------------------------------------------------------

    # IMPORTANT:
    # Use local_model, not private_model[0].
    # private_model is an Opacus GradSampleModule wrapper.
    # local_model contains the original Sequential model.
    #
    # Both references point to the same trained parameters.

    return {
        "domain": domain,
        "client_id": client_id,
        "samples": len(X),

        "encoder_state": copy.deepcopy(
            local_model[0].state_dict()
        ),

        "shared_state": copy.deepcopy(
            local_model[1].state_dict()
        ),

        "average_loss": (
            total_loss / max(total_steps, 1)
        ),

        "epsilon": epsilon,
        "steps": total_steps
    }


# ============================================================
# 5. SAMPLE-WEIGHTED FEDAVG
# ============================================================

def fedavg(state_dicts, sample_counts):

    total_samples = sum(sample_counts)

    aggregated_state = {}

    for key in state_dicts[0].keys():

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
# 6. SERVER AGGREGATION
# ============================================================

def aggregate_client_updates(global_models, updates):

    # UAV encoders: aggregate UAV clients only.
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

    # UNSW encoders: aggregate UNSW clients only.
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

    # Shared classifier: aggregate all ten clients.
    shared_state = fedavg(
        [u["shared_state"] for u in updates],
        [u["samples"] for u in updates]
    )

    global_models["shared"].load_state_dict(
        shared_state
    )


# ============================================================
# 7. VALIDATION
# ============================================================

def evaluate_domain(domain, global_models):

    path = (
        PROCESSED_DIR
        / domain
        / "validation.npz"
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

    loader = DataLoader(
        TensorDataset(X, y),
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

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            logits = shared(
                encoder(batch_X)
            ).squeeze(1)

            loss = criterion(
                logits,
                batch_y
            )

            predictions = (
                torch.sigmoid(logits) >= 0.5
            ).float()

            total_loss += (
                loss.item() * len(batch_y)
            )

            correct += (
                predictions == batch_y
            ).sum().item()

            total += len(batch_y)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total
    }


# ============================================================
# 8. FEDERATED DP TRAINING
# ============================================================

def main():

    torch.manual_seed(42)
    np.random.seed(42)

    global_models = initialize_global_models()

    # Create one persistent RDP accountant per client.
    # Create one persistent PRV accountant per client.
    client_accountants = {
    (domain, client_id): PRVAccountant()
    for domain in ["uav", "unsw"]
    for client_id in range(1, N_CLIENTS + 1)
    }

    history = []
    privacy_history = []

    print("\nDP FEDERATED TRAINING STARTED")

    print("Partition:", PARTITION)
    print("Clients per domain:", N_CLIENTS)
    print("Global rounds:", ROUNDS)
    print("Local epochs:", LOCAL_EPOCHS)

    print("Noise multiplier:", NOISE_MULTIPLIER)
    print("Maximum gradient norm:", MAX_GRAD_NORM)
    print("Delta:", DELTA)

    # --------------------------------------------------------
    # GLOBAL COMMUNICATION ROUNDS
    # --------------------------------------------------------

    for round_number in range(1, ROUNDS + 1):

        print("\n" + "=" * 60)
        print(f"GLOBAL ROUND {round_number}")
        print("=" * 60)

        updates = []

        # Train all UAV and UNSW clients.
        for domain in ["uav", "unsw"]:

            for client_id in range(1, N_CLIENTS + 1):

                update = train_local_client(
                    domain,
                    client_id,
                    global_models,
                    client_accountants
                )

                updates.append(update)

                privacy_history.append({
                    "round": round_number,
                    "domain": domain,
                    "client_id": client_id,
                    "epsilon": update["epsilon"],
                    "delta": DELTA,
                    "steps_this_round": update["steps"],
                    "noise_multiplier": NOISE_MULTIPLIER,
                    "max_grad_norm": MAX_GRAD_NORM
                })

                print(
                    f"{domain.upper()} Client {client_id}: "
                    f"samples={update['samples']}, "
                    f"loss={update['average_loss']:.4f}, "
                    f"epsilon={update['epsilon']:.4f}, "
                    f"steps={update['steps']}"
                )

        # Server aggregates DP-trained model updates.
        aggregate_client_updates(
            global_models,
            updates
        )

        print("\nSERVER AGGREGATION COMPLETED")

        # Validate both domains.
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

    # ========================================================
    # 9. FINAL PRIVACY ACCOUNTING
    # ========================================================

    final_privacy = []

    for (domain, client_id), accountant in client_accountants.items():

        epsilon = accountant.get_epsilon(
            delta=DELTA
        )

        final_privacy.append({
            "domain": domain,
            "client_id": client_id,
            "epsilon": epsilon,
            "delta": DELTA
        })

    privacy_df = pd.DataFrame(final_privacy)

    maximum_epsilon = privacy_df["epsilon"].max()

    # ========================================================
    # 10. SAVE DP MODEL AND RESULTS
    # ========================================================

    torch.save(
        {
            "uav_encoder": global_models["uav_encoder"].state_dict(),
            "unsw_encoder": global_models["unsw_encoder"].state_dict(),
            "shared": global_models["shared"].state_dict(),
            "partition": PARTITION,
            "rounds": ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "noise_multiplier": NOISE_MULTIPLIER,
            "max_grad_norm": MAX_GRAD_NORM,
            "delta": DELTA,
            "max_client_epsilon": float(maximum_epsilon)
        },
        RESULTS_DIR / "hfl_dp_global_model.pt"
    )

    pd.DataFrame(history).to_csv(
        RESULTS_DIR / "hfl_dp_training_history.csv",
        index=False
    )

    pd.DataFrame(privacy_history).to_csv(
        RESULTS_DIR / "hfl_dp_privacy_history.csv",
        index=False
    )

    privacy_df.to_csv(
        RESULTS_DIR / "hfl_dp_final_privacy.csv",
        index=False
    )

    # ========================================================
    # 11. DISPLAY FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("DP FEDERATED TRAINING COMPLETED")
    print("=" * 60)

    print("\nFINAL CLIENT PRIVACY RESULTS")

    print(
        privacy_df.to_string(index=False)
    )

    print(
        "\nMaximum client epsilon:",
        f"{maximum_epsilon:.4f}"
    )

    print("\nSaved files:")

    print(
        RESULTS_DIR / "hfl_dp_global_model.pt"
    )

    print(
        RESULTS_DIR / "hfl_dp_training_history.csv"
    )

    print(
        RESULTS_DIR / "hfl_dp_privacy_history.csv"
    )

    print(
        RESULTS_DIR / "hfl_dp_final_privacy.csv"
    )


if __name__ == "__main__":
    main()
