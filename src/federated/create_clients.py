
from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
CLIENTS_OUT = PROCESSED / "clients"

N_CLIENTS = 5
RANDOM_STATE = 42

# Lower alpha creates stronger non-IID label skew.
DIRICHLET_ALPHA = 0.3

DOMAINS = ["uav", "unsw"]


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

def load_training_data(domain):

    path = PROCESSED / domain / "train.npz"

    data = np.load(path)

    X = data["X"]
    y = data["y"]
    original_labels = data["original_labels"]

    return X, y, original_labels


# ============================================================
# 3. IID PARTITIONING
# ============================================================

def create_iid_partition(X, y, original_labels, n_clients, rng):

    # Randomly shuffle all training record indices.
    indices = rng.permutation(len(X))

    # Divide indices into approximately equal-sized groups.
    client_indices = np.array_split(indices, n_clients)

    return client_indices


# ============================================================
# 4. NON-IID PARTITIONING
# ============================================================

def create_noniid_partition(
    X,
    y,
    original_labels,
    n_clients,
    alpha,
    rng
):

    client_indices = [[] for _ in range(n_clients)]

    # Group records by their original traffic category.
    unique_labels = np.unique(original_labels)

    for label in unique_labels:

        # Find indices belonging to this original category.
        label_indices = np.where(original_labels == label)[0]

        # Shuffle records within this category.
        rng.shuffle(label_indices)

        # Draw client proportions from a Dirichlet distribution.
        proportions = rng.dirichlet(
            np.repeat(alpha, n_clients)
        )

        # Convert proportions into integer record counts.
        counts = rng.multinomial(
            len(label_indices),
            proportions
        )

        # Assign each category's records to clients.
        start = 0

        for client_id, count in enumerate(counts):

            end = start + count

            assigned_indices = label_indices[start:end]

            client_indices[client_id].extend(
                assigned_indices.tolist()
            )

            start = end

    # Shuffle each client's final collection.
    for indices in client_indices:
        rng.shuffle(indices)

    return [
        np.asarray(indices, dtype=np.int64)
        for indices in client_indices
    ]


# ============================================================
# 5. SAVE CLIENT DATA
# ============================================================

def save_clients(
    domain,
    X,
    y,
    original_labels,
    client_indices,
    partition_type
):

    output_dir = CLIENTS_OUT / partition_type / domain
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "domain": domain,
        "partition_type": partition_type,
        "number_of_clients": len(client_indices),
        "clients": []
    }

    all_indices = []

    for client_id, indices in enumerate(client_indices):

        client_number = client_id + 1

        client_X = X[indices]
        client_y = y[indices]
        client_original_labels = original_labels[indices]

        # Save this client's private local training data.
        np.savez_compressed(
            output_dir / f"client_{client_number}.npz",
            X=client_X.astype(np.float32),
            y=client_y.astype(np.int64),
            original_labels=client_original_labels
        )

        all_indices.extend(indices.tolist())

        # Original category distribution.
        category_counts = (
            pd.Series(client_original_labels)
            .value_counts()
            .to_dict()
        )

        # Binary target distribution.
        binary_counts = {
            str(int(label)): int(count)
            for label, count in zip(
                *np.unique(client_y, return_counts=True)
            )
        }

        client_info = {
            "client_id": client_number,
            "samples": int(len(indices)),
            "features": int(client_X.shape[1]),
            "binary_class_counts": binary_counts,
            "original_category_counts": {
                str(k): int(v)
                for k, v in category_counts.items()
            }
        }

        metadata["clients"].append(client_info)

        print(
            f"Client {client_number}: "
            f"samples={len(indices)}, "
            f"features={client_X.shape[1]}, "
            f"binary_classes={binary_counts}"
        )

    # Check every training record is assigned exactly once.
    all_indices = np.asarray(all_indices)

    assert len(all_indices) == len(X), (
        "Some training records are missing or duplicated."
    )

    assert len(np.unique(all_indices)) == len(X), (
        "A training record was assigned to multiple clients."
    )

    assert np.array_equal(
        np.sort(all_indices),
        np.arange(len(X))
    ), "Client partitions do not cover the complete training set."

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("Partition coverage: PASS")
    print("Saved to:", output_dir)


# ============================================================
# 6. CREATE CLIENTS FOR EACH DOMAIN
# ============================================================

def process_domain(domain):

    print("\n" + "=" * 60)
    print(f"CREATING CLIENTS: {domain.upper()}")
    print("=" * 60)

    X, y, original_labels = load_training_data(domain)

    print("Training data shape:", X.shape)
    print("Number of clients:", N_CLIENTS)

    # IID partition.
    iid_rng = np.random.default_rng(RANDOM_STATE)

    iid_indices = create_iid_partition(
        X, y, original_labels, N_CLIENTS, iid_rng
    )

    print("\n--- IID PARTITION ---")

    save_clients(
        domain,
        X,
        y,
        original_labels,
        iid_indices,
        "iid"
    )

    # Non-IID partition.
    noniid_rng = np.random.default_rng(RANDOM_STATE)

    noniid_indices = create_noniid_partition(
        X,
        y,
        original_labels,
        N_CLIENTS,
        DIRICHLET_ALPHA,
        noniid_rng
    )

    print("\n--- NON-IID PARTITION ---")

    save_clients(
        domain,
        X,
        y,
        original_labels,
        noniid_indices,
        "noniid"
    )


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":

    for domain in DOMAINS:
        process_domain(domain)

    print("\n" + "=" * 60)
    print("ALL FEDERATED CLIENTS CREATED SUCCESSFULLY")
    print("=" * 60)
