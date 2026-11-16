
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Make model files importable.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))

from domain_encoder import DomainEncoder
from shared_network import SharedNetwork


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = torch.device("cpu")
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 1

CLIENTS_DIR = ROOT / "data" / "processed" / "clients" / "iid"


# --------------------------------------------------
# Load one real client
# --------------------------------------------------

def load_client(domain):

    path = CLIENTS_DIR / domain / "client_1.npz"

    data = np.load(path)

    X = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.float32)

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    return X, y, loader


# --------------------------------------------------
# Local training test
# --------------------------------------------------

def train_client(domain, input_dim):

    print("\n" + "=" * 55)
    print(f"LOCAL TRAINING: {domain.upper()} CLIENT 1")
    print("=" * 55)

    X, y, loader = load_client(domain)

    encoder = DomainEncoder(input_dim=input_dim).to(DEVICE)
    shared = SharedNetwork(embedding_dim=32).to(DEVICE)

    # Both encoder and shared network are locally trainable.
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(shared.parameters()),
        lr=LEARNING_RATE
    )

    criterion = nn.BCEWithLogitsLoss()

    # Record initial loss on a fixed batch.
    encoder.eval()
    shared.eval()

    with torch.no_grad():
        initial_embedding = encoder(X[:BATCH_SIZE].to(DEVICE))
        initial_logits = shared(initial_embedding)

        initial_loss = criterion(
            initial_logits.squeeze(1),
            y[:BATCH_SIZE].to(DEVICE)
        ).item()

    # Local training.
    encoder.train()
    shared.train()

    total_loss = 0.0
    total_samples = 0

    for epoch in range(EPOCHS):

        for batch_X, batch_y in loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            # Forward propagation.
            embeddings = encoder(batch_X)
            logits = shared(embeddings)

            # Calculate classification loss.
            loss = criterion(
                logits.squeeze(1),
                batch_y
            )

            # Backpropagation.
            loss.backward()

            # Update local model parameters.
            optimizer.step()

            total_loss += loss.item() * len(batch_X)
            total_samples += len(batch_X)

    average_train_loss = total_loss / total_samples

    # Evaluate again on the same fixed batch.
    encoder.eval()
    shared.eval()

    with torch.no_grad():
        final_embedding = encoder(X[:BATCH_SIZE].to(DEVICE))
        final_logits = shared(final_embedding)

        final_loss = criterion(
            final_logits.squeeze(1),
            y[:BATCH_SIZE].to(DEVICE)
        ).item()

    print("Input shape:", tuple(X.shape))
    print("Target shape:", tuple(y.shape))
    print("Embedding shape:", tuple(final_embedding.shape))
    print("Logits shape:", tuple(final_logits.shape))

    print(f"Initial fixed-batch loss: {initial_loss:.4f}")
    print(f"Average training loss: {average_train_loss:.4f}")
    print(f"Final fixed-batch loss: {final_loss:.4f}")

    print("Local training completed successfully.")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    torch.manual_seed(42)
    np.random.seed(42)

    train_client("uav", input_dim=15)

    train_client("unsw", input_dim=191)

    print("\nBoth local training tests completed.")
