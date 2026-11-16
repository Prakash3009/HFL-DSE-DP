
import numpy as np
from pathlib import Path
import math

clients_dir = Path("data/processed/clients/noniid")
batch_size = 256

for domain in ["uav", "unsw"]:
    print(f"\n{domain.upper()} CLIENT SAMPLING RATES")

    for client_id in range(1, 6):
        path = (
            clients_dir
            / domain
            / f"client_{client_id}.npz"
        )

        data = np.load(path)

        n = len(data["X"])

        # Opacus uses sample_rate = 1 / len(original DataLoader)
        original_loader_batches = math.ceil(n / batch_size)
        sample_rate = 1 / original_loader_batches

        print(
            f"Client {client_id}: "
            f"N={n}, "
            f"loader_batches={original_loader_batches}, "
            f"q={sample_rate:.8f}"
        )
