
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine
from opacus.accountants import PRVAccountant

torch.manual_seed(42)

X = torch.randn(1024, 10)
y = torch.randint(0, 2, (1024,)).float()

dataset = TensorDataset(X, y)

criterion = nn.BCEWithLogitsLoss()

# Persistent accountant across rounds
persistent_accountant = PRVAccountant()


def train_one_round(round_number):

    model = nn.Sequential(
        nn.Linear(10, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01
    )

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        drop_last=False
    )

    engine = PrivacyEngine(secure_mode=False)

    private_model, private_optimizer, private_loader = (
        engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=True
        )
    )

    private_model.train()

    for batch_X, batch_y in private_loader:

        private_optimizer.zero_grad()

        logits = private_model(batch_X).squeeze(1)
        loss = criterion(logits, batch_y)

        loss.backward()
        private_optimizer.step()

    # Round-level accountant
    round_accountant = engine.accountant

    # Compose this round into persistent accountant
    for noise, sample_rate, steps in round_accountant.history:

        for _ in range(steps):

            persistent_accountant.step(
                noise_multiplier=noise,
                sample_rate=sample_rate
            )

    round_epsilon = round_accountant.get_epsilon(delta=1e-5)

    cumulative_epsilon = persistent_accountant.get_epsilon(
        delta=1e-5
    )

    print(f"\nROUND {round_number}")
    print("Round events:", round_accountant.history)
    print(f"Round epsilon: {round_epsilon:.6f}")
    print(f"Cumulative epsilon: {cumulative_epsilon:.6f}")


print("\nTWO-ROUND PRV COMPOSITION TEST")

train_one_round(1)
train_one_round(2)

print("\nFINAL ACCOUNTING")
print("Persistent history:", persistent_accountant.history)
print(
    "Final cumulative epsilon:",
    persistent_accountant.get_epsilon(delta=1e-5)
)

