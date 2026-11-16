import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine

from src.models.domain_encoder import DomainEncoder
from src.models.shared_network import SharedNetwork


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DEVICE = torch.device("cpu")

INPUT_DIM = 15
BATCH_SIZE = 256
EPOCHS = 1

MAX_GRAD_NORM = 1.0
NOISE_MULTIPLIER = 1.0
DELTA = 1e-5

# --------------------------------------------------
# 2. Create sample dataset
# --------------------------------------------------

X = torch.randn(1024, INPUT_DIM)
y = torch.randint(0, 2, (1024, 1)).float()

dataset = TensorDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False
)

# --------------------------------------------------
# 3. Create complete local model
# --------------------------------------------------

encoder = DomainEncoder(input_dim=INPUT_DIM)
shared = SharedNetwork()

model = nn.Sequential(
    encoder,
    shared
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# --------------------------------------------------
# 4. Attach PrivacyEngine
# --------------------------------------------------

privacy_engine = PrivacyEngine()

model, optimizer, loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=loader,
    noise_multiplier=NOISE_MULTIPLIER,
    max_grad_norm=MAX_GRAD_NORM,
    poisson_sampling=True
)

# --------------------------------------------------
# 5. Train with DP-SGD
# --------------------------------------------------

model.train()

for epoch in range(EPOCHS):

    total_loss = 0.0

    for batch_X, batch_y in loader:

        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(batch_X)

        loss = criterion(logits, batch_y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}: "
        f"Average batch loss = {total_loss / len(loader):.4f}"
    )

# --------------------------------------------------
# 6. Calculate privacy expenditure
# --------------------------------------------------

epsilon = privacy_engine.get_epsilon(DELTA)

print("\n" + "=" * 50)
print("DP-SGD TEST COMPLETED")
print("=" * 50)

print("Noise multiplier:", NOISE_MULTIPLIER)
print("Maximum gradient norm:", MAX_GRAD_NORM)
print("Delta:", DELTA)
print(f"Calculated epsilon: {epsilon:.4f}")