
import torch

from domain_encoder import DomainEncoder
from shared_network import SharedNetwork


# Use CPU for this experiment.
device = torch.device("cpu")

# Create domain-specific encoders.
uav_encoder = DomainEncoder(input_dim=15).to(device)
unsw_encoder = DomainEncoder(input_dim=191).to(device)

# Create one shared classifier.
shared_network = SharedNetwork(embedding_dim=32).to(device)

# Simulate a batch of 64 UAV records.
uav_input = torch.randn(64, 15).to(device)

# Simulate a batch of 64 UNSW records.
unsw_input = torch.randn(64, 191).to(device)

# Pass data through the corresponding encoders.
uav_embedding = uav_encoder(uav_input)
unsw_embedding = unsw_encoder(unsw_input)

# Both embeddings go through the SAME shared network.
uav_logits = shared_network(uav_embedding)
unsw_logits = shared_network(unsw_embedding)

print("UAV input shape:", uav_input.shape)
print("UAV embedding shape:", uav_embedding.shape)
print("UAV output shape:", uav_logits.shape)

print("\nUNSW input shape:", unsw_input.shape)
print("UNSW embedding shape:", unsw_embedding.shape)
print("UNSW output shape:", unsw_logits.shape)

# Verify expected dimensions.
assert uav_embedding.shape == (64, 32)
assert unsw_embedding.shape == (64, 32)

assert uav_logits.shape == (64, 1)
assert unsw_logits.shape == (64, 1)

print("\nAll model shape checks passed!")
