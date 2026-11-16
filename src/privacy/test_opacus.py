import torch
from opacus import PrivacyEngine

from src.models.domain_encoder import DomainEncoder
from src.models.shared_network import SharedNetwork

# CPU execution
device = torch.device("cpu")

# Test UAV dimensions
input_dim = 15
batch_size = 64

encoder = DomainEncoder(input_dim=input_dim).to(device)
shared = SharedNetwork().to(device)

# Create sample input and labels
x = torch.randn(batch_size, input_dim)
y = torch.randint(0, 2, (batch_size, 1)).float()

# Forward pass
embedding = encoder(x)
logits = shared(embedding)

print("Input shape:", x.shape)
print("Embedding shape:", embedding.shape)
print("Output shape:", logits.shape)

# Check model compatibility with Opacus
model = torch.nn.Sequential(encoder, shared)

from opacus.validators import ModuleValidator

errors = ModuleValidator.validate(model, strict=False)

print("\nOpacus compatibility errors:")
print(errors)

if not errors:
    print("\nModel passed Opacus compatibility validation.")