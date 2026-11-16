
import torch
import torch.nn as nn


class DomainEncoder(nn.Module):

    def __init__(self, input_dim, embedding_dim=32):

        super().__init__()

        self.encoder = nn.Sequential(

            # Layer 1: original features -> 64
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            # Layer 2: 64 -> common embedding
            nn.Linear(64, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):

        return self.encoder(x)
