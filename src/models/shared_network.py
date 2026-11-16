
import torch
import torch.nn as nn


class SharedNetwork(nn.Module):

    def __init__(self, embedding_dim=32):

        super().__init__()

        self.classifier = nn.Sequential(

            # 32 -> 32
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),

            nn.Dropout(0.2),

            # 32 -> 16
            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Dropout(0.2),

            # 16 -> 1
            nn.Linear(16, 1)
        )

    def forward(self, x):

        return self.classifier(x)
