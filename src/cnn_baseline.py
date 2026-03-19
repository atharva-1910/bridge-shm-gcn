from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BridgeCNN(nn.Module):
    """
    1D CNN baseline for bridge SHM binary classification.

    Expected input:
    - x: tensor of shape (batch_size, 25) representing a flat feature vector.
    Output:
    - logits: tensor of shape (batch_size, 2) for classes (0=healthy, 1=damaged).
    """

    def __init__(self) -> None:
        super().__init__()

        # Conv block 1: extract local patterns from the 25-length feature sequence.
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # BatchNorm improves stability and convergence for the 16-channel activations.
        self.bn1 = nn.BatchNorm1d(16)

        # Conv block 2: increase representation capacity to 32 channels.
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # BatchNorm for 32-channel activations.
        self.bn2 = nn.BatchNorm1d(32)

        # Conv block 3: further increase capacity to 64 channels.
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # BatchNorm for 64-channel activations.
        self.bn3 = nn.BatchNorm1d(64)

        # Adaptive pooling: convert sequence length to a fixed length of 8.
        self.pool = nn.AdaptiveAvgPool1d(8)

        # MLP head: classify the pooled feature map into 2 classes.
        self.fc1 = nn.Linear(64 * 8, 128)  # flatten channels×length -> hidden
        self.drop1 = nn.Dropout(p=0.3)  # dropout regularization
        self.fc2 = nn.Linear(128, 64)  # second hidden layer
        self.drop2 = nn.Dropout(p=0.3)  # dropout regularization
        self.fc3 = nn.Linear(64, 2)  # final logits for 2 classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning logits of shape (batch_size, 2).
        """
        # 1) Reshape to (batch_size, 1, 25) so Conv1d can operate over the feature axis.
        x = x.view(x.size(0), 1, 25)

        # 2) Conv block 1: Conv -> BN -> ReLU.
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # 3) Conv block 2: Conv -> BN -> ReLU.
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # 4) Conv block 3: Conv -> BN -> ReLU.
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # 5) Pool to a fixed length, then flatten to a vector per sample.
        x = self.pool(x)  # (batch, 64, 8)
        x = torch.flatten(x, start_dim=1)  # (batch, 64*8)

        # 6) MLP classifier: Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear.
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop2(x)

        x = self.fc3(x)
        return x


def get_cnn_summary() -> BridgeCNN:
    """
    Create a BridgeCNN instance, print trainable parameter count, and return it.
    """
    model = BridgeCNN()

    # Count trainable parameters (parameters with requires_grad=True).
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    return model
