from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GCNConv, global_mean_pool


class BridgeGCN(nn.Module):
    """
    GCN-based binary classifier for bridge structural health monitoring.

    Expected input:
    - x: node features of shape (num_nodes_in_batch, 5)
         (each graph has 5 nodes × 5 features per node)
    - edge_index: graph connectivity in COO format, shape (2, num_edges)
    - batch: graph assignment vector mapping each node to a graph id
    """

    def __init__(self) -> None:
        super().__init__()

        # Layer 1: Graph convolution maps 5-D node features to 32-D embeddings.
        self.conv1 = GCNConv(5, 32)
        # BatchNorm stabilizes training by normalizing node embeddings across the batch.
        self.bn1 = BatchNorm(32)
        # Dropout regularizes the model to reduce overfitting.
        self.drop1 = nn.Dropout(p=0.3)

        # Layer 2: Graph convolution increases capacity to 64-D node embeddings.
        self.conv2 = GCNConv(32, 64)
        # BatchNorm for the 64-D embeddings.
        self.bn2 = BatchNorm(64)
        # Dropout regularization.
        self.drop2 = nn.Dropout(p=0.3)

        # Layer 3: Final graph convolution refines 64-D node embeddings.
        self.conv3 = GCNConv(64, 64)

        # FC1: Graph-level MLP head compresses pooled 64-D graph vector to 32-D.
        self.fc1 = nn.Linear(64, 32)
        # Dropout regularization in the classifier head.
        self.drop_fc = nn.Dropout(p=0.3)

        # FC2: Final classifier outputs logits for 2 classes (0=healthy, 1=damaged).
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns class logits of shape (num_graphs_in_batch, 2).
        """
        # --- GCN block 1 ---
        x = self.conv1(x, edge_index)  # message passing over edges (5 -> 32)
        x = self.bn1(x)  # normalize node embeddings
        x = F.relu(x)  # non-linearity
        x = self.drop1(x)  # dropout regularization

        # --- GCN block 2 ---
        x = self.conv2(x, edge_index)  # (32 -> 64)
        x = self.bn2(x)  # normalize node embeddings
        x = F.relu(x)  # non-linearity
        x = self.drop2(x)  # dropout regularization

        # --- GCN block 3 ---
        x = self.conv3(x, edge_index)  # (64 -> 64)
        x = F.relu(x)  # final non-linearity

        # Pooling: aggregate node embeddings to a single graph embedding per sample.
        x = global_mean_pool(x, batch)  # (num_nodes -> num_graphs)

        # --- MLP classifier head ---
        x = self.fc1(x)  # (64 -> 32)
        x = F.relu(x)  # non-linearity
        x = self.drop_fc(x)  # dropout regularization
        x = self.fc2(x)  # (32 -> 2) logits

        return x


def get_model_summary() -> BridgeGCN:
    """
    Create a BridgeGCN instance, print trainable parameter count, and return it.
    """
    model = BridgeGCN()

    # Count trainable parameters (parameters with requires_grad=True).
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    return model
