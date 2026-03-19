from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data


def build_graph(row: np.ndarray, label: int) -> Data:
    """
    Build a 5-node bridge SHM graph for a single sample.

    Args:
        row: numpy array of 25 features (5 features per node × 5 nodes).
        label: integer class label (0=healthy, 1=alert).

    Node definitions (each node gets 5 features):
        Node 0 (Structural):    indices 0-4
        Node 1 (Dynamic):       indices 5-9
        Node 2 (Load):          indices 10-14
        Node 3 (Environmental): indices 15-19
        Node 4 (Health):        indices 20-24
    """
    # 1) Validate inputs early to catch shape/type issues.
    row = np.asarray(row)
    if row.shape[0] != 25:
        raise ValueError(f"Expected row with 25 features, got shape {row.shape}.")
    if int(label) not in (0, 1):
        raise ValueError("label must be 0 or 1.")

    # 2) Slice the 25-D feature vector into 5 nodes × 5 features each.
    #    This produces a (5, 5) node feature matrix.
    node_features = np.stack(
        [
            row[0:5],   # Structural
            row[5:10],  # Dynamic
            row[10:15],  # Load
            row[15:20],  # Environmental
            row[20:25],  # Health
        ],
        axis=0,
    )

    # 3) Create node feature tensor x of shape (5, 5).
    x = torch.tensor(node_features, dtype=torch.float32)

    # 4) Define the directed edges, then include reverse edges to make them bidirectional.
    #    Physical causality / coupling edges:
    #      Environmental ↔ Structural  (3↔0)
    #      Load ↔ Structural           (2↔0)
    #      Load ↔ Dynamic              (2↔1)
    #      Structural ↔ Health         (0↔4)
    #      Dynamic ↔ Health            (1↔4)
    #      Environmental ↔ Dynamic     (3↔1)
    edges: Tuple[Tuple[int, int], ...] = (
        (3, 0),
        (0, 3),
        (2, 0),
        (0, 2),
        (2, 1),
        (1, 2),
        (0, 4),
        (4, 0),
        (1, 4),
        (4, 1),
        (3, 1),
        (1, 3),
    )

    # 5) Create edge_index tensor of shape (2, num_edges).
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # 6) Create label tensor y from the provided label.
    y = torch.tensor([int(label)], dtype=torch.long)

    # 7) Return a torch_geometric Data object.
    return Data(x=x, edge_index=edge_index, y=y)


def visualize_graph() -> None:
    """
    Visualize the fixed 5-node bridge graph and save it as outputs/bridge_graph.png.
    """
    # 1) Define node labels and colors in the requested order.
    node_labels = {
        0: "Structural",
        1: "Dynamic",
        2: "Load",
        3: "Environmental",
        4: "Health",
    }
    node_colors = {
        0: "blue",    # structural
        1: "orange",  # dynamic
        2: "green",   # load
        3: "cyan",    # environmental
        4: "red",     # health
    }

    # 2) Build the graph structure (undirected for visualization clarity).
    G = nx.Graph()
    G.add_nodes_from(node_labels.keys())
    G.add_edges_from(
        [
            (3, 0),  # Environmental - Structural
            (2, 0),  # Load - Structural
            (2, 1),  # Load - Dynamic
            (0, 4),  # Structural - Health
            (1, 4),  # Dynamic - Health
            (3, 1),  # Environmental - Dynamic
        ]
    )

    # 3) Choose a layout so the plot is stable and readable.
    pos = nx.spring_layout(G, seed=42)

    # 4) Draw nodes, edges, and labels.
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.8)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=[node_colors[n] for n in G.nodes()],
        node_size=1400,
        linewidths=1.5,
        edgecolors="black",
    )
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color="white")
    plt.axis("off")
    plt.tight_layout()

    # 5) Save the figure to outputs/bridge_graph.png (create outputs/ if needed).
    out_path = Path("outputs") / "bridge_graph.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
