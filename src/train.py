from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

# Allow running this file directly via: `python3 src/train.py`
# When executed this way, Python does not treat `src/` as an installed package,
# so we add the project root (parent of `src/`) to sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset
from torch_geometric.loader import DataLoader as GeoDataLoader

from src.cnn_baseline import BridgeCNN
from src.gcn_model import BridgeGCN
from src.graph_builder import build_graph
from src.preprocess import get_feature_groups, load_and_preprocess


# -----------------------------
# Configuration / paths
# -----------------------------
DATASET_CSV_PATH = "data/raw/bridge_digital_twin_dataset.csv"

# Fixed 25-feature schema (5 features per node × 5 nodes) in the exact order required.
FEATURE_COLUMNS = [
    "Strain_microstrain",
    "Deflection_mm",
    "Crack_Propagation_mm",
    "Corrosion_Level_percent",
    "Displacement_mm",
    "Vibration_ms2",
    "Tilt_deg",
    "Modal_Frequency_Hz",
    "Seismic_Activity_ms2",
    "Bearing_Joint_Forces_kN",
    "Vehicle_Load_tons",
    "Traffic_Volume_vph",
    "Axle_Counts_pmin",
    "Impact_Events_g",
    "Dynamic_Load_Distribution_percent",
    "Temperature_C",
    "Humidity_percent",
    "Wind_Speed_ms",
    "Precipitation_mmh",
    "Water_Level_m",
    "Structural_Health_Index_SHI",
    "Fatigue_Accumulation_au",
    "Anomaly_Detection_Score",
    "Energy_Dissipation_au",
    "Acoustic_Emissions_levels",
]

LABEL_COLUMN = "Maintenance_Alert"


def _ensure_dirs() -> None:
    """Create output directories if they don't exist."""
    Path("outputs").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)


def _load_xy_with_fixed_schema(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the CSV and produce X/y using ONLY FEATURE_COLUMNS (in-order) and LABEL_COLUMN.

    Notes:
    - We still call load_and_preprocess() (per project API), but we use a fixed-schema
      preprocessing here to guarantee the exact 25-column ordering required for graph nodes.
    """
    # Call the shared preprocessing function (as requested) so the project entrypoint uses it.
    # We don't rely on its X output here because we must preserve a strict 25-column order.
    _ = load_and_preprocess(csv_path)

    # Load raw CSV so we can select columns by name and enforce ordering.
    df = pd.read_csv(csv_path)

    # Drop Timestamp if present (non-feature time column).
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    # Drop prediction/future columns if present (leakage prevention).
    leakage_cols = [
        "SHI_Predicted_24h_Ahead",
        "SHI_Predicted_7d_Ahead",
        "SHI_Predicted_30d_Ahead",
    ]
    existing_leakage_cols = [c for c in leakage_cols if c in df.columns]
    if existing_leakage_cols:
        df = df.drop(columns=existing_leakage_cols)

    # Ensure required columns exist.
    missing = [c for c in FEATURE_COLUMNS + [LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in CSV: {missing}")

    # Select X with exact column order and y from label column.
    X_df = df[FEATURE_COLUMNS].copy()
    # Coerce label to numeric; drop rows with missing/invalid labels so astype(int) is safe.
    y_series = pd.to_numeric(df[LABEL_COLUMN], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    valid_mask = y_series.notna()
    if not bool(valid_mask.all()):
        X_df = X_df.loc[valid_mask].copy()
        y_series = y_series.loc[valid_mask]
    y = y_series.astype(int).to_numpy()

    # Coerce to numeric, impute missing values with column mean, and standardize features.
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.fillna(X_df.mean(numeric_only=True))

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.to_numpy()).astype(np.float32, copy=False)

    return X, y


def _accuracy_from_logits(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    """Compute accuracy given logits and integer labels."""
    preds = torch.argmax(logits, dim=1)
    return (preds == y_true).float().mean().item()


@torch.no_grad()
def _eval_gcn(model: BridgeGCN, loader: GeoDataLoader, device: torch.device) -> float:
    """Evaluate GCN accuracy on a torch_geometric DataLoader."""
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == batch.y).sum().item())
        total += int(batch.y.numel())
    return correct / max(total, 1)


def train_gcn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> BridgeGCN:
    """
    Train a BridgeGCN for 100 epochs and print progress every 10 epochs.
    """
    # Convert each row into a graph Data object.
    train_graphs = [build_graph(row, int(lbl)) for row, lbl in zip(X_train, y_train)]
    test_graphs = [build_graph(row, int(lbl)) for row, lbl in zip(X_test, y_test)]

    # Use torch_geometric DataLoader to batch multiple graphs.
    train_loader = GeoDataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = GeoDataLoader(test_graphs, batch_size=32, shuffle=False)

    model = BridgeGCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 101):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass over the batched graphs.
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)

            # Backpropagation.
            loss.backward()
            optimizer.step()

            # Accumulate metrics.
            epoch_loss += float(loss.item()) * int(batch.y.numel())
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == batch.y).sum().item())
            total += int(batch.y.numel())

        train_loss = epoch_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        test_acc = _eval_gcn(model, test_loader, device)

        # Print progress every 10 epochs as requested.
        if epoch % 10 == 0:
            print(
                f"[GCN] Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}"
            )

    return model


@torch.no_grad()
def _eval_cnn(model: BridgeCNN, loader: TorchDataLoader, device: torch.device) -> float:
    """Evaluate CNN accuracy on a standard PyTorch DataLoader."""
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == yb).sum().item())
        total += int(yb.numel())
    return correct / max(total, 1)


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> BridgeCNN:
    """
    Train a BridgeCNN for 100 epochs and print progress every 10 epochs.
    """
    # Convert numpy arrays into torch tensors.
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # Use TensorDataset + DataLoader for mini-batching.
    train_loader = TorchDataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True
    )
    test_loader = TorchDataLoader(
        TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False
    )

    model = BridgeCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 101):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            # Forward pass over the batch of flat feature vectors.
            logits = model(xb)
            loss = criterion(logits, yb)

            # Backpropagation.
            loss.backward()
            optimizer.step()

            # Accumulate metrics.
            epoch_loss += float(loss.item()) * int(yb.numel())
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.numel())

        train_loss = epoch_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        test_acc = _eval_cnn(model, test_loader, device)

        # Print progress every 10 epochs as requested.
        if epoch % 10 == 0:
            print(
                f"[CNN] Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}"
            )

    return model


@torch.no_grad()
def _predict_gcn(
    model: BridgeGCN, X: np.ndarray, y: np.ndarray, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Get predictions and labels for the GCN on a numpy test set."""
    graphs = [build_graph(row, int(lbl)) for row, lbl in zip(X, y)]
    loader = GeoDataLoader(graphs, batch_size=32, shuffle=False)

    model.eval()
    all_preds: list[int] = []
    all_y: list[int] = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        all_preds.extend(preds)
        all_y.extend(batch.y.detach().cpu().numpy().tolist())
    return np.asarray(all_preds), np.asarray(all_y)


@torch.no_grad()
def _predict_cnn(
    model: BridgeCNN, X: np.ndarray, y: np.ndarray, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Get predictions and labels for the CNN on a numpy test set."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    loader = TorchDataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=False)

    model.eval()
    all_preds: list[int] = []
    all_y: list[int] = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        all_preds.extend(preds)
        all_y.extend(yb.detach().cpu().numpy().tolist())
    return np.asarray(all_preds), np.asarray(all_y)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute accuracy, F1, precision, and recall (binary)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def _save_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: str, title: str
) -> None:
    """Save a confusion matrix plot as a PNG under outputs/."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Healthy (0)", "Damaged (1)"],
        yticklabels=["Healthy (0)", "Damaged (1)"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _print_comparison_table(gcn: Dict[str, float], cnn: Dict[str, float]) -> None:
    """Print the comparison table in the requested box-drawing style."""
    def pct(x: float) -> str:
        return f"{(x * 100):6.2f}%"

    def dec(x: float) -> str:
        return f"{x:5.3f}"

    print("┌─────────┬──────────┬────────┬───────────┬────────┐")
    print("│ Model   │ Accuracy │ F1     │ Precision │ Recall │")
    print("├─────────┼──────────┼────────┼───────────┼────────┤")
    print(
        f"│ GCN     │ {pct(gcn['accuracy'])}  │ {dec(gcn['f1'])}  │"
        f"  {dec(gcn['precision'])}   │ {dec(gcn['recall'])}  │"
    )
    print(
        f"│ CNN     │ {pct(cnn['accuracy'])}  │ {dec(cnn['f1'])}  │"
        f"  {dec(cnn['precision'])}   │ {dec(cnn['recall'])}  │"
    )
    print("└─────────┴──────────┴────────┴───────────┴────────┘")


def main() -> None:
    # Ensure output folders exist for plots and model checkpoints.
    _ensure_dirs()

    # Optional: load feature groups (useful for sanity-checking the dataset schema).
    _ = get_feature_groups()

    # -----------------------------
    # DATA LOADING + SPLIT
    # -----------------------------
    # Load X/y using exactly the 25 specified columns (in the required order).
    X, y = _load_xy_with_fixed_schema(DATASET_CSV_PATH)

    # Stratified 80/20 train-test split to preserve class balance.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Choose device (GPU if available, else CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # GCN TRAINING
    # -----------------------------
    gcn_model = train_gcn(X_train, y_train, X_test, y_test, device=device)

    # -----------------------------
    # CNN TRAINING
    # -----------------------------
    cnn_model = train_cnn(X_train, y_train, X_test, y_test, device=device)

    # -----------------------------
    # EVALUATION (test set)
    # -----------------------------
    gcn_pred, gcn_true = _predict_gcn(gcn_model, X_test, y_test, device=device)
    cnn_pred, cnn_true = _predict_cnn(cnn_model, X_test, y_test, device=device)

    gcn_metrics = _compute_metrics(gcn_true, gcn_pred)
    cnn_metrics = _compute_metrics(cnn_true, cnn_pred)

    # Print clean comparison table.
    _print_comparison_table(gcn_metrics, cnn_metrics)

    # Save confusion matrices as PNGs.
    _save_confusion_matrix(
        gcn_true,
        gcn_pred,
        out_path="outputs/gcn_confusion_matrix.png",
        title="GCN Confusion Matrix",
    )
    _save_confusion_matrix(
        cnn_true,
        cnn_pred,
        out_path="outputs/cnn_confusion_matrix.png",
        title="CNN Confusion Matrix",
    )

    # Save trained models (state_dict) for later inference/retraining.
    torch.save(gcn_model.state_dict(), "models/gcn_model.pth")
    torch.save(cnn_model.state_dict(), "models/cnn_model.pth")
    print("Saved models to models/ and confusion matrices to outputs/.")


if __name__ == "__main__":
    main()
