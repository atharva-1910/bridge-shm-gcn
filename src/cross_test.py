from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import sys
from pathlib import Path as _Path

# Allow running directly via: `python3 src/cross_test.py`
# Add project root (parent of `src/`) to sys.path so `from src...` imports work.
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader as GeoDataLoader

from src.gcn_model import BridgeGCN
from src.graph_builder import build_graph
from src.preprocess import get_feature_groups, load_and_preprocess


# Fixed 25-feature schema (in the exact order required)
FEATURE_COLS = [
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


def load_dataset(
    filepath: str, scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Load a dataset CSV and return scaled X, y, and the scaler used.

    - Loads CSV from filepath
    - Selects only FEATURE_COLS as X (in-order)
    - Uses Maintenance_Alert as y
    - If scaler is None: fit a new StandardScaler on X
    - If scaler provided: transform X with the existing scaler
    """
    # Call shared preprocessing function (keeps project API consistent); we don't rely on its X output
    # here because we must enforce FEATURE_COLS ordering explicitly.
    _ = load_and_preprocess(filepath)

    df = pd.read_csv(filepath)

    # Ensure required columns exist.
    missing = [c for c in FEATURE_COLS + ["Maintenance_Alert"] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {filepath}: {missing}")

    # Select feature matrix and label vector.
    X_df = df[FEATURE_COLS].copy()
    y_series = pd.to_numeric(df["Maintenance_Alert"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )

    # Drop rows with missing labels so metric computations and training are well-defined.
    valid_mask = y_series.notna()
    if not bool(valid_mask.all()):
        X_df = X_df.loc[valid_mask].copy()
        y_series = y_series.loc[valid_mask]

    # Coerce feature columns to numeric and impute missing values using column means.
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.fillna(X_df.mean(numeric_only=True))

    X = X_df.to_numpy(dtype=np.float32, copy=False)
    y = y_series.astype(int).to_numpy()

    # Fit or apply the scaler.
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32, copy=False)
    else:
        X_scaled = scaler.transform(X).astype(np.float32, copy=False)

    return X_scaled, y, scaler


def build_graph_list(X: np.ndarray, y: np.ndarray) -> List:
    """
    Convert each row of X and its corresponding y into a torch_geometric Data graph.

    Uses build_graph(row, label) for each sample.
    """
    return [build_graph(row, int(lbl)) for row, lbl in zip(X, y)]


@torch.no_grad()
def evaluate_model(model: BridgeGCN, graph_list: List) -> Tuple[float, float, float, float]:
    """
    Run the model on all graphs and compute standard classification metrics.

    Returns:
        accuracy, f1, precision, recall
    """
    loader = GeoDataLoader(graph_list, batch_size=32, shuffle=False)

    model.eval()
    all_preds: List[int] = []
    all_true: List[int] = []

    for batch in loader:
        logits = model(batch.x, batch.edge_index, batch.batch)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        all_preds.extend(preds)
        all_true.extend(batch.y.cpu().numpy().tolist())

    y_true = np.asarray(all_true)
    y_pred = np.asarray(all_preds)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    return acc, f1, prec, rec


def _train_gcn(train_graphs: List, device: torch.device) -> BridgeGCN:
    """
    Train a BridgeGCN on the provided graph list (simple fixed training loop).
    """
    loader = GeoDataLoader(train_graphs, batch_size=32, shuffle=True)
    model = BridgeGCN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Fixed training schedule (kept consistent across experiments).
    for epoch in range(1, 101):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()

    return model.cpu()


def _format_summary_table(rows: List[Tuple[str, float, float]]) -> str:
    """
    Create the requested box-drawing summary table.
    Each row is (experiment_name, accuracy, f1).
    """
    lines = []
    lines.append("┌─────────────────────────────────┬──────────┬────────┐")
    lines.append("│ Experiment                      │ Accuracy │   F1   │")
    lines.append("├─────────────────────────────────┼──────────┼────────┤")
    for name, acc, f1 in rows:
        lines.append(f"│ {name:<31} │ {acc*100:6.2f}%  │ {f1:5.3f}  │")
    lines.append("└─────────────────────────────────┴──────────┴────────┘")
    return "\n".join(lines)


if __name__ == "__main__":
    # Ensure output directory exists for saving the results table.
    Path("outputs").mkdir(parents=True, exist_ok=True)

    # Touch the feature grouping API so this script stays aligned with project structure.
    _ = get_feature_groups()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    real_path = "data/raw/bridge_digital_twin_dataset.csv"
    synth_path = "data/synthetic/synthetic_bridge_data.csv"

    results_rows: List[Tuple[str, float, float]] = []

    # -----------------------------
    # EXPERIMENT 1: Train REAL -> Test SYNTHETIC
    # -----------------------------
    print("\nEXPERIMENT 1 - Train on REAL, Test on SYNTHETIC")
    X_real, y_real, real_scaler = load_dataset(real_path, scaler=None)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
    )
    model_real = _train_gcn(build_graph_list(Xr_tr, yr_tr), device=device)

    # Load synthetic and transform with the SAME scaler fitted on real.
    X_synth_scaled, y_synth, _ = load_dataset(synth_path, scaler=real_scaler)
    synth_graphs = build_graph_list(X_synth_scaled, y_synth)
    acc1, f11, p1, r1 = evaluate_model(model_real, synth_graphs)
    print(f"Accuracy: {acc1*100:.2f}% | F1: {f11:.3f} | Precision: {p1:.3f} | Recall: {r1:.3f}")
    results_rows.append(("Train Real    → Test Synthetic", acc1, f11))

    # -----------------------------
    # EXPERIMENT 2: Train SYNTHETIC -> Test REAL
    # -----------------------------
    print("\nEXPERIMENT 2 - Train on SYNTHETIC, Test on REAL")
    X_synth, y_synth2, synth_scaler = load_dataset(synth_path, scaler=None)
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
        X_synth, y_synth2, test_size=0.2, random_state=42, stratify=y_synth2
    )
    model_synth = _train_gcn(build_graph_list(Xs_tr, ys_tr), device=device)

    # Load real and transform with the SAME scaler fitted on synthetic.
    X_real_scaled, y_real2, _ = load_dataset(real_path, scaler=synth_scaler)
    real_graphs = build_graph_list(X_real_scaled, y_real2)
    acc2, f12, p2, r2 = evaluate_model(model_synth, real_graphs)
    print(f"Accuracy: {acc2*100:.2f}% | F1: {f12:.3f} | Precision: {p2:.3f} | Recall: {r2:.3f}")
    results_rows.append(("Train Synth   → Test Real", acc2, f12))

    # -----------------------------
    # EXPERIMENT 3: Train COMBINED -> Test COMBINED (held-out)
    # -----------------------------
    print("\nEXPERIMENT 3 - Train on COMBINED, Test on BOTH (held-out combined split)")
    # Combine (unscaled) by reloading with a fresh scaler fitted on combined X.
    X_real_u, y_real_u, _ = load_dataset(real_path, scaler=None)
    X_synth_u, y_synth_u, _ = load_dataset(synth_path, scaler=None)

    X_combined = np.vstack([X_real_u, X_synth_u])
    y_combined = np.concatenate([y_real_u, y_synth_u])

    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
    )
    model_combined = _train_gcn(build_graph_list(Xc_tr, yc_tr), device=device)
    acc3, f13, p3, r3 = evaluate_model(model_combined, build_graph_list(Xc_te, yc_te))
    print(f"Accuracy: {acc3*100:.2f}% | F1: {f13:.3f} | Precision: {p3:.3f} | Recall: {r3:.3f}")
    results_rows.append(("Train Combined→ Test Combined", acc3, f13))

    # -----------------------------
    # FINAL SUMMARY TABLE + SAVE
    # -----------------------------
    table = _format_summary_table(results_rows)
    print("\n" + table)

    out_path = Path("outputs") / "cross_dataset_results.txt"
    out_path.write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved summary table to {out_path.as_posix()}")

