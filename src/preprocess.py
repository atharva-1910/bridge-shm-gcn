from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a bridge SHM CSV dataset and preprocess it for modeling.

    Steps:
    - Load CSV
    - Drop non-feature/time column if present
    - Drop future/prediction leakage columns if present
    - Separate label (y) from features (X)
    - Impute missing values with column means
    - Standardize features (zero mean, unit variance)

    Returns:
        X: numpy array of standardized features
        y: numpy array of binary labels (0=healthy, 1=alert)
    """
    # 1) Load the CSV file into a DataFrame.
    df = pd.read_csv(filepath)

    # 2) Drop the Timestamp column if it exists (often not a useful numeric feature).
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    # 3) Drop known future/prediction columns if they exist (to prevent target leakage).
    leakage_cols = [
        "SHI_Predicted_24h_Ahead",
        "SHI_Predicted_7d_Ahead",
        "SHI_Predicted_30d_Ahead",
    ]
    existing_leakage_cols = [c for c in leakage_cols if c in df.columns]
    if existing_leakage_cols:
        df = df.drop(columns=existing_leakage_cols)

    # 4) Use Maintenance_Alert as the label y (binary: 0=healthy, 1=alert).
    if "Maintenance_Alert" not in df.columns:
        raise KeyError(
            "Expected 'Maintenance_Alert' column to exist for labels, but it was not found."
        )
    # Coerce labels to numeric, treat non-numeric/inf as missing, and drop rows with missing labels.
    y_series = pd.to_numeric(df["Maintenance_Alert"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    valid_mask = y_series.notna()
    if not bool(valid_mask.all()):
        df = df.loc[valid_mask].copy()
        y_series = y_series.loc[valid_mask]
    y = y_series.astype(int).to_numpy()

    # 5) Drop Maintenance_Alert from the feature set X.
    X_df = df.drop(columns=["Maintenance_Alert"])

    # 6) Ensure feature columns are numeric; coerce non-numeric values to NaN.
    #    This makes mean-imputation and scaling reliable.
    X_df = X_df.apply(pd.to_numeric, errors="coerce")

    # 7) Fill missing values with the column mean (computed ignoring NaNs).
    X_df = X_df.fillna(X_df.mean(numeric_only=True))

    # 8) Normalize all feature columns using StandardScaler.
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.to_numpy())

    # 9) Return NumPy arrays for downstream GCN/CNN pipelines.
    return X.astype(np.float32, copy=False), y.astype(np.int64, copy=False)


def get_feature_groups() -> Dict[str, List[str]]:
    """
    Return domain-based feature groups for analysis/visualization.
    """
    return {
        "structural": [
            "Strain_microstrain",
            "Deflection_mm",
            "Crack_Propagation_mm",
            "Corrosion_Level_percent",
            "Displacement_mm",
        ],
        "dynamic": [
            "Vibration_ms2",
            "Tilt_deg",
            "Modal_Frequency_Hz",
            "Seismic_Activity_ms2",
            "Bearing_Joint_Forces_kN",
        ],
        "load": [
            "Vehicle_Load_tons",
            "Traffic_Volume_vph",
            "Axle_Counts_pmin",
            "Impact_Events_g",
            "Dynamic_Load_Distribution_percent",
        ],
        "environmental": [
            "Temperature_C",
            "Humidity_percent",
            "Wind_Speed_ms",
            "Precipitation_mmh",
            "Water_Level_m",
        ],
        "health": [
            "Structural_Health_Index_SHI",
            "Fatigue_Accumulation_au",
            "Anomaly_Detection_Score",
            "Energy_Dissipation_au",
            "Acoustic_Emissions_levels",
        ],
    }


def generate_synthetic_data() -> pd.DataFrame:
    """
    Generate synthetic bridge vibration data matching the real dataset schema.

    - Creates 500 healthy samples (label=0) and 500 damaged samples (label=1)
    - Uses the exact 25 feature columns in the exact order requested
    - Shuffles rows, saves to data/synthetic/synthetic_bridge_data.csv, and returns the DataFrame
    """
    feature_columns = [
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

    # Healthy distribution parameters (mean, std) for each feature.
    healthy_params = {
        "Strain_microstrain": (150, 20),
        "Deflection_mm": (5, 1),
        "Crack_Propagation_mm": (0.1, 0.05),
        "Corrosion_Level_percent": (5, 2),
        "Displacement_mm": (2, 0.5),
        "Vibration_ms2": (0.5, 0.1),
        "Tilt_deg": (0.2, 0.05),
        "Modal_Frequency_Hz": (15, 1),
        "Seismic_Activity_ms2": (0.01, 0.005),
        "Bearing_Joint_Forces_kN": (200, 20),
        "Vehicle_Load_tons": (10, 2),
        "Traffic_Volume_vph": (500, 50),
        "Axle_Counts_pmin": (20, 3),
        "Impact_Events_g": (0.3, 0.05),
        "Dynamic_Load_Distribution_percent": (50, 5),
        "Temperature_C": (25, 5),
        "Humidity_percent": (60, 10),
        "Wind_Speed_ms": (5, 2),
        "Precipitation_mmh": (2, 1),
        "Water_Level_m": (3, 0.5),
        "Structural_Health_Index_SHI": (85, 5),
        "Fatigue_Accumulation_au": (0.2, 0.05),
        "Anomaly_Detection_Score": (0.1, 0.02),
        "Energy_Dissipation_au": (0.3, 0.05),
        "Acoustic_Emissions_levels": (20, 5),
    }

    # Damaged distribution overrides (mean, std) for select features.
    damaged_overrides = {
        "Strain_microstrain": (300, 50),
        "Deflection_mm": (15, 3),
        "Crack_Propagation_mm": (2, 0.5),
        "Corrosion_Level_percent": (35, 8),
        "Vibration_ms2": (2.5, 0.5),
        "Modal_Frequency_Hz": (10, 2),
        "Structural_Health_Index_SHI": (40, 10),
        "Fatigue_Accumulation_au": (0.8, 0.1),
        "Anomaly_Detection_Score": (0.7, 0.1),
        "Acoustic_Emissions_levels": (80, 15),
    }

    rng = np.random.default_rng()
    n_healthy = 500
    n_damaged = 500

    # Generate healthy samples using the specified distributions.
    healthy_data = {
        col: rng.normal(loc=healthy_params[col][0], scale=healthy_params[col][1], size=n_healthy)
        for col in feature_columns
    }
    healthy_df = pd.DataFrame(healthy_data, columns=feature_columns)
    healthy_df["Maintenance_Alert"] = 0

    # Generate damaged samples:
    # - Use override parameters for the specified damage-indicative features
    # - For all other features, keep the healthy mean but increase noise (std) by 20%
    damaged_data = {}
    for col in feature_columns:
        if col in damaged_overrides:
            mean, std = damaged_overrides[col]
        else:
            mean, std = healthy_params[col]
            std = std * 1.2  # 20% noise increase for all other features
        damaged_data[col] = rng.normal(loc=mean, scale=std, size=n_damaged)

    damaged_df = pd.DataFrame(damaged_data, columns=feature_columns)
    damaged_df["Maintenance_Alert"] = 1

    # Combine healthy and damaged samples, then shuffle.
    df = pd.concat([healthy_df, damaged_df], ignore_index=True)
    df = df.sample(frac=1.0, random_state=None).reset_index(drop=True)

    # Save to the requested path.
    out_dir = "data/synthetic"
    out_path = f"{out_dir}/synthetic_bridge_data.csv"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("Generated 1000 synthetic samples (500 healthy, 500 damaged)")
    print("Saved to data/synthetic/synthetic_bridge_data.csv")

    return df


if __name__ == "__main__":
    generate_synthetic_data()
