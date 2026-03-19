import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import pandas as pd
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
from src.graph_builder import build_graph
from src.gcn_model import BridgeGCN
from src.cnn_baseline import BridgeCNN
import torch.nn.functional as F


st.set_page_config(
    page_title="Bridge SHM — Digital Twin GNN Monitor",
    page_icon="🌉",
    layout="wide",
)


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


def draw_bridge_diagram(prediction=None, confidence=None):
    """Draw a steel truss suspension bridge with colored sensor nodes."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.axis("off")

    # --- Water / ground ---
    ax.add_patch(plt.Rectangle((0, 0), 14, 0.5, color="#0a3d62", alpha=0.85, zorder=1))
    # water shimmer lines
    for wx in np.linspace(0.5, 13.5, 18):
        ax.plot([wx, wx + 0.4], [0.25, 0.25], color="#1e90ff", linewidth=0.8, alpha=0.5, zorder=2)

    # --- Left support pillar ---
    pillar_pts_l = np.array([[0.7, 0.5], [1.3, 0.5], [1.6, 2.0], [0.4, 2.0]])
    ax.add_patch(plt.Polygon(pillar_pts_l, color="#4a4a5a", zorder=3))
    # --- Right support pillar ---
    pillar_pts_r = np.array([[12.7, 0.5], [13.3, 0.5], [13.6, 2.0], [12.4, 2.0]])
    ax.add_patch(plt.Polygon(pillar_pts_r, color="#4a4a5a", zorder=3))

    # --- Bridge deck ---
    ax.plot([1.0, 13.0], [2.0, 2.0], color="#888899", linewidth=10,
            solid_capstyle="round", zorder=4)
    # road markings
    for rx in np.linspace(2.5, 12.0, 12):
        ax.plot([rx, rx + 0.4], [2.0, 2.0], color="#ffff88", linewidth=1.5,
                alpha=0.6, zorder=5)

    # --- Truss diagonals below deck ---
    truss_color = "#556677"
    ax.plot([1.0, 13.0], [1.25, 1.25], color=truss_color, linewidth=2, zorder=3)
    for i in np.linspace(1.0, 12.0, 13):
        ax.plot([i, i + 0.85], [2.0, 1.25], color=truss_color, linewidth=1.8, zorder=3)
        ax.plot([i + 0.85, i + 0.85], [1.25, 2.0], color=truss_color, linewidth=1.8, zorder=3)

    # --- Left tower ---
    ax.plot([2.2, 2.2], [2.0, 5.8], color="#aaaacc", linewidth=6, solid_capstyle="round", zorder=4)
    ax.plot([1.7, 2.7], [5.8, 5.8], color="#aaaacc", linewidth=5, zorder=4)
    # tower cross braces
    for ty in [3.0, 4.0, 5.0]:
        ax.plot([1.9, 2.5], [ty, ty + 0.5], color="#9999bb", linewidth=1.5, zorder=4)
        ax.plot([2.5, 1.9], [ty, ty + 0.5], color="#9999bb", linewidth=1.5, zorder=4)

    # --- Right tower ---
    ax.plot([11.8, 11.8], [2.0, 5.8], color="#aaaacc", linewidth=6, solid_capstyle="round", zorder=4)
    ax.plot([11.3, 12.3], [5.8, 5.8], color="#aaaacc", linewidth=5, zorder=4)
    for ty in [3.0, 4.0, 5.0]:
        ax.plot([11.5, 12.1], [ty, ty + 0.5], color="#9999bb", linewidth=1.5, zorder=4)
        ax.plot([12.1, 11.5], [ty, ty + 0.5], color="#9999bb", linewidth=1.5, zorder=4)

    # --- Suspension cables from left tower ---
    cable_color = "#ccccdd"
    for cx in [3.0, 4.5, 6.0, 7.5]:
        ax.plot([2.2, cx], [5.8, 2.0], color=cable_color, linewidth=1.8, alpha=0.75, zorder=4)

    # --- Suspension cables from right tower ---
    for cx in [7.5, 9.0, 10.5, 12.0]:
        ax.plot([11.8, cx], [5.8, 2.0], color=cable_color, linewidth=1.8, alpha=0.75, zorder=4)

    # --- Sensor node colors ---
    if prediction == 1:  # DAMAGED
        nc = {
            "structural": "#ff3333",
            "dynamic":    "#ff3333",
            "load":       "#ff8800",
            "env":        "#44cc44",
            "health":     "#ff3333",
        }
        glow = "#ff0000"
    else:  # HEALTHY or no prediction
        nc = {
            "structural": "#22cc55",
            "dynamic":    "#22cc55",
            "load":       "#22cc55",
            "env":        "#22cc55",
            "health":     "#22cc55",
        }
        glow = "#00ff88"

    def draw_sensor(x, y, color, label, icon, key_offset=0):
        # glow ring
        ax.plot(x, y, "o", markersize=30, color=color, alpha=0.2, zorder=6)
        # main dot
        ax.plot(x, y, "o", markersize=20, color=color,
                markeredgecolor="white", markeredgewidth=2.0, zorder=7)
        # icon text inside dot
        ax.text(x, y, icon, fontsize=9, ha="center", va="center",
                color="white", fontweight="bold", zorder=8)
        # label above
        ax.text(x, y + 0.55, label, color="white", fontsize=8,
                ha="center", fontweight="bold", zorder=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a2e",
                          edgecolor=color, alpha=0.8, linewidth=1.2))

    # Node 0 — Structural (left deck area)
    draw_sensor(3.5, 2.0, nc["structural"], "Structural", "⚙")
    # Node 1 — Dynamic (center deck)
    draw_sensor(7.0, 2.0, nc["dynamic"], "Dynamic", "📳")
    # Node 2 — Load (right deck)
    draw_sensor(10.0, 2.0, nc["load"], "Load", "🚛")
    # Node 3 — Environmental (top left tower)
    draw_sensor(2.2, 5.8, nc["env"], "Environmental", "🌤")
    # Node 4 — Health (top right tower)
    draw_sensor(11.8, 5.8, nc["health"], "Health", "💗")

    # --- Title ---
    if prediction is not None:
        status = "🔴  DAMAGED" if prediction == 1 else "🟢  HEALTHY"
        conf_str = f"  —  {confidence:.1f}% confident" if confidence else ""
        title_col = "#ff4444" if prediction == 1 else "#44ee88"
        ax.set_title(f"Bridge Status: {status}{conf_str}",
                     color=title_col, fontsize=16, fontweight="bold", pad=10)
    else:
        ax.set_title("Steel Truss Bridge  —  Live Sensor Domain Monitor",
                     color="#aaddff", fontsize=14, fontweight="bold", pad=10)

    plt.tight_layout()
    return fig


@st.cache_resource
def load_models():
    gcn = BridgeGCN()
    cnn = BridgeCNN()
    gcn_path = "models/gcn_model.pth"
    cnn_path = "models/cnn_model.pth"
    if os.path.exists(gcn_path):
        gcn.load_state_dict(torch.load(gcn_path, map_location="cpu", weights_only=False))
        gcn.eval()
    if os.path.exists(cnn_path):
        cnn.load_state_dict(torch.load(cnn_path, map_location="cpu", weights_only=False))
        cnn.eval()
    return gcn, cnn


def _load_full_dataset() -> pd.DataFrame:
    return pd.read_csv("data/raw/bridge_digital_twin_dataset.csv")


def _fit_scaler(df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[FEATURE_COLS].astype(float).values)
    return scaler


def _run_gcn(gcn: BridgeGCN, x_row_scaled: np.ndarray) -> tuple:
    data = build_graph(x_row_scaled.astype(np.float32), 0)
    batch = torch.zeros(data.x.size(0), dtype=torch.long)
    logits = gcn(data.x, data.edge_index, batch)
    probs = F.softmax(logits, dim=1).squeeze(0)
    pred = int(torch.argmax(probs).item())
    conf = float(probs[pred].item())
    return pred, conf


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌉 Bridge SHM Dashboard")
    st.caption("Digital Twin-based monitoring using GCN and CNN models")
    st.divider()
    st.markdown("### 🧠 Model Performance")
    a, b = st.columns(2)
    c, d = st.columns(2)
    a.metric("GCN Accuracy", "99.92%")
    b.metric("GCN F1", "0.929")
    c.metric("CNN Accuracy", "99.95%")
    d.metric("CNN F1", "0.959")
    st.divider()
    st.info(
        "Graph Convolutional Networks model the bridge as a graph "
        "where sensors are nodes and physical connections are edges. "
        "This captures structural relationships between monitoring domains."
    )

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Live Prediction",
    "🕸️ Sensor Graph",
    "📈 Model Comparison",
    "📊 Data Explorer",
    "ℹ️ About",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🔍 Live Prediction")
    st.write("Test the GCN model on sample rows from the digital twin dataset.")

    # Default bridge diagram (no prediction yet)
    default_bridge = draw_bridge_diagram()
    st.pyplot(default_bridge, use_container_width=True)
    plt.close()

    st.divider()

    gcn, cnn = load_models()

    if "sample_df" not in st.session_state:
        st.session_state.sample_df = None

    if st.button("🎲 Load Sample Data"):
        with st.spinner("Loading sample rows..."):
            df_full = _load_full_dataset()
            st.session_state.sample_df = (
                df_full.sample(n=5, random_state=None)[FEATURE_COLS].reset_index(drop=True)
            )
        st.success("Loaded 5 random samples.")

    if st.session_state.sample_df is not None:
        df_full = _load_full_dataset()
        scaler = _fit_scaler(df_full)
        X_scaled = scaler.transform(
            st.session_state.sample_df[FEATURE_COLS].astype(float).values
        )

        summary_rows = []

        for i in range(5):
            row_scaled = X_scaled[i]
            pred, conf = _run_gcn(gcn, row_scaled)

            shi_scaled = float(row_scaled[20])
            shi_value = float(np.clip(shi_scaled * 20.0 + 85.0, 0.0, 100.0))
            conf_pct = conf * 100.0

            st.markdown(f"#### Sample {i + 1}")

            # Bridge diagram updated with prediction
            pred_bridge = draw_bridge_diagram(prediction=pred, confidence=conf_pct)
            st.pyplot(pred_bridge, use_container_width=True)
            plt.close()

            left, right = st.columns(2)
            with left:
                if pred == 0:
                    st.markdown(
                        f"""
                        <div style="background:#1a7a1a;padding:25px;border-radius:15px;
                        text-align:center;font-size:32px;color:white;font-weight:bold;
                        margin:10px 0;">🟢 HEALTHY<br>
                        <span style="font-size:18px;">Confidence: {conf_pct:.1f}%</span></div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="background:#8b0000;padding:25px;border-radius:15px;
                        text-align:center;font-size:32px;color:white;font-weight:bold;
                        margin:10px 0;">🔴 DAMAGED<br>
                        <span style="font-size:18px;">Confidence: {conf_pct:.1f}%</span></div>
                        """,
                        unsafe_allow_html=True,
                    )

            with right:
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=shi_value,
                        title={"text": "Structural Health Index"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "steps": [
                                {"range": [0, 40],  "color": "#ff4444"},
                                {"range": [40, 70], "color": "#ffaa00"},
                                {"range": [70, 100],"color": "#44bb44"},
                            ],
                            "threshold": {
                                "line": {"color": "black", "width": 4},
                                "thickness": 0.75,
                                "value": 70,
                            },
                        },
                    )
                )
                fig_gauge.update_layout(
                    height=260, margin=dict(l=20, r=20, t=50, b=10)
                )
                st.plotly_chart(
                    fig_gauge, use_container_width=True, key=f"gauge_shi_{i}"
                )

            summary_rows.append({
                "Sample": i + 1,
                "GCN Prediction": "Healthy" if pred == 0 else "Damaged",
                "Confidence": f"{conf_pct:.1f}%",
                "SHI Value": round(shi_value, 2),
            })

            st.divider()

        st.subheader("📋 Predictions Summary")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SENSOR GRAPH
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🕸️ Bridge Monitoring Graph")
    st.write("5 monitoring domains modeled as graph nodes")

    # Bridge visual on top
    st.markdown("### 🌉 Bridge with Sensor Locations")
    bridge_sensor_fig = draw_bridge_diagram()
    st.pyplot(bridge_sensor_fig, use_container_width=True)
    plt.close()

    st.divider()

    # NetworkX graph below
    G = nx.Graph()
    nodes = ["Structural", "Dynamic", "Load", "Environmental", "Health"]
    for n in nodes:
        G.add_node(n)

    edges = [
        ("Structural", "Environmental"),
        ("Structural", "Load"),
        ("Dynamic", "Load"),
        ("Health", "Structural"),
        ("Health", "Dynamic"),
        ("Dynamic", "Environmental"),
    ]
    G.add_edges_from(edges)

    pos = {
        "Structural":    (0, 0),
        "Dynamic":       (2, 0),
        "Load":          (1, 2),
        "Environmental": (-1, 1),
        "Health":        (3, 1),
    }
    colors = ["#4C72B0", "#DD8452", "#55A868", "#64B5CD", "#C44E52"]

    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=4000)
    nx.draw_networkx_edges(G, pos, width=3, alpha=0.85)
    nx.draw_networkx_labels(G, pos, font_size=11, font_color="white", font_weight="bold")
    plt.title("Bridge Sensor Domain Graph", fontsize=16, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color=colors[0], label="Structural"),
        mpatches.Patch(color=colors[1], label="Dynamic"),
        mpatches.Patch(color=colors[2], label="Load"),
        mpatches.Patch(color=colors[3], label="Environmental"),
        mpatches.Patch(color=colors[4], label="Health"),
    ]
    plt.legend(handles=legend_patches, loc="lower left")
    plt.axis("off")
    st.pyplot(plt)
    plt.close()

    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown("**Structural** — Strain, Deflection, Crack, Corrosion")
    c2.markdown("**Dynamic** — Vibration, Tilt, Modal Freq, Seismic")
    c3.markdown("**Load** — Vehicle Load, Traffic, Axle Counts")
    c4.markdown("**Environmental** — Temp, Humidity, Wind, Rain")
    c5.markdown("**Health** — SHI, Fatigue, Anomaly Score")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📈 GCN vs CNN Model Comparison")

    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=[99.92, 92.9, 95.8, 90.2],
        theta=["Accuracy", "F1", "Precision", "Recall"],
        fill="toself",
        name="GCN",
        line_color="blue",
    ))
    radar.add_trace(go.Scatterpolar(
        r=[99.95, 95.9, 100.0, 92.2],
        theta=["Accuracy", "F1", "Precision", "Recall"],
        fill="toself",
        name="CNN",
        line_color="orange",
    ))
    radar.update_layout(
        title="GCN vs CNN — Performance Radar",
        polar=dict(radialaxis=dict(range=[85, 100], visible=True)),
        showlegend=True,
        height=520,
        margin=dict(l=30, r=30, t=70, b=30),
    )
    st.plotly_chart(radar, use_container_width=True, key="radar_comparison")

    st.divider()
    left, right = st.columns(2)
    with left:
        df_cmp = pd.DataFrame(
            [
                ["GCN", "99.92%", 0.929, 0.958, 0.902],
                ["CNN", "99.95%", 0.959, 1.000, 0.922],
            ],
            columns=["Model", "Accuracy", "F1", "Precision", "Recall"],
        )
        st.dataframe(df_cmp, use_container_width=True)
    with right:
        st.markdown(
            "**Why GCN is better architecturally**\n\n"
            "- Structural systems have interacting domains (load ↔ dynamics ↔ health).\n"
            "- GCN encodes these relationships as edges and performs message passing.\n"
            "- This is a more faithful inductive bias than a flat feature vector."
        )

    st.divider()
    cm1, cm2 = st.columns(2)
    with cm1:
        st.image("outputs/gcn_confusion_matrix.png", caption="GCN Confusion Matrix")
    with cm2:
        st.image("outputs/cnn_confusion_matrix.png", caption="CNN Confusion Matrix")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    df = _load_full_dataset()

    total   = len(df)
    healthy = int((df["Maintenance_Alert"] == 0).sum())
    damaged = int((df["Maintenance_Alert"] == 1).sum())
    rate    = (damaged / max(total, 1)) * 100.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Samples",  f"{total}")
    m2.metric("Healthy Count",  f"{healthy}")
    m3.metric("Damaged Count",  f"{damaged}")
    m4.metric("Damage Rate %",  f"{rate:.2f}%")

    st.divider()

    row1_left, row1_right = st.columns(2)
    with row1_left:
        pie_df = df["Maintenance_Alert"].value_counts().reset_index()
        pie_df.columns = ["Maintenance_Alert", "Count"]
        pie_df["Label"] = pie_df["Maintenance_Alert"].map({0: "Healthy", 1: "Damaged"})
        fig_pie = px.pie(
            pie_df,
            names="Label",
            values="Count",
            color="Label",
            color_discrete_map={"Healthy": "green", "Damaged": "red"},
            title="Maintenance Alert Distribution",
        )
        st.plotly_chart(fig_pie, use_container_width=True, key="pie_chart")

    with row1_right:
        fig_line = px.line(
            df["Structural_Health_Index_SHI"].head(500),
            title="Structural Health Index — First 500 Rows",
            labels={"index": "Row", "value": "SHI"},
        )
        st.plotly_chart(fig_line, use_container_width=True, key="shi_line")

    st.divider()

    row2_left, row2_right = st.columns(2)
    with row2_left:
        df_sc = df[
            ["Structural_Health_Index_SHI", "Anomaly_Detection_Score",
             "Fatigue_Accumulation_au", "Maintenance_Alert"]
        ].copy()
        df_sc["Fatigue_Accumulation_au"] = (
            pd.to_numeric(df_sc["Fatigue_Accumulation_au"], errors="coerce")
            .fillna(0).clip(lower=0)
        )
        fig_scatter = px.scatter(
            df_sc,
            x="Structural_Health_Index_SHI",
            y="Anomaly_Detection_Score",
            color="Maintenance_Alert",
            size="Fatigue_Accumulation_au",
            color_discrete_map={0: "green", 1: "red"},
            title="Health Index vs Anomaly Score",
        )
        st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_health")

    with row2_right:
        corr_with_label = df[FEATURE_COLS].corrwith(df["Maintenance_Alert"])
        top10 = corr_with_label.reindex(
            corr_with_label.abs().sort_values(ascending=False).head(10).index
        )
        bar_df = pd.DataFrame({"Feature": top10.index, "Correlation": top10.values})
        bar_df["Direction"] = np.where(bar_df["Correlation"] >= 0, "Positive", "Negative")
        fig_imp = px.bar(
            bar_df,
            x="Feature",
            y="Correlation",
            color="Direction",
            color_discrete_map={"Positive": "red", "Negative": "blue"},
            title="Top Features Driving Damage Alerts",
        )
        st.plotly_chart(fig_imp, use_container_width=True, key="feature_importance")

    st.divider()

    corr = df[FEATURE_COLS].corr()
    fig_corr = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Feature Correlation Matrix",
    )
    st.plotly_chart(fig_corr, use_container_width=True, key="correlation_heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("ℹ️ About This Project")

    # Bridge diagram at top of About tab too
    about_bridge = draw_bridge_diagram()
    st.pyplot(about_bridge, use_container_width=True)
    plt.close()

    st.divider()

    left, right = st.columns(2)
    with left:
        st.info(
            "Modern bridges are evolving into intelligent systems that can sense, "
            "analyze, and adapt. This dataset represents the digital twin of a smart "
            "bridge — a virtual model that mirrors the real bridge behavior in real time. "
            "The data combines IoT-based sensor readings and simulated parameters to "
            "capture how the bridge responds to stress, traffic, and environmental conditions."
        )
        st.markdown(
            "**Key Applications:**\n"
            "- Predicting structural health and remaining useful life\n"
            "- Detecting anomalies and early signs of damage\n"
            "- Forecasting environmental or load-induced degradation\n"
            "- Developing AI-driven digital twin systems\n"
            "- Designing real-time monitoring dashboards"
        )

    with right:
        st.success(
            "🧠 **Our Novel Contribution:** We extended the original CNN-based SHM "
            "paper by modeling the bridge as a graph. GCN captures relationships "
            "between sensor domains which is architecturally more meaningful for "
            "structural systems than treating each sensor independently."
        )
        stats_df = pd.DataFrame(
            [
                ["Dataset",       "Digital Twin Bridge SHM (Kaggle)"],
                ["Features",      "25 sensor features"],
                ["Graph Nodes",   "5 monitoring domains"],
                ["Graph Edges",   "12 physical connections"],
                ["GCN Accuracy",  "99.92%"],
                ["CNN Accuracy",  "99.95%"],
                ["Cross-dataset", "99.28% (combined)"],
            ],
            columns=["Component", "Detail"],
        )
        st.dataframe(stats_df, use_container_width=True)

    st.divider()
    st.subheader("📊 Cross-Dataset Generalization Results")

    cross_df = pd.DataFrame({
        "Experiment": [
            "Train Real → Test Synthetic",
            "Train Synth → Test Real",
            "Train Combined → Test Combined",
        ],
        "Accuracy": [49.0, 0.61, 99.28],
    })
    cross_df["Bucket"] = np.where(cross_df["Accuracy"] > 90, ">90%", "<50%")
    fig_cross = px.bar(
        cross_df,
        x="Experiment",
        y="Accuracy",
        color="Bucket",
        color_discrete_map={">90%": "green", "<50%": "red"},
        title="Cross-Dataset Generalization",
        text="Accuracy",
    )
    fig_cross.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_cross.add_annotation(
        x=1, y=95,
        text="Domain shift detected between real and synthetic data",
        showarrow=False,
        font=dict(color="black"),
        bgcolor="rgba(255,255,255,0.7)",
    )
    fig_cross.update_layout(
        yaxis_range=[0, 110],
        height=420,
        margin=dict(l=20, r=20, t=70, b=20),
    )
    st.plotly_chart(fig_cross, use_container_width=True, key="cross_dataset_chart")

    st.caption(
        "Deep Learning Project | IEEE Paper Implementation with GCN Extension "
        "| Bridge Structural Health Monitoring"
    )