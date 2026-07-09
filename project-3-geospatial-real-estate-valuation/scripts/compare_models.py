"""
Compare XGBoost Baseline vs Spatial Attention Model vs GNN (GAT) Model on the holdout set.

Usage
-----
    python scripts/compare_models.py

Requires
--------
    - models/xgboost_regressor.pkl
    - models/spatial_attention_model.pth
    - models/spatial_model_scaler.pkl
    - models/gat_model.pth
    - models/gat_model_scaler.pkl
    - data/processed/kc_house_data_engineered.csv
    - data/processed/kc_house_data_knn_edges.csv
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from spatial_model import SpatialAttentionRegressor  # noqa: E402
from gnn_model import GATValuationModel  # noqa: E402

# ─────────────────────────── constants ───────────────────────────
RANDOM_SEED = 42
K_NEIGHBORS = 5

DROP_COLS = ["geometry", "lat", "long", "zipcode", "id", "date",
             "price", "price_upper_cap", "is_price_outlier"]
TARGET_COL = "price_normalized"

ENGINEERED_CSV = os.path.join(PROJECT_ROOT, "data", "processed",
                              "kc_house_data_engineered.csv")
KNN_EDGES_CSV  = os.path.join(PROJECT_ROOT, "data", "processed",
                              "kc_house_data_knn_edges.csv")
XGBOOST_PATH   = os.path.join(PROJECT_ROOT, "models", "xgboost_regressor.pkl")
SPATIAL_PATH   = os.path.join(PROJECT_ROOT, "models", "spatial_attention_model.pth")
SCALER_PATH    = os.path.join(PROJECT_ROOT, "models", "spatial_model_scaler.pkl")
GAT_PATH       = os.path.join(PROJECT_ROOT, "models", "gat_model.pth")
GAT_SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "gat_model_scaler.pkl")
REPORT_PATH    = os.path.join(PROJECT_ROOT, "docs", "model_comparison.md")


def load_data():
    df = pd.read_csv(ENGINEERED_CSV)
    ids = df["id"].astype(str).values if "id" in df.columns else np.arange(len(df)).astype(str)
    drop = [c for c in DROP_COLS if c in df.columns]
    # Drop spatial features/embeddings to match original model features and prevent shape mismatches
    spatial_cols = [c for c in df.columns if c.startswith("spatial_emb_") or c.startswith("local_")]
    drop.extend(spatial_cols)
    feature_cols = [c for c in df.columns if c not in drop and c != TARGET_COL]
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)
    return df, X, y, ids, feature_cols


def build_neighbor_index(ids, edges_csv, k):
    edges = pd.read_csv(edges_csv)
    adj = {str(i): [] for i in ids}
    for src, tgt in zip(edges["source_id"], edges["target_id"]):
        src_str = str(src)
        if src_str in adj and len(adj[src_str]) < k:
            adj[src_str].append(str(tgt))
    return adj


def build_edge_index(ids, edges_csv):
    """Build PyG edge_index tensor from the KNN edges CSV."""
    edges = pd.read_csv(edges_csv)
    id_to_idx = {str(pid): idx for idx, pid in enumerate(ids)}
    
    src_indices = []
    tgt_indices = []
    
    for src, tgt in zip(edges["source_id"], edges["target_id"]):
        src_str, tgt_str = str(src), str(tgt)
        if src_str in id_to_idx and tgt_str in id_to_idx:
            src_indices.append(id_to_idx[src_str])
            tgt_indices.append(id_to_idx[tgt_str])
            
    return torch.tensor([src_indices, tgt_indices], dtype=torch.long)


def make_neighbor_tensor(X_scaled, id_to_idx, ids, adj, k):
    n, feat_dim = X_scaled.shape
    neighbor_arr = np.zeros((n, k, feat_dim), dtype=np.float32)
    mask_arr = np.ones((n, k), dtype=bool)
    for i, pid in enumerate(ids):
        for j, nb_id in enumerate(adj.get(str(pid), [])[:k]):
            if nb_id in id_to_idx:
                neighbor_arr[i, j] = X_scaled[id_to_idx[nb_id]]
                mask_arr[i, j] = False
    return torch.from_numpy(neighbor_arr), torch.from_numpy(mask_arr)


def evaluate_xgboost(X_test_raw, y_test):
    """Evaluate the saved XGBoost model on the raw (unscaled) holdout features."""
    if not os.path.exists(XGBOOST_PATH):
        return None, None
    model = joblib.load(XGBOOST_PATH)
    # XGBoost was trained on the raw (unscaled) features
    preds = model.predict(X_test_raw)
    mape = mean_absolute_percentage_error(y_test, preds)
    rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
    return mape, rmse


def evaluate_spatial(X_scaled, y, ids, adj, idx_test, y_mean, y_std):
    """Evaluate the saved Spatial Attention model on the scaled holdout features."""
    if not os.path.exists(SPATIAL_PATH):
        return None, None

    id_to_idx = {str(pid): i for i, pid in enumerate(ids)}
    neighbor_features, neighbor_mask = make_neighbor_tensor(
        X_scaled, id_to_idx, ids, adj, K_NEIGHBORS
    )

    X_tensor = torch.from_numpy(X_scaled)
    y_test = y[idx_test]
    X_test = X_tensor[idx_test]
    nb_feat_test = neighbor_features[idx_test]
    nb_mask_test = neighbor_mask[idx_test]

    input_dim = X_scaled.shape[1]
    model = SpatialAttentionRegressor(input_dim=input_dim, hidden_dim=128, dropout=0.2)
    model.load_state_dict(torch.load(SPATIAL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    with torch.no_grad():
        preds, _ = model(X_test, nb_feat_test, nb_mask_test)

    # Denormalize predictions to raw USD scale
    preds_np = preds.numpy() * y_std + y_mean
    mape = mean_absolute_percentage_error(y_test, preds_np)
    rmse = float(np.sqrt(np.mean((y_test - preds_np) ** 2)))
    return mape, rmse


def evaluate_gat(X_scaled, y, edge_index, idx_test, y_mean, y_std):
    """Evaluate the GAT model on the holdout split."""
    if not os.path.exists(GAT_PATH):
        return None, None
        
    input_dim = X_scaled.shape[1]
    model = GATValuationModel(
        in_feats=input_dim,
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        dropout=0.2
    )
    model.load_state_dict(torch.load(GAT_PATH, map_location="cpu", weights_only=True))
    model.eval()
    
    X_tensor = torch.from_numpy(X_scaled)
    with torch.no_grad():
        preds, _ = model(X_tensor, edge_index)
        
    preds_np = preds[idx_test].numpy() * y_std + y_mean
    y_test = y[idx_test]
    
    mape = mean_absolute_percentage_error(y_test, preds_np)
    rmse = float(np.sqrt(np.mean((y_test - preds_np) ** 2)))
    return mape, rmse


def write_report(xgb_mape, xgb_rmse, spatial_mape, spatial_rmse, gat_mape, gat_rmse):
    models = {
        "XGBoost Baseline": (xgb_mape, xgb_rmse),
        "Spatial Attention (Manual)": (spatial_mape, spatial_rmse),
        "GNN (Graph Attention Network)": (gat_mape, gat_rmse)
    }
    
    # Filter out models that weren't run/trained
    valid_models = {k: v for k, v in models.items() if v[0] is not None}
    
    best_model_name = "N/A"
    if valid_models:
        best_model_name = min(valid_models.keys(), key=lambda k: valid_models[k][0])

    lines = [
        "# Model Comparison: XGBoost vs Spatial Attention vs GNN",
        "",
        "All metrics computed on the same **20% holdout split** (random_state=42).",
        "",
        "## Results",
        "",
        "| Model | MAPE | RMSE | Status |",
        "|-------|------|------|--------|",
    ]

    for model_name, (mape, rmse) in models.items():
        if mape is not None:
            lines.append(f"| {model_name} | {mape * 100:.2f}% | {rmse:,.0f} | Trained & Evaluated |")
        else:
            lines.append(f"| {model_name} | N/A | N/A | Not Found |")

    lines.extend([
        "",
        "## Verdict",
        "",
    ])

    if best_model_name != "N/A":
        best_mape = valid_models[best_model_name][0]
        xgb_val = valid_models.get("XGBoost Baseline", (None, None))[0]
        if xgb_val is not None:
            pct_improvement = (xgb_val - best_mape) / xgb_val * 100
            lines.append(f"✅ **{best_model_name}** is the top-performing model, achieving **{best_mape * 100:.2f}% MAPE**.")
            lines.append(f"This represents a **{pct_improvement:.2f}% relative improvement** over the tabular XGBoost baseline.")
        else:
            lines.append(f"✅ **{best_model_name}** is the top-performing model, achieving **{best_mape * 100:.2f}% MAPE**.")
    else:
        lines.append("No models have been trained or evaluated yet.")

    lines.extend([
        "",
        "## Regenerate",
        "",
        "```powershell",
        "python scripts/train_spatial_model.py",
        "python scripts/train_gnn_model.py",
        "python scripts/compare_models.py",
        "```",
    ])

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nReport saved to {REPORT_PATH}")


def main():
    print("=" * 60)
    print("Model Comparison: XGBoost vs Spatial vs GNN (GAT)")
    print("=" * 60)

    df, X, y, ids, feature_cols = load_data()
    adj = build_neighbor_index(ids, KNN_EDGES_CSV, K_NEIGHBORS)
    edge_index = build_edge_index(ids, KNN_EDGES_CSV)

    # Shared holdout split
    idx_all = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=0.2, random_state=RANDOM_SEED
    )

    # 1. Evaluate XGBoost
    print("\n-- XGBoost Baseline --")
    xgb_mape, xgb_rmse = evaluate_xgboost(X[idx_test], y[idx_test])
    if xgb_mape is not None:
        print(f"  MAPE : {xgb_mape * 100:.2f}%")
        print(f"  RMSE : {xgb_rmse:,.0f}")
    else:
        print("  Model not found. Run run_pipeline.py first.")

    # 2. Evaluate Spatial Attention (Manual)
    print("\n-- Spatial Attention Model --")
    spatial_scaler_pkg = None
    y_mean_spatial, y_std_spatial = 0.0, 1.0
    if os.path.exists(SCALER_PATH):
        scalers_dict = joblib.load(SCALER_PATH)
        if isinstance(scalers_dict, dict):
            spatial_scaler = scalers_dict["feature_scaler"]
            y_mean_spatial = scalers_dict["y_mean"]
            y_std_spatial = scalers_dict["y_std"]
        else:
            spatial_scaler = scalers_dict
        
        X_scaled_spatial = spatial_scaler.transform(X).astype(np.float32)
        spatial_mape, spatial_rmse = evaluate_spatial(
            X_scaled_spatial, y, ids, adj, idx_test, y_mean_spatial, y_std_spatial
        )
        print(f"  MAPE : {spatial_mape * 100:.2f}%")
        print(f"  RMSE : {spatial_rmse:,.0f}")
    else:
        spatial_mape, spatial_rmse = None, None
        print("  Scaler/Model not found. Run scripts/train_spatial_model.py first.")

    # 3. Evaluate GNN (GAT)
    print("\n-- GNN (Graph Attention Network) --")
    y_mean_gat, y_std_gat = 0.0, 1.0
    if os.path.exists(GAT_SCALER_PATH):
        scalers_dict_gat = joblib.load(GAT_SCALER_PATH)
        gat_scaler = scalers_dict_gat["feature_scaler"]
        y_mean_gat = scalers_dict_gat["y_mean"]
        y_std_gat = scalers_dict_gat["y_std"]
        
        X_scaled_gat = gat_scaler.transform(X).astype(np.float32)
        gat_mape, gat_rmse = evaluate_gat(
            X_scaled_gat, y, edge_index, idx_test, y_mean_gat, y_std_gat
        )
        print(f"  MAPE : {gat_mape * 100:.2f}%")
        print(f"  RMSE : {gat_rmse:,.0f}")
    else:
        gat_mape, gat_rmse = None, None
        print("  Scaler/Model not found. Run scripts/train_gnn_model.py first.")

    # 4. Write Markdown Report
    write_report(xgb_mape, xgb_rmse, spatial_mape, spatial_rmse, gat_mape, gat_rmse)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
