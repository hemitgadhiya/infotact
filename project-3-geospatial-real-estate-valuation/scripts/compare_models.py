"""
Compare XGBoost Baseline vs Spatial Attention Model on the holdout set.

Usage
-----
    python scripts/compare_models.py

Requires
--------
    - models/xgboost_regressor.pkl
    - models/spatial_attention_model.pth
    - models/spatial_model_scaler.pkl
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


def evaluate_xgboost(X_test_raw, y_test, feature_cols):
    """Evaluate the saved XGBoost model on the raw (unscaled) holdout features."""
    if not os.path.exists(XGBOOST_PATH):
        return None, None
    model = joblib.load(XGBOOST_PATH)
    # XGBoost was trained on the raw (unscaled) features
    preds = model.predict(X_test_raw)
    mape = mean_absolute_percentage_error(y_test, preds)
    rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
    return mape, rmse


def evaluate_spatial(X_scaled, y, ids, adj, feature_cols, idx_test, y_mean, y_std):
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


def write_report(xgb_mape, xgb_rmse, spatial_mape, spatial_rmse):
    spatial_beats = (
        spatial_mape is not None and xgb_mape is not None and spatial_mape < xgb_mape
    )
    improvement = (
        f"{(xgb_mape - spatial_mape) / xgb_mape * 100:.2f}%"
        if (spatial_mape is not None and xgb_mape is not None)
        else "N/A"
    )

    lines = [
        "# Model Comparison: XGBoost Baseline vs Spatial Attention Model",
        "",
        "All metrics computed on the same **20% holdout split** (random_state=42).",
        "",
        "## Results",
        "",
        "| Model | MAPE | RMSE |",
        "|-------|------|------|",
        f"| XGBoost Baseline | {xgb_mape * 100:.2f}% | {xgb_rmse:,.0f} |"
        if xgb_mape is not None else "| XGBoost Baseline | N/A | N/A |",
        f"| Spatial Attention | {spatial_mape * 100:.2f}% | {spatial_rmse:,.0f} |"
        if spatial_mape is not None else "| Spatial Attention | N/A (not trained) | N/A |",
        "",
        "## Verdict",
        "",
    ]

    if spatial_beats:
        lines += [
            f"✅ **Spatial model outperforms XGBoost** — MAPE improved by **{improvement}** "
            f"({xgb_mape * 100:.2f}% → {spatial_mape * 100:.2f}%).",
            "",
            "This confirms that incorporating neighborhood graph context via attention "
            "adds predictive value beyond standard tabular features.",
        ]
    elif spatial_mape is not None:
        lines += [
            f"⚠️ XGBoost baseline still leads by **{improvement}** in MAPE. "
            "Consider training the spatial model for more epochs or tuning hyper-parameters.",
        ]
    else:
        lines += [
            "ℹ️ Spatial model not yet trained. Run `scripts/train_spatial_model.py` first.",
        ]

    lines += [
        "",
        "## Regenerate",
        "",
        "```powershell",
        "python scripts/train_spatial_model.py",
        "python scripts/compare_models.py",
        "```",
    ]

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nReport saved to {REPORT_PATH}")


def main():
    print("=" * 60)
    print("Model Comparison: XGBoost vs Spatial Attention")
    print("=" * 60)

    df, X, y, ids, feature_cols = load_data()
    adj = build_neighbor_index(ids, KNN_EDGES_CSV, K_NEIGHBORS)

    # Shared holdout split
    idx_all = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=0.2, random_state=RANDOM_SEED
    )

    # Scale for spatial model
    scaler: StandardScaler | None = None
    y_mean = 0.0
    y_std = 1.0
    if os.path.exists(SCALER_PATH):
        scalers_dict = joblib.load(SCALER_PATH)
        if isinstance(scalers_dict, dict):
            scaler = scalers_dict["feature_scaler"]
            y_mean = scalers_dict["y_mean"]
            y_std = scalers_dict["y_std"]
        else:
            scaler = scalers_dict
    else:
        scaler = StandardScaler().fit(X[idx_train])

    X_scaled = scaler.transform(X).astype(np.float32)

    print("\n-- XGBoost Baseline --")
    xgb_mape, xgb_rmse = evaluate_xgboost(X[idx_test], y[idx_test], feature_cols)
    if xgb_mape is not None:
        print(f"  MAPE : {xgb_mape * 100:.2f}%")
        print(f"  RMSE : {xgb_rmse:,.0f}")
    else:
        print("  Model not found. Run run_pipeline.py first.")

    print("\n-- Spatial Attention Model --")
    spatial_mape, spatial_rmse = evaluate_spatial(
        X_scaled, y, ids, adj, feature_cols, idx_test, y_mean, y_std
    )
    if spatial_mape is not None:
        print(f"  MAPE : {spatial_mape * 100:.2f}%")
        print(f"  RMSE : {spatial_rmse:,.0f}")
    else:
        print("  Model not found. Run scripts/train_spatial_model.py first.")

    write_report(xgb_mape, xgb_rmse, spatial_mape, spatial_rmse)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
