"""
Train the Graph Attention Network (GAT) Valuation Model using PyTorch Geometric.

Usage
-----
    python scripts/train_gnn_model.py

Requires
--------
    - data/processed/kc_house_data_engineered.csv
    - data/processed/kc_house_data_knn_edges.csv
    - PyTorch Geometric installed in the virtual environment
"""

from __future__ import annotations

import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from gnn_model import GATValuationModel  # noqa: E402

# ─────────────────────────── constants ───────────────────────────
RANDOM_SEED = 42
HIDDEN_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 3
DROPOUT = 0.2
LEARNING_RATE = 1e-3
EPOCHS = 150
PATIENCE = 15           # early stopping patience

DROP_COLS = ["geometry", "lat", "long", "zipcode", "id", "date",
             "price", "price_upper_cap", "is_price_outlier"]
TARGET_COL = "price_normalized"

ENGINEERED_CSV = os.path.join(PROJECT_ROOT, "data", "processed",
                              "kc_house_data_engineered.csv")
KNN_EDGES_CSV = os.path.join(PROJECT_ROOT, "data", "processed",
                             "kc_house_data_knn_edges.csv")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models",
                               "gat_model.pth")
SCALER_SAVE_PATH = os.path.join(PROJECT_ROOT, "models",
                                "gat_model_scaler.pkl")


# ─────────────────────────── data helpers ────────────────────────

def load_features(csv_path: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load and split engineered features into X / y arrays."""
    df = pd.read_csv(csv_path)

    drop = [c for c in DROP_COLS if c in df.columns]
    # Drop spatial features/embeddings to prevent 2-hop target leakage in GNN/Attention
    spatial_cols = [c for c in df.columns if c.startswith("spatial_emb_") or c.startswith("local_")]
    drop.extend(spatial_cols)
    
    feature_cols = [c for c in df.columns if c not in drop and c != TARGET_COL]

    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)
    ids = df["id"].astype(str).values if "id" in df.columns else np.arange(len(df)).astype(str)

    return df, X, y, ids, feature_cols


def build_edge_index(ids: np.ndarray, edges_csv: str) -> torch.Tensor:
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


# ─────────────────────────── training ────────────────────────────

def train() -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("Graph Attention Network (GAT) — Training")
    print("=" * 60)

    # ── 1. Load data ──
    if not os.path.exists(ENGINEERED_CSV):
        raise FileNotFoundError(f"Engineered CSV not found at {ENGINEERED_CSV}.")
    if not os.path.exists(KNN_EDGES_CSV):
        raise FileNotFoundError(f"KNN edges CSV not found at {KNN_EDGES_CSV}.")

    df_raw, X, y, ids, feature_cols = load_features(ENGINEERED_CSV)
    print(f"Dataset loaded: {len(X):,} properties, {X.shape[1]} features")

    # ── 2. Train / holdout split (same seed as XGBoost baseline) ──
    idx_all = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=0.2, random_state=RANDOM_SEED
    )

    # ── 3. Scale features and target ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    
    y_mean = float(y[idx_train].mean())
    y_std = float(y[idx_train].std())
    y_scaled = ((y - y_mean) / y_std).astype(np.float32)
    
    scalers = {
        "feature_scaler": scaler,
        "y_mean": y_mean,
        "y_std": y_std
    }
    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    joblib.dump(scalers, SCALER_SAVE_PATH)
    print(f"Scalers saved to {SCALER_SAVE_PATH}")

    # ── 4. Build edge index ──
    print("Building PyTorch Geometric edge index...")
    edge_index = build_edge_index(ids, KNN_EDGES_CSV)
    print(f"Graph constructed with {edge_index.shape[1]:,} directed edges.")

    # ── 5. Convert to tensors ──
    X_tensor = torch.from_numpy(X_scaled)
    y_tensor = torch.from_numpy(y_scaled)

    # ── 6. Build GAT model ──
    input_dim = X_scaled.shape[1]
    model = GATValuationModel(
        in_feats=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5
    )
    criterion = nn.HuberLoss()

    best_val_mape = float("inf")
    patience_counter = 0

    print(f"\nTraining for up to {EPOCHS} epochs (patience={PATIENCE})...")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val MAPE':>10}")
    print("-" * 34)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass on the full graph
        preds, _ = model(X_tensor, edge_index)
        loss = criterion(preds[idx_train], y_tensor[idx_train])
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # ── validation ──
        model.eval()
        with torch.no_grad():
            val_preds, _ = model(X_tensor, edge_index)
            
        # Denormalize predictions and targets to calculate correct MAPE in USD scale
        val_preds_usd = val_preds[idx_test].numpy() * y_std + y_mean
        y_test_usd = y[idx_test]
        
        val_mape = mean_absolute_percentage_error(y_test_usd, val_preds_usd)
        scheduler.step(val_mape)

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6}  {loss.item():>12.4f}  {val_mape * 100:>9.2f}%")

        # ── early stopping ──
        if val_mape < best_val_mape:
            best_val_mape = val_mape
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print("\n" + "=" * 60)
    print(f"GNN Training complete.")
    print(f"Best holdout MAPE : {best_val_mape * 100:.2f}%")
    print(f"Model saved to    : {MODEL_SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train()
