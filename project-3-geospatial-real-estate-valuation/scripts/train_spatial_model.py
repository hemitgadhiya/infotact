"""
Train the Spatial Attention Valuation Model.

Usage
-----
    python scripts/train_spatial_model.py

Requires
--------
    - data/processed/kc_house_data_engineered.csv  (run run_pipeline.py first)
    - PyTorch installed in the virtual environment
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from spatial_model import SpatialAttentionRegressor  # noqa: E402

# ─────────────────────────── constants ───────────────────────────
RANDOM_SEED = 42
K_NEIGHBORS = 5         # must match the K used when building knn edges
HIDDEN_DIM = 128
DROPOUT = 0.2
LEARNING_RATE = 1e-3
EPOCHS = 60
BATCH_SIZE = 512
PATIENCE = 10           # early stopping patience

DROP_COLS = ["geometry", "lat", "long", "zipcode", "id", "date",
             "price", "price_upper_cap", "is_price_outlier"]
TARGET_COL = "price_normalized"

ENGINEERED_CSV = os.path.join(PROJECT_ROOT, "data", "processed",
                              "kc_house_data_engineered.csv")
KNN_EDGES_CSV = os.path.join(PROJECT_ROOT, "data", "processed",
                             "kc_house_data_knn_edges.csv")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models",
                               "spatial_attention_model.pth")
SCALER_SAVE_PATH = os.path.join(PROJECT_ROOT, "models",
                                "spatial_model_scaler.pkl")


# ─────────────────────────── data helpers ────────────────────────

def load_features(csv_path: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Load and split engineered features into X / y arrays."""
    df = pd.read_csv(csv_path)

    drop = [c for c in DROP_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop and c != TARGET_COL]

    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)
    ids = df["id"].astype(str).values if "id" in df.columns else np.arange(len(df)).astype(str)

    return df, X, y, ids, feature_cols


def build_neighbor_index(
    ids: np.ndarray, edges_csv: str, k: int
) -> dict[str, list[str]]:
    """Build {source_id -> [neighbor_id, ...]} mapping from the KNN edges CSV."""
    edges = pd.read_csv(edges_csv)
    adj: dict[str, list[str]] = {str(i): [] for i in ids}
    for src, tgt in zip(edges["source_id"], edges["target_id"]):
        src_str = str(src)
        if src_str in adj and len(adj[src_str]) < k:
            adj[src_str].append(str(tgt))
    return adj


def make_neighbor_tensor(
    X_scaled: np.ndarray,
    id_to_idx: dict[str, int],
    ids: np.ndarray,
    adj: dict[str, list[str]],
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns
    -------
    neighbor_features : FloatTensor (n, k, feat_dim)
    neighbor_mask     : BoolTensor  (n, k)  — True where neighbor is padded
    """
    n, feat_dim = X_scaled.shape
    neighbor_arr = np.zeros((n, k, feat_dim), dtype=np.float32)
    mask_arr = np.ones((n, k), dtype=bool)  # start all masked

    for i, prop_id in enumerate(ids):
        neighbors = adj.get(str(prop_id), [])
        for j, nb_id in enumerate(neighbors[:k]):
            if nb_id in id_to_idx:
                nb_idx = id_to_idx[nb_id]
                neighbor_arr[i, j] = X_scaled[nb_idx]
                mask_arr[i, j] = False  # valid neighbor

    return torch.from_numpy(neighbor_arr), torch.from_numpy(mask_arr)


# ─────────────────────────── training ────────────────────────────

def train() -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("Spatial Attention Regressor — Training")
    print("=" * 60)

    # ── 1. Load data ──
    if not os.path.exists(ENGINEERED_CSV):
        raise FileNotFoundError(
            f"Engineered CSV not found at {ENGINEERED_CSV}.\n"
            "Run  python run_pipeline.py  first."
        )
    if not os.path.exists(KNN_EDGES_CSV):
        raise FileNotFoundError(
            f"KNN edges CSV not found at {KNN_EDGES_CSV}.\n"
            "Run  python run_pipeline.py  first."
        )

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

    # ── 4. Build adjacency and neighbor tensors ──
    adj = build_neighbor_index(ids, KNN_EDGES_CSV, K_NEIGHBORS)
    id_to_idx = {str(pid): i for i, pid in enumerate(ids)}

    print(f"Building neighbor tensors (K={K_NEIGHBORS})...")
    neighbor_features, neighbor_mask = make_neighbor_tensor(
        X_scaled, id_to_idx, ids, adj, K_NEIGHBORS
    )

    # ── 5. Convert to tensors ──
    X_tensor = torch.from_numpy(X_scaled)
    y_tensor = torch.from_numpy(y_scaled)

    X_train = X_tensor[idx_train]
    y_train = y_tensor[idx_train]
    nb_feat_train = neighbor_features[idx_train]
    nb_mask_train = neighbor_mask[idx_train]

    X_test = X_tensor[idx_test]
    y_test = y_tensor[idx_test]
    nb_feat_test = neighbor_features[idx_test]
    nb_mask_test = neighbor_mask[idx_test]

    # ── 6. Build model ──
    input_dim = X_scaled.shape[1]
    model = SpatialAttentionRegressor(
        input_dim=input_dim, hidden_dim=HIDDEN_DIM, dropout=DROPOUT
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5
    )
    criterion = nn.MSELoss()

    n_train = len(idx_train)
    n_batches = (n_train + BATCH_SIZE - 1) // BATCH_SIZE
    best_val_mape = float("inf")
    patience_counter = 0

    print(f"\nTraining for up to {EPOCHS} epochs (batch_size={BATCH_SIZE}, patience={PATIENCE})...")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val MAPE':>10}")
    print("-" * 34)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0

        for b in range(n_batches):
            batch_idx = perm[b * BATCH_SIZE: (b + 1) * BATCH_SIZE]
            bx = X_train[batch_idx]
            by = y_train[batch_idx]
            bnf = nb_feat_train[batch_idx]
            bnm = nb_mask_train[batch_idx]

            optimizer.zero_grad()
            preds, _ = model(bx, bnf, bnm)
            loss = criterion(preds, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item() * len(batch_idx)

        avg_loss = epoch_loss / n_train

        # ── validation ──
        model.eval()
        with torch.no_grad():
            val_preds, _ = model(X_test, nb_feat_test, nb_mask_test)
        # Denormalize predictions and targets to calculate correct MAPE in USD scale
        val_preds_usd = val_preds.numpy() * y_std + y_mean
        y_test_usd = y_test.numpy() * y_std + y_mean
        val_mape = mean_absolute_percentage_error(
            y_test_usd, val_preds_usd
        )
        scheduler.step(val_mape)

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6}  {avg_loss:>12.2f}  {val_mape * 100:>9.2f}%")

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
    print(f"Training complete.")
    print(f"Best holdout MAPE : {best_val_mape * 100:.2f}%")
    print(f"Model saved to    : {MODEL_SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train()
