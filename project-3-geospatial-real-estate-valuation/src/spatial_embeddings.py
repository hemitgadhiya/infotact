"""Spatial embedding generation from the KNN graph representing neighborhood pricing context."""

from __future__ import annotations

import logging
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def compute_neighborhood_features(
    df: pd.DataFrame,
    adj: dict[str, list[tuple[str, float]]],
    id_col: str = "id",
    price_col: str = "price_normalized",
    sqft_col: str = "sqft_living",
    age_col: str = "house_age",
    epsilon: float = 0.01,
) -> pd.DataFrame:
    """Compute raw neighborhood statistics for each property from its nearest neighbors.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame of property records.
    adj : dict
        Mapping of source_id -> list of tuples (target_id, distance_km).
    id_col : str, default 'id'
        Column name of the unique identifier.
    price_col : str, default 'price_normalized'
        Column name of the target price to aggregate.
    sqft_col : str, default 'sqft_living'
        Column name of the square footage.
    age_col : str, default 'house_age'
        Column name of the house age.
    epsilon : float, default 0.01
        Value added to physical distance to prevent division by zero in weighting.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the raw neighborhood pricing context features with the 'id' column.
    """
    # Create dictionary lookups for O(1) retrieval speed
    id_to_price = df.set_index(id_col)[price_col].to_dict()
    id_to_sqft = df.set_index(id_col)[sqft_col].to_dict()
    id_to_age = df.set_index(id_col)[age_col].to_dict()

    features = []
    for node_id in df[id_col].astype(str):
        neighbors = adj.get(node_id, [])
        if not neighbors:
            features.append({
                id_col: node_id,
                "local_price_mean": 0.0,
                "local_price_median": 0.0,
                "local_price_std": 0.0,
                "local_price_min": 0.0,
                "local_price_max": 0.0,
                "local_price_weighted_mean": 0.0,
                "local_price_per_sqft_mean": 0.0,
                "local_price_per_sqft_std": 0.0,
                "local_sqft_living_mean": 0.0,
                "local_sqft_living_std": 0.0,
                "local_house_age_mean": 0.0,
                "local_distance_mean": 0.0,
                "local_distance_std": 0.0,
            })
            continue

        neighbor_prices = []
        neighbor_sqfts = []
        neighbor_ages = []
        neighbor_dists = []
        neighbor_price_per_sqft = []
        weights = []
        weighted_prices = []

        for nb_id, dist in neighbors:
            if nb_id in id_to_price:
                p = id_to_price[nb_id]
                sqft = id_to_sqft[nb_id]
                age = id_to_age[nb_id]

                neighbor_prices.append(p)
                neighbor_sqfts.append(sqft)
                neighbor_ages.append(age)
                neighbor_dists.append(dist)

                p_per_sqft = p / max(1.0, sqft)
                neighbor_price_per_sqft.append(p_per_sqft)

                w = 1.0 / (dist + epsilon)
                weights.append(w)
                weighted_prices.append(w * p)

        if not neighbor_prices:
            features.append({
                id_col: node_id,
                "local_price_mean": 0.0,
                "local_price_median": 0.0,
                "local_price_std": 0.0,
                "local_price_min": 0.0,
                "local_price_max": 0.0,
                "local_price_weighted_mean": 0.0,
                "local_price_per_sqft_mean": 0.0,
                "local_price_per_sqft_std": 0.0,
                "local_sqft_living_mean": 0.0,
                "local_sqft_living_std": 0.0,
                "local_house_age_mean": 0.0,
                "local_distance_mean": 0.0,
                "local_distance_std": 0.0,
            })
            continue

        prices_arr = np.array(neighbor_prices)
        sqfts_arr = np.array(neighbor_sqfts)
        ages_arr = np.array(neighbor_ages)
        dists_arr = np.array(neighbor_dists)
        p_per_sqft_arr = np.array(neighbor_price_per_sqft)

        sum_w = sum(weights)
        weighted_mean_price = sum(weighted_prices) / sum_w if sum_w > 0 else np.mean(prices_arr)

        features.append({
            id_col: node_id,
            "local_price_mean": float(np.mean(prices_arr)),
            "local_price_median": float(np.median(prices_arr)),
            "local_price_std": float(np.std(prices_arr)) if len(prices_arr) > 1 else 0.0,
            "local_price_min": float(np.min(prices_arr)),
            "local_price_max": float(np.max(prices_arr)),
            "local_price_weighted_mean": float(weighted_mean_price),
            "local_price_per_sqft_mean": float(np.mean(p_per_sqft_arr)),
            "local_price_per_sqft_std": float(np.std(p_per_sqft_arr)) if len(p_per_sqft_arr) > 1 else 0.0,
            "local_sqft_living_mean": float(np.mean(sqfts_arr)),
            "local_sqft_living_std": float(np.std(sqfts_arr)) if len(sqfts_arr) > 1 else 0.0,
            "local_house_age_mean": float(np.mean(ages_arr)),
            "local_distance_mean": float(np.mean(dists_arr)),
            "local_distance_std": float(np.std(dists_arr)) if len(dists_arr) > 1 else 0.0,
        })

    return pd.DataFrame(features)


def generate_spatial_embeddings(
    df: pd.DataFrame,
    edges_df: pd.DataFrame,
    n_components: int = 8,
    id_col: str = "id",
    price_col: str = "price_normalized",
    scaler_path: str | None = None,
    pca_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate neighborhood spatial features and compress them into dense embeddings.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing property records.
    edges_df : pd.DataFrame
        KNN graph edges containing columns: ['source_id', 'target_id', 'distance_km'].
    n_components : int, default 8
        Number of PCA components to keep for dense embeddings.
    id_col : str, default 'id'
        Column name representing the unique property identifier.
    price_col : str, default 'price_normalized'
        Column name representing normalized price.
    scaler_path : str, optional
        Path to save/load the StandardScaler.
    pca_path : str, optional
        Path to save/load the PCA model.

    Returns
    -------
    embeddings_df : pd.DataFrame
        DataFrame with columns [id_col, 'spatial_emb_0', ..., 'spatial_emb_{n-1}'].
    raw_features_df : pd.DataFrame
        DataFrame containing the raw neighborhood pricing context features.
    """
    # 1. Build adjacency mapping
    ids = df[id_col].astype(str).values
    adj = {str(i): [] for i in ids}
    for _, row in edges_df.iterrows():
        src = str(row["source_id"])
        tgt = str(row["target_id"])
        dist = float(row["distance_km"])
        if src in adj:
            adj[src].append((tgt, dist))

    # 2. Extract raw neighborhood features
    raw_features_df = compute_neighborhood_features(
        df=df,
        adj=adj,
        id_col=id_col,
        price_col=price_col,
    )

    # 3. Fit/apply Scaler and PCA
    feature_cols = [c for c in raw_features_df.columns if c != id_col]
    X = raw_features_df[feature_cols].values

    # Handle loading existing scaler/pca
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        if scaler_path:
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(scaler, scaler_path)

    if pca_path and os.path.exists(pca_path):
        pca = joblib.load(pca_path)
        X_pca = pca.transform(X_scaled)
    else:
        actual_components = min(n_components, X_scaled.shape[1])
        pca = PCA(n_components=actual_components)
        X_pca = pca.fit_transform(X_scaled)
        if pca_path:
            os.makedirs(os.path.dirname(pca_path), exist_ok=True)
            joblib.dump(pca, pca_path)

    emb_cols = [f"spatial_emb_{i}" for i in range(X_pca.shape[1])]
    embeddings_df = pd.DataFrame(X_pca, columns=emb_cols)
    embeddings_df.insert(0, id_col, raw_features_df[id_col].values)

    return embeddings_df, raw_features_df
