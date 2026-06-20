"""Graph construction module for geospatially located real estate properties."""

import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

logger = logging.getLogger(__name__)


def build_knn_graph(
    df: pd.DataFrame,
    k: int = 5,
    earth_radius: float = 6371.0,
    id_col: str = "id",
    lat_col: str = "lat",
    long_col: str = "long",
) -> dict:
    """Convert a dataset of properties with latitude and longitude into a K-Nearest Neighbors graph.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing property records.
    k : int, default 5
        Number of nearest neighbors to connect for each node.
    earth_radius : float, default 6371.0
        Earth radius in kilometers. Use 3958.8 for miles.
    id_col : str, default 'id'
        Column name representing the unique property identifier.
    lat_col : str, default 'lat'
        Column name representing latitude in degrees.
    long_col : str, default 'long'
        Column name representing longitude in degrees.

    Returns
    -------
    dict
        A dictionary containing:
        - "edges_df": pd.DataFrame of graph edges with columns:
            `['source_id', 'target_id', 'distance_km']`
        - "adjacency_list": dict mapping source_id to a list of tuples `(target_id, distance_km)`
        - "k_used": int, the number of neighbors actually used (may be capped based on dataset size)
    """
    # 1. Input validations
    for col in [lat_col, long_col]:
        if col not in df.columns:
            raise KeyError(f"Geospatial coordinate column '{col}' not found in DataFrame.")

    # Validate coordinate ranges
    if not df[lat_col].between(-90.0, 90.0).all():
        raise ValueError("Latitude values must be between -90 and 90 degrees.")
    if not df[long_col].between(-180.0, 180.0).all():
        raise ValueError("Longitude values must be between -180 and 180 degrees.")

    # Determine unique identifiers
    if id_col in df.columns:
        node_ids = df[id_col].astype(str).values
    else:
        logger.warning(
            f"Unique identifier column '{id_col}' not found in DataFrame. Using row index as ID."
        )
        node_ids = np.array([str(idx) for idx in df.index])

    n_samples = len(df)
    if n_samples == 0:
        return {
            "edges_df": pd.DataFrame(columns=["source_id", "target_id", "distance_km"]),
            "adjacency_list": {},
            "k_used": 0,
        }

    # 2. Configure K based on number of samples
    # A node cannot connect to itself, so the maximum number of neighbors is n_samples - 1.
    max_k = max(0, n_samples - 1)
    k_used = min(k, max_k)

    if k_used < k:
        logger.warning(
            f"Requested K={k} but dataset only has {n_samples} rows. Capping K to {k_used}."
        )

    if k_used == 0:
        return {
            "edges_df": pd.DataFrame(columns=["source_id", "target_id", "distance_km"]),
            "adjacency_list": {node_id: [] for node_id in node_ids},
            "k_used": 0,
        }

    # 3. Fit BallTree with Haversine metric
    # BallTree haversine metric requires coordinates in radians, in the order [lat, long]
    coords_rad = np.radians(df[[lat_col, long_col]].values)
    tree = BallTree(coords_rad, metric="haversine")

    # Query for K + 1 nearest neighbors because the nearest neighbor to a point is always itself
    # (distance 0.0), which we will filter out to prevent self-loops.
    distances_rad, indices = tree.query(coords_rad, k=k_used + 1)

    # Convert distance from radians to kilometers (or miles based on earth_radius)
    distances_km = distances_rad * earth_radius

    # 4. Construct edge list and adjacency list
    edges = []
    adj_list = {}

    for i in range(n_samples):
        source_id = node_ids[i]
        adj_list[source_id] = []

        # Filter out self-loop (where index is the current node i)
        node_neighbors = []
        for neighbor_idx, dist in zip(indices[i], distances_km[i]):
            if neighbor_idx == i:
                continue
            node_neighbors.append((node_ids[neighbor_idx], float(dist)))
            
            # Stop once we have gathered k_used neighbors
            if len(node_neighbors) == k_used:
                break

        # Append edges and populate adjacency list
        for target_id, dist in node_neighbors:
            edges.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "distance_km": dist,
                }
            )
            adj_list[source_id].append((target_id, dist))

    edges_df = pd.DataFrame(edges)

    return {
        "edges_df": edges_df,
        "adjacency_list": adj_list,
        "k_used": k_used,
    }
