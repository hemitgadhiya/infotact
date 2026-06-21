"""
Spatial Attention Valuation Model.

Architecture:
    1. A shared MLP tabular encoder: encodes each property's tabular features into
       a compact embedding vector.
    2. A neighbor attention aggregator: for a target property, attends over its
       K-nearest neighbors' embeddings and produces a weighted context vector.
    3. A final regression head: concatenates [target_embedding, context_vector]
       and predicts price_normalized.

This avoids heavy GNN library dependencies (DGL / PyTorch Geometric) and is
fully compatible with standard CPU PyTorch, making it easy to install and run
on Windows without CUDA.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TabularEncoder(nn.Module):
    """
    Multi-layer perceptron that encodes a flat property feature vector into an
    embedding of dimension `hidden_dim`.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch_size, input_dim)

        Returns
        -------
        Tensor of shape (batch_size, hidden_dim)
        """
        return self.net(x)


class NeighborAttentionAggregator(nn.Module):
    """
    Single-head dot-product attention over neighbor embeddings.

    Given a target embedding and a set of neighbor embeddings, computes
    attention weights and returns the weighted sum of neighbor embeddings
    (the context vector).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # Learnable query projection from target embedding
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Learnable key projection from neighbor embeddings
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** 0.5

    def forward(
        self,
        target_emb: torch.Tensor,
        neighbor_embs: torch.Tensor,
        neighbor_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        target_emb   : Tensor of shape (batch_size, hidden_dim)
        neighbor_embs: Tensor of shape (batch_size, k, hidden_dim)
        neighbor_mask: BoolTensor of shape (batch_size, k), True = padded / invalid neighbor

        Returns
        -------
        context : Tensor of shape (batch_size, hidden_dim)   — aggregated neighbor context
        weights : Tensor of shape (batch_size, k)            — attention weights
        """
        # Query: (batch_size, 1, hidden_dim)
        q = self.query_proj(target_emb).unsqueeze(1)
        # Keys: (batch_size, k, hidden_dim)
        k = self.key_proj(neighbor_embs)

        # Dot-product attention scores: (batch_size, 1, k) -> (batch_size, k)
        scores = torch.bmm(q, k.transpose(1, 2)).squeeze(1) / self.scale

        if neighbor_mask is not None:
            scores = scores.masked_fill(neighbor_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        # Replace NaN (all-masked rows) with uniform weights
        weights = torch.nan_to_num(weights, nan=0.0)

        # Weighted sum: (batch_size, hidden_dim)
        context = torch.bmm(weights.unsqueeze(1), neighbor_embs).squeeze(1)
        return context, weights


class SpatialAttentionRegressor(nn.Module):
    """
    Full spatial attention model for real estate price prediction.

    Workflow
    --------
    1. Encode *all* properties in the batch with ``TabularEncoder``.
    2. For each target property, gather neighbor embeddings from step 1.
    3. Run ``NeighborAttentionAggregator`` to produce a context vector.
    4. Concatenate [target_emb, context_vector] and regress to price.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = TabularEncoder(input_dim, hidden_dim, dropout)
        self.attention = NeighborAttentionAggregator(hidden_dim)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        target_features: torch.Tensor,
        neighbor_features: torch.Tensor,
        neighbor_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        target_features  : Tensor (batch_size, input_dim)        — target property features
        neighbor_features: Tensor (batch_size, k, input_dim)     — neighbor property features
        neighbor_mask    : BoolTensor (batch_size, k), optional  — True = missing/padded neighbor

        Returns
        -------
        predictions : Tensor (batch_size,)  — predicted price_normalized values
        attn_weights: Tensor (batch_size, k) — attention weights per neighbor
        """
        batch_size, k, feat_dim = neighbor_features.shape

        # Encode target: (batch_size, hidden_dim)
        target_emb = self.encoder(target_features)

        # Encode neighbors: flatten to (batch_size*k, feat_dim), encode, reshape
        neighbor_flat = neighbor_features.view(batch_size * k, feat_dim)
        neighbor_embs = self.encoder(neighbor_flat).view(batch_size, k, -1)

        # Attend over neighbors
        context, attn_weights = self.attention(target_emb, neighbor_embs, neighbor_mask)

        # Concatenate and regress
        combined = torch.cat([target_emb, context], dim=1)
        predictions = self.regressor(combined).squeeze(1)

        return predictions, attn_weights
