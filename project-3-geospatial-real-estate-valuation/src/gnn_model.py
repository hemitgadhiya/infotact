"""
Graph Attention Network (GAT) for Real Estate Valuation using PyTorch Geometric.

Architecture:
    1. Input projection: maps raw node features to hidden_dim.
    2. Three GATConv layers with multi-head attention for message-passing
       over the KNN neighborhood graph.
    3. Residual (skip) connections for stable gradient flow.
    4. An MLP regression head that predicts price_normalized.

This model uses PyG's GATConv to perform true graph-based message-passing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GATValuationModel(nn.Module):
    """
    Multi-layer Graph Attention Network for property price regression.

    Parameters
    ----------
    in_feats : int
        Dimensionality of input node features.
    hidden_dim : int
        Hidden embedding size per attention head.
    num_heads : int
        Number of attention heads in intermediate GAT layers.
    num_layers : int
        Total number of GATConv layers (minimum 2).
    dropout : float
        Dropout rate applied after each GAT layer and in the MLP head.
    """

    def __init__(
        self,
        in_feats: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # ── GAT convolution layers ──────────────────────────────────
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skip_projs = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                # First layer: raw features → hidden_dim * num_heads
                layer_in = in_feats
                layer_out = hidden_dim
                heads = num_heads
                self.gat_layers.append(
                    GATConv(
                        layer_in,
                        layer_out,
                        heads=heads,
                        dropout=dropout,
                        concat=True,
                        add_self_loops=True,
                    )
                )
                out_dim = hidden_dim * num_heads
                self.norms.append(nn.BatchNorm1d(out_dim))
                # Skip projection to match concatenated head dimension
                self.skip_projs.append(
                    nn.Linear(layer_in, out_dim) if layer_in != out_dim else nn.Identity()
                )

            elif i < num_layers - 1:
                # Intermediate layers: hidden_dim * num_heads → hidden_dim * num_heads
                layer_in = hidden_dim * num_heads
                layer_out = hidden_dim
                heads = num_heads
                self.gat_layers.append(
                    GATConv(
                        layer_in,
                        layer_out,
                        heads=heads,
                        dropout=dropout,
                        concat=True,
                        add_self_loops=True,
                    )
                )
                out_dim = hidden_dim * num_heads
                self.norms.append(nn.BatchNorm1d(out_dim))
                self.skip_projs.append(nn.Identity())  # Dimensions already match

            else:
                # Final GAT layer: hidden_dim * num_heads → hidden_dim  (average heads)
                layer_in = hidden_dim * num_heads
                layer_out = hidden_dim
                heads = 1  # Single head, averaged
                self.gat_layers.append(
                    GATConv(
                        layer_in,
                        layer_out,
                        heads=heads,
                        dropout=dropout,
                        concat=False,
                        add_self_loops=True,
                    )
                )
                out_dim = hidden_dim
                self.norms.append(nn.BatchNorm1d(out_dim))
                self.skip_projs.append(
                    nn.Linear(layer_in, out_dim) if layer_in != out_dim else nn.Identity()
                )

        # ── MLP regression head ─────────────────────────────────────
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._last_attention_weights = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """
        Forward pass through the GAT model.

        Parameters
        ----------
        x : Tensor (num_nodes, in_feats)
            Input node features.
        edge_index : Tensor (2, num_edges)
            Edge index representing connections.
        return_attention : bool
            If True, collect and return per-layer attention weights.

        Returns
        -------
        predictions : Tensor (num_nodes,)
            Predicted price values (one per node).
        attention_weights : list[tuple[Tensor, Tensor]] | None
            Per-layer attention weight tuples (edge_index, weights) if return_attention=True,
            otherwise None.
        """
        h = x
        attn_weights = [] if return_attention else None

        for i, (gat_layer, norm, skip) in enumerate(
            zip(self.gat_layers, self.norms, self.skip_projs)
        ):
            identity = skip(h)

            if return_attention:
                h_new, (edge_index_out, attn) = gat_layer(h, edge_index, return_attention_weights=True)
                attn_weights.append((edge_index_out, attn))
            else:
                h_new = gat_layer(h, edge_index)

            # BatchNorm + residual + activation
            h_new = norm(h_new)
            h_new = h_new + identity
            h_new = torch.nn.functional.elu(h_new)

            if i < self.num_layers - 1:
                h_new = self.dropout(h_new)

            h = h_new

        # Store attention weights for later analysis (e.g., top-K neighbors)
        self._last_attention_weights = attn_weights

        # Regression head
        predictions = self.regressor(h).squeeze(-1)

        return predictions, attn_weights

    def get_neighbor_attention(
        self,
        node_idx: int,
        layer_idx: int = -1,
    ) -> list[tuple[int, float]]:
        """
        Extract attention weights for a specific node's neighbors.

        Parameters
        ----------
        node_idx : int
            Index of the target node.
        layer_idx : int
            Which GAT layer's attention to return (-1 = last layer).

        Returns
        -------
        list of (neighbor_idx, attention_weight) tuples, sorted by weight descending.
        """
        if self._last_attention_weights is None:
            raise RuntimeError(
                "No attention weights available. Run forward() with return_attention=True first."
            )

        edge_index, attn = self._last_attention_weights[layer_idx]
        # Average across heads
        if len(attn.shape) > 1:
            attn_avg = attn.mean(dim=-1)
        else:
            attn_avg = attn

        # Find edges where the target node is the destination (edge_index[1] is target)
        edge_mask = edge_index[1] == node_idx
        neighbor_indices = edge_index[0][edge_mask].tolist()
        neighbor_weights = attn_avg[edge_mask].tolist()

        # Sort by weight descending
        pairs = sorted(
            zip(neighbor_indices, neighbor_weights), key=lambda x: x[1], reverse=True
        )
        return pairs
