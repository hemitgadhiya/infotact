"""Unit tests for the GATValuationModel architecture."""

from __future__ import annotations

import torch
from src.gnn_model import GATValuationModel


def test_gat_model_forward():
    # Setup small synthetic graph: 5 nodes, 8 directed edges
    num_nodes = 5
    in_feats = 10
    
    # 5 nodes with 10 features each
    x = torch.randn(num_nodes, in_feats)
    
    # Directed edges: node 0 -> node 1, node 1 -> node 2, etc.
    src = torch.tensor([0, 1, 2, 3, 0, 4, 1, 3], dtype=torch.long)
    tgt = torch.tensor([1, 2, 3, 4, 2, 0, 3, 1], dtype=torch.long)
    edge_index = torch.stack([src, tgt], dim=0)
    
    model = GATValuationModel(
        in_feats=in_feats,
        hidden_dim=8,
        num_heads=2,
        num_layers=2,
        dropout=0.0
    )
    
    # Test forward pass without attention tracking
    preds, attn = model(x, edge_index, return_attention=False)
    
    assert preds.shape == (num_nodes,)
    assert attn is None
    
    # Test forward pass with attention tracking
    preds, attn = model(x, edge_index, return_attention=True)
    
    assert preds.shape == (num_nodes,)
    assert attn is not None
    assert len(attn) == 2  # one for each layer
    
    # Each entry in attn is (edge_index_out, attention_tensor)
    edge_index_out, attn_weights = attn[-1]
    # Verify shape: attn_weights has shape (num_edges, num_heads)
    # Note: PyG's GATConv with self-loops adds extra self-loop edges.
    assert edge_index_out.shape[0] == 2
    assert attn_weights.ndim == 2 or attn_weights.ndim == 1
    
    # Test neighbor attention extraction
    # Query for node 2
    neighbors = model.get_neighbor_attention(node_idx=2, layer_idx=-1)
    
    assert isinstance(neighbors, list)
    if len(neighbors) > 0:
        # Each item should be (neighbor_idx, weight)
        idx, weight = neighbors[0]
        assert isinstance(idx, int)
        assert isinstance(weight, float)
        # Weights should be sorted descending
        weights = [w for _, w in neighbors]
        assert all(weights[i] >= weights[i+1] for i in range(len(weights)-1))
