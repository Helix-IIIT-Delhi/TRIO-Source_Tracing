"""Gated Attention Mechanism"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttention(nn.Module):
    """
    Gated attention mechanism for adaptive weighting of feature streams.

    Computes: fused = gate(x) * x + (1 - gate(y)) * y
    """

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.d_model = d_model

        # Gating networks for each stream
        self.gate_trilllsson = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        self.gate_xvector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, trilllsson, xvector):
        """Compute gated fusion of two feature streams"""
        # Compute gating weights
        if trilllsson.dim() > 2:
            gate_t = self.gate_trilllsson(trilllsson)
            gate_x = self.gate_xvector(xvector)
        else:
            gate_t = self.gate_trilllsson(trilllsson)
            gate_x = self.gate_xvector(xvector)

        # Fused representation
        fused = gate_t * trilllsson + (1 - gate_x) * xvector

        return fused
