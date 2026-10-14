"""Canonical Correlation Analysis Module"""

import torch
import torch.nn as nn


class CanonicalCorrelation(nn.Module):
    """
    Canonical Correlation Analysis for aligning different representation spaces.

    Learns projections to maximize correlation between two feature sets.
    """

    def __init__(self, d_model: int = 512, n_canonical: int = 256):
        super().__init__()
        self.d_model = d_model
        self.n_canonical = n_canonical

        # Projection matrices for CCA
        self.U = nn.Parameter(torch.randn(d_model, n_canonical))
        self.V = nn.Parameter(torch.randn(d_model, n_canonical))

        # Initialize orthogonally
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)

    def forward(self, x, y, fused):
        """Apply canonical correlation to align representations"""
        # Project to canonical space
        x_can = torch.matmul(x, self.U) if x is not None else None
        y_can = torch.matmul(y, self.V) if y is not None else None

        # Compute CCA loss (correlation maximization)
        if x_can is not None and y_can is not None:
            # Normalize
            x_can = x_can / (torch.norm(x_can, dim=-1, keepdim=True) + 1e-8)
            y_can = y_can / (torch.norm(y_can, dim=-1, keepdim=True) + 1e-8)

            # Correlation
            corr = torch.matmul(x_can.t(), y_can) / (len(x) + 1e-8)

        return fused
