"""Canonical Correlation Analysis Module"""

import torch
import torch.nn as nn


class CanonicalCorrelation(nn.Module):
    """
    Canonical Correlation Analysis for aligning different representation spaces.

    Learns projections U and V to maximize correlation between:
    U^T x and V^T y

    ρ = max E[<U^T X, V^T Y>] / sqrt(Var(U^T X) Var(V^T Y))
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

    def forward(self, x, y):
        """
        Compute canonical correlation and aligned representations.

        Args:
            x: Features from source [batch, d_model]
            y: Features from target [batch, d_model]

        Returns:
            cca_loss: Negative canonical correlation (for minimization)
        """
        if x is None or y is None:
            return torch.tensor(0.0)

        # Project to canonical space
        x_can = torch.matmul(x, self.U)  # [batch, n_canonical]
        y_can = torch.matmul(y, self.V)  # [batch, n_canonical]

        # Center the projections (subtract mean)
        x_can_centered = x_can - x_can.mean(dim=0)
        y_can_centered = y_can - y_can.mean(dim=0)

        # Compute covariance matrices
        n = x_can.size(0)

        # Covariance between x and y canonical spaces
        cov_xy = torch.matmul(x_can_centered.t(), y_can_centered) / (n - 1)

        # Auto-covariance
        cov_xx = torch.matmul(x_can_centered.t(), x_can_centered) / (n - 1)
        cov_yy = torch.matmul(y_can_centered.t(), y_can_centered) / (n - 1)

        # Add regularization for numerical stability
        eps = 1e-8
        cov_xx = cov_xx + eps * torch.eye(cov_xx.size(0), device=cov_xx.device)
        cov_yy = cov_yy + eps * torch.eye(cov_yy.size(0), device=cov_yy.device)

        # Compute canonical correlation
        # ρ = ||cov_xy|| / sqrt(||cov_xx|| * ||cov_yy||)
        # Use Frobenius norm for matrix norm
        trace_cov_xy = torch.trace(cov_xy @ cov_xy.t())
        trace_cov_xx = torch.trace(cov_xx @ cov_xx.t())
        trace_cov_yy = torch.trace(cov_yy @ cov_yy.t())

        # Canonical correlation (higher is better, so negate for loss)
        canonical_correlation = trace_cov_xy / torch.sqrt(trace_cov_xx * trace_cov_yy + eps)

        # Return negative correlation as loss (to minimize)
        cca_loss = -canonical_correlation

        return cca_loss
