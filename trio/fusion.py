"""Paralinguistic Feature Fusion Module"""

import torch
import torch.nn as nn
from .attention import GatedAttention
from .correlation import CanonicalCorrelation


class ParalinguisticFusion(nn.Module):
    """
    Fuses TRILLsson and x-vector representations using gated attention
    and canonical correlation analysis.
    """

    def __init__(
        self,
        d_model: int = 512,
        use_trilllsson: bool = True,
        use_xvector: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_trilllsson = use_trilllsson
        self.use_xvector = use_xvector

        if use_trilllsson and use_xvector:
            # Gated attention for both streams
            self.gated_attention = GatedAttention(d_model)

            # Canonical correlation analysis
            self.cca = CanonicalCorrelation(d_model)

            # Self-attention for refinement
            self.self_attention = nn.MultiheadAttention(
                d_model, num_heads=8, batch_first=True
            )

    def forward(self, trilllsson_feat, xvector_feat):
        """Fuse paralinguistic and speaker features"""

        if self.use_trilllsson and self.use_xvector:
            # Apply gated attention
            fused = self.gated_attention(trilllsson_feat, xvector_feat)

            # Apply canonical correlation
            fused = self.cca(trilllsson_feat, xvector_feat, fused)

            # Self-attention refinement
            if fused.dim() == 2:
                fused = fused.unsqueeze(1)

            attn_out, _ = self.self_attention(fused, fused, fused)
            fused = fused + attn_out

            if attn_out.size(1) == 1:
                fused = fused.squeeze(1)

            return fused

        elif self.use_trilllsson:
            return trilllsson_feat
        elif self.use_xvector:
            return xvector_feat
        else:
            raise ValueError("At least one feature stream must be enabled")
