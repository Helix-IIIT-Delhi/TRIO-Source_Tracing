"""TRIO Model Architecture"""

import torch
import torch.nn as nn
from .fusion import ParalinguisticFusion


class TRIOModel(nn.Module):
    """
    TRIO: Gated Canonical Correlation Attention Network

    Fuses paralinguistic and speaker representations for source tracing.

    Args:
        num_sources: Number of synthesis systems to classify
        trilllsson_dim: TRILLsson feature dimension
        xvector_dim: x-vector feature dimension
        hidden_dim: Hidden dimension for fusion
        use_trilllsson: Enable TRILLsson fusion
        use_xvector: Enable x-vector fusion
    """

    def __init__(
        self,
        num_sources: int = 10,
        trilllsson_dim: int = 2048,
        xvector_dim: int = 512,
        hidden_dim: int = 512,
        use_trilllsson: bool = True,
        use_xvector: bool = True,
    ):
        super().__init__()
        self.num_sources = num_sources
        self.use_trilllsson = use_trilllsson
        self.use_xvector = use_xvector

        # Feature projections
        if use_trilllsson:
            self.trilllsson_proj = nn.Linear(trilllsson_dim, hidden_dim)
        if use_xvector:
            self.xvector_proj = nn.Linear(xvector_dim, hidden_dim)

        # Fusion module
        self.fusion = ParalinguisticFusion(
            hidden_dim,
            use_trilllsson=use_trilllsson,
            use_xvector=use_xvector
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_sources)
        )

    def forward(self, trilllsson_feat=None, xvector_feat=None):
        """Forward pass for source tracing"""
        # Project features if provided
        if trilllsson_feat is not None and self.use_trilllsson:
            trilllsson = self.trilllsson_proj(trilllsson_feat)
        else:
            trilllsson = None

        if xvector_feat is not None and self.use_xvector:
            xvector = self.xvector_proj(xvector_feat)
        else:
            xvector = None

        # Fuse representations
        fused = self.fusion(trilllsson, xvector)

        # Global pooling if sequence features
        if fused.dim() > 2:
            fused = fused.mean(dim=1)

        # Classify
        logits = self.classifier(fused)

        return {
            "logits": logits,
            "predictions": torch.argmax(logits, dim=1),
            "confidence": torch.softmax(logits, dim=1).max(dim=1).values,
            "source_id": torch.argmax(logits, dim=1),
        }
