"""Training Losses for TRIO"""

import torch
import torch.nn as nn


class TrioLoss(nn.Module):
    """Combined loss for TRIO: classification + canonical correlation"""

    def __init__(self, lambda_cca: float = 0.1):
        super().__init__()
        self.lambda_cca = lambda_cca
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets, cca_score=0.0):
        """Compute combined loss"""
        # Classification loss
        ce = self.ce_loss(logits, targets)

        # CCA regularization (maximize correlation)
        cca = -cca_score if cca_score > 0 else torch.tensor(0.0)

        # Combined loss
        total_loss = ce + self.lambda_cca * cca

        return total_loss
