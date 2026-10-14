"""Utility Functions for TRIO"""

import torch
import numpy as np


def extract_paralinguistic_features(audio, model_type='trilllsson'):
    """Extract paralinguistic features from audio"""
    # Placeholder for feature extraction
    # In practice, use pre-trained models like TRILLsson or x-vector
    return torch.randn(1, 2048) if model_type == 'trilllsson' else torch.randn(1, 512)


def normalize_features(features, eps=1e-8):
    """L2 normalize feature vectors"""
    return features / (torch.norm(features, dim=-1, keepdim=True) + eps)


def compute_accuracy(predictions, targets):
    """Compute classification accuracy"""
    correct = (predictions == targets).float().mean()
    return correct.item()


def seed_everything(seed=42):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
