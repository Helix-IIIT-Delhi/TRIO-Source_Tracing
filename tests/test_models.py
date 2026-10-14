"""Unit tests for TRIO models"""

import torch
from trio import TRIOModel


def test_trio_model():
    """Test TRIO model initialization and forward pass"""
    model = TRIOModel(num_sources=10)

    # Create dummy inputs
    trilllsson = torch.randn(4, 2048)
    xvector = torch.randn(4, 512)

    # Forward pass
    output = model(trilllsson, xvector)

    assert output['logits'].shape == (4, 10)
    assert output['predictions'].shape == (4,)
    assert output['confidence'].shape == (4,)
    print("✓ TRIO model test passed")


if __name__ == '__main__':
    test_trio_model()
