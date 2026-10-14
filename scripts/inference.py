#!/usr/bin/env python3
"""Inference script for TRIO"""

import argparse
import torch
from trio import TRIOModel


def inference():
    parser = argparse.ArgumentParser(description='Run TRIO inference')
    parser.add_argument('--audio', required=True, help='Path to audio file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--output-format', default='json')
    args = parser.parse_args()

    # Load model
    model = TRIOModel(num_sources=10)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    # Run inference
    print(f"Audio: {args.audio}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output format: {args.output_format}")


if __name__ == '__main__':
    inference()
