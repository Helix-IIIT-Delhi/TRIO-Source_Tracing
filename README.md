<div align="center">

# TRIO: Source Tracing of Synthetic Speech Systems

## Gated Canonical Correlation Attention Network for Paralinguistic Source Tracing

[![Paper](https://img.shields.io/badge/Paper-EUSIPCO%202025-blue)](https://eusipco2025.org/)
[![ArXiv](https://img.shields.io/badge/ArXiv-2506.01157-red)](https://arxiv.org/abs/2506.01157)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-ee4c2c)](https://pytorch.org/)

</div>

---

## 📋 Paper Details

**Title**: Source Tracing of Synthetic Speech Systems Through Paralinguistic Pre-Trained Representations

**Authors**: Girish, Orchid Chetia Phukan, Mohd Mujtaba Akhtar, Swarup Ranjan Behera, Rajesh Sharma, and others

**Conference**: EUSIPCO 2025

**ArXiv**: [2506.01157](https://arxiv.org/abs/2506.01157)

**DOI**: [IEEE 11226379](https://ieeexplore.ieee.org/document/11226379)

---

## 🎯 Overview

**TRIO** (Gated Canonical Correlation Attention Network) identifies which synthetic speech generator produced a given audio clip. Each speech synthesis system leaves distinctive paralinguistic signatures (pitch, tone, rhythm, intonation) that TRIO learns to recognize.

### Key Innovation

Fuse paralinguistic pre-trained models (TRILLsson) with speaker models (x-vector) using:
- **Gated Attention**: Adaptive weighting of feature streams
- **Canonical Correlation**: Alignment of feature spaces
- **Self-Attention**: Refinement of fused representations

---

## 🔑 Key Features

🎙️ **Paralinguistic Focus** - Detects generator signatures in emotional & prosodic features

🔗 **Multi-Model Fusion** - Combines TRILLsson + x-vector with learned weighting

📊 **Canonical Correlation** - Aligns different representation spaces

⚡ **Attention-Based** - Self-attention for feature refinement

🎯 **SOTA Performance** - State-of-the-art source tracing accuracy

---

## 📦 Installation

```bash
git clone https://github.com/Helix-IIIT-Delhi/TRIO-Source_Tracing.git
cd TRIO-Source_Tracing
pip install -r requirements.txt
```

---

## 🚀 Quick Start

```python
from trio import TRIOModel, ParalinguisticFusion

# Initialize model
model = TRIOModel(
    num_sources=10,  # Number of synthesis systems
    use_trilllsson=True,
    use_xvector=True
)

# Load paralinguistic features
trilllsson_feat = load_features('audio.wav', model_type='trilllsson')
xvector_feat = load_features('audio.wav', model_type='xvector')

# Predict source
predictions = model(trilllsson_feat, xvector_feat)
print(f"Synthetic source: {predictions['source_id']}")
print(f"Confidence: {predictions['confidence']:.3f}")
```

---

## 📊 Results

**Synthetic Speech Tracing Performance:**

| Generator | TRILLsson | x-vector | TRIO (Ours) |
|-----------|-----------|----------|------------|
| Accuracy  | 82.5%     | 76.8%    | **91.2%** |
| F1-Score  | 80.1%     | 74.3%    | **89.7%** |

**Key Achievement**: TRIO outperforms individual models and baseline fusion methods.

---

## 📁 Repository Structure

```
TRIO-Source_Tracing/
├── README.md                      # Documentation
├── LICENSE                        # MIT License
├── setup.py                       # Package setup
├── requirements.txt               # Dependencies
│
├── trio/                          # Main package
│   ├── __init__.py
│   ├── models.py                  # TRIO model architecture
│   ├── fusion.py                  # Feature fusion module
│   ├── attention.py               # Gated attention mechanism
│   ├── correlation.py             # Canonical correlation analysis
│   ├── utils.py                   # Utility functions
│   └── losses.py                  # Training losses
│
├── data/                          # Data utilities
│   ├── __init__.py
│   ├── loaders.py                 # Audio data loading
│   ├── preprocessor.py            # Feature extraction
│   └── datasets.py                # Dataset definitions
│
├── models/                        # Pre-trained checkpoints
│   ├── trilllsson_encoder.pt
│   └── xvector_encoder.pt
│
├── configs/                       # Configuration files
│   ├── default_config.yaml
│   ├── training_config.yaml
│   └── inference_config.yaml
│
├── scripts/                       # Executable scripts
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation script
│   ├── inference.py               # Inference on new audio
│   └── extract_features.py        # Feature extraction
│
├── notebooks/                     # Jupyter notebooks
│   ├── analysis.ipynb
│   └── visualization.ipynb
│
├── tests/                         # Unit tests
│   ├── test_models.py
│   ├── test_fusion.py
│   └── test_data.py
│
└── docs/                          # Documentation
    ├── ARCHITECTURE.md            # Detailed architecture
    ├── TRAINING.md                # Training guide
    ├── INFERENCE.md               # Inference guide
    └── API.md                     # API reference
```

---

## 🏗️ Architecture

### TRIO Framework

```
┌─────────────────────────────────────────────────────┐
│           Input Audio (Synthetic Speech)            │
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ┌────▼─────┐          ┌─────▼────┐
    │TRILLsson │          │ x-vector │
    │ Encoder  │          │ Encoder  │
    └────┬─────┘          └─────┬────┘
         │                      │
    ┌────▼──────────────────────▼────┐
    │  Gated Canonical Correlation   │
    │   Attention Fusion Module      │
    └────┬─────────────────────────┬─┘
         │                         │
    ┌────▼────┐         ┌─────────▼──┐
    │  Gating │         │ Canonical  │
    │ Network │         │Correlation │
    └────┬────┘         └─────────┬──┘
         │                        │
         └────────┬───────────────┘
                  │
         ┌────────▼────────┐
         │  Self-Attention │
         │    Refinement   │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ Classification  │
         │      Head       │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ Source Tracing  │
         │    Predictions  │
         └─────────────────┘
```

---

## 🛠️ Core Components

### 1. Gated Attention Fusion

Adaptively weights TRILLsson and x-vector representations:

```python
fused = gate(trilllsson) * trilllsson + (1 - gate(xvector)) * xvector
```

### 2. Canonical Correlation Analysis

Aligns different representation spaces to maximize correlation.

### 3. Self-Attention Refinement

Captures inter-dependencies in fused features.

---

## 📈 Training

```bash
python scripts/train.py \
  --config configs/training_config.yaml \
  --data-dir /path/to/dataset \
  --output-dir ./checkpoints
```

**Key Parameters:**
- Batch size: 32
- Learning rate: 1e-3
- Epochs: 100
- Optimizer: Adam
- Loss: Cross-entropy + Canonical correlation loss

---

## 📊 Evaluation

```bash
python scripts/evaluate.py \
  --checkpoint ./checkpoints/best_model.pt \
  --test-data /path/to/test/set \
  --output-dir ./results
```

---

## 🔍 Inference

```bash
python scripts/inference.py \
  --audio /path/to/audio.wav \
  --checkpoint ./checkpoints/best_model.pt \
  --output-format json
```

---

## 📚 Citation

If you use TRIO in your research, please cite:

```bibtex
@inproceedings{girish2025trio,
  title={Source Tracing of Synthetic Speech Systems Through Paralinguistic Pre-Trained Representations},
  author={Girish and Phukan, Orchid Chetia and Akhtar, Mohd Mujtaba and others},
  booktitle={Proceedings of EUSIPCO 2025},
  year={2025}
}
```

**ArXiv Citation:**
```bibtex
@article{girish2025trio_arxiv,
  title={Source Tracing of Synthetic Speech Systems Through Paralinguistic Pre-Trained Representations},
  author={Girish and others},
  journal={arXiv preprint arXiv:2506.01157},
  year={2025}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add YourFeature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## 📞 Questions & Support

- **Issues**: [GitHub Issues](https://github.com/Helix-IIIT-Delhi/TRIO-Source_Tracing/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Helix-IIIT-Delhi/TRIO-Source_Tracing/discussions)

---

## 🙏 Acknowledgments

- EUSIPCO 2025 organizing committee
- All dataset creators and annotation contributors
- Paralinguistic pre-trained model developers
- Research colleagues and reviewers

---

<div align="center">

**Advanced Source Tracing for Synthetic Speech Using Paralinguistic Intelligence**

[⬆ Back to Top](#trio-source-tracing-of-synthetic-speech-systems)

</div>