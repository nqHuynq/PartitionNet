<div align="center">

# PartitionNet (v28-eq-vector)
### Taxon-Permutation-Equivariant Neural Architecture for Phylogenetic Quartet Partitioning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Permutation Equivariance](https://img.shields.io/badge/Equivariance-100%25%20Exact-brightgreen.svg)](#-permutation-equivariance-guarantee)

</div>

---

## 📖 Overview

**PartitionNet** is a deep learning framework designed for phylogenetic quartet topology inference. Given multiple sequence alignment (MSA) chunks of 4 taxa $\mathcal{T} = \{t_1, t_2, t_3, t_4\}$, PartitionNet predicts the true unrooted evolutionary tree topology among the 3 canonical unrooted topologies:

| Topology | Canonical Partition | Bipolar Vector Representation ($y \in \{-1, +1\}^4$) |
|:---:|:---:|:---:|
| **Topology 1** | $(t_1, t_2) \mid (t_3, t_4)$ | `[+1, +1, -1, -1]` $\equiv$ `[-1, -1, +1, +1]` |
| **Topology 2** | $(t_1, t_3) \mid (t_2, t_4)$ | `[+1, -1, +1, -1]` $\equiv$ `[-1, +1, -1, +1]` |
| **Topology 3** | $(t_1, t_4) \mid (t_2, t_3)$ | `[+1, -1, -1, +1]` $\equiv$ `[-1, +1, +1, -1]` |

---

## 🔒 Permutation Equivariance Guarantee

A major challenge in quartet topology inference is **taxon order sensitivity**: classical neural networks can produce completely different predictions if the taxa sequences are shuffled.

PartitionNet solves this fundamentally by constructing an **equivariant vector read-out**:
$$\mathbf{z}(\pi(\mathbf{x})) = \pi(\mathbf{z}(\mathbf{x})) \quad \forall \pi \in \mathcal{S}_4$$

- **Pair Feature Builder**: Computes symmetric pairwise embeddings for all unordered pairs $\{i, j\}$ ($P = \binom{n}{2} = 6$).
- **Relational Triangle Layer**: Pair $(i, j)$ aggregates context across all intermediate taxa $k$ using pair tensor contraction:
  $$T_{ij} = \frac{1}{n-2} \sum_{k \neq i, j} H_{ik} \odot H_{jk}$$
- **Content-Based Reference Anchor**: Chooses reference taxon $r$ dynamically via sequence content hash rather than indexing order, preserving exact permutation equivariance.

---

## 🏗️ Repository Architecture

```text
PartitionNet/
├── configs/
│   ├── default.yaml            # Standard hyperparams & training setup
│   └── kaggle.yaml             # Preset for Kaggle GPU environments
├── partitionnet/                # Core modular package
│   ├── __init__.py             # Public API exports
│   ├── config.py               # Dataclass configuration definitions
│   ├── data/
│   │   ├── dataset.py          # NumericRAMDataset for HDF5 sequence data
│   │   ├── sampler.py          # TrainChunkBasedDynamicSampler (OOM-safe)
│   │   └── collate.py          # ChunkCollator with preallocated buffer
│   ├── models/
│   │   ├── layers.py           # PairFeatureBuilder, DilatedContractingBlock, ChunkAttentionAggregation
│   │   ├── partitionnet.py     # PartitionNetEq (permutation-equivariant model)
│   │   └── loss.py             # StableSymmetricBipolarLoss
│   └── utils/
│       ├── memory.py           # Seed, memory trimming, state dict cleaning
│       ├── metrics.py          # Topology decoding, exact match, Hamming loss
│       └── invariance.py       # 24-taxon permutation test suite
├── notebooks/
│   └── partitionnet_v28_tutorial.ipynb  # Interactive walkthrough tutorial
├── scripts/
│   ├── train.py                # Command-line training pipeline
│   └── evaluate.py             # Checkpoint evaluation & permutation testing
├── tests/
│   ├── test_model.py           # Unit tests for shapes and equivariance
│   └── test_loss.py            # Unit tests for bipolar loss symmetry
├── pyproject.toml              # Modern Python packaging
├── requirements.txt            # Package dependencies
└── README.md                   # Project documentation
```

---

## ⚡ Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/nqHuynq/PartitionNet.git
cd PartitionNet

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies and PartitionNet package
pip install -e .
```

---

## 🚀 Quickstart

### Python API

```python
import torch
from partitionnet import PartitionNetEq
from partitionnet.utils.invariance import check_permutation_invariance

# Instantiate model
model = PartitionNetEq(window=2048)
model.eval()

# Synthetic input: (Batch=2, Chunks=2, Taxa=4, Window=2048)
x = torch.randint(0, 4, (2, 2, 4, 2048), dtype=torch.uint8)

# Forward pass
logits = model(x)  # Shape: (2, 4)

# Verify permutation equivariance
worst_diff, agreement = check_permutation_invariance(model, x)
print(f"Max logit diff under permutations: {worst_diff:.3e}")
print(f"Topology agreement: {agreement * 100:.1f}%")
```

---

## 🏋️‍♂️ Training & Evaluation

### Training with CLI

```bash
python scripts/train.py \
  --data-dir ./data/IQNet_Numeric_H5 \
  --epochs 35 \
  --batch-size 32 \
  --lr 1e-4 \
  --accum-steps 16 \
  --output-dir ./outputs
```

### Evaluation & Permutation Testing

```bash
python scripts/evaluate.py \
  --checkpoint ./outputs/pnet_v28_eq_best.pth \
  --test-file ./data/IQNet_Numeric_H5/iqnet_test.h5 \
  --perm-test \
  --output-json ./outputs/eval_results.json
```

---

## 🧪 Loss Function: StableSymmetricBipolarLoss

Since phylogenetic partitions are sign-symmetric ($y \equiv -y$), standard BCE or MSE is ill-suited. PartitionNet uses **StableSymmetricBipolarLoss**:

$$\mathcal{L} = \min\big(\text{softplus}(-\mathbf{y} \cdot \mathbf{z}), \text{softplus}(\mathbf{y} \cdot \mathbf{z})\big) + \lambda_{\text{penalty}} \left( \frac{1}{4} \sum_{i=1}^4 \tanh\left(\frac{z_i}{\tau}\right) \right)^2 + \lambda_{\text{pair}} \mathcal{L}_{\text{pair}}$$

1. **Symmetric Margin Loss**: Invariant to global label flip.
2. **Balanced Bipolar Constraint**: Penalizes unbalanced splits away from $2 \mid 2$ partitions.
3. **Auxiliary Pair Loss**: Directly supervises pairwise relational edges ($y_i y_j > 0$).

---

## 🔬 Benchmark Results

| Metric | Anchor Model (v27) | PartitionNet (v28-eq-vector) |
|:---|:---:|:---:|
| **Topology Accuracy** | 82.04% | **83.18%** (+1.14%) |
| **Exact Match** | 0.8170 | **0.8295** |
| **Hamming Loss** | 0.0915 | **0.0852** |
| **Permutation Topo Agreement** | ~85.2% | **100.00%** |
| **Permutation Equivariance Error** | > 1.2e-1 | **< 1e-6 (Exact)** |

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
