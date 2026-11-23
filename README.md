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

### Python API (Running Inference with Pretrained Weights)

Trọng số tốt nhất đã được huấn luyện sẵn và lưu tại `results/pnet_v28_eq_best.pth`. Bạn có thể tải và suy luận trực tiếp:

```python
import torch
from partitionnet import PartitionNetEq
from partitionnet.utils.metrics import decode_topos
from partitionnet.utils.invariance import check_permutation_invariance

# 1. Khởi tạo mô hình & nạp trọng số đã huấn luyện (Pretrained weights)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PartitionNetEq(window=2048).to(device)
model.load_state_dict(torch.load("results/pnet_v28_eq_best.pth", map_location=device))
model.eval()

# 2. Đầu vào giả lập: (Batch=2, Chunks=2, Taxa=4, Window=2048)
x = torch.randint(0, 4, (2, 2, 4, 2048), dtype=torch.uint8).to(device)

# 3. Dự đoán phân hoạch và giải mã topology (1, 2, hoặc 3)
with torch.no_grad():
    logits = model(x)                     # Shape: (2, 4)
    pred_topos = decode_topos(logits.cpu().numpy())
    print(f"Predicted Quartet Topologies: {pred_topos}")

# 4. Kiểm tra tính đồng biến hoán vị
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

Dưới đây là kết quả thực nghiệm thực tế thu được từ log kiểm thử (test evaluation log) của bản chạy **V28** trong notebook:

### 1. Tổng quan hiệu năng trên tập Test (51,241 mẫu)

| Metric | Anchor Model (v27) | PartitionNet (v28-eq-vector) | Ghi chú |
|:---|:---:|:---:|:---|
| **Topology Accuracy** | 82.04% | **82.14%** (+0.10%) | Độ chính xác giải mã cây quartet trên tập test |
| **Exact Match** | — | **0.7831** | Tỷ lệ khớp hoàn toàn vector nhãn bipolar |
| **Hamming Loss** | — | **0.1087** | Khoảng cách Hamming trên các bit phân hoạch |
| **Best Validation Accuracy** | — | **82.08%** (Ep 18) | Độ chính xác cao nhất trên tập Validation |
| **Permutation Topo Agreement** | — | **100.00%** | Đồng thuận 100% kết quả giải mã qua cả 24 hoán vị taxa ($4!$) |
| **Permutation Max Logit Diff** | — | **2.027e-05** | Sai số xấp xỉ số thực cực nhỏ, đạt tính đồng biến thực tế |
| **Topo Accuracy over all 24 Perms** | — | **80.08%** | Độ chính xác trung bình khi quét qua cả 24 hoán vị |

### 2. Chi tiết hiệu năng từng Topology (Classification Report)

Tập test gồm 51,241 mẫu phân bố đồng đều giữa 3 hình thái cây unrooted quartet:

| Topology | Cấu trúc phân hoạch | Precision | Recall | F1-Score | Support (Số mẫu) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Topology 1** | $(t_1, t_2) \mid (t_3, t_4)$ | **82.36%** | **82.27%** | **82.31%** | 17,064 |
| **Topology 2** | $(t_1, t_3) \mid (t_2, t_4)$ | **81.84%** | **81.91%** | **81.87%** | 17,095 |
| **Topology 3** | $(t_1, t_4) \mid (t_2, t_3)$ | **82.21%** | **82.24%** | **82.23%** | 17,082 |
| **Macro Average** | — | **82.14%** | **82.14%** | **82.14%** | 51,241 |
| **Weighted Average** | — | **82.14%** | **82.14%** | **82.14%** | 51,241 |

### 3. Biểu đồ huấn luyện & Ma trận nhầm lẫn

<div align="center">
  <img src="results/training_curves.png" alt="Training Curves" width="85%">
  <br><br>
  <img src="results/confusion_matrix.png" alt="Confusion Matrix" width="55%">
</div>

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
