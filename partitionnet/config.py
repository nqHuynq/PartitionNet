from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import os
import torch


@dataclass
class ModelConfig:
    embed_dim: int = 16
    dim: int = 32
    window_size: int = 2048
    center: bool = True
    pair_channels: int = 64
    num_heads: int = 4
    dropout: float = 0.3


@dataclass
class LossConfig:
    lambda_penalty: float = 0.5
    temperature: float = 3.0
    lambda_pair: float = 0.5


@dataclass
class TrainingConfig:
    seed: int = 9999
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Training Loop
    max_epochs: int = 35
    max_epochs_this_run: int = 35
    patience: int = 7
    accum_steps: int = 16
    max_lr: float = 1e-4
    sanity_lr: float = 5e-4
    weight_decay: float = 1e-3
    pct_start: float = 0.1
    
    # Memory and Time Controls
    ram_safety_limit: float = 90.0
    max_run_time_sec: float = 9.0 * 3600
    num_workers: int = 0
    
    # Batching & Bucketing
    window_size: int = 2048
    max_chunks_limit: int = 50
    chunk_buckets: Tuple[int, ...] = (4, 8, 16, 24, 32, 50)
    max_batch_size_train: int = 32
    max_chunks_per_batch_train: int = 32
    max_batch_size_eval: int = 64
    max_chunks_per_batch_eval: int = 64
    
    # Data Paths
    num_train_parts: int = 8
    base_h5_path: str = "/kaggle/input/datasets/huykaizer243/evonaps-numeric/IQNet_Numeric_H5"
    model_input_dir: str = "/kaggle/input/models/nguyenabel/partitionnet-v28-eq-vector/keras/default/4"
    
    # Checkpoints and Artifacts
    output_dir: str = "/kaggle/working"
    checkpoint_name: str = "pnet_v28_eq_chkpt.pth"
    best_model_name: str = "pnet_v28_eq_best.pth"
    log_name: str = "training_log_eq.csv"
    metrics_name: str = "pnet_v28_eq_results.json"
    preds_name: str = "pnet_v28_eq_preds.npz"

    def resolve_paths(self):
        """Auto-detect directories if running inside Kaggle or local environments."""
        if not os.path.exists(self.base_h5_path):
            for root, _, files in os.walk("/kaggle/input"):
                if any(f.endswith(".h5") for f in files):
                    self.base_h5_path = root
                    break
        os.makedirs(self.output_dir, exist_ok=True)
