import os
import random
import ctypes
import psutil
import numpy as np
import torch

try:
    libc = ctypes.CDLL('libc.so.6')
    libc.mallopt(-3, 65536)
    libc.mallopt(-4, 65536)
except Exception:
    libc = None


def set_seed(seed: int = 9999):
    """Set random seed for reproducibility across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def trim_memory():
    """Trigger glibc malloc_trim if available to return freed heap to OS."""
    if libc is not None:
        try:
            libc.malloc_trim(0)
        except Exception:
            pass


def get_ram_usage() -> float:
    """Returns current system virtual memory utilization percentage."""
    return psutil.virtual_memory().percent


def clean_state_dict(sd: dict) -> dict:
    """Strip 'module.' or '_orig_mod.' prefixes added by DDP/DataParallel/compile."""
    cleaned = {}
    for k, v in sd.items():
        if k.startswith('module.'):
            k = k[7:]
        if k.startswith('_orig_mod.'):
            k = k[10:]
        cleaned[k] = v
    return cleaned


def resolve_input_file(target_name: str, search_dirs: list = None) -> str:
    """Locates an input artifact across specified directories or Kaggle input roots."""
    if search_dirs is None:
        search_dirs = []
    
    for d in search_dirs:
        direct = os.path.join(d, target_name)
        if os.path.exists(direct):
            return direct
            
    # Fallback to /kaggle/input search if on Kaggle
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if target_name in files:
                return os.path.join(root, target_name)
                
    return ""
