import numpy as np
import torch
from sklearn.metrics import classification_report, hamming_loss, accuracy_score


def bipolar_to_topo(bit_arr) -> int:
    """Maps 4-bit bipolar vector to quartet topology class index (1, 2, or 3)."""
    bit_str = "".join(['1' if x > 0 else '0' for x in bit_arr])
    mapping = {
        "1100": 1, "0011": 1,
        "1010": 2, "0101": 2,
        "1001": 3, "0110": 3
    }
    return mapping.get(bit_str, -1)


def _bits_to_class(bits: torch.Tensor) -> torch.Tensor:
    """Converts (N, 4) bipolar tensor (sign-invariant) into topology class {0, 1, 2} or -1."""
    out = torch.full((bits.shape[0],), -1, dtype=torch.long, device=bits.device)
    for k, ref in enumerate([(1, 1, -1, -1), (1, -1, 1, -1), (1, -1, -1, 1)]):
        t = torch.tensor(ref, dtype=bits.dtype, device=bits.device)
        out[(bits == t).all(1) | (bits == -t).all(1)] = k
    return out


def _top2_bits(z: torch.Tensor) -> torch.Tensor:
    """Constructs bipolar vector by assigning +1 to top-2 logits and -1 to the rest."""
    idx = z.topk(2, dim=1).indices
    bits = -torch.ones_like(z)
    bits.scatter_(1, idx, 1.0)
    return bits


def decode_topos(logits: np.ndarray) -> np.ndarray:
    """Decodes array of raw 4-dim logits into quartet topology labels."""
    top2_idx = np.argsort(-logits, axis=1)[:, :2]
    preds_top2 = np.full_like(logits, -1.0)
    np.put_along_axis(preds_top2, top2_idx, 1.0, axis=1)
    return np.array([bipolar_to_topo(b) for b in preds_top2])


def compute_metrics(logits: np.ndarray, labels: np.ndarray):
    """Computes exact match, Hamming loss, topology accuracy, and classification report."""
    preds_strict = np.where(logits > 0.0, 1.0, -1.0)
    match_orig = (preds_strict == labels).all(axis=1)
    match_flip = (preds_strict == -labels).all(axis=1)
    exact_match = (match_orig | match_flip).mean()

    preds_aligned = np.where(match_flip[:, None], -preds_strict, preds_strict)
    y_true_bin = (labels > 0).astype(int)
    preds_aligned_bin = (preds_aligned > 0).astype(int)
    ham = hamming_loss(y_true_bin, preds_aligned_bin)

    true_topos = [bipolar_to_topo(b) for b in labels]
    pred_topos = decode_topos(logits)
    topo_acc = accuracy_score(true_topos, pred_topos)
    report = classification_report(
        true_topos, pred_topos,
        target_names=['Topo 1', 'Topo 2', 'Topo 3'],
        digits=4,
        output_dict=True
    )

    return exact_match, ham, topo_acc, report
