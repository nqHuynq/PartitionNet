import itertools
import numpy as np
import torch
from torch.utils.data import DataLoader
from partitionnet.utils.metrics import _bits_to_class, _top2_bits


@torch.no_grad()
def check_permutation_invariance(model: torch.nn.Module, x: torch.Tensor, perms=None):
    """Evaluates taxon-permutation equivariance for input tensor x: (B, N_chunks, 4, W).
    
    Verifies that for any permutation p of taxa:
        z(x[:, :, p, :]) == z(x)[:, p]
    Returns (worst_logit_difference, fraction_of_exact_topology_agreement).
    """
    model.eval()
    perms = perms or list(itertools.permutations(range(4)))
    z = model(x).float()
    cls0_bits = _top2_bits(z)
    worst, agree = 0.0, 1.0
    
    for p in perms:
        p = list(p)
        zp = model(x[:, :, p, :]).float()
        worst = max(worst, (zp - z[:, p]).abs().max().item())
        exp_cls = _bits_to_class(cls0_bits[:, p])
        got_cls = _bits_to_class(_top2_bits(zp))
        agree = min(agree, (exp_cls == got_cls).float().mean().item())
        
    return worst, agree


def run_permutation_test(model: torch.nn.Module, dataset, collate_fn, device: str = "cpu", n_samples: int = 512):
    """Runs permutation invariance test on random subset of dataset across all 24 permutations.
    Returns (worst_diff, topology_agreement, overall_perm_accuracy).
    """
    model.eval()
    idx = np.random.RandomState(0).choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    sampler = [list(idx[i:i + 16]) for i in range(0, len(idx), 16)]
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0)
    
    worst, agree, correct, total = 0.0, 1.0, 0, 0
    for inputs, labels in loader:
        inputs = inputs.to(device).clone()
        labels_dev = labels.to(device)
        d, a = check_permutation_invariance(model, inputs)
        worst, agree = max(worst, d), min(agree, a)
        
        with torch.no_grad():
            for p in itertools.permutations(range(4)):
                p = list(p)
                zp = model(inputs[:, :, p, :]).float()
                pred_cls = _bits_to_class(_top2_bits(zp))
                true_cls = _bits_to_class(labels_dev[:, p])
                correct += (pred_cls == true_cls).sum().item()
                total += zp.shape[0]
                
    overall_acc = correct / max(total, 1)
    return worst, agree, overall_acc
