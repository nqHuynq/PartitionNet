import torch
import torch.nn as nn
import torch.nn.functional as F


class StableSymmetricBipolarLoss(nn.Module):
    """Flip-invariant bipolar loss with sum constraint penalty and auxiliary pair supervision.
    
    Since a partition split {A, B} is identical to {B, A}, a target vector y and its
    negation -y represent the exact same phylogenetic quartet tree topology.
    
    This loss computes:
        L = min(logaddexp(0, -y*z), logaddexp(0, y*z)) + lambda_penalty * mean(tanh(z/tau))^2 + L_pair
    """
    def __init__(
        self,
        lambda_penalty: float = 0.5,
        temperature: float = 3.0,
        lambda_pair: float = 0.5
    ):
        super().__init__()
        self.lambda_penalty = lambda_penalty
        self.temperature = temperature
        self.lambda_pair = lambda_pair

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pair_logits: torch.Tensor = None
    ):
        logits_f32 = logits.float()
        targets_f32 = targets.float()

        zeros = torch.zeros_like(logits_f32)
        loss_y = torch.logaddexp(zeros, -targets_f32 * logits_f32).mean(dim=1)
        loss_minus_y = torch.logaddexp(zeros, targets_f32 * logits_f32).mean(dim=1)
        symmetric_margin_loss = torch.min(loss_y, loss_minus_y).mean()

        preds_soft = torch.tanh(logits_f32 / self.temperature)
        sum_preds = preds_soft.sum(dim=1) / 4.0
        constraint_penalty = torch.mean(sum_preds ** 2)

        aux = torch.zeros((), device=logits_f32.device)
        if pair_logits is not None and self.lambda_pair > 0:
            # Auxiliary target: same side (y_i * y_j > 0)
            t_dim = targets_f32.shape[1]
            i0, i1 = torch.triu_indices(t_dim, t_dim, 1, device=targets_f32.device)
            same = (targets_f32[:, i0] * targets_f32[:, i1] > 0).float()
            aux = self.lambda_pair * F.binary_cross_entropy_with_logits(pair_logits.float(), same)

        total_loss = symmetric_margin_loss + (self.lambda_penalty * constraint_penalty) + aux
        return (
            total_loss,
            symmetric_margin_loss.detach(),
            (self.lambda_penalty * constraint_penalty + aux).detach()
        )
