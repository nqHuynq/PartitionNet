import torch
from partitionnet.models.loss import StableSymmetricBipolarLoss


def test_loss_sign_invariance():
    """Verify that target y and -y yield identical loss."""
    loss_fn = StableSymmetricBipolarLoss(lambda_penalty=0.5, temperature=3.0, lambda_pair=0.0)
    
    logits = torch.tensor([[2.5, 2.1, -1.8, -2.0]], dtype=torch.float32)
    target_pos = torch.tensor([[1.0, 1.0, -1.0, -1.0]], dtype=torch.float32)
    target_neg = -target_pos
    
    loss_pos, margin_pos, _ = loss_fn(logits, target_pos)
    loss_neg, margin_neg, _ = loss_fn(logits, target_neg)
    
    assert torch.isclose(loss_pos, loss_neg, atol=1e-6), "Loss must be sign-symmetric with respect to target"
    assert torch.isclose(margin_pos, margin_neg, atol=1e-6), "Margin loss must be sign-symmetric"
