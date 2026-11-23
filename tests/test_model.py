import torch
from partitionnet.models.partitionnet import PartitionNetEq
from partitionnet.utils.invariance import check_permutation_invariance


def test_partitionnet_eq_shapes():
    """Verify output shapes for batch of chunked inputs."""
    model = PartitionNetEq(embed_dim=16, dim=32, window=128)
    model.eval()
    
    B, N_chunks, n_taxa, W = 2, 3, 4, 128
    dummy_x = torch.randint(0, 4, (B, N_chunks, n_taxa, W), dtype=torch.uint8)
    
    with torch.no_grad():
        out = model(dummy_x)
        assert out.shape == (B, n_taxa), f"Expected shape {(B, n_taxa)}, got {out.shape}"
        
        out_z, out_pairs = model(dummy_x, return_pairs=True)
        assert out_z.shape == (B, n_taxa)
        assert out_pairs.shape == (B, 6), f"Expected 6 pairs for 4 taxa, got {out_pairs.shape}"


def test_permutation_equivariance():
    """Verify that permuting taxa permutes logits identically."""
    torch.manual_seed(42)
    model = PartitionNetEq(embed_dim=16, dim=32, window=128)
    model.eval()
    
    B, N_chunks, n_taxa, W = 2, 2, 4, 128
    dummy_x = torch.randint(0, 4, (B, N_chunks, n_taxa, W), dtype=torch.uint8)
    
    worst_diff, topo_agree = check_permutation_invariance(model, dummy_x)
    assert worst_diff < 1e-4, f"Permutation equivariance violated! Max diff: {worst_diff}"
    assert topo_agree == 1.0, f"Topology agreement under permutation should be 1.0, got {topo_agree}"


if __name__ == "__main__":
    test_partitionnet_eq_shapes()
    test_permutation_equivariance()
    print("All model tests passed!")

