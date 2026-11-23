import torch
import torch.nn as nn
import torch.nn.functional as F
from partitionnet.models.layers import (
    PairFeatureBuilder,
    DilatedContractingBlock,
    ChunkAttentionAggregation
)


class PartitionNetEq(nn.Module):
    """Taxon-permutation-EQUIVARIANT bipolar-vector predictor for quartet topology inference.
    
    Given sequence alignment chunks x of shape (B, N_chunks, n, W):
      - Produces z of shape (B, n) bipolar logits (where sign(z_i) identifies the split of taxon i).
      - Guarantee: x[:, :, perm, :] -> z[:, perm] exactly.
      
    Pipeline:
      1) Pair network: symmetric pair feature builder -> shared conv backbone -> positional attention.
      2) Chunk aggregation: attention-based pooling across alignment chunks.
      3) Relational triangle layer: pair (i, j) aggregates across (i, k) and (j, k) via tensor contraction.
      4) Equivariant vector readout: content-based hash anchor selects reference taxon r dynamically.
    """
    def __init__(
        self,
        embed_dim: int = 16,
        dim: int = 32,
        window: int = 2048,
        center: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        self.dim = dim
        self.center = center
        self.window = window
        
        self.builder = PairFeatureBuilder(embed_dim=embed_dim, out_channels=64)
        self.backbone = DilatedContractingBlock(in_channels=64, mid_channels=64, out_channels=dim)
        self.pos_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.chunk_aggregator = ChunkAttentionAggregation(embed_dim=dim)
        
        self.tri = nn.Linear(dim, dim)
        self.rel_head = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1)
        )
        
        # Content-based hash weights for dynamic reference taxon selection
        g = torch.Generator().manual_seed(12345)
        ref = torch.randn(window, 5, generator=g)
        ref[:, 4] = 0.0  # zero out padding token
        self.register_buffer('ref_w', ref)

    def forward(self, x_chunks: torch.Tensor, return_pairs: bool = False):
        """
        x_chunks: (B, N_chunks, n, W) with tokens in {0, 1, 2, 3} and padding 4.
        """
        b, n, t, w = x_chunks.shape
        i0, i1 = torch.triu_indices(t, t, 1, device=x_chunks.device)
        P = i0.numel()
        chunk_mask = (x_chunks == 4).all(dim=-1).all(dim=-1)  # (b, n)

        # 1. Pair features across all chunks
        h = self.builder(x_chunks.reshape(b * n, t, w))       # (b*n, P, C, W)
        c = h.shape[2]
        
        # 2. Shared convolutional contracting backbone
        h = self.backbone(h.reshape(b * n * P, c, w))         # (b*n*P, dim, L)
        h = h.transpose(1, 2)
        
        # 3. Positional self-attention over sequence length
        h, _ = self.pos_attn(h, h, h)
        h = h.max(dim=1)[0].view(b, n, P, self.dim)           # (b, n, P, dim)
        
        # 4. Chunk attention pooling
        h = h.permute(0, 2, 1, 3).reshape(b * P, n, self.dim)
        m = chunk_mask.unsqueeze(1).expand(b, P, n).reshape(b * P, n)
        pf = self.chunk_aggregator(h, m).view(b, P, self.dim) # (b, P, dim)

        # 5. Relational triangle layer over taxon pairs
        H = pf.new_zeros(b, t, t, self.dim)
        H[:, i0, i1] = pf
        H[:, i1, i0] = pf
        T = torch.einsum('bikd,bjkd->bijd', H, H) / max(t - 2, 1)
        pf = pf + self.tri(T[:, i0, i1])
        
        s = self.rel_head(torch.cat([pf, pf.mean(1, keepdim=True).expand_as(pf)], -1)).squeeze(-1).float()  # (b, P)

        S = s.new_zeros(b, t, t)
        S[:, i0, i1] = s
        S[:, i1, i0] = s

        # 6. Content-based reference taxon anchor (permutation-equivariant)
        pos = torch.arange(w, device=x_chunks.device)
        q = self.ref_w[pos, x_chunks.long()].sum(dim=(1, 3))  # (b, t)
        r = q.argmax(dim=1)
        ar = torch.arange(b, device=x_chunks.device)
        z = S[ar, :, r]                                       # z_i = S_{i, r}
        
        is_r = F.one_hot(r, t).bool()
        conf = (z.abs() * (~is_r)).sum(1, keepdim=True) / max(t - 1, 1)
        z = torch.where(is_r, conf.expand_as(z), z)           # z_r = +confidence
        
        if self.center:
            z = z - z.mean(dim=-1, keepdim=True)
            
        z = torch.clamp(z, -15.0, 15.0)
        return (z, s) if return_pairs else z
