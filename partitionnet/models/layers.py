import torch
import torch.nn as nn
import torch.nn.functional as F


class PairFeatureBuilder(nn.Module):
    """Per site, per unordered pair {i, j}: computes features symmetric in (i, j)
    and valid for any n taxa.
    
    Output shape: (B, P, C, W), where P = n * (n - 1) / 2.
    """
    def __init__(self, embed_dim: int = 16, out_channels: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(5, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(3 * embed_dim + 4, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n, W) uint8, with 4 representing padding.
        """
        x = x.long()
        B, n, W = x.shape
        i0, i1 = torch.triu_indices(n, n, 1, device=x.device)
        mask = (x != 4).any(dim=1).float()  # symmetric across taxa
        
        e = self.embedding(x) * mask[:, None, :, None]  # (B, n, W, d)
        ei, ej = e[:, i0], e[:, i1]
        ctx = e.sum(1, keepdim=True).expand_as(ei)      # all-taxa symmetric context
        
        valid = (x != 4).float()
        frac = (x.unsqueeze(2) == x.unsqueeze(1)).float().sum(2) * valid / n  # share of taxa with same state
        fi, fj = frac[:, i0], frac[:, i1]
        
        eq = ((x[:, i0] == x[:, i1]) & (x[:, i0] != 4)).float()
        scal = torch.stack([eq, fi + fj, fi * fj, (fi - fj).abs()], dim=-1)
        
        f = torch.cat([ei + ej, ei * ej, ctx, scal], dim=-1)  # symmetric in (i, j)
        h = self.norm(self.mlp(f)) * mask[:, None, :, None]
        return h.permute(0, 1, 3, 2).contiguous()            # (B, P, C, W)


class DilatedContractingBlock(nn.Module):
    """Hierarchical 1D multi-dilation convolutional block with residual connection."""
    def __init__(self, in_channels: int = 64, mid_channels: int = 64, out_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, 7, padding=3),
            nn.BatchNorm1d(mid_channels),
            nn.SiLU(),
            nn.MaxPool1d(4, 4)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, 5, padding=4, dilation=2),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.MaxPool1d(4, 4)
        )
        self.res_proj = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=16),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.conv2(self.conv1(x)) + self.res_proj(x))


class ChunkAttentionAggregation(nn.Module):
    """Learned attention pooling across chunks/windows with padding mask handling."""
    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        s = self.attn(feats).squeeze(-1).masked_fill(mask, -1e4)
        w = torch.nan_to_num(F.softmax(s, dim=1), nan=0.0)
        return (feats * w.unsqueeze(-1)).sum(dim=1)
