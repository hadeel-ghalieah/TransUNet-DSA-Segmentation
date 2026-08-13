# # models/dsa_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, window_size=5):
        super(SparseAttention, self).__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.window_size = window_size
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, inner_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv3d(inner_dim, dim, kernel_size=1)

    def forward(self, x):
        B, C, D, H, W = x.shape

        # Step 1: Extract QKV
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = [t.view(B, self.heads, self.dim_head, D, H, W) for t in qkv]

        out = torch.zeros_like(v)

        stride = self.window_size  # non-overlapping blocks

        for d in range(0, D, stride):
            for h in range(0, H, stride):
                for w in range(0, W, stride):
                    d_slice = slice(d, min(d + stride, D))
                    h_slice = slice(h, min(h + stride, H))
                    w_slice = slice(w, min(w + stride, W))

                    q_blk = q[:, :, :, d_slice, h_slice, w_slice]
                    k_blk = k[:, :, :, d_slice, h_slice, w_slice]
                    v_blk = v[:, :, :, d_slice, h_slice, w_slice]

                    q_flat = q_blk.flatten(-3)
                    k_flat = k_blk.flatten(-3)
                    v_flat = v_blk.flatten(-3)

                    attn_scores = torch.einsum('bhcN,bhcM->bhNM', q_flat, k_flat) * self.scale
                    attn_probs = F.softmax(attn_scores, dim=-1)
                    out_blk = torch.einsum('bhNM,bhcM->bhcN', attn_probs, v_flat)
                    out_blk = out_blk.view_as(q_blk)

                    out[:, :, :, d_slice, h_slice, w_slice] = out_blk

        out = out.view(B, -1, D, H, W)
        return self.to_out(out)

class DSA_TransUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(DSA_TransUNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.sparse_attention = SparseAttention(
            dim=64,
            heads=4,
            dim_head=32,
            window_size=5
        )

        self.decoder = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.sparse_attention(x)
        x = self.decoder(x)
        return x
