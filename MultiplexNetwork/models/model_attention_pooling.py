import torch
import torch.nn as nn
import torch.nn.functional as F

# file: MultiplexNetwork/models/model_attention_pooling.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CohortEvidenceAttentionPooling(nn.Module):
    """
    Attention-based pooling that maps multiple medical embeddings (B, N, D_med)
    to a prompt embedding sequence (B, prompt_len, dim_qwen) and produces
    attention scores for neighbor evidence loss.
    """

    def __init__(
        self,
        dim_medclip: int,
        dim_qwen: int,
        prompt_len: int = 32,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_neighbors: int = 512,  # 用于 role embedding
    ):
        super().__init__()

        assert dim_qwen % n_heads == 0, "dim_qwen must be divisible by n_heads"

        self.dim_medclip = dim_medclip
        self.dim_qwen = dim_qwen
        self.prompt_len = prompt_len
        self.n_heads = n_heads

        # --- projection ---
        self.proj = nn.Linear(dim_medclip, dim_qwen)

        # --- role / position embedding（中心病人 vs Top-K 邻居）---
        self.role_emb = nn.Embedding(max_neighbors, dim_qwen)

        # --- learnable prompt queries ---
        self.prompt_queries = nn.Parameter(torch.randn(prompt_len, dim_qwen) * 0.02)

        # --- prompt FiLM（由中心病人调制）---
        self.prompt_film = nn.Linear(dim_qwen, dim_qwen * 2)

        # --- temperature for attention ---
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # --- multi-head attention ---
        self.mha = nn.MultiheadAttention(
            embed_dim=dim_qwen, num_heads=n_heads, batch_first=True, dropout=dropout
        )

        # --- post processing ---
        self.layer_norm = nn.LayerNorm(dim_qwen)
        self.dropout = nn.Dropout(dropout)

        # --- simple attention for evidence loss ---
        self.evidence_attn = nn.Linear(dim_qwen, 1)

    def forward(self, embs: torch.Tensor):
        """
        embs: [B, N, dim_medclip]
        returns:
            out: [B, prompt_len, dim_qwen]
            attn_scores: [B, N-1]  # 仅对邻居的注意力分布
        """
        if embs.ndim != 3:
            raise ValueError(f"Expected embs ndim=3, got {embs.ndim}")

        B, N, _ = embs.shape  # [B, TopK+1, D_medclip]
        device = embs.device

        # --- project medical embeddings ---
        kv = self.proj(embs)  # [B, N, dim_qwen]

        # --- add role / position embedding ---
        pos_ids = torch.arange(N, device=device).unsqueeze(0)  # [1, N]
        kv = kv + self.role_emb(pos_ids)  # [B, N, dim_qwen]

        # --- prepare prompt queries ---
        q = self.prompt_queries.unsqueeze(0).expand(B, -1, -1)  # [B, L, dim_qwen]

        # --- center-conditioned FiLM ---
        center = kv[:, 0, :]  # [B, dim_qwen]
        gamma, beta = self.prompt_film(center).chunk(2, dim=-1)
        q = q * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        # --- attention with temperature ---
        attn_out, _ = self.mha(q, kv / self.temperature, kv, need_weights=False)

        # --- residual + norm ---
        out = self.layer_norm(attn_out + q)
        out = self.dropout(out)

        # --- evidence attention score ---
        # kv[:, 0] is the target patient and has no entry in neighbor_labels.
        # Exclude it before normalizing so the evidence distribution is [B, K].
        if N < 2:
            raise ValueError("Evidence attention requires at least one neighbor")
        neighbor_kv = kv[:, 1:, :]
        evidence_weights = self.evidence_attn(neighbor_kv).squeeze(-1)  # [B, N-1]
        attn_scores = torch.softmax(evidence_weights, dim=1)  # [B, N-1]

        return out, attn_scores
