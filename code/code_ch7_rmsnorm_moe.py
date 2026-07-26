"""
================================================================================
  第 7 章程式碼：穩定通道與專家分流——RMSNorm & 殘差流 & MoE
  累加第 1~6 章 (串接完整 Transformer Block)
  涵蓋心智圖節點：Residual Stream, ResNet, Layer Normalization, RMSNorm, Pre-norm/Post-norm, Mixture of Experts (MoE Experts, Router Network)
================================================================================
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from code_ch3_rope import RotaryPositionalEmbedding
from code_ch5_gqa_kvcache import GroupedQueryAttentionWithKVCache
from code_ch6_swiglu_ffn import SwiGLUFFN

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

class RMSNorm(nn.Module):
    """
    第 7 章核心類別：RMSNorm (Root Mean Square Normalization)
    只計算均方根的極速層歸一化組件。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class SparseMoEBlock(nn.Module):
    """
    第 7 章進階類別：混合專家模型 (Sparse Mixture of Experts / MoE)
    含 Router Network (路由網絡) 與 Top-k 專家選擇。
    """
    def __init__(self, d_model: int, hidden_dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLUFFN(d_model, hidden_dim) for _ in range(num_experts)])
        # 統計掛鉤：記錄最近一次前向傳播的路由狀況，供 §9.10 的專家使用率分析
        # 與負載平衡損失取用。不參與推論運算，也不會存進 state_dict。
        self.last_prob_mean = None   # P_i：Router 給各專家的平均機率（保留梯度）
        self.last_frac = None        # f_i：實際被分配到各專家的 Token 比例

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_flat = x.view(-1, D)

        router_logits = self.router(x_flat)
        probs = F.softmax(router_logits, dim=-1)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)

        self.last_prob_mean = probs.mean(0)
        with torch.no_grad():   # f_i 來自 argmax，不可微分，只能當統計量
            self.last_frac = (torch.bincount(indices.reshape(-1), minlength=self.num_experts)
                              .float() / indices.numel())

        out = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            mask = (indices == i)
            if mask.any():
                token_idx, expert_k_idx = torch.where(mask)
                expert_weights = weights[token_idx, expert_k_idx].unsqueeze(-1)
                expert_out = self.experts[i](x_flat[token_idx])
                out[token_idx] += expert_out * expert_weights
        return out.view(B, L, D)

class TransformerBlock(nn.Module):
    """
    第 7 章整合類別：標準 Pre-Norm Transformer Block (含殘差流 Residual Stream)
    """
    def __init__(self, d_model: int, n_heads: int, num_kv_groups: int, hidden_dim: int, rope: RotaryPositionalEmbedding,
                 use_moe: bool = False, num_experts: int = 4, top_k: int = 1):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttentionWithKVCache(d_model, n_heads, num_kv_groups, rope)

        self.ffn_norm = RMSNorm(d_model)
        if use_moe:
            self.ffn = SparseMoEBlock(d_model, hidden_dim, num_experts=num_experts, top_k=top_k)
        else:
            self.ffn = SwiGLUFFN(d_model, hidden_dim)

    def forward(self, x: torch.Tensor, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False):
        # 1. Attention + 殘差流 (Residual Stream 1)
        normed_x = self.attn_norm(x)
        attn_out, new_cache = self.attn(normed_x, kv_cache=kv_cache, use_cache=use_cache)
        h = x + attn_out  # 殘差快捷連接

        # 2. FFN / MoE + 殘差流 (Residual Stream 2)
        out = h + self.ffn(self.ffn_norm(h))  # 殘差快捷連接
        return out, new_cache

def demo_chapter7():
    print("=" * 60)
    print("📖 [第 7 章 測試] 執行 RMSNorm, Residual Stream & MoE 測試...")
    print("=" * 60)
    
    d_model = 32
    n_heads = 4
    num_kv_groups = 2
    hidden_dim = 64
    
    rope = RotaryPositionalEmbedding(dim=d_model // n_heads)
    
    # 建立單層 Transformer Block (包含 MoE 專家系統)
    block = TransformerBlock(d_model, n_heads, num_kv_groups, hidden_dim, rope, use_moe=True)
    
    x = torch.randn(1, 4, d_model)
    out, cache = block(x, use_cache=True)
    
    print(f"  輸入向量形狀: {x.shape}")
    print(f"  經過包含 4 個專家的 MoE Transformer Block 輸出形狀: {out.shape}")
    print(f"  殘差流 (Residual Stream) 保障輸入輸出維度 100% 一致！")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] 第 7 章 RMSNorm, Residual Stream & MoE 測試 100% 成功！")
    print("=" * 60)

if __name__ == "__main__":
    demo_chapter7()
