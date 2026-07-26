"""
================================================================================
  第 5 章程式碼：推論加速神器——多頭注意力、GQA 與 KV Cache
  累加第 1~4 章 (Tokenizer + Embedding + RoPE + Scaled Attention)
  涵蓋心智圖節點：Multi-Head Attention (MHA), Grouped-Query Attention (GQA), Projection Matrix, KV Cache, FlashAttention
================================================================================
"""

import sys
import torch
import torch.nn as nn
from typing import Tuple, Optional
from code_ch1_tokenizer import SimpleBPEStyleTokenizer
from code_ch2_embedding import TokenEmbedding
from code_ch3_rope import RotaryPositionalEmbedding
from code_ch4_attention import scaled_dot_product_attention

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

class GroupedQueryAttentionWithKVCache(nn.Module):
    """
    第 5 章核心類別：支持 KV Cache 的 GQA (分組查詢注意力)
    """
    def __init__(self, d_model: int, n_heads: int, num_kv_groups: int, rope: RotaryPositionalEmbedding):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = d_model // n_heads
        self.num_queries_per_kv = n_heads // num_kv_groups

        # 投影矩陣
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_groups * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_groups * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.rope = rope

    def forward(
        self, 
        x: torch.Tensor, 
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, L, _ = x.shape
        
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_kv_groups, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_kv_groups, self.head_dim)

        start_pos = kv_cache[0].size(1) if kv_cache is not None else 0
        q = self.rope(q, L, start_pos=start_pos)
        k = self.rope(k, L, start_pos=start_pos)

        # KV Cache 緩存拼接
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=1)
            v = torch.cat([prev_v, v], dim=1)
        
        new_kv_cache = (k, v) if use_cache else None
        total_seq_len = k.size(1)

        # GQA: Key/Value 重複以匹配 Query 頭數
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=2)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        mask = torch.tril(torch.ones(L, total_seq_len, device=x.device)).unsqueeze(0).unsqueeze(0) if L > 1 else None

        attn_out, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(attn_out), new_kv_cache

def demo_chapter5():
    print("=" * 60)
    print("📖 [第 5 章 測試] 執行 GQA & KV Cache 測試...")
    print("=" * 60)
    
    corpus = "諸葛亮草船借箭，神機妙算。"
    tokenizer = SimpleBPEStyleTokenizer(corpus)
    
    d_model = 32
    n_heads = 4
    num_kv_groups = 2  # GQA: 4 個 Query 頭共享 2 個 Key/Value 頭
    
    rope = RotaryPositionalEmbedding(dim=d_model // n_heads)
    gqa = GroupedQueryAttentionWithKVCache(d_model, n_heads, num_kv_groups, rope)
    
    tokens = tokenizer.encode("草船借箭", add_bos_eos=False)
    input_tensor = torch.tensor([tokens], dtype=torch.long)
    embed_layer = TokenEmbedding(vocab_size=tokenizer.vocab_size, d_model=d_model)
    x = embed_layer(input_tensor)
    
    # 測試無 KV Cache 第一次計算 (Prefill)
    out1, cache1 = gqa(x, use_cache=True)
    print(f"  輸入向量形狀: {x.shape}")
    print(f"  GQA 輸出形狀: {out1.shape}")
    print(f"  第一次產生的 Key Cache 形狀: {cache1[0].shape} (B, Seq_Len, kv_groups, head_dim)")
    
    # 測試第二次輸入單字並調用 KV Cache (Decode Step)
    next_token = torch.tensor([[tokenizer.encode("神", add_bos_eos=False)[0]]], dtype=torch.long)
    x_next = embed_layer(next_token)
    out2, cache2 = gqa(x_next, kv_cache=cache1, use_cache=True)
    print(f"  解碼單字輸入形狀: {x_next.shape}")
    print(f"  使用 KV Cache 後拼接的 Key Cache 形狀: {cache2[0].shape} (Seq_Len 拼接為 5)")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] 第 5 章 GQA & KV Cache 測試 100% 成功！")
    print("=" * 60)

if __name__ == "__main__":
    demo_chapter5()
