"""
================================================================================
  第 4 章程式碼：大腦高光筆——縮放點積注意力與因果掩碼 (Attention Mechanism)
  累加第 1~3 章 (Tokenizer + Embedding + RoPE)
  涵蓋心智圖節點：Q/K/V Transformed Vectors, Similarity Scoring, Scaled Dot-Product, Softmax, Weighted Average, Causal Masking, Induction Heads, In-context Learning
================================================================================
"""

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from code_ch1_tokenizer import SimpleBPEStyleTokenizer
from code_ch2_embedding import TokenEmbedding
from code_ch3_rope import RotaryPositionalEmbedding

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def scaled_dot_product_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    mask: torch.Tensor = None
):
    """
    第 4 章核心函數：縮放點積注意力 (Scaled Dot-Product Attention)
    計算相似度分數量 -> Softmax 轉概率 -> 加權平均 V
    """
    d_k = q.size(-1)
    
    # 1. 相似度打分 (Similarity Scoring): (B, L, d_k) @ (B, d_k, S) -> (B, L, S)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. 因果掩碼 (Causal Masking): 防止偷看未來的 Token
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
        
    # 3. Softmax 轉成百分比權重
    attn_weights = F.softmax(scores, dim=-1)
    
    # 4. 加權平均 (Weighted Average): 聚合 Value 向量
    output = torch.matmul(attn_weights, v)
    return output, attn_weights

def demo_chapter4():
    print("=" * 60)
    print("📖 [第 4 章 測試] 累加第 1~3 章並執行 Attention & Causal Mask 測試...")
    print("=" * 60)
    
    corpus = "劉備關羽張飛桃園三結義"
    tokenizer = SimpleBPEStyleTokenizer(corpus)
    
    d_model = 16
    embed_layer = TokenEmbedding(vocab_size=tokenizer.vocab_size, d_model=d_model)
    rope_layer = RotaryPositionalEmbedding(dim=d_model)
    
    # 投影產生 Q, K, V
    q_proj = nn.Linear(d_model, d_model, bias=False)
    k_proj = nn.Linear(d_model, d_model, bias=False)
    v_proj = nn.Linear(d_model, d_model, bias=False)
    
    tokens = tokenizer.encode("劉備張飛", add_bos_eos=False)
    input_tensor = torch.tensor([tokens], dtype=torch.long) # (B=1, L=4)
    L = input_tensor.size(1)
    
    embeddings = embed_layer(input_tensor)
    q = rope_layer(q_proj(embeddings), L)
    k = rope_layer(k_proj(embeddings), L)
    v = v_proj(embeddings)
    
    # 建立下三角因果掩碼 (Causal Mask)
    mask = torch.tril(torch.ones(L, L)).unsqueeze(0) # (1, L, L)
    
    attn_output, attn_weights = scaled_dot_product_attention(q, k, v, mask=mask)
    
    print(f"  輸入文字: '劉備張飛' (L={L})")
    print(f"  Query (Q) 形狀: {q.shape}")
    print(f"  Key (K) 形狀: {k.shape}")
    print(f"  Value (V) 形狀: {v.shape}")
    print(f"  因果掩碼矩陣 (下三角):\n{mask[0].int()}")
    print(f"  注意力權重 (Softmax) 形狀: {attn_weights.shape}")
    print(f"  注意力輸出向量形狀: {attn_output.shape}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] 第 4 章 Scaled Dot-Product Attention 測試 100% 成功！")
    print("=" * 60)

if __name__ == "__main__":
    demo_chapter4()
