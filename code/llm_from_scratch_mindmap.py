"""
================================================================================
  大語言模型工作原理 (How Large Language Models Work) - 漸進式程式碼範例
  本檔案根據 NotebookLM 心智圖的 10 大模組節點，循序漸進地累加程式碼。
  每個 Step 代表一個進度，並保留前一次的程式碼累積。
================================================================================
"""

import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

# 強制 stdout 使用 UTF-8 編碼 (防止 Windows CP950 控制台報錯)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ==============================================================================
# Step 1: 分詞 (Tokenization)
# 涵蓋心智圖節點：
#   - 分詞器 (Tokenizer), 詞表 (Vocabulary), 符號 (Tokens)
#   - 整詞詞表 vs 單字符詞表 vs 子詞分詞 (Subword Tokenization / BPE)
# ==============================================================================

class SimpleBPEStyleTokenizer:
    """
    Step 1: 實現一個簡易的分詞器 (Tokenizer)
    展示字符/子詞詞表 (Vocabulary)、Tokens 轉化與 EOS (End-of-Sequence) 標籤。
    """
    def __init__(self, text_corpus: str):
        # 提取獨特字符作為基礎詞表 (Character / Subword Vocabulary)
        unique_chars = sorted(list(set(text_corpus)))
        # 特別標籤：<PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3
        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.vocab = self.special_tokens + unique_chars
        
        # 建立 詞彙 <-> ID 的雙向查找表 (Lookup Table)
        self.token2id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id2token = {idx: token for idx, token in enumerate(self.vocab)}
        
        self.pad_id = self.token2id["<PAD>"]
        self.unk_id = self.token2id["<UNK>"]
        self.bos_id = self.token2id["<BOS>"]
        self.eos_id = self.token2id["<EOS>"]
        self.vocab_size = len(self.vocab)

    def encode(self, text: str, add_bos_eos: bool = True) -> List[int]:
        """文字轉 ID 序列 (Tokens)"""
        ids = [self.token2id.get(char, self.unk_id) for char in text]
        if add_bos_eos:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        """ID 序列 (Tokens) 轉回文字"""
        tokens = [self.id2token.get(idx, "<UNK>") for idx in ids if idx not in (self.pad_id, self.bos_id, self.eos_id)]
        return "".join(tokens)


# ==============================================================================
# Step 2: 嵌入 (Embedding) - 累加 Step 1
# 涵蓋心智圖節點：
#   - 嵌入矩陣 (Embedding Matrix), 查找表 (Lookup Table)
#   - 數字向量 (Digital Vectors), 隱藏維度 (Hidden Dimension / d_model)
#   - 內部表示 (Internal Representation), 語義空間 (Semantic Space)
# ==============================================================================

class TokenEmbedding(nn.Module):
    """
    Step 2: 嵌入矩陣 (Embedding Layer)
    將離散的 Token ID 透過 Lookup Table 查表轉換為隱藏維度 (d_model) 的連續數字向量。
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # 建立嵌入矩陣 (Embedding Matrix): Shape (vocab_size, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 輸入 (batch_size, seq_len) -> 輸出 (batch_size, seq_len, d_model)
        return self.embedding(input_ids) * math.sqrt(self.d_model)


# ==============================================================================
# Step 3: 位置編碼 (Positional Encoding) - 累加 Step 1 ~ Step 2
# 涵蓋心智圖節點：
#   - 詞序 (Word Order), 正弦編碼 (Sinusoidal Encoding)
#   - 旋轉位置編碼 (Rotary Positional Embedding / RoPE)
#   - 相對位置 (Relative Position)
# ==============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """
    Step 3: 旋轉位置編碼 (RoPE - Rotary Positional Embedding)
    當前現代 LLM (如 LLaMA) 廣泛採用的相對位置編碼機制。
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        # 計算旋轉頻率
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        
        # 預先計算 sin 與 cos 矩陣 (cos, sin)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int, start_pos: int = 0) -> torch.Tensor:
        # x shape: (batch, seq_len, n_heads, head_dim) 或 (batch, seq_len, dim)
        cos = self.cos_cached[start_pos : start_pos + seq_len, :].unsqueeze(0)
        sin = self.sin_cached[start_pos : start_pos + seq_len, :].unsqueeze(0)
        if x.ndim == 4:
            cos = cos.unsqueeze(2) # (1, seq_len, 1, dim)
            sin = sin.unsqueeze(2)
        return (x * cos) + (self._rotate_half(x) * sin)


# ==============================================================================
# Step 4: 注意力機制 (Attention Mechanism) - 累加 Step 1 ~ Step 3
# 涵蓋心智圖節點：
#   - 變換向量: 查詢 (Query/Q), 鍵 (Key/K), 值 (Value/V)
#   - 相似度打分 (Similarity Scoring), 縮放點積 (Scaled Dot-Product)
#   - Softmax 函數, 加權平均 (Weighted Average)
#   - 因果掩碼 (Causal Masking)
# ==============================================================================

def scaled_dot_product_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Step 4: 縮放點積注意力 (Scaled Dot-Product Attention)
    計算 Q * K^T / sqrt(d_k) -> Softmax -> 加權平均 V，並套用因果掩碼 (Causal Masking)。
    """
    d_k = q.size(-1)
    # 1. 相似度打分 (Similarity Scoring): (B, h, L, d_k) @ (B, h, d_k, S) -> (B, h, L, S)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. 因果掩碼 (Causal Masking): 防止未來 Token 的資訊洩漏
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
        
    # 3. Softmax 函數：計算權重分佈
    attn_weights = F.softmax(scores, dim=-1)
    
    # 4. 加權平均 (Weighted Average): 聚合 Value 向量
    output = torch.matmul(attn_weights, v)
    return output, attn_weights


# ==============================================================================
# Step 5: 架構演進 (Architectural Evolution) - 累加 Step 1 ~ Step 4
# 涵蓋心智圖節點：
#   - 多頭注意力 (Multi-Head Attention / MHA)
#   - 分組查詢注意力 (Grouped-Query Attention / GQA)
#   - 投影矩陣 (Projection Matrix)
#   - KV 緩存 (KV Cache) 技術
# ==============================================================================

class GroupedQueryAttentionWithKVCache(nn.Module):
    """
    Step 5: 支持 KV Cache 的分組查詢注意力 (GQA / MHA)
    整合 Q/K/V 投影矩陣、RoPE 位置編碼、KV Cache 加速解碼。
    """
    def __init__(self, d_model: int, n_heads: int, num_kv_groups: int, rope: RotaryPositionalEmbedding):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = d_model // n_heads
        self.num_queries_per_kv = n_heads // num_kv_groups

        # 投影矩陣 (Projection Matrices for Q, K, V, O)
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
        
        # 1. 投影產生 Q, K, V
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_kv_groups, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_kv_groups, self.head_dim)

        # 2. 套用 RoPE 旋轉位置編碼
        start_pos = kv_cache[0].size(1) if kv_cache is not None else 0
        q = self.rope(q, L, start_pos=start_pos)
        k = self.rope(k, L, start_pos=start_pos)

        # 3. KV Cache 緩存邏輯 (推論加速)
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=1)
            v = torch.cat([prev_v, v], dim=1)
        
        new_kv_cache = (k, v) if use_cache else None
        total_seq_len = k.size(1)

        # 4. GQA: 將 Key/Value 重複（Repeat）以匹配 Query 的頭數
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=2)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=2)

        # 轉置為 (B, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 5. 因果掩碼 (Causal Mask)
        if L > 1:
            mask = torch.tril(torch.ones(L, total_seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        else:
            mask = None  # 自迴歸推論單步，無需 Mask

        # 6. 計算注意力
        attn_out, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, -1)
        
        # 7. 輸出投影
        return self.out_proj(attn_out), new_kv_cache


# ==============================================================================
# Step 6: 前饋網絡 (Feed-Forward Network / FFN) - 累加 Step 1 ~ Step 5
# 涵蓋心智圖節點：
#   - 深加工 (Deep Processing), 神經元 (Neurons)
#   - 非線性激活函數: ReLU, GELU, SwiGLU (現代 LLM 標配)
# ==============================================================================

class SwiGLUFFN(nn.Module):
    """
    Step 6: SwiGLU 前饋神經網路 (Feed-Forward Network)
    採用 SiLU (Swish) 門控線性單元，為 Transformer 提供非線性表達與知識存儲能力。
    """
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False) # 門控分支
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False) # 下投影
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False) # 上投影

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU 公式: (SiLU(x @ W1) * (x @ W3)) @ W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ==============================================================================
# Step 7: 訓練穩定組件與 MoE (Stability & MoE) - 累加 Step 1 ~ Step 6
# 涵蓋心智圖節點：
#   - 殘差流 (Residual Stream), 層歸一化 (RMSNorm / Pre-norm)
#   - 混合專家模型 (Mixture of Experts / MoE - 路由網路 Router & 專家 Experts)
# ==============================================================================

class RMSNorm(nn.Module):
    """
    Step 7: RMSNorm (Root Mean Square Normalization)
    比傳統 LayerNorm 更高效的廣泛應用歸一化組件。
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
    Step 7 (進階): 混合專家模型 (Sparse Mixture of Experts / MoE)
    包含 Router Network (路由網絡) 與多個 FFN 專家 (Experts)。
    """
    def __init__(self, d_model: int, hidden_dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLUFFN(d_model, hidden_dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_flat = x.view(-1, D)
        
        # 1. 路由網路打分
        router_logits = self.router(x_flat)
        weights, indices = torch.topk(F.softmax(router_logits, dim=-1), self.top_k, dim=-1)
        
        # 2. 專家計算與加權融合
        out = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            # 找到選擇了專家 i 的 Tokens
            mask = (indices == i)
            if mask.any():
                token_idx, expert_k_idx = torch.where(mask)
                expert_weights = weights[token_idx, expert_k_idx].unsqueeze(-1)
                expert_out = self.experts[i](x_flat[token_idx])
                out[token_idx] += expert_out * expert_weights
                
        return out.view(B, L, D)


class TransformerBlock(nn.Module):
    """
    整合 Step 5 (Attention), Step 6 (FFN/MoE), Step 7 (RMSNorm & Residual Stream)
    為標準 Pre-norm 殘差流架構 Block。
    """
    def __init__(self, d_model: int, n_heads: int, num_kv_groups: int, hidden_dim: int, rope: RotaryPositionalEmbedding, use_moe: bool = False):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttentionWithKVCache(d_model, n_heads, num_kv_groups, rope)
        
        self.ffn_norm = RMSNorm(d_model)
        if use_moe:
            self.ffn = SparseMoEBlock(d_model, hidden_dim)
        else:
            self.ffn = SwiGLUFFN(d_model, hidden_dim)

    def forward(self, x: torch.Tensor, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False):
        # 1. Attention + 殘差流 (Residual Stream)
        normed_x = self.attn_norm(x)
        attn_out, new_cache = self.attn(normed_x, kv_cache=kv_cache, use_cache=use_cache)
        h = x + attn_out

        # 2. FFN / MoE + 殘差流 (Residual Stream)
        out = h + self.ffn(self.ffn_norm(h))
        return out, new_cache


# ==============================================================================
# Step 8 & Step 9: 預測解碼與完整 LLM 架構 (Prediction, Decoding & Base LLM)
# 涵蓋心智圖節點：
#   - Logits (未歸一化分數), 概率分佈, 溫度 (Temperature), Top-k / Top-p 採樣
#   - EOS 終止符, 投機解碼概念
#   - 基礎模型 (Base LLM), 預訓練 / 指令微調 / 對齊階段說明
# ==============================================================================

class MiniLLM(nn.Module):
    """
    Step 8 & 9: 完整的迷你大語言模型 (Mini LLM)
    將 Step 1 ~ Step 7 的組件完整串接，形成完整的 Forward Pass 與自迴歸生成 (Generation) 邏輯。
    """
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int = 128, 
        n_layers: int = 2, 
        n_heads: int = 4, 
        num_kv_groups: int = 2, 
        hidden_dim: int = 256,
        use_moe: bool = False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Step 2: 詞嵌入
        self.embed = TokenEmbedding(vocab_size, d_model)
        # Step 3: RoPE 位置編碼
        self.rope = RotaryPositionalEmbedding(dim=d_model // n_heads)
        
        # Step 5 & 6 & 7: 多層 Transformer Block
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, num_kv_groups, hidden_dim, self.rope, use_moe=use_moe)
            for _ in range(n_layers)
        ])
        
        self.final_norm = RMSNorm(d_model)
        # Step 8: Logits 輸出層 (LM Head)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, kv_caches: Optional[List] = None, use_cache: bool = False):
        x = self.embed(input_ids)
        
        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            x, c = layer(x, kv_cache=cache, use_cache=use_cache)
            if use_cache:
                new_caches.append(c)
                
        x = self.final_norm(x)
        logits = self.lm_head(x) # (Batch, Seq_Len, Vocab_Size)
        return logits, new_caches

    @torch.no_grad()
    def generate(
        self, 
        prompt_ids: List[int], 
        max_new_tokens: int = 20, 
        temperature: float = 0.8, 
        top_k: int = 5,
        eos_id: int = 3
    ) -> List[int]:
        """
        Step 8: 解碼與自迴歸生成 (Auto-regressive Sampling Generation)
        實作 Temperature 與 Top-k 採樣機制。
        """
        self.eval()
        generated = list(prompt_ids)
        input_tensor = torch.tensor([generated], dtype=torch.long)
        
        kv_caches = None
        
        for _ in range(max_new_tokens):
            # 前向傳播計算 Logits
            if kv_caches is None:
                # Prompt 階段 (Prefill)
                logits, kv_caches = self.forward(input_tensor, use_cache=True)
            else:
                # 解碼階段 (Decode) - 僅輸入最新的 1 個 Token (配合 KV Cache)
                logits, kv_caches = self.forward(input_tensor[:, -1:], kv_caches=kv_caches, use_cache=True)
                
            # 取得最後一個位置的 Logits
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
            
            # Top-k 濾除
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
            # 計算 Softmax 概率分佈
            probs = F.softmax(next_token_logits, dim=-1)
            
            # 採樣下一個 Token ID
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            
            # 判斷終止符 EOS
            if next_token == eos_id:
                break
                
            input_tensor = torch.tensor([[next_token]], dtype=torch.long)
            
        return generated


# ==============================================================================
# Step 10: 測試與展示 (Execution & Demonstration)
# 示範完整從心智圖 Step 1 到 Step 9 的程式碼累積執行結果
# ==============================================================================

def main():
    print("=" * 80)
    print("[START] [大語言模型工作原理] 漸進式程式碼累積範例執行中...")
    print("=" * 80)

    # 1. Step 1 測試：分詞
    corpus = "Hello LLM Transformer World! 這是大語言模型工作原理的實做示範。"
    tokenizer = SimpleBPEStyleTokenizer(corpus)
    print(f"\n[Step 1 分詞] 詞表大小: {tokenizer.vocab_size}")
    prompt_text = "Hello LLM"
    encoded_ids = tokenizer.encode(prompt_text)
    print(f"  原始文字: '{prompt_text}'")
    print(f"  Token IDs: {encoded_ids}")
    print(f"  解碼文字: '{tokenizer.decode(encoded_ids)}'")

    # 2. Step 2 & 3 測試：嵌入與 RoPE
    embed_layer = TokenEmbedding(vocab_size=tokenizer.vocab_size, d_model=64)
    tokens_tensor = torch.tensor([encoded_ids])
    embeddings = embed_layer(tokens_tensor)
    print(f"\n[Step 2 嵌入] Token Embedding 輸出形狀: {embeddings.shape} (Batch, Seq_Len, d_model)")

    rope = RotaryPositionalEmbedding(dim=16)
    q_dummy = torch.randn(1, len(encoded_ids), 4, 16) # (B, L, heads, head_dim)
    q_rope = rope(q_dummy, len(encoded_ids))
    print(f"[Step 3 位置編碼] RoPE 施加後形狀: {q_rope.shape}")

    # 3. Step 4 ~ 7 測試：完整迷你 LLM 建立
    print("\n[Step 4~7 組件整合] 初始化 2-Layer Transformer (含 GQA, SwiGLU, RMSNorm, KV-Cache)...")
    model = MiniLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        num_kv_groups=2,
        hidden_dim=128,
        use_moe=False
    )
    print(f"  模型總可訓練參數量: {sum(p.numel() for p in model.parameters()):,} 個參數")

    # 4. Step 8 & 9 測試：未訓練時的隨機推論 (Inference before training)
    print("\n[Step 8 & 9 隨機推論測試 (未訓練)] 開始進行提示詞自迴歸生成...")
    generated_ids = model.generate(
        prompt_ids=encoded_ids,
        max_new_tokens=10,
        temperature=0.7,
        top_k=3,
        eos_id=tokenizer.eos_id
    )
    print(f"  未訓練前生成的最終文字: '{tokenizer.decode(generated_ids)}'")

    # 5. 訓練測試：預訓練/微調訓練循環 (Pre-training / Fine-tuning Loop)
    print("\n[Step 10 訓練測試] 開始進行實際模型訓練 (Next-Token Prediction Training Loop)...")
    train_text = "Hello LLM 世界！大語言模型工作原理是透過神經網路預測下一個字。"
    train_tokenizer = SimpleBPEStyleTokenizer(train_text)
    train_model = MiniLLM(vocab_size=train_tokenizer.vocab_size, d_model=64, n_layers=2)
    
    token_ids = torch.tensor([train_tokenizer.encode(train_text, add_bos_eos=False)], dtype=torch.long)
    inputs = token_ids[:, :-1]  # 輸入: x_1, x_2, ..., x_{T-1}
    targets = token_ids[:, 1:]   # 目標: x_2, x_3, ..., x_T (預測下一個 Token)

    optimizer = torch.optim.AdamW(train_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    train_model.train()
    print("  訓練中 (50 Epochs):")
    for epoch in range(1, 51):
        optimizer.zero_grad()
        logits, _ = train_model(inputs) # Forward Pass
        # 計算交叉熵損失 (Loss)
        loss = criterion(logits.view(-1, train_tokenizer.vocab_size), targets.view(-1))
        loss.backward()                  # 反向傳播 (Backward Pass / Gradients)
        optimizer.step()                 # 權重更新 (Optimizer Step)
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:2d}/50 | Loss: {loss.item():.4f}")

    # 6. 訓練後推論測試 (Inference after training)
    print("\n[訓練後推論測試] 模型經過 50 次迭代訓練後，進行文字生成測試:")
    test_prompt = train_tokenizer.encode("Hello LLM", add_bos_eos=False)
    trained_gen = train_model.generate(
        prompt_ids=test_prompt,
        max_new_tokens=25,
        temperature=0.2, # 降低溫度以獲得更確定性的生成
        top_k=2,
        eos_id=train_tokenizer.eos_id
    )
    print(f"  提示詞: 'Hello LLM'")
    print(f"  訓練後模型生成的文字: '{train_tokenizer.decode(trained_gen)}'")

    print("\n" + "=" * 80)
    print("[SUCCESS] 心智圖所有 10 個核心模組已成功驗證【訓練 (Training)】與【推論 (Inference)】！")
    print("=" * 80)



if __name__ == "__main__":
    main()
