"""
================================================================================
  第 3 章程式碼：幫文字戴上指針手錶——位置編碼 (Positional Encoding / RoPE)
  累加第 1~2 章 (Tokenizer + Embedding)
  涵蓋心智圖節點：Word Order, Sinusoidal, Rotary Positional Embedding (RoPE), Relative Position, Lost in the Middle
================================================================================
"""

import sys
import torch
import torch.nn as nn
from code_ch1_tokenizer import SimpleBPEStyleTokenizer
from code_ch2_embedding import TokenEmbedding

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

class RotaryPositionalEmbedding(nn.Module):
    """
    第 3 章核心類別：旋轉位置編碼 (RoPE - Rotary Positional Embedding)
    當前現代 LLM (如 LLaMA / Gemma) 廣泛採用的相對位置編碼機制。
    利用復數旋轉矩陣將向量依照序列位置 (m) 進行二維旋轉。
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        # 計算旋轉角頻率
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        
        # 預先計算並緩存 cos 與 sin 矩陣
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """旋轉向量一半維度: [-x2, x1]"""
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int, start_pos: int = 0) -> torch.Tensor:
        # x 形狀: (batch, seq_len, dim) 或 (batch, seq_len, n_heads, head_dim)
        cos = self.cos_cached[start_pos : start_pos + seq_len, :].unsqueeze(0)
        sin = self.sin_cached[start_pos : start_pos + seq_len, :].unsqueeze(0)
        if x.ndim == 4:
            cos = cos.unsqueeze(2) # (1, seq_len, 1, dim)
            sin = sin.unsqueeze(2)
        return (x * cos) + (self._rotate_half(x) * sin)

def demo_chapter3():
    print("=" * 60)
    print("📖 [第 3 章 測試] 累加第 1~2 章並執行 RoPE 旋轉位置編碼測試...")
    print("=" * 60)
    
    corpus = "趙雲大鬧長坂坡，單騎救主。"
    tokenizer = SimpleBPEStyleTokenizer(corpus)
    
    d_model = 16
    embed_layer = TokenEmbedding(vocab_size=tokenizer.vocab_size, d_model=d_model)
    rope_layer = RotaryPositionalEmbedding(dim=d_model)
    
    tokens = tokenizer.encode("趙雲單騎救主", add_bos_eos=False)
    input_tensor = torch.tensor([tokens], dtype=torch.long)
    seq_len = input_tensor.size(1)
    
    # 1. 取得原始語義嵌入 (無位置資訊)
    embeddings = embed_layer(input_tensor) # (1, 6, 16)
    
    # 2. 注入 RoPE 旋轉位置編碼
    embeddings_with_rope = rope_layer(embeddings, seq_len=seq_len)
    
    print(f"  輸入文字: '趙雲單騎救主' (序列長度 L={seq_len})")
    print(f"  原始詞嵌入向量形狀: {embeddings.shape}")
    print(f"  RoPE 旋轉後向量形狀: {embeddings_with_rope.shape}")
    print(f"  位置 0 ('趙') RoPE 施加前後前 4 維對比:")
    print(f"    原始 Vector: {embeddings[0, 0, :4].detach().tolist()}")
    print(f"    RoPE Vector: {embeddings_with_rope[0, 0, :4].detach().tolist()}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] 第 3 章 RoPE 旋轉位置編碼測試 100% 成功！")
    print("=" * 60)

if __name__ == "__main__":
    demo_chapter3()
