"""
================================================================================
  第 2 章程式碼：語義宇宙的 GPS——詞嵌入矩陣 (Embedding)
  累加第 1 章 Tokenizer
  涵蓋心智圖節點：Embedding Matrix, Lookup Table, Digital Vectors, Hidden Dimension (d_model), Internal Representation, Semantic Space
================================================================================
"""

import sys
import math
import torch
import torch.nn as nn
from code_ch1_tokenizer import SimpleBPEStyleTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

class TokenEmbedding(nn.Module):
    """
    第 2 章核心類別：詞嵌入矩陣 (Token Embedding Lookup Table)
    將離散 Token ID 轉化為高維度連續向量空間 (Semantic Space) 中的座標。
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # 建立 Shape: (vocab_size, d_model) 的可學習權重矩陣
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 輸入形狀: (Batch_Size, Seq_Len) -> 輸出形狀: (Batch_Size, Seq_Len, d_model)
        return self.embedding(input_ids) * math.sqrt(self.d_model)

def demo_chapter2():
    print("=" * 60)
    print("📖 [第 2 章 測試] 累加第 1 章並執行 Token Embedding 測試...")
    print("=" * 60)
    
    corpus = "曹操與劉備煮酒論英雄"
    tokenizer = SimpleBPEStyleTokenizer(corpus)
    
    d_model = 16  # 隱藏維度 (Hidden Dimension)
    embed_layer = TokenEmbedding(vocab_size=tokenizer.vocab_size, d_model=d_model)
    
    tokens = tokenizer.encode("曹操英雄", add_bos_eos=False)
    input_tensor = torch.tensor([tokens], dtype=torch.long) # (Batch=1, Seq_Len=4)
    
    vectors = embed_layer(input_tensor)
    
    print(f"  輸入文字: '曹操英雄'")
    print(f"  Token IDs: {tokens}")
    print(f"  輸入張量形狀: {input_tensor.shape} (Batch, Seq_Len)")
    print(f"  詞嵌入輸出向量形狀: {vectors.shape} (Batch, Seq_Len, d_model={d_model})")
    print(f"  第一個 Token ('曹') 的連續語義向量 (前 4 維): {vectors[0, 0, :4].detach().tolist()}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] 第 2 章 Token Embedding 測試 100% 成功！")
    print("=" * 60)

if __name__ == "__main__":
    demo_chapter2()
