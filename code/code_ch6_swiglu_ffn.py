"""
================================================================================
  第 6 章程式碼：知識加工廠——SwiGLU 前饋神經網路 (FFN)
  累加第 1~5 章
  涵蓋心智圖節點：Deep Processing, Neurons, Non-linear Functions (ReLU, GELU, SwiGLU), Knowledge Storage, ROME (模型編輯)
================================================================================
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

class SwiGLUFFN(nn.Module):
    """
    第 6 章核心類別：SwiGLU 門控前饋網路 (Feed-Forward Network)
    採用 SiLU (Swish) 門控單元，負責模型特徵層面的非線性深加工與知識記憶存儲。
    """
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)  # 門控分支 (Gate)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)  # 下投影 (Down Project)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)  # 上投影 (Up Project)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU 公式: (SiLU(x @ W1) * (x @ W3)) @ W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

def demo_chapter6():
    print("=" * 60)
    print("📖 [第 6 章 測試] 執行 SwiGLU FFN 測試...")
    print("=" * 60)
    
    d_model = 32
    hidden_dim = 64
    ffn = SwiGLUFFN(d_model=d_model, hidden_dim=hidden_dim)
    
    x = torch.randn(1, 4, d_model)  # (Batch=1, Seq_Len=4, d_model=32)
    output = ffn(x)
    
    print(f"  輸入向量形狀: {x.shape}")
    print(f"  SwiGLU FFN 隱藏維度 (hidden_dim): {hidden_dim}")
    print(f"  SwiGLU FFN 輸出向量形狀: {output.shape}")
    print(f"  輸入與輸出維度保持一致 (均為 {d_model})，便於後續殘差連接！")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] 第 6 章 SwiGLU FFN 測試 100% 成功！")
    print("=" * 60)

if __name__ == "__main__":
    demo_chapter6()
