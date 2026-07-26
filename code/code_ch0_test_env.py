"""
================================================================================
  第 0 章程式碼：環境準備與極速驗證 (純 CPU 驗證)
================================================================================
"""

import sys
import torch

# 強制 UTF-8 輸出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def verify_environment():
    print("=" * 60)
    print("🛠️ [第 0 章 驗證] 正在測試本地 PyTorch CPU 環境...")
    print("=" * 60)
    
    # 檢查 PyTorch 版本
    print(f"  PyTorch 版本: {torch.__version__}")
    
    # 測試 CPU 張量維度與基礎運算
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([10.0, 20.0, 30.0, 40.0])
    z = x + y
    
    print(f"  張量 x: {x.tolist()}")
    print(f"  張量 y: {y.tolist()}")
    print(f"  向量相加結果 (x + y): {z.tolist()}")
    
    # 矩陣乘法測試 (Matrix Multiplication)
    mat_a = torch.randn(2, 4)
    mat_b = torch.randn(4, 2)
    mat_c = torch.matmul(mat_a, mat_b)
    print(f"  2x4 與 4x2 矩陣乘法測試成功，輸出形狀: {mat_c.shape}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] 🎉 恭喜！PyTorch CPU 環境驗證成功，本地端可順暢進行模型訓練！")
    print("=" * 60)

if __name__ == "__main__":
    verify_environment()
