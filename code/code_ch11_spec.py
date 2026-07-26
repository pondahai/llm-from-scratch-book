"""
================================================================================
  第 11 章程式碼：MiniLLM 技術規格說明書與參數量計算器 (Model Card & Spec)
================================================================================
"""

import sys
import torch
from corpus_util import build_tokenizer
from code_ch8_decoding import MiniLLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def analyze_minillm_spec():
    print("=" * 70)
    print("📋 [第 11 章] DIY MiniLLM 官方技術規格說明書 (Model Card)")
    print("=" * 70)
    
    # 規格超參數 (Hyperparameters)
    # 詞表大小不寫死，一律走全專案唯一的分詞器入口 (V = 6,178)，
    # 才不會跟書中 §11.1 的規格表與隨書檢查點對不上。
    vocab_size = build_tokenizer().vocab_size
    d_model = 128
    n_layers = 3
    n_heads = 4
    num_kv_groups = 2
    hidden_dim = 256
    max_seq_len = 2048
    
    model = MiniLLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        num_kv_groups=num_kv_groups,
        hidden_dim=hidden_dim
    )
    
    # 計算各模組參數量 (Parameter Breakdown)
    embed_params = sum(p.numel() for p in model.embed.parameters())
    layer_params = sum(p.numel() for p in model.layers.parameters())
    norm_params = sum(p.numel() for p in model.final_norm.parameters())
    head_params = sum(p.numel() for p in model.lm_head.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    print("\n1. 📊 模型核心超參數 (Model Configuration):")
    print(f"   • 詞表大小 (Vocab Size): {vocab_size:,}")
    print(f"   • 隱藏層維度 (d_model): {d_model}")
    print(f"   • Transformer 層數 (n_layers): {n_layers}")
    print(f"   • Query 注意力頭數 (n_heads): {n_heads}")
    print(f"   • Key/Value 分組數 (num_kv_groups): {num_kv_groups} (GQA 2:1)")
    print(f"   • FFN 隱藏維度 (hidden_dim): {hidden_dim} (SwiGLU)")
    print(f"   • 最長上下文序列 (Max Sequence Length): {max_seq_len} Tokens (RoPE)")
    
    print("\n2. 🧮 參數量細部拆解 (Parameter Breakdown):")
    print(f"   • Token Embedding 嵌入層 : {embed_params:>10,} 個參數 ({embed_params/total_params*100:.1f}%)")
    print(f"   • {n_layers}-Layer Transformer Block: {layer_params:>10,} 個參數 ({layer_params/total_params*100:.1f}%)")
    print(f"   • Final RMSNorm 歸一化層  : {norm_params:>10,} 個參數 ({norm_params/total_params*100:.1f}%)")
    print(f"   • LM Head 分數輸出層     : {head_params:>10,} 個參數 ({head_params/total_params*100:.1f}%)")
    print(f"   --------------------------------------------------")
    print(f"   • 模型總參數量 (Total Params) : {total_params:>10,} 個參數 (約 {total_params/10000:.0f} 萬)")
    
    print("\n3. 💻 本地 CPU 運算資源分析 (Resource Analysis):")
    fp32_mb = (total_params * 4) / (1024 * 1024)
    print(f"   • 模型單精度 (FP32) 記憶體佔用: {fp32_mb:.2f} MB")
    print(f"   • 推論前向傳播單步 FLOPs      : ~{total_params * 2 / 1e6:.2f} MFLOPs")
    print(f"   • 硬體要求 (Hardware Requirement): 任意 CPU / 128MB RAM 以上即可運行！")
    
    print("\n" + "=" * 70)
    print("[SUCCESS] 第 11 章 DIY MiniLLM 技術規格分析完成！")
    print("=" * 70)

if __name__ == "__main__":
    analyze_minillm_spec()
