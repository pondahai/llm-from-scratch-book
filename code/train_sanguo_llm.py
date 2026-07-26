"""
================================================================================
  四大名著《三國演義》文本 - MiniLLM 預訓練範例
  本腳本讀取 local 文本 data/sanguo_yanyi.txt
  演示如何使用傳統公有領域文本進行大語言模型 (LLM) 預訓練。
================================================================================
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional

# 強制 stdout 使用 UTF-8 編碼
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 匯入先前心智圖構建的 MiniLLM 模組
from llm_from_scratch_mindmap import (
    SimpleBPEStyleTokenizer,
    TokenEmbedding,
    RotaryPositionalEmbedding,
    TransformerBlock,
    RMSNorm,
    MiniLLM
)

class TextDataset(Dataset):
    """將小說長文本分割為指定長度 (seq_len) 的批次數據集"""
    def __init__(self, token_ids: List[int], seq_len: int = 64):
        self.seq_len = seq_len
        # 切分為多個 (seq_len + 1) 長度的片段
        num_samples = (len(token_ids) - 1) // seq_len
        self.data = []
        for i in range(num_samples):
            chunk = token_ids[i * seq_len : (i + 1) * seq_len + 1]
            self.data.append(chunk)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = torch.tensor(self.data[idx], dtype=torch.long)
        x = chunk[:-1]  # 輸入序列 (Input)
        y = chunk[1:]   # 目標序列 (Target / Next Token)
        return x, y


def train_sanguo_demo():
    sanguo_path = r"data/sanguo_yanyi.txt"
    if not os.path.exists(sanguo_path):
        print(f"找不到檔案: {sanguo_path}")
        return

    print("=" * 80)
    print("📖 [四大名著 LLM 預訓練] 開始載入《三國演義》文本...")
    print("=" * 80)

    # 1. 讀取三國演義文本
    with open(sanguo_path, "r", encoding="utf-8", errors="ignore") as f:
        full_text = f.read()

    print(f"  三國演義總字數: {len(full_text):,} 字")
    
    # 2. 構建詞表與 Tokenizer
    print("  構建詞表 (Vocabulary)...")
    tokenizer = SimpleBPEStyleTokenizer(full_text[:50000]) # 採樣建立詞表
    print(f"  詞表大小: {tokenizer.vocab_size} 個 Token")

    # 取前 20,000 字作為演示預訓練文本集
    train_corpus = full_text[:20000]
    token_ids = tokenizer.encode(train_corpus, add_bos_eos=False)
    
    seq_len = 64
    batch_size = 16
    dataset = TextDataset(token_ids, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"  建立數據集: 共有 {len(dataset)} 個序列樣本 (Sequence Length={seq_len})")

    # 3. 初始化 MiniLLM 模型
    model = MiniLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=3,
        n_heads=4,
        num_kv_groups=2,
        hidden_dim=256
    )
    print(f"  MiniLLM 模型參數量: {sum(p.numel() for p in model.parameters()):,} 個")

    # 4. 開始預訓練 Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("\n🚀 開始預訓練 (Next-Token Prediction)...")
    model.train()
    epochs = 3
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for step, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch}/{epochs} | 平均 預訓練 Loss: {avg_loss:.4f}")

    # 5. 測試文字生成 (Inference / Generation)
    print("\n📜 預訓練完成！進行三國風提示詞生成測試:")
    prompts = ["宴桃園", "曹操", "劉備"]
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt, add_bos_eos=False)
        gen_ids = model.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=30,
            temperature=0.7,
            top_k=5,
            eos_id=tokenizer.eos_id
        )
        generated_text = tokenizer.decode(gen_ids)
        print(f"  提示詞: '{prompt}' -> 生成續寫: '{generated_text}'")

    print("\n" + "=" * 80)
    print("[SUCCESS] 《三國演義》開放文本已成功進行 MiniLLM 預訓練驗證！")
    print("=" * 80)

if __name__ == "__main__":
    train_sanguo_demo()
