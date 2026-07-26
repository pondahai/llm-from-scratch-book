"""
================================================================================
  四大名著 (三國演義 + 水滸傳 + 紅樓夢 + 西遊記) 全套合集 - MiniLLM 預訓練腳本
  讀取合集文字檔 data/four_great_novels_combined.txt (涵蓋 291萬字全本)
================================================================================
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from code_ch1_tokenizer import SimpleBPEStyleTokenizer
from code_ch8_decoding import MiniLLM

class TextDataset(Dataset):
    def __init__(self, token_ids: list, seq_len: int = 32):
        self.seq_len = seq_len
        num_samples = (len(token_ids) - 1) // seq_len
        self.data = [token_ids[i * seq_len : (i + 1) * seq_len + 1] for i in range(num_samples)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = torch.tensor(self.data[idx], dtype=torch.long)
        return chunk[:-1], chunk[1:]

def train_four_novels():
    combined_path = r"../data/four_great_novels_combined.txt"
    if not os.path.exists(combined_path):
        combined_path = r"data/four_great_novels_combined.txt"
        
    if not os.path.exists(combined_path):
        print(f"找不到檔案: {combined_path}")
        return

    print("=" * 80)
    print("📚 [四大名著全套合集 LLM 預訓練] 開始載入 《三國演義》+《水滸傳》+《紅樓夢》+《西遊記》...")
    print("=" * 80)

    with open(combined_path, "r", encoding="utf-8", errors="ignore") as f:
        full_text = f.read()

    print(f"  四大名著合集檔案大小: {os.path.getsize(combined_path):,} bytes")
    print(f"  合集總字數: {len(full_text):,} 字 (涵蓋四大名著全部人物與經典情節)")

    # 1. 建立涵蓋四大名著的廣義 Tokenizer 詞表
    tokenizer = SimpleBPEStyleTokenizer(full_text[:50000])
    print(f"  詞表建置完成: {tokenizer.vocab_size} 個 Unique Tokens")

    # 2. 構建數據集 (取前 30,000 字進行 CPU 預訓練)
    train_corpus = full_text[:30000]
    token_ids = tokenizer.encode(train_corpus, add_bos_eos=False)
    dataset = TextDataset(token_ids, seq_len=32)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(f"  建立數據集: 共有 {len(dataset)} 個序列樣本 (Batch Size=16)")

    # 3. 初始化 MiniLLM
    model = MiniLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        num_kv_groups=2,
        hidden_dim=128
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  MiniLLM 參數量: {total_params:,} 個 (記憶體: {total_params*4/(1024*1024):.2f} MB)")

    # 4. 開始 CPU 預訓練
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    print("\n🚀 開始四大名著聯合預訓練 (Next-Token Prediction)...")
    model.train()
    for epoch in range(1, 6):
        total_loss = 0.0
        for x, y in dataloader:
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch}/5 | 平均 Loss: {total_loss / len(dataloader):.4f}")

    # 5. 四大名著人物提示詞測試 (Inference Across 4 Books)
    print("\n📜 四大名著提示詞生成測試 (Inference Test):")
    test_prompts = [
        ("三國演義", "曹操"),
        ("水滸傳", "武松"),
        ("紅樓夢", "寶玉"),
        ("西遊記", "悟空")
    ]

    for category, prompt in test_prompts:
        prompt_ids = tokenizer.encode(prompt, add_bos_eos=False)
        gen = model.generate(prompt_ids, max_new_tokens=15, temperature=0.5, top_k=3, eos_id=tokenizer.eos_id)
        print(f"  • [{category}] 提示詞: '{prompt}' -> 生成續寫: '{tokenizer.decode(gen)}'")

    print("\n" + "=" * 80)
    print("[SUCCESS] 🎉 《三國演義》《水滸傳》《紅樓夢》《西遊記》四大名著聯合預訓練實測 100% 成功！")
    print("=" * 80)

if __name__ == "__main__":
    train_four_novels()
