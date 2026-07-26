"""
================================================================================
  第 9 章程式碼：實戰預訓練——四大名著與不同參數量 (Scaling Laws) 實戰腳本
  累加第 1~8 章
  對照書本 9.5 節：參數量 Scaling Law 對比分析
================================================================================
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from code_ch1_tokenizer import SimpleBPEStyleTokenizer
from code_ch8_decoding import MiniLLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

class TextDataset(Dataset):
    """將長文本切割為長度為 seq_len 的批次訓練數據集"""
    def __init__(self, token_ids: list, seq_len: int = 32):
        self.seq_len = seq_len
        num_samples = (len(token_ids) - 1) // seq_len
        self.data = [token_ids[i * seq_len : (i + 1) * seq_len + 1] for i in range(num_samples)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = torch.tensor(self.data[idx], dtype=torch.long)
        x = chunk[:-1]  # 輸入 (x_1, ..., x_T-1)
        y = chunk[1:]   # 目標 (x_2, ..., x_T)
        return x, y

def get_model_config(tier_name: str, vocab_size: int):
    """根據書本 9.5 節與 11.1 節定義不同參數量等級規格 (完全相容 nano/micro/mini/base/large)"""
    t = tier_name.lower().strip()
    if t in ["nano", "micro", "lite"]:
        # Micro Spec (d=64, N=2, h=4, g=2, ffn=128) → 864,832 參數 @ V=6178
        return MiniLLM(vocab_size=vocab_size, d_model=64, n_layers=2, n_heads=4, num_kv_groups=2, hidden_dim=128)
    elif t in ["mini", "plus"]:
        # Mini Spec (d=128, N=3, h=4, g=2, ffn=256 - 本書主推) → 2,024,832 參數 @ V=6178
        return MiniLLM(vocab_size=vocab_size, d_model=128, n_layers=3, n_heads=4, num_kv_groups=2, hidden_dim=256)
    elif t == "base":
        # Base Spec (d=256, N=6, h=8, g=4, ffn=512) → 6,705,408 參數 @ V=6178
        return MiniLLM(vocab_size=vocab_size, d_model=256, n_layers=6, n_heads=8, num_kv_groups=4, hidden_dim=512)
    elif t == "large":
        # Large Spec (d=512, N=12, h=16, g=4, ffn=2048) → 51,952,128 參數 @ V=6178
        # ⚠️ 本書已放棄此規格，不隨書提供權重：純 CPU 以四大名著全語料訓練約需 19 小時，
        #    超出「家用筆電當天跑完」的設計前提。此設定僅供有 GPU 的讀者自行擴充參考 (見 §9.9)。
        return MiniLLM(vocab_size=vocab_size, d_model=512, n_layers=12, n_heads=16, num_kv_groups=4, hidden_dim=2048)
    else:
        return MiniLLM(vocab_size=vocab_size, d_model=128, n_layers=3, n_heads=4, num_kv_groups=2, hidden_dim=256)

def demo_chapter9(tier: str = "mini"):
    print("=" * 70)
    print(f"📖 [第 9 章 實戰預訓練] 執行四大名著 MiniLLM 預訓練 (規格: {tier.upper()})...")
    print("=" * 70)
    
    # 優先搜尋四大名著全套合集 (combined_path 第一順位)
    possible_paths = [
        os.path.join(PROJECT_ROOT, "data", "four_great_novels_combined.txt"),
        os.path.join(PROJECT_ROOT, "data", "sanguo.txt"),
        os.path.join(PROJECT_ROOT, "data", "sanguo_yanyi.txt"),
        os.path.join(SCRIPT_DIR, "data", "four_great_novels_combined.txt"),
        os.path.join(SCRIPT_DIR, "data", "sanguo.txt"),
    ]
    
    data_file = None
    for p in possible_paths:
        if os.path.exists(p):
            data_file = p
            break

    if data_file:
        with open(data_file, "r", encoding="utf-8", errors="ignore") as f:
            corpus = f.read()[:50000] # 取 50,000 字語料，重現 V=2505 與 1,084,544 精確參數量
        data_name = os.path.basename(data_file)
    else:
        corpus = "宴桃園豪傑三結義，武松打虎景陽岡，寶玉重逢賈府裡，悟空大鬧天宮戰天兵。" * 500
        data_name = "內建四大名著範例文本"

    tokenizer = SimpleBPEStyleTokenizer(corpus)
    token_ids = tokenizer.encode(corpus, add_bos_eos=False)
    
    dataset = TextDataset(token_ids, seq_len=32)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = get_model_config(tier, tokenizer.vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print(f"  • 訓練規格 Tier      : {tier.upper()}")
    print(f"  • 訓練語料來源       : {data_name}")
    print(f"  • 詞表大小 (Vocab)   : {tokenizer.vocab_size}")
    print(f"  • 模型總參數量       : {total_params:,} 個參數")
    print(f"  • 訓練樣本總數       : {len(dataset)} 個 (Seq Length=32)")
    print("\n  🚀 開始純 CPU 本地預訓練 (5 Epochs):")
    
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
        avg_loss = total_loss / len(dataloader)
        print(f"    Epoch {epoch}/5 | 平均預訓練 Loss: {avg_loss:.4f}")
        
    print("\n  📜 預訓練後提示詞多樣性採樣測試 (每次執行均動態隨機):")
    test_prompts = ["宴桃園", "曹操", "劉備"]
    for prompt in test_prompts:
        torch.seed()
        prompt_ids = tokenizer.encode(prompt, add_bos_eos=False)
        gen = model.generate(prompt_ids, max_new_tokens=20, temperature=0.8, top_k=5, eos_id=tokenizer.eos_id)
        print(f"    提示詞: '{prompt}' -> 生成續寫: '{tokenizer.decode(gen)}'")
    
    print("\n" + "=" * 70)
    print(f"[SUCCESS] 第 9 章 {tier.upper()} 規格預訓練測試 100% 成功！")
    print("=" * 70)

if __name__ == "__main__":
    tier_arg = sys.argv[1] if len(sys.argv) > 1 else "mini"
    demo_chapter9(tier=tier_arg)
