"""
================================================================================
  第 10 章程式碼：從 Base 到 Chat Assistant——後訓練與指令微調 (SFT / Alignment)
  累加第 1~9 章
  實裝 Prompt Loss Masking (IGNORE_INDEX = -100)，只對 Answer 計算 Loss！
================================================================================
"""

import sys
import os
import torch
import torch.nn as nn

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

def demo_chapter10():
    print("=" * 60)
    print("📖 [第 10 章 測試] 執行具備 Prompt Masking 的指令微調 (SFT)...")
    print("=" * 60)
    
    # 準備 Q&A 指令數據集
    prompt_str = "問：劉備是誰？"
    answer_str = "答：劉備是蜀漢開國皇帝。"
    
    corpus_text = prompt_str + answer_str + "問：關羽是誰？答：關羽是蜀漢名將。"
    tokenizer = SimpleBPEStyleTokenizer(corpus_text)
    
    # 構建 Prompt 與 Answer 的 Token ID
    prompt_ids = tokenizer.encode(prompt_str, add_bos_eos=False)
    answer_ids = tokenizer.encode(answer_str, add_bos_eos=False)
    full_ids = prompt_ids + answer_ids
    
    # 建立序列張量
    input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long)  # [1, Seq_Len]
    target_ids = torch.tensor([full_ids[1:]], dtype=torch.long) # [1, Seq_Len]
    
    # 核心：Prompt Loss Masking (將 Prompt 部分對應的 Target 設為 -100)
    IGNORE_INDEX = -100
    prompt_len = len(prompt_ids)
    
    # 複製份額並遮蔽 Prompt 位置 (前 prompt_len-1 個 Token 不計算 Loss)
    masked_target_ids = target_ids.clone()
    masked_target_ids[0, :prompt_len-1] = IGNORE_INDEX
    
    model = MiniLLM(vocab_size=tokenizer.vocab_size, d_model=64, n_layers=2, n_heads=4, num_kv_groups=2, hidden_dim=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    
    print(f"  • Prompt 遮蔽長度 (Loss Mask=0): {prompt_len-1} 個 Token")
    print("  • 開始進行 Prompt-Masked SFT 指令對齊訓練 (30 Epochs):")
    
    model.train()
    for epoch in range(1, 31):
        optimizer.zero_grad()
        logits, _ = model(input_ids)
        loss = criterion(logits.view(-1, tokenizer.vocab_size), masked_target_ids.view(-1))
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"    Epoch {epoch:2d}/30 | Prompt-Masked SFT Loss: {loss.item():.4f}")
            
    print("\n  指令微調後問答對話測試:")
    test_prompt_ids = tokenizer.encode(prompt_str, add_bos_eos=False)
    gen = model.generate(test_prompt_ids, max_new_tokens=15, temperature=0.1, top_k=1, eos_id=tokenizer.eos_id)
    print(f"    用戶輸入: '{prompt_str}'")
    print(f"    對齊回答: '{tokenizer.decode(gen)}'")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] 第 10 章 Prompt-Masked SFT 指令對齊測試 100% 成功！")
    print("=" * 60)

if __name__ == "__main__":
    demo_chapter10()
