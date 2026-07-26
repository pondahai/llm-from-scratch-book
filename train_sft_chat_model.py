"""
================================================================================
  MiniLLM SFT 標準指令微調腳本 (Prompt Masking + Answer Only Loss)
  採用大模型業界標準 SFT 規範：
  只計算【回答 (Output)】部分的 CrossEntropyLoss，掩碼 (Mask) 提示詞【問句 (Instruction)】部分。
================================================================================
"""

import sys
import os
import json
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), "code"))
from code_ch1_tokenizer import SimpleBPEStyleTokenizer
from code_ch8_decoding import MiniLLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def train_sft():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    base_ckpt_path = os.path.join(ckpt_dir, "mini_model.pt")
    chat_ckpt_path = os.path.join(ckpt_dir, "mini_chat_model.pt")
    data_path = os.path.join(base_dir, "data", "sft_chat_dataset.json")
    corpus_path = os.path.join(base_dir, "data", "four_great_novels_combined.txt")

    print("=" * 80)
    print("🚀 [標準 SFT 模式啟動] 採用 Prompt Masking 進行精確指令微調...")
    print("=" * 80)

    # 1. 讀取 Tokenizer —— 一律由完整語料建立 (V=6178)，與預訓練檢查點保持一致
    from corpus_util import build_tokenizer
    tokenizer = build_tokenizer()
    print(f"  • Tokenizer 詞表大小: {tokenizer.vocab_size} Tokens")

    # 2. 讀取 SFT Q&A 數據集
    with open(data_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    print(f"  • SFT 問答對總數: {len(qa_data)} 組")

    # 3. 載入 Base 預訓練模型
    state_dict = torch.load(base_ckpt_path, map_location="cpu")
    ckpt_vocab_size = state_dict["embed.embedding.weight"].shape[0]
    model = MiniLLM(vocab_size=ckpt_vocab_size, d_model=128, n_layers=3, n_heads=4, num_kv_groups=2, hidden_dim=256)
    model.load_state_dict(state_dict)
    print(f"  • 成功載入 Base LLM 預訓練權重: {base_ckpt_path}")

    # 4. 構建含 Prompt Masking 的 SFT 訓練張量
    IGNORE_INDEX = -100
    sft_batches = []

    for item in qa_data:
        prompt_str = item["instruction"]
        ans_str = item["output"]
        
        prompt_ids = tokenizer.encode(prompt_str, add_bos_eos=False)
        ans_ids = tokenizer.encode(ans_str, add_bos_eos=False) + [tokenizer.eos_id]
        
        full_ids = prompt_ids + ans_ids
        
        input_ids = full_ids[:-1]
        target_ids = full_ids[1:]
        
        prompt_len = len(prompt_ids)
        target_masked = [IGNORE_INDEX if i < prompt_len - 1 else target_ids[i] for i in range(len(target_ids))]
        
        sft_batches.append((
            torch.tensor([input_ids], dtype=torch.long),
            torch.tensor([target_masked], dtype=torch.long)
        ))

    # 5. 開始 SFT 訓練 Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    print("\n  🎯 開始微調 (Loss 僅在答案 Tokens 上計算，Prompt 100% 掩碼)...")
    model.train()

    for epoch in range(1, 51):
        total_loss = 0.0
        for x, y in sft_batches:
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, ckpt_vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(sft_batches)
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:2d}/50 | Answer-Only SFT 平均 Loss: {avg_loss:.4f}")

    # 6. 保存 SFT Chat Checkpoint
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), chat_ckpt_path)
    print(f"\n  💾 SFT Chat 專用模型權重已成功導出存檔: {chat_ckpt_path}")

    # 7. 比對真實推論測試
    print("\n" + "=" * 80)
    print("🔬 [標準 SFT 對齊效果實證比對] 呼叫真實 PyTorch model.generate() 驗證回答:")
    print("=" * 80)

    test_prompts = [item["instruction"] for item in qa_data]

    model.eval()
    for prompt in test_prompts:
        prompt_ids = tokenizer.encode(prompt, add_bos_eos=False)
        torch.manual_seed(42)
        gen_ids = model.generate(prompt_ids, max_new_tokens=25, temperature=0.1, top_k=1, eos_id=tokenizer.eos_id)
        out_text = tokenizer.decode(gen_ids)
        print(f"  • 用戶輸入 : '{prompt}'")
        print(f"    SFT Chat 對齊後回答: '{out_text}'\n")

    print("=" * 80)
    print("[SUCCESS] 標準 SFT 後訓練與 Chat 專用 Checkpoint 導出 100% 成功！")
    print("=" * 80)

if __name__ == "__main__":
    train_sft()
