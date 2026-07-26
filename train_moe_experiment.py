"""
================================================================================
  MoE 單變數對照實驗 (供書稿 §9.10)

  三組實驗，除了 FFN 結構之外【所有條件完全相同】：
    ① dense      稠密 SwiGLU (現行 Mini 規格)          2,024,832 參數
    ② moe        4 專家 Top-1，無負載平衡損失           2,911,104 參數
    ③ moe_bal    4 專家 Top-1，加上負載平衡損失 α=0.01  2,911,104 參數

  採 Top-1 而非 Top-2 是刻意的：Top-1 時每個 Token 只過一個專家，
  【每 Token 的 FFN 計算量與稠密版完全相同 (98,304)】，
  差別只有總參數量多了 44%。這樣才是「固定計算量、只變容量」的單變數對照；
  若用 Top-2，計算量會是稠密版的兩倍，比較就不公平了 (詳見書稿 §9.10)。

  ①之所以要重跑而不直接引用 §9.9 的數字：原本的訓練沒有固定隨機種子，
  三組必須用同一顆種子才能把差異歸因於結構而非初始化運氣。

  用法：
      python train_moe_experiment.py            # 跑全部三組
      python train_moe_experiment.py dense moe  # 只跑指定的組
================================================================================
"""

import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from corpus_util import build_tokenizer, load_full_corpus, PROJECT_ROOT
from code_ch7_rmsnorm_moe import SparseMoEBlock
from code_ch8_decoding import MiniLLM

CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

SEQ_LEN, BATCH, EPOCHS, LR = 32, 16, 5, 1e-3
SEED = 1234                      # 三組共用，確保初始化與資料順序一致
SPEC = (128, 3, 4, 2, 256)       # Mini 規格：d_model, n_layers, n_heads, kv_groups, d_ffn
NUM_EXPERTS, TOP_K = 4, 1
AUX_ALPHA = 0.01                 # Switch Transformer 論文的預設值
PROMPTS = ["宴桃園", "武松", "寶玉", "悟空"]

ARMS = {
    #  key       檔名                          use_moe  平衡損失
    "dense":    ("moe_ctrl_dense_model.pt",    False,   False),
    "moe":      ("moe_top1_model.pt",          True,    False),
    "moe_bal":  ("moe_top1_balanced_model.pt", True,    True),
}


class TextDataset(Dataset):
    def __init__(self, token_ids, seq_len=SEQ_LEN):
        n = (len(token_ids) - 1) // seq_len
        self.data = [token_ids[i * seq_len:(i + 1) * seq_len + 1] for i in range(n)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        c = torch.tensor(self.data[i], dtype=torch.long)
        return c[:-1], c[1:]


def load_balancing_loss(model):
    """
    Switch Transformer 的負載平衡損失：α · N · Σ f_i · P_i

      f_i = 實際被分配到專家 i 的 Token 比例（來自 argmax，不可微分）
      P_i = Router 給專家 i 的平均機率（可微分）

    兩者相乘才有梯度可傳：哪個專家被超用，就把它的機率往下壓。
    四個專家完全平均時此項最小。
    """
    total = 0.0
    for m in model.modules():
        if isinstance(m, SparseMoEBlock) and m.last_prob_mean is not None:
            total = total + m.num_experts * (m.last_frac * m.last_prob_mean).sum()
    return AUX_ALPHA * total


def expert_usage(model):
    """回傳每一層 MoE 的專家使用比例 f_i（取最近一次前向傳播的統計）"""
    return [m.last_frac.tolist() for m in model.modules()
            if isinstance(m, SparseMoEBlock) and m.last_frac is not None]


def run(arm, tokenizer, corpus):
    ckpt_name, use_moe, use_balance = ARMS[arm]
    d, L, h, g, ff = SPEC

    ids = tokenizer.encode(corpus, add_bos_eos=False)
    torch.manual_seed(SEED)      # 固定初始化
    gen = torch.Generator().manual_seed(SEED)    # 固定每個 Epoch 的洗牌順序
    dl = DataLoader(TextDataset(ids), batch_size=BATCH, shuffle=True, generator=gen)

    model = MiniLLM(vocab_size=tokenizer.vocab_size, d_model=d, n_layers=L,
                    n_heads=h, num_kv_groups=g, hidden_dim=ff,
                    use_moe=use_moe, num_experts=NUM_EXPERTS, top_k=TOP_K)
    n_par = sum(p.numel() for p in model.parameters())
    # 每個 Token 實際跑過的 FFN 參數量：Top-1 時就是單一專家的大小
    ffn_per_expert = 3 * d * ff
    activated = ffn_per_expert * (TOP_K if use_moe else 1) * L

    print(f"\n{'=' * 78}")
    print(f"[{arm}] use_moe={use_moe} 平衡損失={use_balance} | 參數 {n_par:,} | "
          f"每 Token 激活 FFN {activated:,} | 樣本 {len(dl.dataset):,} | {EPOCHS} Epochs")
    print(f"{'=' * 78}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    t0 = time.time()
    losses = []
    model.train()
    for ep in range(1, EPOCHS + 1):
        tot = aux_tot = 0.0
        for x, y in dl:
            opt.zero_grad()
            logits, _ = model(x)
            loss = crit(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            ce = loss.item()
            if use_balance:
                aux = load_balancing_loss(model)
                aux_tot += aux.item()      # .item() 而非 float()，否則 PyTorch 會對
                loss = loss + aux          # 「把帶梯度的張量轉純量」發出警告洗版
            loss.backward()
            opt.step()
            tot += ce                      # 只累計交叉熵，三組的 Loss 才可比
        losses.append(tot / len(dl))
        msg = f"   [{arm}] Epoch {ep}/{EPOCHS}  Loss {losses[-1]:.4f}  ({time.time() - t0:.0f}s)"
        if use_balance:
            msg += f"  aux {aux_tot / len(dl):.4f}"
        print(msg, flush=True)

    elapsed = time.time() - t0
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, ckpt_name))
    print(f"   [{arm}] 已存檔 {ckpt_name} | 耗時 {elapsed / 60:.1f} 分", flush=True)

    usage = expert_usage(model) if use_moe else []
    for i, u in enumerate(usage, 1):
        print(f"   [{arm}] 第 {i} 層專家使用率: "
              + " ".join(f"E{j}={v * 100:.1f}%" for j, v in enumerate(u, 1)), flush=True)

    model.eval()
    samples = {}
    for p in PROMPTS:
        pid = tokenizer.encode(p, add_bos_eos=False)
        torch.manual_seed(42)
        out = model.generate(pid, max_new_tokens=24, temperature=0.8,
                             top_k=5, eos_id=tokenizer.eos_id)
        samples[p] = tokenizer.decode(out)
        print(f"   [{arm}] '{p}' -> {samples[p]!r}", flush=True)

    return dict(arm=arm, params=n_par, activated=activated, epochs=EPOCHS,
                losses=losses, final_loss=losses[-1], minutes=elapsed / 60,
                expert_usage=usage, samples=samples, checkpoint=ckpt_name)


def main():
    wanted = [a for a in sys.argv[1:] if a in ARMS] or list(ARMS)
    tok = build_tokenizer()
    corpus = load_full_corpus()
    print(f"統一詞表 V = {tok.vocab_size} | 全語料 {len(corpus):,} 字 | 種子 {SEED}", flush=True)

    results = []
    for arm in wanted:
        results.append(run(arm, tok, corpus))
        out = os.path.join(PROJECT_ROOT, "moe_experiment_results.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)

    print(f"\n{'=' * 78}\n彙總 (供書稿 §9.10 填表)\n{'=' * 78}")
    for r in results:
        print(f"  {r['arm']:9s} 參數 {r['params']:>10,} | 激活 {r['activated']:>8,} | "
              f"{r['minutes']:6.1f} 分 | Loss {r['losses'][0]:.4f} -> {r['final_loss']:.4f}")
    print(f"\n完整結果：moe_experiment_results.json", flush=True)


if __name__ == "__main__":
    main()
