"""
================================================================================
  擴語料實驗 + 各規格檢查點訓練 (統一詞表 V=6178)

  產出四個檢查點：
    A 組  sanguo_mini_model.pt  Mini @《三國演義》5 萬字   50 Epochs
    B 組  mini_model.pt         Mini @ 四大名著 291 萬字    5 Epochs
          micro_model.pt        Micro @ 四大名著 291 萬字   5 Epochs
          base_model.pt         Base  @ 四大名著 291 萬字   5 Epochs

  A 與 B 使用【完全相同的詞表與模型結構】，唯一差異是訓練資料量 (58 倍)，
  因此兩者的 Loss 與生成品質可以直接比較。
  Large 規格 (48.2M) 在純 CPU 上需約 19 小時，本實驗不納入。
================================================================================
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from corpus_util import build_tokenizer, load_full_corpus, load_sanguo_sample, PROJECT_ROOT
from code_ch8_decoding import MiniLLM

CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
SEQ_LEN, BATCH = 32, 16


class TextDataset(Dataset):
    def __init__(self, token_ids, seq_len=SEQ_LEN):
        n = (len(token_ids) - 1) // seq_len
        self.data = [token_ids[i * seq_len:(i + 1) * seq_len + 1] for i in range(n)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        c = torch.tensor(self.data[i], dtype=torch.long)
        return c[:-1], c[1:]


SPECS = {   # name: (d_model, n_layers, n_heads, kv_groups, d_ffn)
    "micro": (64, 2, 4, 2, 128),
    "mini":  (128, 3, 4, 2, 256),
    "base":  (256, 6, 8, 4, 512),
}

PROMPTS = ["宴桃園", "武松", "寶玉", "悟空"]


def train(tag, ckpt_name, spec, corpus, epochs, tokenizer):
    d, L, h, g, ff = spec
    ids = tokenizer.encode(corpus, add_bos_eos=False)
    dl = DataLoader(TextDataset(ids), batch_size=BATCH, shuffle=True)
    model = MiniLLM(vocab_size=tokenizer.vocab_size, d_model=d, n_layers=L,
                    n_heads=h, num_kv_groups=g, hidden_dim=ff)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*78}\n[{tag}] 語料 {len(corpus):,} 字 | 樣本 {len(dl.dataset):,} | "
          f"參數 {n_par:,} | {epochs} Epochs\n{'='*78}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    t0 = time.time()
    first = last = None
    model.train()
    for ep in range(1, epochs + 1):
        tot = 0.0
        for x, y in dl:
            opt.zero_grad()
            logits, _ = model(x)
            loss = crit(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            loss.backward()
            opt.step()
            tot += loss.item()
        last = tot / len(dl)
        if ep == 1:
            first = last
        print(f"   [{tag}] Epoch {ep}/{epochs}  Loss {last:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    path = os.path.join(CKPT_DIR, ckpt_name)
    torch.save(model.state_dict(), path)
    print(f"   [{tag}] 已存檔 {ckpt_name} | 起始 {first:.4f} -> 最終 {last:.4f} "
          f"| 耗時 {elapsed:.1f}s ({elapsed/60:.1f} 分)", flush=True)

    model.eval()
    for p in PROMPTS:
        pid = tokenizer.encode(p, add_bos_eos=False)
        torch.manual_seed(42)
        out = model.generate(pid, max_new_tokens=24, temperature=0.8,
                             top_k=5, eos_id=tokenizer.eos_id)
        print(f"   [{tag}] '{p}' -> {tokenizer.decode(out)!r}", flush=True)

    return dict(tag=tag, params=n_par, epochs=epochs, first=first,
                last=last, seconds=elapsed, chars=len(corpus))


def main():
    tok = build_tokenizer()
    full = load_full_corpus()
    sanguo = load_sanguo_sample(50000)
    print(f"統一詞表 V = {tok.vocab_size}", flush=True)
    print(f"全語料 {len(full):,} 字 | A 組 {len(sanguo):,} 字", flush=True)

    results = []
    results.append(train("A·三國5萬", "sanguo_mini_model.pt", SPECS["mini"], sanguo, 50, tok))
    results.append(train("B·全語料MINI", "mini_model.pt", SPECS["mini"], full, 5, tok))
    results.append(train("MICRO全語料", "micro_model.pt", SPECS["micro"], full, 5, tok))
    results.append(train("BASE全語料", "base_model.pt", SPECS["base"], full, 5, tok))

    print(f"\n{'='*78}\n彙總 (供書稿 §9.8 / 擴語料實驗填表)\n{'='*78}", flush=True)
    for r in results:
        print(f"  {r['tag']:14s} 參數 {r['params']:>10,} | 語料 {r['chars']:>9,} 字 | "
              f"{r['epochs']:>2} Ep | {r['seconds']/60:6.1f} 分 | "
              f"Loss {r['first']:.4f} -> {r['last']:.4f}", flush=True)


if __name__ == "__main__":
    main()
