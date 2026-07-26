"""
================================================================================
  驗證集評估 (供書稿 §9.8 / §9.9 / §9.10)

  在【模型從未見過的文本】上計算 Loss。訓練 Loss 低可能只是把訓練資料背起來，
  驗證 Loss 才能分辨「學會」與「背熟」。

  驗證集：data/rulin_waishi.txt（《儒林外史》，見 fetch_validation_corpus.py）

  兩個必須注意的地方：

  1. **未知字**。詞表 V=6,178 是從四大名著算出來的，《儒林外史》裡沒出現過的
     字會變成 <UNK>。若把這些位置算進去，量到的其實是「模型多會預測 <UNK>」，
     不是語言能力。因此【目標為 <UNK> 的位置一律排除】，並回報未知字比例。

  2. **絕對值沒有意義**。驗證 Loss 一定比訓練 Loss 高，因為文本沒看過、
     作者與文風也不同。有意義的是【同一份驗證集上、不同模型之間的高低】。

  用法：
      python eval_validation.py                    # 評估所有已知檢查點
      python eval_validation.py mini_model.pt      # 只評估指定的
================================================================================
"""

import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from corpus_util import build_tokenizer, PROJECT_ROOT
from code_ch8_decoding import MiniLLM

CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
VAL_FILE = os.path.join(PROJECT_ROOT, "data", "rulin_waishi.txt")
SEQ_LEN, BATCH = 32, 32

# 檢查點 -> (規格, 是否 MoE)。規格為 (d_model, n_layers, n_heads, kv_groups, d_ffn)
MICRO, MINI, BASE = (64, 2, 4, 2, 128), (128, 3, 4, 2, 256), (256, 6, 8, 4, 512)
CKPTS = {
    "micro_model.pt":              (MICRO, False),
    "mini_model.pt":               (MINI,  False),
    "base_model.pt":               (BASE,  False),
    "sanguo_mini_model.pt":        (MINI,  False),
    "moe_ctrl_dense_model.pt":     (MINI,  False),
    "moe_top1_model.pt":           (MINI,  True),
    "moe_top1_balanced_model.pt":  (MINI,  True),
}


def load_val_ids(tok):
    text = open(VAL_FILE, encoding="utf-8", errors="ignore").read()
    ids = tok.encode(text, add_bos_eos=False)
    unk = sum(1 for i in ids if i == tok.unk_id)
    return text, ids, unk


def evaluate(ckpt, spec, use_moe, tok, ids):
    d, L, h, g, ff = spec
    model = MiniLLM(vocab_size=tok.vocab_size, d_model=d, n_layers=L, n_heads=h,
                    num_kv_groups=g, hidden_dim=ff,
                    use_moe=use_moe, num_experts=4, top_k=1)
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, ckpt), map_location="cpu"))
    model.eval()

    n = (len(ids) - 1) // SEQ_LEN
    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for b in range(0, n, BATCH):
            rows = [ids[i * SEQ_LEN:(i + 1) * SEQ_LEN + 1]
                    for i in range(b, min(b + BATCH, n))]
            c = torch.tensor(rows, dtype=torch.long)
            x, y = c[:, :-1], c[:, 1:]
            logits, _ = model(x)
            # ignore_index：目標是 <UNK> 的位置不計入，否則量到的是「多會猜 UNK」
            loss = F.cross_entropy(logits.reshape(-1, tok.vocab_size), y.reshape(-1),
                                   ignore_index=tok.unk_id, reduction="sum")
            total_loss += loss.item()
            total_tok += int((y != tok.unk_id).sum())
    mean = total_loss / total_tok
    return {"checkpoint": ckpt, "val_loss": mean, "val_ppl": math.exp(mean),
            "scored_tokens": total_tok}


def main():
    if not os.path.isfile(VAL_FILE):
        raise SystemExit("找不到驗證集，請先執行 fetch_validation_corpus.py")

    tok = build_tokenizer()
    text, ids, unk = load_val_ids(tok)
    print(f"驗證集《儒林外史》{len(text):,} 字 | 詞表 V={tok.vocab_size}")
    print(f"未知字 {unk:,} 個（{unk / len(ids) * 100:.2f}%）——這些位置一律排除，不計入 Loss\n")

    wanted = [a for a in sys.argv[1:]] or list(CKPTS)
    results = []
    for name in wanted:
        if name not in CKPTS:
            print(f"  ⚠ 略過未知的檢查點 {name}")
            continue
        if not os.path.isfile(os.path.join(CKPT_DIR, name)):
            print(f"  ⚠ 找不到 {name}，略過")
            continue
        spec, use_moe = CKPTS[name]
        r = evaluate(name, spec, use_moe, tok, ids)
        results.append(r)
        print(f"  {name:30s} 驗證 Loss {r['val_loss']:.4f}  困惑度 {r['val_ppl']:7.1f}", flush=True)

    out = os.path.join(PROJECT_ROOT, "validation_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"validation_set": os.path.basename(VAL_FILE),
                   "chars": len(text), "unk_rate": unk / len(ids),
                   "results": results}, f, ensure_ascii=False, indent=1)
    print(f"\n結果：{os.path.relpath(out, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
