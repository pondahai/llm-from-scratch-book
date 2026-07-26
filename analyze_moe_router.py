"""
================================================================================
  MoE 路由分析 (供書稿 §9.10)

  訓練腳本印出的專家使用率，是【最後一個 batch】的快照——只有 512 個 Token，
  拿來下結論太薄。這支腳本改用整份語料做一次純前向掃描，聚合出穩定的分佈。

  除了各專家的佔比，另外報告一個好用的單一指標：

      有效專家數 = exp(H)，H 為使用率分佈的資訊熵

  四位專家完全平均時為 4.00，全部擠在一位專家時為 1.00。
  它把「坍縮到什麼程度」壓縮成一個可以直接比較的數字。

  用法：
      python analyze_moe_router.py                 # 分析全部 MoE 檢查點
      python analyze_moe_router.py moe_top1_model.pt
================================================================================
"""

import json
import math
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from corpus_util import build_tokenizer, load_full_corpus, PROJECT_ROOT
from code_ch7_rmsnorm_moe import SparseMoEBlock
from code_ch8_decoding import MiniLLM

CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
SEQ_LEN, BATCH = 32, 16
SPEC = (128, 3, 4, 2, 256)
NUM_EXPERTS, TOP_K = 4, 1
DEFAULT_CKPTS = ["moe_top1_model.pt", "moe_top1_balanced_model.pt"]


def effective_experts(fracs):
    """exp(資訊熵)：完全平均為專家數，完全坍縮為 1"""
    h = -sum(p * math.log(p) for p in fracs if p > 0)
    return math.exp(h)


def analyze(ckpt_name, tokenizer, ids, max_batches=None):
    d, L, h, g, ff = SPEC
    model = MiniLLM(vocab_size=tokenizer.vocab_size, d_model=d, n_layers=L,
                    n_heads=h, num_kv_groups=g, hidden_dim=ff,
                    use_moe=True, num_experts=NUM_EXPERTS, top_k=TOP_K)
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, ckpt_name), map_location="cpu"))
    model.eval()

    blocks = [m for m in model.modules() if isinstance(m, SparseMoEBlock)]
    counts = torch.zeros(len(blocks), NUM_EXPERTS)

    n = (len(ids) - 1) // SEQ_LEN
    total_tokens = 0
    with torch.no_grad():
        for b in range(0, n, BATCH):
            rows = [ids[i * SEQ_LEN:(i + 1) * SEQ_LEN] for i in range(b, min(b + BATCH, n))]
            x = torch.tensor(rows, dtype=torch.long)
            model(x)
            for j, blk in enumerate(blocks):
                # last_frac 是比例，乘回 Token 數才能跨 batch 累加
                counts[j] += blk.last_frac * x.numel()
            total_tokens += x.numel()
            if max_batches and b // BATCH + 1 >= max_batches:
                break

    print(f"\n{ckpt_name}   掃描 {total_tokens:,} 個 Token")
    layers = []
    for j in range(len(blocks)):
        frac = (counts[j] / counts[j].sum()).tolist()
        eff = effective_experts(frac)
        layers.append({"layer": j + 1, "fractions": frac, "effective_experts": eff})
        bar = " ".join(f"E{k + 1}={p * 100:5.1f}%" for k, p in enumerate(frac))
        print(f"   第 {j + 1} 層  {bar}   有效專家數 {eff:.2f} / {NUM_EXPERTS}")
    avg = sum(l["effective_experts"] for l in layers) / len(layers)
    print(f"   三層平均有效專家數 {avg:.2f} / {NUM_EXPERTS}")
    return {"checkpoint": ckpt_name, "tokens": total_tokens,
            "layers": layers, "mean_effective_experts": avg}


def main():
    names = [a for a in sys.argv[1:]] or DEFAULT_CKPTS
    names = [n for n in names if os.path.isfile(os.path.join(CKPT_DIR, n))]
    if not names:
        raise SystemExit("找不到任何 MoE 檢查點，請先跑 train_moe_experiment.py")

    tok = build_tokenizer()
    ids = tok.encode(load_full_corpus(), add_bos_eos=False)
    print(f"語料 {len(ids):,} Token | 詞表 V={tok.vocab_size}")

    results = [analyze(n, tok, ids) for n in names]
    out = os.path.join(PROJECT_ROOT, "moe_router_analysis.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\n結果：{os.path.relpath(out, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
