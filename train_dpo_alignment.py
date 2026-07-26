"""
================================================================================
  DPO 直接偏好優化 (Direct Preference Optimization) 實作

  這是「預訓練 → SFT → 對齊」三階段的最後一步。

  DPO 相對於 RLHF 的關鍵簡化：
    RLHF：人類標偏好 → 訓練獎勵模型 → 用強化學習推動主模型   (三個模型、流程不穩)
    DPO ：人類標偏好 → 【直接】用一個損失函數調主模型        (兩個模型、就是普通訓練)

  DPO 的損失函數只做一件事：
    讓「好答案」相對於參考模型變得更可能，
    讓「壞答案」相對於參考模型變得更不可能，
    而且用 sigmoid 把兩者的差距壓成一個可微分的分數。

    L = -log σ( β · [ (logπ(chosen) - logπ_ref(chosen))
                    - (logπ(rejected) - logπ_ref(rejected)) ] )

  參考模型 (π_ref) 是 SFT 後的凍結副本，用來當「不要跑太遠」的錨。
================================================================================
"""

import os
import sys
import json
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from corpus_util import build_tokenizer, PROJECT_ROOT
from code_ch8_decoding import MiniLLM

CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
SFT_CKPT = os.path.join(CKPT_DIR, "mini_chat_model.pt")
DPO_CKPT = os.path.join(CKPT_DIR, "mini_dpo_model.pt")
DATA = os.path.join(PROJECT_ROOT, "data", "dpo_preference_dataset.json")

BETA = 0.1          # DPO 溫度：越大越不准偏離參考模型
LR = 5e-6           # 對齊階段要用很小的學習率，否則會摧毀 SFT 學到的格式
EPOCHS = 30
IGNORE = -100


def sequence_logprob(model, input_ids, target_ids, prompt_len, vocab_size):
    """算出【只有答案那一段】的對數機率總和 (prompt 部分不計)。"""
    logits, _ = model(input_ids)
    logp = F.log_softmax(logits, dim=-1)
    # 取出每個位置上「實際那個 token」的對數機率
    picked = logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # [1, L]
    mask = torch.zeros_like(picked)
    mask[:, prompt_len - 1:] = 1.0      # 與 SFT 相同的錯位邏輯 (見 §10.2)
    return (picked * mask).sum()


def build_pair(tok, item):
    p = tok.encode(item["prompt"], add_bos_eos=False)
    out = {}
    for key in ("chosen", "rejected"):
        a = tok.encode(item[key], add_bos_eos=False) + [tok.eos_id]
        full = p + a
        out[key] = (
            torch.tensor([full[:-1]], dtype=torch.long),
            torch.tensor([full[1:]], dtype=torch.long),
            len(p),
        )
    return out


def main():
    tok = build_tokenizer()
    V = tok.vocab_size
    print("=" * 78)
    print("🎯 [DPO 直接偏好優化] 三階段對齊的最後一步")
    print("=" * 78)
    print(f"  • 詞表大小       : {V}")

    if not os.path.exists(SFT_CKPT):
        raise FileNotFoundError(f"找不到 SFT 模型 {SFT_CKPT}，請先執行 train_sft_chat_model.py")

    def load():
        m = MiniLLM(vocab_size=V, d_model=128, n_layers=3, n_heads=4,
                    num_kv_groups=2, hidden_dim=256)
        m.load_state_dict(torch.load(SFT_CKPT, map_location="cpu"))
        return m

    policy = load()                       # 要被調整的模型
    ref = load()                          # 凍結的參考模型
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    data = json.load(open(DATA, encoding="utf-8"))
    pairs = [build_pair(tok, it) for it in data]
    print(f"  • 偏好三元組     : {len(pairs)} 組 (問題 / 較好答案 / 較差答案)")
    print(f"  • β = {BETA} | lr = {LR} | {EPOCHS} Epochs\n")

    opt = torch.optim.AdamW(policy.parameters(), lr=LR)
    t0 = time.time()
    first = last = None

    for ep in range(1, EPOCHS + 1):
        tot_loss = 0.0
        n_correct = 0
        policy.train()
        for pr in pairs:
            opt.zero_grad()
            terms = {}
            for key in ("chosen", "rejected"):
                x, y, plen = pr[key]
                lp_pol = sequence_logprob(policy, x, y, plen, V)
                with torch.no_grad():
                    lp_ref = sequence_logprob(ref, x, y, plen, V)
                terms[key] = lp_pol - lp_ref     # 相對參考模型的變化量

            margin = terms["chosen"] - terms["rejected"]
            loss = -F.logsigmoid(BETA * margin)
            loss.backward()
            opt.step()
            tot_loss += loss.item()
            if margin.item() > 0:
                n_correct += 1

        avg = tot_loss / len(pairs)
        acc = n_correct / len(pairs) * 100
        if ep == 1:
            first = avg
        last = avg
        if ep % 5 == 0 or ep == 1:
            print(f"   Epoch {ep:2d}/{EPOCHS} | DPO Loss {avg:.4f} | 偏好正確率 {acc:5.1f}%")

    torch.save(policy.state_dict(), DPO_CKPT)
    print(f"\n  ✅ 已存檔 {os.path.basename(DPO_CKPT)} | "
          f"Loss {first:.4f} -> {last:.4f} | 耗時 {time.time()-t0:.1f}s")

    # ---- 對齊前後對照 ----
    print("\n" + "=" * 78)
    print("🔬 [對齊前後對照] 同一個提示詞，SFT 模型 vs DPO 模型")
    print("=" * 78)
    sft = load()
    sft.eval()
    policy.eval()
    probes = ["問：劉備是誰？答：", "問：武松是誰？答：",
              "問：孫悟空是誰？答：", "問：今天天氣如何？答："]
    for q in probes:
        ids = tok.encode(q, add_bos_eos=False)
        torch.manual_seed(42)
        a = tok.decode(sft.generate(ids, max_new_tokens=22, temperature=0.7,
                                    top_k=5, eos_id=tok.eos_id))
        torch.manual_seed(42)
        b = tok.decode(policy.generate(ids, max_new_tokens=22, temperature=0.7,
                                       top_k=5, eos_id=tok.eos_id))
        print(f"\n  提示：{q}")
        print(f"    SFT : {a}")
        print(f"    DPO : {b}")
    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
