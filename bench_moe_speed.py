"""
================================================================================
  MoE 速度受控量測 (供書稿 §9.10)

  【為什麼需要這支腳本】
  train_moe_experiment.py 印出的「總耗時」不可靠：三組前後跨了四個多小時，
  期間機器的負載、時脈與散熱狀態都不同，量到的差異可能只是機器狀態的差異。
  要比較「MoE 比稠密慢多少」，必須讓三組在【同一段時間、同一個機器狀態】下
  交錯執行。

  作法：三組輪流各跑一輪，重複多輪 (ABC ABC ABC)，取每步耗時的中位數。
  這樣即使中途時脈變動，也會平均落在三組身上，不會系統性地偏袒任何一組。

  用法：
      python bench_moe_speed.py
================================================================================
"""

import json
import os
import statistics
import sys
import time

import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from corpus_util import build_tokenizer, load_full_corpus, PROJECT_ROOT
from code_ch8_decoding import MiniLLM
from train_moe_experiment import load_balancing_loss

SEQ_LEN, BATCH, LR = 32, 16, 1e-3
SPEC = (128, 3, 4, 2, 256)
ROUNDS, STEPS, WARMUP = 3, 40, 5
PROMPT = "宴桃園"

CONFIGS = {
    "dense":   dict(use_moe=False, balance=False),
    "moe":     dict(use_moe=True,  balance=False),
    "moe_bal": dict(use_moe=True,  balance=True),
}


def build(cfg, V):
    d, L, h, g, ff = SPEC
    torch.manual_seed(1234)
    return MiniLLM(vocab_size=V, d_model=d, n_layers=L, n_heads=h,
                   num_kv_groups=g, hidden_dim=ff,
                   use_moe=cfg["use_moe"], num_experts=4, top_k=1)


def train_steps(model, cfg, batches, V):
    """回傳每一步的耗時 (秒)，前 WARMUP 步捨棄"""
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    model.train()
    times = []
    for i, (x, y) in enumerate(batches):
        t0 = time.perf_counter()
        opt.zero_grad()
        logits, _ = model(x)
        # y 來自切片，記憶體不連續，要用 reshape 而非 view
        loss = crit(logits.view(-1, V), y.reshape(-1))
        if cfg["balance"]:
            loss = loss + load_balancing_loss(model)
        loss.backward()
        opt.step()
        times.append(time.perf_counter() - t0)
    return times[WARMUP:]


def gen_time(model, prompt_ids, eos_id):
    model.eval()
    torch.manual_seed(42)
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(prompt_ids, max_new_tokens=24, temperature=0.8, top_k=5, eos_id=eos_id)
    return time.perf_counter() - t0


def main():
    tok = build_tokenizer()
    ids = tok.encode(load_full_corpus(), add_bos_eos=False)
    V = tok.vocab_size

    n = STEPS
    batches = []
    for b in range(n):
        rows = [ids[(b * BATCH + k) * SEQ_LEN:(b * BATCH + k + 1) * SEQ_LEN + 1]
                for k in range(BATCH)]
        c = torch.tensor(rows, dtype=torch.long)
        batches.append((c[:, :-1], c[:, 1:]))

    models = {k: build(c, V) for k, c in CONFIGS.items()}
    prompt_ids = tok.encode(PROMPT, add_bos_eos=False)

    step_times = {k: [] for k in CONFIGS}
    gen_times = {k: [] for k in CONFIGS}

    print(f"交錯量測：{ROUNDS} 輪 × {STEPS} 步（每輪捨棄前 {WARMUP} 步暖機）\n")
    for r in range(1, ROUNDS + 1):
        for name, cfg in CONFIGS.items():      # 每一輪三組都跑，機器狀態才公平
            step_times[name] += train_steps(models[name], cfg, batches, V)
            gen_times[name].append(gen_time(models[name], prompt_ids, tok.eos_id))
        print(f"  第 {r}/{ROUNDS} 輪完成", flush=True)

    base = statistics.median(step_times["dense"])
    base_gen = statistics.median(gen_times["dense"])
    print(f"\n{'組別':<9}{'每步中位數':>12}{'相對稠密':>10}{'生成 24 字':>12}{'相對稠密':>10}")
    results = {}
    for name in CONFIGS:
        m = statistics.median(step_times[name])
        gm = statistics.median(gen_times[name])
        results[name] = {"step_median_s": m, "step_ratio": m / base,
                         "gen_median_s": gm, "gen_ratio": gm / base_gen,
                         "samples": len(step_times[name])}
        print(f"{name:<9}{m * 1000:>10.1f} ms{m / base:>9.2f}x{gm * 1000:>10.1f} ms{gm / base_gen:>9.2f}x")

    out = os.path.join(PROJECT_ROOT, "moe_speed_benchmark.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"rounds": ROUNDS, "steps_per_round": STEPS,
                   "batch": BATCH, "seq_len": SEQ_LEN, "results": results},
                  f, ensure_ascii=False, indent=1)
    print(f"\n結果：{os.path.relpath(out, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
