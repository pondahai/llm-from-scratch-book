"""
================================================================================
  第 10.4 節程式碼：動態量化 (Dynamic Quantization) 實測

  把 FP32 權重壓成 INT8，實際量三件事：
    1. 檔案大小縮小多少
    2. 推論速度快多少
    3. 生成品質掉多少

  PyTorch 內建的 quantize_dynamic 只需要一行，適用於 nn.Linear——
  而 Transformer 的參數幾乎全在 nn.Linear 裡 (Q/K/V/O 投影 + SwiGLU 三個分支 + LM Head)。
================================================================================
"""

import os
import sys
import time
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from corpus_util import build_tokenizer, PROJECT_ROOT
from code_ch8_decoding import MiniLLM

CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "mini_model.pt")
TMP_FP32 = os.path.join(PROJECT_ROOT, "checkpoints", "_tmp_fp32.pt")
TMP_INT8 = os.path.join(PROJECT_ROOT, "checkpoints", "_tmp_int8.pt")
PROMPTS = ["宴桃園", "武松", "寶玉", "悟空"]


def file_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def timed_generate(model, tokenizer, prompt, n=24, seed=42):
    ids = tokenizer.encode(prompt, add_bos_eos=False)
    torch.manual_seed(seed)
    t0 = time.time()
    out = model.generate(ids, max_new_tokens=n, temperature=0.8,
                         top_k=5, eos_id=tokenizer.eos_id)
    return tokenizer.decode(out), (time.time() - t0) * 1000


def main():
    tok = build_tokenizer()
    model = MiniLLM(vocab_size=tok.vocab_size, d_model=128, n_layers=3,
                    n_heads=4, num_kv_groups=2, hidden_dim=256)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    print("=" * 78)
    print("🔬 [第 10.4 節] INT8 動態量化實測")
    print("=" * 78)

    # ---- 這就是全部的量化程式碼：一行 ----
    qmodel = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    qmodel.eval()

    torch.save(model.state_dict(), TMP_FP32)
    torch.save(qmodel.state_dict(), TMP_INT8)
    fp32_mb, int8_mb = file_mb(TMP_FP32), file_mb(TMP_INT8)

    print(f"\n【1】檔案大小")
    print(f"   FP32 原始 : {fp32_mb:6.2f} MB")
    print(f"   INT8 量化 : {int8_mb:6.2f} MB   → 縮小為 {int8_mb/fp32_mb*100:.0f}%"
          f" (壓縮 {fp32_mb/int8_mb:.2f} 倍)")

    print(f"\n【2】推論速度 (每個提示詞生成 24 tokens，取 3 次平均)")
    tot_f = tot_q = 0.0
    for p in PROMPTS:
        tf = sum(timed_generate(model, tok, p)[1] for _ in range(3)) / 3
        tq = sum(timed_generate(qmodel, tok, p)[1] for _ in range(3)) / 3
        tot_f += tf
        tot_q += tq
        print(f"   '{p}'  FP32 {tf:7.1f} ms | INT8 {tq:7.1f} ms | {tf/tq:.2f}x")
    print(f"   平均加速: {tot_f/tot_q:.2f}x")

    print(f"\n【3】生成品質對照 (相同亂數種子)")
    for p in PROMPTS:
        f_txt, _ = timed_generate(model, tok, p)
        q_txt, _ = timed_generate(qmodel, tok, p)
        same = "完全相同" if f_txt == q_txt else "略有差異"
        print(f"   '{p}' [{same}]")
        print(f"      FP32: {f_txt}")
        print(f"      INT8: {q_txt}")

    os.remove(TMP_FP32)
    os.remove(TMP_INT8)
    print("\n" + "=" * 78)
    print("[SUCCESS] 量化實測完成")
    print("=" * 78)


if __name__ == "__main__":
    main()
