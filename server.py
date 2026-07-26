"""
================================================================================
  MiniLLM 全功能前後端實時動態聯動 API 伺服器 (Full-Stack Real-Time API Server)
  支持前端與 Python/PyTorch 後端 100% 動態聯動：
  前端切換哪一個模型規格 (Micro / Mini / Base / Large)，後端就即時載入並執行該 PyTorch 模型推論！
================================================================================
"""

import sys
import os
import json
import time
import math
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

# 強制 stdout/stderr 使用 UTF-8 編碼
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 匯入專案 PyTorch 模型與 Tokenizer
sys.path.append(os.path.join(os.path.dirname(__file__), "code"))
from code_ch8_decoding import MiniLLM
# 詞表一律透過 corpus_util 由【完整語料】建立 (V=6178)，
# 不可在此自行切片自建，否則會與訓練好的權重對不上 (詳見 corpus_util.py 的說明)。
from corpus_util import build_tokenizer, load_full_corpus

# 全域預訓練語料與 Tokenizer 快取
TOKENIZER = None
CORPUS_TEXT = ""

def init_corpus_and_tokenizer():
    global TOKENIZER, CORPUS_TEXT
    if TOKENIZER is not None:
        return
    CORPUS_TEXT = load_full_corpus()
    TOKENIZER = build_tokenizer()
    print(f"✅ [後端初始化] 語料 {len(CORPUS_TEXT):,} 字 | 統一詞表 {TOKENIZER.vocab_size} Tokens", flush=True)

# 初始化
init_corpus_and_tokenizer()

# 後端模型快取 Cache
MODEL_CACHE = {}

def get_real_pytorch_model(tier_name: str):
    """根據前端切換的規格 (Micro/Mini/Base/Large)，動態實例化並快取 PyTorch 模型"""
    global MODEL_CACHE, TOKENIZER
    if tier_name in MODEL_CACHE:
        return MODEL_CACHE[tier_name]

    vocab_size = TOKENIZER.vocab_size

    if tier_name == "micro":
        # Micro 規格 (864,832 參數 @ V=6178)
        model = MiniLLM(vocab_size=vocab_size, d_model=64, n_layers=2, n_heads=4, num_kv_groups=2, hidden_dim=128)
    elif tier_name in ("mini", "mini_chat", "mini_dpo"):
        # Mini 規格 (2,024,832 參數 @ V=6178 - 主推 Base / Chat)
        model = MiniLLM(vocab_size=vocab_size, d_model=128, n_layers=3, n_heads=4, num_kv_groups=2, hidden_dim=256)
    elif tier_name == "base":
        # Base 規格 (6.71M 參數 @ V=6178)
        model = MiniLLM(vocab_size=vocab_size, d_model=256, n_layers=6, n_heads=8, num_kv_groups=4, hidden_dim=512)
    elif tier_name == "sanguo_mini":
        # A 組對照：與 Mini 完全相同的結構，但只用《三國演義》5 萬字訓練
        model = MiniLLM(vocab_size=vocab_size, d_model=128, n_layers=3, n_heads=4, num_kv_groups=2, hidden_dim=256)
    elif tier_name == "moe_mini":
        # §9.10 MoE 對照：與 Mini 同結構，但 FFN 換成 4 專家 Top-1 (2,911,104 參數)。
        # 採 Top-1 是為了讓每個 Token 的 FFN 計算量與 Mini 完全相同 (294,912)，
        # 唯一的變數是容量。權重為加了負載平衡損失的那一版。
        model = MiniLLM(vocab_size=vocab_size, d_model=128, n_layers=3, n_heads=4,
                        num_kv_groups=2, hidden_dim=256,
                        use_moe=True, num_experts=4, top_k=1)
    elif tier_name == "large":
        # Large 規格 (51.9M) 在純 CPU 上訓練需約 19 小時，本書不隨附此檢查點。
        raise ValueError(
            "Large 規格 (51.9M) 未隨書提供權重：在純 CPU 上以完整語料訓練約需 19 小時。\n"
            "本書改以 Micro / Mini / Base 三種規格示範「擴模型」軸，"
            "並以 Mini 的 5 萬字 vs 291 萬字對照示範「擴語料」軸。\n"
            "若你有 GPU 想自行訓練，只需在 train_corpus_experiment.py 的 SPECS 加回 "
            "large: (512, 12, 16, 4, 2048) 即可。"
        )
    else:
        model = MiniLLM(vocab_size=vocab_size, d_model=128, n_layers=3, n_heads=4, num_kv_groups=2, hidden_dim=256)

    # 檢查是否已有預訓練好的 Checkpoint 檔案 (.pt)
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    special = {"mini_chat": "mini_chat_model.pt", "mini_dpo": "mini_dpo_model.pt",
               "moe_mini": "moe_top1_balanced_model.pt"}
    ckpt_filename = special.get(tier_name, f"{tier_name}_model.pt")
    ckpt_path = os.path.join(ckpt_dir, ckpt_filename)

    # 權重載入必須成功；絕不可退回隨機權重繼續服務。
    # 未訓練的模型仍會「正常」回應，只是輸出全是亂碼 (詳見書中 9.7 節)，
    # 若在此處靜默吞掉例外，讀者會誤以為是模型能力不足，而不是權重根本沒載入。
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"找不到 {tier_name.upper()} 規格的權重檔：{ckpt_path}\n"
            f"隨書權重不隨 Git 倉庫發佈 (檔案較大)，請至 GitHub Releases 下載後解壓到 checkpoints/：\n"
            f"  https://github.com/pondahai/llm-from-scratch-mindmap/releases\n"
            f"或自行訓練：`python train_corpus_experiment.py` (micro/mini/base 三規格與 A/B 對照組)、\n"
            f"`python train_sft_chat_model.py` (Chat 規格)、`python train_dpo_alignment.py` (DPO 規格)。"
        )

    state_dict = torch.load(ckpt_path, map_location="cpu")
    ckpt_vocab = state_dict["embed.embedding.weight"].shape[0]
    if ckpt_vocab != vocab_size:
        raise ValueError(
            f"詞表大小不一致，拒絕載入 {ckpt_path}：\n"
            f"  • Checkpoint 訓練時的詞表 = {ckpt_vocab}\n"
            f"  • 目前 Tokenizer 的詞表   = {vocab_size}\n"
            f"這代表這份 Checkpoint 不是用目前的詞表訓練的。\n"
            f"本專案的詞表已釘死於 data/vocab.json (V = 6,178)，一律由 code/corpus_util.py 載入；\n"
            f"請確認 data/vocab.json 未被更動，或改用 GitHub Releases 上與本版本配對的權重檔。"
        )

    model.load_state_dict(state_dict)
    print(f"📦 [載入已訓練權重 Checkpoint] 成功載入 {ckpt_path} (詞表 {ckpt_vocab})", flush=True)

    model.eval()
    MODEL_CACHE[tier_name] = model
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🚀 [後端動態載入 PyTorch 模型] 規格: {tier_name.upper()} | 參數量: {total_params:,} 個參數", flush=True)
    return model

# 推論本身要序列化：torch.manual_seed 是行程層級的全域狀態，
# 兩個請求同時跑會互相打亂隨機序列，破壞「同種子必得同結果」的可複現性。
# 執行緒化只是為了不讓閒置連線卡住 accept 迴圈，不是為了並行推論。
GEN_LOCK = threading.Lock()

class LinkedLLMRequestHandler(BaseHTTPRequestHandler):
    # 瀏覽器常會先開一條 TCP 連線卻不送請求（預連線）。沒有逾時的話，
    # 處理該連線的執行緒會永遠卡在讀請求行上；配合下面的 ThreadingHTTPServer，
    # 這種連線最多佔用一條執行緒 30 秒後就被回收。
    timeout = 30

    def do_GET(self):
        try:
            req_path = self.path.split("?")[0]
            if req_path in ("/", "/index.html"):
                file_path = os.path.join(os.path.dirname(__file__), "index.html")
                with open(file_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                self.wfile.flush()
            else:
                file_path = os.path.join(os.path.dirname(__file__), req_path.lstrip("/"))
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    with open(file_path, "rb") as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                    self.wfile.flush()
                else:
                    self.send_error(404, "File Not Found")
        except Exception as e:
            traceback.print_exc()

    def do_POST(self):
        print(f"📥 [收到 POST 請求] Path: {self.path}", flush=True)
        try:
            if self.path == "/api/generate":
                length = int(self.headers.get("content-length", self.headers.get("Content-Length", 0)))
                body = self.rfile.read(length)
                payload = json.loads(body.decode("utf-8"))

                prompt = payload.get("prompt", "宴桃園")
                tier = payload.get("tier", "mini")
                temperature = float(payload.get("temperature", 0.8))
                top_k = int(payload.get("top_k", 5))
                max_tokens = int(payload.get("max_tokens", 30))

                seed_raw = payload.get("seed", None)

                print(f"   參數: prompt='{prompt}', tier='{tier}', temp={temperature}, top_k={top_k}, seed={seed_raw}", flush=True)
                start_time = time.time()

                # 1. 用真實 Tokenizer 將 Prompt 轉為 Token IDs
                prompt_ids = TOKENIZER.encode(prompt, add_bos_eos=False)

                # 2~4 全程持鎖：載入模型會動到 MODEL_CACHE，播種與生成則共用
                #    行程層級的 RNG，這三步之間插進任何一個並行請求都會出錯。
                with GEN_LOCK:
                    # 2. 取得真實 PyTorch 後端模型
                    model = get_real_pytorch_model(tier)
                    total_params = sum(p.numel() for p in model.parameters())

                    # 3. 設定隨機種子——必須緊貼 generate() 之前。
                    #    模型首次載入時的隨機初始化會消耗 RNG，若在取得模型前播種，
                    #    冷啟動的第一次請求會與之後（走 MODEL_CACHE）的結果不一致。
                    #    未指定 seed 時隨機播種，但一律回報實際種子，讓任何一次生成都能複現。
                    if seed_raw is None or seed_raw == "":
                        seed = torch.seed()          # 隨機播種，並回傳實際使用的種子
                    else:
                        seed = int(seed_raw)
                        torch.manual_seed(seed)      # 指定種子：同參數必得同結果

                    # 4. 呼叫真實 PyTorch generate 推論
                    gen_ids = model.generate(
                        prompt_ids=prompt_ids,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        eos_id=TOKENIZER.eos_id
                    )

                # 5. 反解碼為文字與 Token 列表
                generated_text = TOKENIZER.decode(gen_ids)
                token_strings = [TOKENIZER.id2token.get(idx, "<UNK>") for idx in gen_ids if idx not in (TOKENIZER.pad_id, TOKENIZER.bos_id, TOKENIZER.eos_id)]

                latency_ms = int((time.time() - start_time) * 1000)
                print(f"✨ [生成完成] 結果: '{generated_text}' | 耗時: {latency_ms} ms", flush=True)

                response_data = {
                    "status": "success",
                    "prompt": prompt,
                    "tier": tier,
                    "total_params": f"{total_params:,}",
                    "generated_text": generated_text,
                    "tokens": token_strings,
                    "latency_ms": latency_ms,
                    "seed": seed
                }

                response_bytes = json.dumps(response_data, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(response_bytes)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(response_bytes)
                self.wfile.flush()
            else:
                self.send_error(404, "Endpoint not found")
        except Exception as e:
            traceback.print_exc()
            err_bytes = json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False).encode("utf-8")
            try:
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(err_bytes)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(err_bytes)
                self.wfile.flush()
            except Exception:
                pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

def run_server(port=8080):
    server_address = ("", port)
    # 必須用 ThreadingHTTPServer，不能用單執行緒的 HTTPServer：
    # 單執行緒版本一次只服務一條連線，瀏覽器的預連線（開了 TCP 卻遲遲不送請求）
    # 會讓 accept 迴圈整個停住，行程還活著、port 還在 listen，但頁面就是連不進來。
    httpd = ThreadingHTTPServer(server_address, LinkedLLMRequestHandler)
    httpd.daemon_threads = True
    print("=" * 80, flush=True)
    print(f"🔥 [前後端實時動態聯動 API 伺服器已啟動] Listening on http://localhost:{port}", flush=True)
    print(f"👉 切換模型或輸入提示詞，前端將直接呼叫 PyTorch 後端實時推論！", flush=True)
    print("=" * 80, flush=True)
    httpd.serve_forever()

if __name__ == "__main__":
    run_server(8080)
