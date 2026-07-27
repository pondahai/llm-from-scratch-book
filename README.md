# 📘 《大語言模型工作原理與實作》— 從心智圖到 PyTorch 積木式打造

> * **作者**：pondahai
> * **協同開發**：Gemini 3.6 Flash · Claude Opus5（校稿與實驗驗證）
> * **進度**：第 0 章 ~ 第 11 章 & 附錄完成，全書 90 頁

用純 CPU、不到 3 百萬參數，從零打造一個真的會寫古文的大語言模型。
**每一塊積木都手寫、每一個數字都實測。**

---

## 🚀 快速開始

```bash
# 1. 安裝 CPU 版 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. 下載模型權重（不在 Git 倉庫裡，見下方說明）
#    到 Releases 下載需要的 .pt 權重檔，直接放進 checkpoints/
#    https://github.com/pondahai/llm-from-scratch-book/releases

# 3. 直接用權重生成文字（不用訓練，秒開）
python server.py
# 開啟 http://localhost:8080
```

想自己從頭訓練？

```bash
python train_corpus_experiment.py    # 預訓練 4 個模型（含 A/B 對照組）
python train_sft_chat_model.py       # SFT 指令對齊
python train_dpo_alignment.py        # DPO 偏好對齊
```

---

## 📦 隨書附的模型權重

> 📥 **權重不在 Git 倉庫裡，請到 [Releases](https://github.com/pondahai/llm-from-scratch-book/releases) 下載**，直接放進 `checkpoints/` 即可（是獨立的 .pt 檔，不需要解壓）。
>
> 九份權重合計約 90 MB。二進位檔一旦進了 Git 歷史就永久拿不掉，之後每次 `clone` 都得整包拖下來，
> 所以改用 Releases 附件發佈。`server.py` 找不到權重時會直接印出下載網址。

全部使用**同一份詞表**（`data/vocab.json`，V=6,178），可互換載入。

| 檔案 | 參數量 | 大小 | 說明 |
| :--- | ---: | ---: | :--- |
| `micro_model.pt` | 864,832 | 3.3 MB | 擴模型軸 · 最小規格 |
| `mini_model.pt` | 2,024,832 | 7.7 MB | **本書主推**（B 組：四大名著全語料） |
| `base_model.pt` | 6,705,408 | 25.6 MB | 擴模型軸 · 最大規格 |
| `sanguo_mini_model.pt` | 2,024,832 | 7.7 MB | A 組對照：只讀《三國演義》5 萬字 |
| `mini_chat_model.pt` | 2,024,832 | 7.7 MB | SFT 指令對齊版 |
| `mini_dpo_model.pt` | 2,024,832 | 7.7 MB | DPO 偏好對齊版 |
| `moe_ctrl_dense_model.pt` | 2,024,832 | 7.7 MB | §9.10 稠密對照組（種子固定 1234） |
| `moe_top1_model.pt` | 2,911,104 | 11.1 MB | §9.10 MoE Top-1，**無平衡損失（坍縮版）** |
| `moe_top1_balanced_model.pt` | 2,911,104 | 11.1 MB | §9.10 MoE Top-1 + 負載平衡損失 |

> 最後三份是 §9.10 那組單變數對照實驗的完整權重。三者除 FFN 結構外所有條件相同，
> 每個 Token 的 FFN 計算量也完全相同（294,912）——下載這三份就能直接驗證書中那張三欄表，
> 不必自己跑三個半小時：
>
> ```bash
> python analyze_moe_router.py   # 逐層專家使用率與有效專家數（2.11 vs 3.99）
> python eval_validation.py      # 在《儒林外史》上重考
> python bench_moe_speed.py      # 交錯量測速度（1.00x / 1.27x / 1.38x）
> ```

> ⚠️ **Large 規格（51.9M）不隨書提供**：純 CPU 以完整語料訓練約需 19 小時，且檔案 184 MB 超過 GitHub 單檔上限。若你有 GPU 想自行訓練，在 `train_corpus_experiment.py` 的 `SPECS` 加回 `"large": (512, 12, 16, 4, 2048)` 即可。

### 為什麼有 `data/vocab.json`？

詞表是從語料**算**出來的。語料檔只要有任何差異（少一本書、換行符不同、編碼不同），算出來的詞表就不一樣，你下載的檢查點就載入不了。

把詞表釘死成一個檔案就沒有這個風險——這也是 Hugging Face 上每個模型旁邊都有 `tokenizer.json` 的原因。
`code/corpus_util.py` 會優先讀它，找不到才從語料重建。

---

## 🧪 書中的兩軸實驗（皆為實測數據）

**擴語料軸**（§9.8）— 同模型、同詞表，只改資料量：

| | A 組（三國 5 萬字） | B 組（四大名著 291 萬字） |
| :--- | :--- | :--- |
| 最終 Loss | 0.2616 | 3.9724 |
| 輸入「武松」 | 劉表、涿、漢禪（硬套三國） | **李逵**、那婦人房裏（水滸） |
| 輸入「悟空」 | 玄德、曹仁、關羽（硬套三國） | **行者道**（西遊） |

**Loss 好看 15 倍的那個，其實只是在默寫。**

**擴模型軸**（§9.9）— 同語料、同輪數：

| 規格 | 參數量 | 耗時 | 訓練 Loss | 驗證困惑度 |
| :--- | ---: | ---: | ---: | ---: |
| Micro | 864,832 | 31.7 分 | 4.35 | 164 |
| Mini | 2,024,832 | 61.9 分 | 3.97 | 144 |
| Base | 6,705,408 | 200.2 分 | 3.50 | 137 |

**MoE 軸**（§9.10）— 同語料、**同計算量**，只改 FFN 結構：

| 組別 | 參數量 | 每 Token 激活 FFN | 訓練 Loss | 驗證 Loss | 有效專家數 | 每步耗時 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 稠密（對照） | 2,024,832 | 294,912 | 3.9623 | 4.9667 | — | 1.00x |
| MoE Top-1 | 2,911,104 | 294,912 | 3.9417 | 4.9611 | 2.11 / 4 | 1.27x |
| MoE + 平衡損失 | 2,911,104 | 294,912 | 3.9046 | 4.9553 | 3.99 / 4 | 1.38x |

**訓練 Loss 改善 0.058，到了沒看過的文本上只剩 0.011——80% 是背出來的。**

> 📏 **驗證集**：《儒林外史》33 萬字（`data/rulin_waishi.txt`），完全沒有參與訓練。
> 未知字率 0.58%，評估時排除。驗證 Loss 的絕對值沒有意義，只比較同一份考卷上的相對高低。
>
> 最極端的對照是 §9.8 的 A 組（只讀《三國演義》5 萬字）：**訓練困惑度 1.3，驗證困惑度 4,139,802。**

測試環境：Intel i7-1185G7（4 核 8 緒）/ 16 GB RAM / **100% 純 CPU，無顯卡**

---

## 📂 資料夾結構

> 本倉庫只收錄隨書配對的程式碼與資料。書稿本體（PDF / EPUB）與排版工具鏈不在版控範圍內，
> 書籍透過 mooPub / Readmoo 獨家發行。

```text
llm_book_project/
├── server.py                            # 全棧 API 伺服器（含權重守門檢查）
├── index.html                           # Web 視覺化研發儀表板
├── train_corpus_experiment.py           # 擴語料 A/B 對照 + 三規格預訓練
├── train_sft_chat_model.py              # SFT 指令微調（Prompt Loss Masking）
├── train_dpo_alignment.py               # DPO 偏好對齊
│
├── code/                                # 逐章積木（純 CPU 驗證通過）
│   ├── corpus_util.py                   # ★ 全專案唯一的語料/詞表入口
│   ├── code_ch1_tokenizer.py            # 第 1 章：分詞器與詞表
│   ├── code_ch2_embedding.py            # 第 2 章：詞嵌入矩陣
│   ├── code_ch3_rope.py                 # 第 3 章：RoPE 旋轉位置編碼
│   ├── code_ch4_attention.py            # 第 4 章：縮放點積注意力
│   ├── code_ch5_gqa_kvcache.py          # 第 5 章：GQA 與 KV Cache
│   ├── code_ch6_swiglu_ffn.py           # 第 6 章：SwiGLU 前饋網路
│   ├── code_ch7_rmsnorm_moe.py          # 第 7 章：RMSNorm / 殘差流 / MoE
│   ├── code_ch8_decoding.py             # 第 8 章：完整 MiniLLM 與自迴歸生成
│   ├── code_ch9_pretraining.py          # 第 9 章：預訓練
│   ├── code_ch10_posttraining.py        # 第 10 章：SFT
│   ├── code_ch10_quantization.py        # 第 10.4 節：INT8 動態量化實測
│   ├── code_ch11_spec.py                # 第 11 章：Model Card 規格分析
│   │
│   │                                    # ↓ §9.6「4-Step 體驗路徑」四階段用到的腳本
│   ├── llm_from_scratch_mindmap.py      # 階段 1：Nano 規格極速初體驗（78,912 參數 / 約 5 秒）
│   ├── train_sanguo_llm.py              # 階段 2：《三國演義》前 2 萬字文風訓練（約 13 秒）
│   ├── train_all_four_novels_live.py    # 階段 3：四大名著合集聯合預訓練
│   └── fetch_wikisource_books.py        # 維基文庫語料自動化爬取腳本
│
├── train_moe_experiment.py              # §9.10 MoE 三組對照實驗
├── analyze_moe_router.py                # §9.10 專家使用率統計（掃全語料）
├── bench_moe_speed.py                   # §9.10 速度交錯量測
├── fetch_validation_corpus.py           # 抓取驗證集《儒林外史》
├── eval_validation.py                   # 在驗證集上評估所有檢查點
│
├── data/
│   ├── vocab.json                       # ★ 統一詞表（V=6,178）
│   ├── four_great_novels_combined.txt   # 四大名著合集（291.5 萬字，公有領域）
│   ├── rulin_waishi.txt                 # ★ 驗證集《儒林外史》（33 萬字，未參與訓練）
│   ├── sft_chat_dataset.json            # SFT 問答資料
│   └── dpo_preference_dataset.json      # DPO 偏好資料（24 組三元組）
│
└── checkpoints/                         # 預訓練權重（空的，請從 Releases 下載後放於此）
```

---

## 📖 授權 (License)

| 對象 | 授權 |
| :--- | :--- |
| **本倉庫的程式碼與模型權重** | [MIT License](LICENSE) — 可自由使用、修改、商用 |
| **四大名著語料** | 取自**維基文庫（Wikisource）**，屬**公有領域**，可自由使用。`code/fetch_wikisource_books.py` 為原始爬取腳本 |
| **書籍內容本身**（文字、插圖、排版） | **著作權所有 · 翻印必究**，不在 MIT 授權範圍內，亦不包含於本倉庫 |

換句話說：**程式碼隨你拿去用，書的內容不行。** 本倉庫只提供隨書配對的可執行程式碼與權重，
書稿本體透過 mooPub / Readmoo 獨家發行。

---

## 🔍 這本書的一個特色：不美化數據

書中多處記錄了「實驗結果與預期不符」的過程，例如：

* **§10.4 量化**：教科書說壓縮到 25%，實測只到 55%，而且**速度反而慢 2.4 倍**
* **§10.5 DPO**：偏好正確率練到 100%，但生成結果一個字都沒變；調高學習率則模型直接崩潰吐空字串

這些都保留了原始數據與成因分析，而不是換成漂亮的範例。
**看模型實際輸出什麼，不要只看訓練曲線**——這是全書重複最多次的一句話。
