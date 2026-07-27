# 🗺️ 隨書程式碼倉庫 · 後續規劃 (Roadmap)

本檔只記錄**這個倉庫**的後續工作。書稿本身的進度與出版事宜不在此列。

書中已實作的部分見 [README.md](README.md)；每一項規劃都附上它對應到書裡的哪一節，
方便你判斷需要先讀哪些章節。

---

## 已完成

### 💬 SFT 指令微調與 DPO 偏好對齊（§10.1 ~ §10.3）

- 問答對指令資料集 `data/sft_chat_dataset.json`
- SFT 微調腳本 `train_sft_chat_model.py`（讀取 Base 權重 `mini_model.pt`，Prompt Loss Masking）
- DPO 偏好對齊 `train_dpo_alignment.py` 與 `data/dpo_preference_dataset.json`（24 組三元組）
- 對齊前後的實證比對：未對齊的 Base 只會續寫小說（`昔黃巾餘人...`），
  SFT 之後才會正面回答（`劉備字玄德，是蜀漢開國皇帝。`）

### 🧬 MoE 對照實驗（§9.10）

- `train_moe_experiment.py` 三組單變數對照（稠密 / MoE Top-1 / MoE + 負載平衡損失）
- `analyze_moe_router.py` 掃全語料的逐層專家使用率
- `bench_moe_speed.py` 訓練與生成的交錯速度量測
- `eval_validation.py` 在《儒林外史》驗證集上重考所有檢查點

---

## 規劃中

### ⚡ 4-bit / 8-bit 量化（延伸 §10.4）

書中 `code/code_ch10_quantization.py` 已做 INT8 動態量化的實測
（壓到 55%、速度反而慢 2.4 倍，成因分析見 §10.4）。
下一步是補上權重層級的 INT4/INT8 靜態量化示範，看看在純 CPU 上
記憶體頻寬的瓶頸能不能真的換到速度。

### 🚀 投機解碼 (Speculative Decoding)

以 `micro_model.pt`（864,832 參數）當草稿模型，加速 `base_model.pt`（6,705,408）的生成。
兩者共用同一份詞表 `data/vocab.json`，天生就能配對，是這個倉庫少見的現成優勢。

> 原本規劃用 `large_model.pt` 當目標模型，已隨 Large 規格一併放棄
> （純 CPU 訓練約需 19 小時、檔案 184 MB 超過 GitHub 單檔上限，見 README）。

### 📦 GGUF 匯出與 `llama.cpp` 執行

編寫權重轉換腳本，把 MiniLLM 的 `state_dict` 匯出成 GGUF，
用 `llama.cpp` 原生 C++ 跑起來。這件事的價值在於驗證書中的架構描述是否精確——
GGUF 要求你把每一個張量的名字、形狀與排列講清楚，含糊不得。

---

## 歡迎的貢獻

這個倉庫的定位是**教學用的最小可讀實作**，所以 PR 的取捨標準跟一般專案不同：

- ✅ 修正錯誤、補上讓程式更好懂的註解、修正書中程式碼與倉庫不一致之處
- ✅ 在不同平台（Linux / macOS / 其他 CPU）上的相容性修正
- ⚠️ 效能最佳化請先開 Issue 討論——**如果會讓程式碼變難讀，通常不會併入**
- ❌ 引入新的重量級相依套件（本倉庫刻意只依賴 PyTorch）
